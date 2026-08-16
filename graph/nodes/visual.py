"""Visual Planning Agent + Asset/Visual Agent — real logic.

visual_planning_node routes each beat to real stock footage (BROLL) or an animated-mascot scene
(AI_IMAGE) based on content — concrete/real-world beats get stock search keywords, abstract/
conceptual beats get an AI-image prompt (used only as backdrop-scene context; the mascot itself
comes from the shared character pack, not a per-beat generation). asset_visual_node sources
accordingly, falling back from stock -> mascot character animation -> a "no asset" placeholder
(never crashes the run for a sourcing miss; video_assembly_node renders a solid-color placeholder
clip for beats with no asset).
"""
import httpx
from pydantic import BaseModel, Field

from agents.schemas.script import ScriptOutput
from agents.schemas.visual import (
    AssetOutput,
    AssetType,
    SceneType,
    SceneVisualPlan,
    SourcedAsset,
    VisualPlanOutput,
)
from core.llm import call_structured
from core.logging import get_logger
from graph.nodes._helpers import log_and_trace
from graph.state import PipelineState
from tools import character_assets as character_assets_tool
from tools import stock_media as stock_media_tool
from tools.media_paths import media_path

logger = get_logger("graph.nodes.visual")

STAGE_PLAN = "visual_planning"
STAGE_ASSET = "asset_visual"


# --- Visual Planning: LLM routes each beat to stock footage or an AI-image prompt ---


class _ScenePlanDraft(BaseModel):
    beat_index: int
    scene_type: SceneType
    description: str
    search_keywords: list[str] = Field(default_factory=list)
    ai_image_prompt: str | None = None
    transition: str = "cut"


class _VisualPlanDraft(BaseModel):
    scenes: list[_ScenePlanDraft]


_VISUAL_PLANNING_SYSTEM_PROMPT = (
    "You are a video visual director. For each script beat, choose how it should be shown: BROLL "
    "(real-world stock footage/photos) for concrete, demonstrable, real-world content — give 2-5 "
    "generic search_keywords good for stock-footage search (avoid proper nouns/brand names that "
    "won't have stock matches). AI_IMAGE for abstract, conceptual, or hard-to-film content — give "
    "a descriptive ai_image_prompt instead. Only use CHART or MOTION_GRAPHIC for beats that "
    "explicitly present numeric/statistical data; prefer AI_IMAGE for anything else abstract, "
    "since no chart-rendering pipeline exists downstream of this plan."
)


def visual_planning_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_PLAN, "start")

    script = ScriptOutput.model_validate(state["script_output"])
    beats_listing = "\n".join(
        f'- beat_index={b.index} type={b.beat_type} voiceover="{b.voiceover_text}" visual_cue="{b.visual_cue}"'
        for b in script.beats
    )
    draft = call_structured(f"Script beats:\n{beats_listing}", _VisualPlanDraft, system=_VISUAL_PLANNING_SYSTEM_PROMPT)

    valid_indices = {b.index for b in script.beats}
    scenes = []
    for scene_draft in draft.scenes:
        if scene_draft.beat_index not in valid_indices:
            logger.warning("visual_planning.invalid_beat_index", beat_index=scene_draft.beat_index)
            continue
        scenes.append(
            SceneVisualPlan(
                beat_index=scene_draft.beat_index,
                scene_type=scene_draft.scene_type,
                description=scene_draft.description,
                search_keywords=scene_draft.search_keywords,
                ai_image_prompt=scene_draft.ai_image_prompt,
                transition=scene_draft.transition,
            )
        )

    output = VisualPlanOutput(scenes=scenes)
    return {
        "visual_plan": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_PLAN, "complete", scenes=len(scenes))],
    }


# --- Asset sourcing: stock first (when it fits the scene), AI-image fallback, graceful miss ---


def _download(url: str, path: str) -> None:
    with httpx.stream("GET", url, timeout=60, follow_redirects=True) as response:
        response.raise_for_status()
        with open(path, "wb") as f:
            f.writelines(response.iter_bytes())


def _try_stock(scene: SceneVisualPlan, run_id: str) -> SourcedAsset | None:
    if not scene.search_keywords:
        return None
    try:
        hits = stock_media_tool.search_videos(scene.search_keywords)
        asset_type = AssetType.STOCK_VIDEO
        if not hits:
            hits = stock_media_tool.search_images(scene.search_keywords)
            asset_type = AssetType.STOCK_IMAGE
        if not hits:
            return None

        hit = hits[0]
        ext = ".mp4" if asset_type == AssetType.STOCK_VIDEO else ".jpg"
        path = media_path(run_id, "assets", f"beat_{scene.beat_index}{ext}")
        _download(hit["url"], path)
    except Exception as exc:
        logger.warning("asset_visual.stock_failed", beat_index=scene.beat_index, error=str(exc))
        return None

    return SourcedAsset(
        beat_index=scene.beat_index,
        scene_type=scene.scene_type,
        asset_type=asset_type,
        source=hit["source"],
        local_path=path,
        source_url=hit["url"],
        license=hit.get("license", ""),
        attribution_required=False,  # both Pexels and Pixabay's current licenses don't require it
    )


def _try_character_animation(scene: SceneVisualPlan) -> SourcedAsset | None:
    """No per-beat file to source -- just makes sure the shared mascot pose pack exists (a no-op
    after the first call ever, since tools.character_assets caches it on disk)."""
    try:
        character_assets_tool.get_character_pack()
    except Exception as exc:
        logger.warning("asset_visual.character_pack_failed", beat_index=scene.beat_index, error=str(exc))
        return None

    return SourcedAsset(
        beat_index=scene.beat_index,
        scene_type=scene.scene_type,
        asset_type=AssetType.CHARACTER_ANIMATION,
        source="mascot-character",
        local_path=None,
        license="generated",
        attribution_required=False,
    )


def asset_visual_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_ASSET, "start")

    plan = VisualPlanOutput.model_validate(state["visual_plan"])
    run_id = state["run_id"]

    assets = []
    for scene in plan.scenes:
        asset = None
        if scene.scene_type in (SceneType.BROLL, SceneType.CHART, SceneType.MOTION_GRAPHIC):
            asset = _try_stock(scene, run_id)
        if asset is None:
            asset = _try_character_animation(scene)
        if asset is None:
            logger.error("asset_visual.no_asset_sourced", beat_index=scene.beat_index)
            asset = SourcedAsset(
                beat_index=scene.beat_index,
                scene_type=scene.scene_type,
                asset_type=AssetType.AI_IMAGE,
                source="none",
                local_path=None,
                license="",
                attribution_required=False,
            )
        assets.append(asset)

    output = AssetOutput(assets=assets)
    return {
        "asset_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_ASSET, "complete", assets=len(assets))],
    }
