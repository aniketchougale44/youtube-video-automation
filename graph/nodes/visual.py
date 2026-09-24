"""Visual Planning Agent + Asset/Visual Agent — real logic.

visual_planning_node routes each beat to real stock footage (BROLL) or an animated-mascot scene
(AI_IMAGE) based on content — concrete/real-world beats get stock search keywords, abstract/
conceptual beats get an AI-image prompt (used only as backdrop-scene context; the mascot itself
comes from the shared character pack, not a per-beat generation). The LLM may also route beats to
AI_VIDEO -- the pipeline's only motion-video source. settings.max_ai_video_beats_per_run decides
how aggressively: 0 (no cap) flips the planning prompt so AI_VIDEO becomes the default for every
narrative beat, which is what turns the render into an animated story rather than a mascot
slideshow; a positive value caps it and tells the LLM to reserve those beats for standout moments,
the right shape for the metered hosted generators. _try_ai_video walks
settings.ai_video_provider's generator list (see core/settings.py): "colab" = the free offline Wan
endpoint the operator runs on Google Colab (tools/colab_video.py) and nothing else; "auto" = that
Colab endpoint first, then the hosted chain fal -> Hugging Face -> NVIDIA -> Veo.
settings.enable_ai_video_beats is the kill switch, checked independently of the cap.
asset_visual_node falls back from
stock/AI_VIDEO -> mascot character animation -> a "no asset" placeholder (never crashes the run
for a sourcing miss; video_assembly_node renders a solid-color placeholder clip for beats with no
asset).
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
from core.settings import get_settings
from graph.nodes._helpers import log_and_trace
from graph.state import PipelineState
from tools import character_assets as character_assets_tool
from tools import colab_video as colab_video_tool
from tools import fal_video as fal_video_tool
from tools import hf_video as hf_video_tool
from tools import image_gen as image_gen_tool
from tools import nvidia_video as nvidia_video_tool
from tools import stock_media as stock_media_tool
from tools import veo_video as veo_video_tool
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


_UNCAPPED_AI_VIDEO_PROMPT = (
    "AI_VIDEO — a real generated motion clip — is available and is the DEFAULT choice for this "
    "script. Route every narrative beat to AI_VIDEO and give it a descriptive ai_image_prompt "
    "written as a shot description: subject, what it is doing, setting, lighting, and a consistent "
    "art style repeated verbatim across beats (the generator has no memory between beats, so the "
    "style words are the only thing keeping the look continuous). Only fall back to BROLL for a "
    "beat that genuinely needs real-world documentary footage, or to CHART/MOTION_GRAPHIC for a "
    "beat that explicitly presents numeric data."
)


def _capped_ai_video_prompt(cap: int) -> str:
    return (
        f" AI_VIDEO is also available, but only for at most {cap} beat(s) in the whole script — "
        "reserve it for a single standout, motion-worthy moment (e.g. the hook/opening beat), not "
        "routine content, since each one is a slow, costly real video generation; give a "
        "descriptive ai_image_prompt the same way as for AI_IMAGE. Any beat you don't use it on "
        "should fall back to your normal BROLL/AI_IMAGE/CHART/MOTION_GRAPHIC choice."
    )


def _visual_planning_system_prompt() -> str:
    settings = get_settings()
    if not settings.enable_ai_video_beats:
        return _VISUAL_PLANNING_SYSTEM_PROMPT
    # Uncapped (max_ai_video_beats_per_run=0) flips the whole instruction rather than just raising
    # a number: the base prompt's stock-footage-first framing actively fights "animate the story",
    # and with no stock API keys configured those BROLL beats don't even resolve -- they fall
    # through to the mascot, which is exactly the slideshow this setting exists to get rid of.
    if settings.max_ai_video_beats_per_run <= 0:
        return _UNCAPPED_AI_VIDEO_PROMPT
    return _VISUAL_PLANNING_SYSTEM_PROMPT + _capped_ai_video_prompt(settings.max_ai_video_beats_per_run)


def visual_planning_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_PLAN, "start")
    settings = get_settings()

    script = ScriptOutput.model_validate(state["script_output"])
    beats_listing = "\n".join(
        f'- beat_index={b.index} type={b.beat_type} voiceover="{b.voiceover_text}" visual_cue="{b.visual_cue}"'
        for b in script.beats
    )
    draft = call_structured(f"Script beats:\n{beats_listing}", _VisualPlanDraft, system=_visual_planning_system_prompt())

    valid_indices = {b.index for b in script.beats}
    scenes = []
    for scene_draft in draft.scenes:
        if scene_draft.beat_index not in valid_indices:
            logger.warning("visual_planning.invalid_beat_index", beat_index=scene_draft.beat_index)
            continue
        scene_type = scene_draft.scene_type
        if scene_type == SceneType.AI_VIDEO and not settings.enable_ai_video_beats:
            # structured output still permits the enum value even when the prompt never mentions
            # it -- guard here too so a paid Veo call can never fire while the feature is off.
            logger.warning("visual_planning.ai_video_disabled_downgrading", beat_index=scene_draft.beat_index)
            scene_type = SceneType.AI_IMAGE
        scenes.append(
            SceneVisualPlan(
                beat_index=scene_draft.beat_index,
                scene_type=scene_type,
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


def _ai_video_generators(provider: str) -> list[tuple[str, object]]:
    """(source_label, generate_video_callable) pairs to try in order, per settings.ai_video_provider.
    "colab" -> the free offline Colab endpoint only (no paid API ever hit); "none" -> nothing;
    "auto" (default) -> Colab first (a no-op unless COLAB_VIDEO_URL is set) then the hosted chain
    fal -> Hugging Face -> NVIDIA -> Veo."""
    colab = ("colab", colab_video_tool.generate_video)
    hosted = [
        ("fal", fal_video_tool.generate_video),
        ("huggingface", hf_video_tool.generate_video),
        ("nvidia", nvidia_video_tool.generate_video),
        ("veo", veo_video_tool.generate_video),
    ]
    if provider == "colab":
        return [colab]
    if provider == "none":
        return []
    return [colab, *hosted]


def _try_ai_video(scene: SceneVisualPlan, run_id: str) -> SourcedAsset | None:
    """Real AI video generation for a beat the visual-planning LLM flagged as standout/motion-
    worthy. Only ever called for beats within settings.max_ai_video_beats_per_run (see
    asset_visual_node). Walks the generator list for settings.ai_video_provider and returns the
    first success; if every one fails (not configured, quota exhausted, timeout, Colab session
    down) it returns None and the beat falls back to character animation like any other sourcing
    miss -- never crashes the run."""
    prompt = scene.ai_image_prompt or scene.description
    path = media_path(run_id, "assets", f"beat_{scene.beat_index}.mp4")

    for source, generate_video in _ai_video_generators(get_settings().ai_video_provider):
        try:
            generate_video(prompt, output_path=path)
        except Exception as exc:
            logger.warning(
                "asset_visual.ai_video_generator_failed",
                beat_index=scene.beat_index, source=source, error=str(exc),
            )
            continue
        return SourcedAsset(
            beat_index=scene.beat_index,
            scene_type=scene.scene_type,
            asset_type=AssetType.AI_VIDEO,
            source=source,
            local_path=path,
            license="generated",
            attribution_required=False,
        )
    return None


def _try_character_animation(scene: SceneVisualPlan, run_id: str) -> SourcedAsset | None:
    """Makes sure the shared mascot pose pack exists (a no-op after the first call ever, since
    tools.character_assets caches it on disk), and generates a per-beat backdrop photo via the
    free/keyless Pollinations fallback (tools.image_gen) so the mascot sits in front of a real
    scene instead of a flat color -- purely cosmetic, so a failed/skipped generation still yields
    a usable asset (local_path=None -> video_assembly_node's flat-color backdrop), never falls
    through to the "no asset" placeholder over an image-gen hiccup."""
    try:
        character_assets_tool.get_character_pack()
    except Exception as exc:
        logger.warning("asset_visual.character_pack_failed", beat_index=scene.beat_index, error=str(exc))
        return None

    backdrop_path = None
    prompt = scene.ai_image_prompt or scene.description
    if prompt:
        candidate_path = media_path(run_id, "assets", f"beat_{scene.beat_index}_backdrop.jpg")
        try:
            image_gen_tool.generate_image(prompt, output_path=candidate_path, size="1920x1080")
            backdrop_path = candidate_path
        except Exception as exc:
            logger.warning("asset_visual.backdrop_image_failed", beat_index=scene.beat_index, error=str(exc))

    return SourcedAsset(
        beat_index=scene.beat_index,
        scene_type=scene.scene_type,
        asset_type=AssetType.CHARACTER_ANIMATION,
        source="mascot-character",
        local_path=backdrop_path,
        license="generated",
        attribution_required=False,
    )


def asset_visual_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_ASSET, "start")
    settings = get_settings()

    plan = VisualPlanOutput.model_validate(state["visual_plan"])
    run_id = state["run_id"]

    # Cap enforced here too (not just in the planning prompt) -- in beat order, regardless of what
    # order the LLM emitted scenes in, so "first N beats" is deterministic. The kill switch is
    # checked separately from the cap rather than folded into it, because cap<=0 means "no cap"
    # (settings.max_ai_video_beats_per_run) while the feature being off must mean "no beats" --
    # collapsing the two would turn the kill switch into its exact opposite.
    ai_video_indices = sorted(s.beat_index for s in plan.scenes if s.scene_type == SceneType.AI_VIDEO)
    if not settings.enable_ai_video_beats:
        ai_video_budget_indices: set[int] = set()
    elif settings.max_ai_video_beats_per_run <= 0:
        ai_video_budget_indices = set(ai_video_indices)
    else:
        ai_video_budget_indices = set(ai_video_indices[: settings.max_ai_video_beats_per_run])

    assets = []
    for scene in plan.scenes:
        asset = None
        if scene.scene_type == SceneType.AI_VIDEO and scene.beat_index in ai_video_budget_indices:
            asset = _try_ai_video(scene, run_id)
        if asset is None and scene.scene_type in (SceneType.BROLL, SceneType.CHART, SceneType.MOTION_GRAPHIC):
            asset = _try_stock(scene, run_id)
        if asset is None:
            asset = _try_character_animation(scene, run_id)
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
