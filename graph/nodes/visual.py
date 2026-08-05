"""Visual Planning Agent + Asset/Visual Agent — stub logic."""
from agents.schemas.script import ScriptOutput
from agents.schemas.visual import (
    AssetOutput,
    AssetType,
    SceneType,
    SceneVisualPlan,
    SourcedAsset,
    VisualPlanOutput,
)
from graph.nodes._helpers import log_and_trace
from graph.state import PipelineState

STAGE_PLAN = "visual_planning"
STAGE_ASSET = "asset_visual"


def visual_planning_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_PLAN, "start")

    script = ScriptOutput.model_validate(state["script_output"])

    # STUB: real logic maps each script beat to a scene type + search keywords / AI-image prompt
    scenes = [
        SceneVisualPlan(
            beat_index=beat.index,
            scene_type=SceneType.BROLL,
            description=f"[STUB] scene for beat {beat.index}",
            search_keywords=["placeholder"],
            transition="cut",
        )
        for beat in script.beats
    ]
    output = VisualPlanOutput(scenes=scenes)

    return {
        "visual_plan": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_PLAN, "complete", scenes=len(scenes))],
    }


def asset_visual_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_ASSET, "start")

    plan = VisualPlanOutput.model_validate(state["visual_plan"])

    # STUB: real logic calls tools.stock_media (Pexels/Pixabay) or tools.image_gen per scene
    assets = [
        SourcedAsset(
            beat_index=scene.beat_index,
            scene_type=scene.scene_type,
            asset_type=AssetType.STOCK_VIDEO,
            source="stub",
            local_path=f"/tmp/stub_asset_{scene.beat_index}.mp4",
            license="stub-license",
            attribution_required=False,
        )
        for scene in plan.scenes
    ]
    output = AssetOutput(assets=assets)

    return {
        "asset_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_ASSET, "complete", assets=len(assets))],
    }
