"""Stock footage must be reachable from every scene type, not only pre-labelled BROLL beats.

Regression test for a live Stage-3 run. asset_visual_node only consulted Pexels/Pixabay when the
planner had labelled a beat BROLL/CHART/MOTION_GRAPHIC. On a real script about STEM rhymes the
planner labelled 1 beat of 8 BROLL and the rest AI_IMAGE (abstract content), so exactly one beat
ever reached the stock API. The other seven -- including AI_VIDEO beats whose generator was
unreachable, and AI_IMAGE beats whose image-gen returned 402 -- dropped straight to a generic
mascot shot while usable footage sat one API call away.

Adding a Pexels key could not fix that, because the code never asked.
"""
import pytest

from agents.schemas.visual import (
    AssetOutput,
    AssetType,
    SceneType,
    SceneVisualPlan,
    VisualPlanOutput,
)
from core.settings import get_settings
from graph.nodes.visual import asset_visual_node
from graph.state import initial_state

pytestmark = pytest.mark.usefixtures("mock_external_apis")

_HIT = [{"url": "https://videos.example.com/clip.mp4", "source": "pexels", "license": "pexels"}]


def _state(scenes: list[SceneVisualPlan]) -> dict:
    st = initial_state(run_id="stock-fallback-test", thread_id="t1")
    st["visual_plan"] = VisualPlanOutput(scenes=scenes).model_dump(mode="json")
    return st


def _serve_stock(monkeypatch, calls: list):
    def _search_videos(keywords, per_page=5):
        calls.append(list(keywords))
        return _HIT

    def _fake_download(url, path):
        with open(path, "wb") as f:
            f.write(b"\x00")

    monkeypatch.setattr("graph.nodes.visual.stock_media_tool.search_videos", _search_videos)
    monkeypatch.setattr("graph.nodes.visual._download", _fake_download)


@pytest.mark.parametrize(
    "scene_type",
    [SceneType.AI_IMAGE, SceneType.BROLL, SceneType.CHART, SceneType.MOTION_GRAPHIC],
)
def test_every_scene_type_can_reach_stock(monkeypatch, scene_type):
    calls: list = []
    _serve_stock(monkeypatch, calls)

    scenes = [SceneVisualPlan(beat_index=0, scene_type=scene_type, description="d",
                              search_keywords=["child reading book"])]
    output = AssetOutput.model_validate(asset_visual_node(_state(scenes))["asset_output"])

    assert output.assets[0].asset_type == AssetType.STOCK_VIDEO, (
        f"{scene_type.value} beat fell back to {output.assets[0].asset_type.value} "
        f"instead of using the stock footage its keywords describe"
    )
    assert calls == [["child reading book"]]


def test_failed_ai_video_uses_stock_before_the_mascot(monkeypatch):
    """The exact production shape: AI_VIDEO beat, generator unreachable, keywords present."""
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_ai_video_beats", True)
    monkeypatch.setattr(settings, "ai_video_provider", "colab")
    monkeypatch.setattr(settings, "max_ai_video_beats_per_run", 3)

    def _dead_generator(*_a, **_k):
        raise RuntimeError("COLAB_VIDEO_URL is not set")

    monkeypatch.setattr("graph.nodes.visual.colab_video_tool.generate_video", _dead_generator)

    calls: list = []
    _serve_stock(monkeypatch, calls)

    scenes = [SceneVisualPlan(beat_index=0, scene_type=SceneType.AI_VIDEO, description="d",
                              ai_image_prompt="a firefly", search_keywords=["firefly night forest"])]
    output = AssetOutput.model_validate(asset_visual_node(_state(scenes))["asset_output"])

    assert output.assets[0].asset_type == AssetType.STOCK_VIDEO, output.assets[0].asset_type
    assert calls, "a failed AI_VIDEO beat never consulted stock"


def test_a_beat_with_no_keywords_still_falls_through_to_the_mascot(monkeypatch):
    """_try_stock is a no-op without keywords, so scenes the planner couldn't describe for a
    stock search must not break -- they just land on the mascot as before."""
    calls: list = []
    _serve_stock(monkeypatch, calls)

    scenes = [SceneVisualPlan(beat_index=0, scene_type=SceneType.AI_IMAGE, description="d",
                              ai_image_prompt="abstract concept", search_keywords=[])]
    output = AssetOutput.model_validate(asset_visual_node(_state(scenes))["asset_output"])

    assert output.assets[0].asset_type == AssetType.CHARACTER_ANIMATION
    assert calls == [], "searched stock with no keywords"
