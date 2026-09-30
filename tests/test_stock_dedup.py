"""Neighbouring beats must not be handed the same stock clip.

_try_stock always took hits[0]. Stock search is keyword-driven and consecutive beats of one script
describe related things, so several beats got the identical file — observed in a real render as
byte-identical footage on beats 0 and 3, which reads as a glitch rather than an edit.
"""
import pytest

from agents.schemas.visual import (
    AssetOutput,
    AssetType,
    SceneType,
    SceneVisualPlan,
    VisualPlanOutput,
)
from graph.nodes.visual import _pick_unused, asset_visual_node
from graph.state import initial_state

pytestmark = pytest.mark.usefixtures("mock_external_apis")


def _hit(n: int) -> dict:
    return {"url": f"https://videos.example.com/clip{n}.mp4", "source": "pexels", "license": "pexels"}


# --- the picker ---


def test_picks_the_first_unused_hit():
    hits = [_hit(1), _hit(2), _hit(3)]
    used = {hits[0]["url"]}
    assert _pick_unused(hits, used, beat_index=1)["url"] == hits[1]["url"]


def test_falls_back_to_the_top_hit_when_everything_is_taken():
    """A repeated shot beats no footage at all."""
    hits = [_hit(1), _hit(2)]
    used = {h["url"] for h in hits}
    assert _pick_unused(hits, used, beat_index=5)["url"] == hits[0]["url"]


def test_an_empty_used_set_takes_the_best_match():
    hits = [_hit(1), _hit(2)]
    assert _pick_unused(hits, set(), beat_index=0)["url"] == hits[0]["url"]


# --- end to end through the node ---


def test_beats_with_identical_keywords_get_different_clips(monkeypatch, tmp_path):
    """The production shape: several beats whose keywords return the same ranked list."""
    shared_hits = [_hit(1), _hit(2), _hit(3)]
    monkeypatch.setattr(
        "graph.nodes.visual.stock_media_tool.search_videos",
        lambda keywords, per_page=5: list(shared_hits),
    )

    downloaded: dict[str, str] = {}

    def _fake_download(url, path):
        downloaded[path] = url
        with open(path, "wb") as f:
            f.write(b"\x00")

    monkeypatch.setattr("graph.nodes.visual._download", _fake_download)

    scenes = [
        SceneVisualPlan(beat_index=i, scene_type=SceneType.BROLL, description="d",
                        search_keywords=["children playing"])
        for i in range(3)
    ]
    state = initial_state(run_id="dedup-test", thread_id="t1")
    state["visual_plan"] = VisualPlanOutput(scenes=scenes).model_dump(mode="json")

    output = AssetOutput.model_validate(asset_visual_node(state)["asset_output"])

    assert all(a.asset_type == AssetType.STOCK_VIDEO for a in output.assets)
    urls = [a.source_url for a in output.assets]
    assert len(set(urls)) == 3, f"beats shared a clip: {urls}"


def test_more_beats_than_hits_still_sources_every_beat(monkeypatch):
    """Running out of distinct clips must not leave a beat with no footage."""
    monkeypatch.setattr(
        "graph.nodes.visual.stock_media_tool.search_videos",
        lambda keywords, per_page=5: [_hit(1), _hit(2)],
    )

    def _fake_download(url, path):
        with open(path, "wb") as f:
            f.write(b"\x00")

    monkeypatch.setattr("graph.nodes.visual._download", _fake_download)

    scenes = [
        SceneVisualPlan(beat_index=i, scene_type=SceneType.BROLL, description="d",
                        search_keywords=["children playing"])
        for i in range(4)
    ]
    state = initial_state(run_id="dedup-test-2", thread_id="t1")
    state["visual_plan"] = VisualPlanOutput(scenes=scenes).model_dump(mode="json")

    output = AssetOutput.model_validate(asset_visual_node(state)["asset_output"])

    assert len(output.assets) == 4
    assert all(a.asset_type == AssetType.STOCK_VIDEO for a in output.assets)
    # the first two are distinct; the last two reuse rather than failing
    assert len({a.source_url for a in output.assets}) == 2
