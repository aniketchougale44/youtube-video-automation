"""Background-music mixing in video_assembly_node, and upload_node's handling of a missing render.

The music bed is cosmetic, so every sourcing/mixing failure has to degrade to a voiceover-only
render rather than fail a multi-minute encode. A missing render file is the opposite: it used to
be reported as a successful upload with a fake video id (scaffolding from when video_assembly_node
was a stub), which recorded a failed run as published and left the Performance Monitor pulling
analytics for an id that never existed.
"""
import os

import pytest

from agents.schemas.publish import (
    MetadataOutput,
    ThumbnailOutput,
    UploadStatus,
    Visibility,
)
from core.settings import get_settings
from graph.nodes.audio_render import _background_music, _music_mood
from graph.nodes.publish import upload_node
from graph.state import initial_state


def _state_with_strategy(category: str | None) -> dict:
    state = initial_state(run_id="music-test", thread_id="t1")
    topic: dict = {
        "title": "t", "description": "d", "search_volume_score": 1.0,
        "competition_score": 1.0, "freshness_score": 1.0, "composite_score": 1.0,
    }
    if category is not None:
        topic["category"] = category
    state["strategy_decision"] = {"selected_topic": topic}
    return state


# --- mood selection ---


def test_music_mood_prefers_the_explicit_setting(monkeypatch):
    monkeypatch.setattr(get_settings(), "background_music_mood", "lo-fi hip hop")
    assert _music_mood(_state_with_strategy("Education")) == "lo-fi hip hop"


def test_music_mood_derives_from_the_strategy_topic_category(monkeypatch):
    monkeypatch.setattr(get_settings(), "background_music_mood", "")
    assert _music_mood(_state_with_strategy("Nursery rhymes")).startswith("Nursery rhymes")


def test_music_mood_falls_back_when_no_category_is_available(monkeypatch):
    monkeypatch.setattr(get_settings(), "background_music_mood", "")
    # no strategy_decision at all -- e.g. a resumed run whose state was trimmed
    state = initial_state(run_id="music-test", thread_id="t1")
    assert _music_mood(state) == "calm ambient instrumental"


# --- music sourcing degrades, never raises ---


def test_background_music_is_skipped_when_disabled(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_background_music", False)

    def _unexpected(*_a, **_k):
        raise AssertionError("must not source audio while the feature is off")

    monkeypatch.setattr("graph.nodes.audio_render.freesound_audio_tool.get_sound", _unexpected)

    assert _background_music(_state_with_strategy("Education"), "run-x", 10.0) == (None, False)


def test_background_music_degrades_to_no_music_on_failure(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_background_music", True)

    def _boom(*_a, **_k):
        raise RuntimeError("freesound exploded")

    monkeypatch.setattr("graph.nodes.audio_render.freesound_audio_tool.get_sound", _boom)

    # must not propagate -- the render is minutes of work and the bed is decoration
    clip, has_music = _background_music(_state_with_strategy("Education"), "run-x", 10.0)
    assert clip is None
    assert has_music is False


# moviepy's FFMPEG_AudioReader.__del__ touches self.proc even when __init__ bailed before
# setting it, so opening a corrupt file emits an unraisable AttributeError from the GC. That is
# moviepy's bug, not a leak on our side -- the clip never opened.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_background_music_degrades_when_the_sourced_file_is_unreadable(monkeypatch, tmp_path):
    """get_sound() promises a file, but a truncated/corrupt one still has to be survivable."""
    settings = get_settings()
    monkeypatch.setattr(settings, "enable_background_music", True)
    monkeypatch.setattr(
        "graph.nodes.audio_render.media_path",
        lambda *parts: os.path.join(str(tmp_path), *parts[1:]),
    )

    def _write_garbage(mood, duration_seconds, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(b"not an audio file")
        return output_path

    monkeypatch.setattr("graph.nodes.audio_render.freesound_audio_tool.get_sound", _write_garbage)

    clip, has_music = _background_music(_state_with_strategy("Education"), "run-x", 10.0)
    assert clip is None
    assert has_music is False


# --- upload_node must not report a fake success ---


def _upload_state(render_path: str) -> dict:
    state = initial_state(run_id="upload-test", thread_id="t1")
    state["metadata_output"] = MetadataOutput(
        title_options=["A Title"], selected_title="A Title", description="d",
        tags=["a", "b", "c", "d", "e"], chapters=[], pinned_comment="c",
    ).model_dump(mode="json")
    state["assembly_output"] = {
        "render_path": render_path, "duration_seconds": 10.0, "resolution": "1920x1080",
        "fps": 30, "has_captions": True, "has_music": True,
    }
    state["thumbnail_output"] = ThumbnailOutput(candidates=[]).model_dump(mode="json")
    return state


def test_upload_reports_failure_when_the_render_is_missing(monkeypatch):
    def _unexpected(**_k):
        raise AssertionError("nothing should be uploaded when there is no render file")

    monkeypatch.setattr("tools.youtube.resumable_upload", _unexpected)

    result = upload_node(_upload_state("does/not/exist.mp4"))["upload_result"]

    assert result["status"] == UploadStatus.FAILED.value
    assert result["youtube_video_id"] is None  # never a fake id -- feedback would query it later
    assert result["quota_units_used"] == 0     # nothing sent, nothing spent
    assert "render file missing" in result["error"]


def test_upload_succeeds_normally_when_the_render_exists(monkeypatch, tmp_path):
    render = tmp_path / "final.mp4"
    render.write_bytes(b"\x00")

    monkeypatch.setattr("tools.youtube.resumable_upload", lambda **_k: "real_video_id")
    monkeypatch.setattr("tools.youtube.get_or_create_playlist", lambda **_k: "pl_1")
    monkeypatch.setattr("tools.youtube.add_video_to_playlist", lambda *_a, **_k: None)

    result = upload_node(_upload_state(str(render)))["upload_result"]

    assert result["status"] == UploadStatus.UPLOADED.value
    assert result["youtube_video_id"] == "real_video_id"
    assert result["visibility"] == Visibility(get_settings().youtube_upload_visibility).value
