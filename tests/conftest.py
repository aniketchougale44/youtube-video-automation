"""Shared test fixtures. Mocks all external I/O (YouTube Data API, Google Trends, LLM calls,
embeddings, TTS, stock media, image generation) so the whole suite runs hermetically — no API
keys, no network, no live Postgres/Redis required. Graph-wiring tests (test_graph_skeleton.py)
only care about retry/escalation/interrupt routing, not real per-stage correctness; that gets its
own focused tests alongside each stage's real implementation (e.g. test_research_agents.py).

voiceover_node/video_assembly_node are replaced outright (not just internally mocked) because real
moviepy encoding is slow and pulls in an ffmpeg dependency that has no place in a fast wiring-only
suite. Every other real node (script_writer, critic_script_qa, fact_check, visual_planning,
asset_visual, metadata_seo, thumbnail, critic_compliance) is left genuinely real and only has its
external calls (LLM, embeddings, stock media, image gen) mocked — critic_script_qa_node and
critic_compliance_node specifically must stay real because the retry/escalation tests exercise
their actual consume_force_reject()/retry_counts control flow, not just their output shape.
"""
import os

import pytest
from PIL import Image

from agents.schemas.common import ContentFormat, ContentType
from agents.schemas.script import BeatType
from agents.schemas.visual import SceneType
from graph.nodes._helpers import log_and_trace
from graph.nodes.publish import _GuidelineSelfCheck, _MetadataDraft
from graph.nodes.research import _StrategyChoice, _TopicIdea, _TrendSynthesis
from graph.nodes.script import _BeatDraft, _ClaimExtraction, _ScriptDraft
from graph.nodes.visual import _ScenePlanDraft, _VisualPlanDraft

FAKE_TRENDING_VIDEOS = [
    {
        "video_id": "vid1",
        "title": "Fake trending video 1",
        "description": "d",
        "category_id": "27",
        "published_at": "2026-08-04T00:00:00Z",
        "channel_title": "c",
        "view_count": 500_000,
        "like_count": 10_000,
        "comment_count": 500,
    },
    {
        "video_id": "vid2",
        "title": "Fake trending video 2",
        "description": "d",
        "category_id": "27",
        "published_at": "2026-08-03T00:00:00Z",
        "channel_title": "c",
        "view_count": 250_000,
        "like_count": 5_000,
        "comment_count": 200,
    },
]

FAKE_EMBEDDING = [0.001] * 1536  # matches EMBEDDING_DIM (tools/embeddings.py)

FAKE_SCRIPT_BEATS = [
    _BeatDraft(beat_type=BeatType.HOOK, voiceover_text="Did you know this fake fact?", visual_cue="cold open"),
    _BeatDraft(beat_type=BeatType.STORY, voiceover_text="Here is the fake story body used for testing purposes.", visual_cue="b-roll"),
    _BeatDraft(beat_type=BeatType.CTA, voiceover_text="Subscribe for more fake content.", visual_cue="end card"),
]


def _fake_call_structured(prompt, output_model, system=None):
    if output_model is _TrendSynthesis:
        return _TrendSynthesis(
            ideas=[
                _TopicIdea(title="Fake Topic A", description="desc A", category="Education", representative_video_ids=["vid1"]),
                _TopicIdea(title="Fake Topic B", description="desc B", category="Education", representative_video_ids=["vid2"]),
            ]
        )
    if output_model is _StrategyChoice:
        return _StrategyChoice(
            selected_topic_title="Fake Topic A",
            content_type=ContentType.EVERGREEN,
            content_format=ContentFormat.LONG_FORM,
            target_length_seconds=480,
            rationale="fake rationale for tests",
        )
    if output_model is _ScriptDraft:
        return _ScriptDraft(working_title="Fake Script Title", hook=FAKE_SCRIPT_BEATS[0].voiceover_text, beats=FAKE_SCRIPT_BEATS, cta=FAKE_SCRIPT_BEATS[-1].voiceover_text)
    if output_model is _ClaimExtraction:
        return _ClaimExtraction(claims=[])  # no claims -> fact_check_node has nothing to verify
    if output_model is _VisualPlanDraft:
        return _VisualPlanDraft(
            scenes=[
                _ScenePlanDraft(beat_index=i, scene_type=SceneType.AI_IMAGE, description="fake scene", ai_image_prompt="fake prompt")
                for i in range(len(FAKE_SCRIPT_BEATS))
            ]
        )
    if output_model is _MetadataDraft:
        return _MetadataDraft(
            title_options=["Fake Title One", "Fake Title Two"],
            selected_title="Fake Title One",
            description="Fake description for testing.",
            tags=["fake", "test", "stub", "demo", "video"],
            pinned_comment="What do you think?",
        )
    if output_model is _GuidelineSelfCheck:
        return _GuidelineSelfCheck(flags=[], risk_notes="")
    raise AssertionError(f"unmocked output_model requested in test: {output_model}")


def _fake_generate_image(prompt: str, output_path: str, size: str = "1024x1024") -> str:
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(output_path)
    return output_path


def _fake_openai_edit(base_image_path: str, prompt: str, output_path: str) -> None:
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(output_path)


def _make_fake_media_path(tmp_dir: str):
    def _fake_media_path(*parts: str) -> str:
        path = os.path.join(tmp_dir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    return _fake_media_path


def _fake_voiceover_node(state: dict) -> dict:
    from agents.schemas.audio_render import VoiceoverOutput, VoiceoverSegment
    from agents.schemas.script import ScriptOutput

    script = ScriptOutput.model_validate(state["script_output"])
    segments = [
        VoiceoverSegment(beat_index=b.index, audio_path=f"/tmp/fake_vo_{b.index}.mp3", duration_seconds=5.0, text=b.voiceover_text)
        for b in script.beats
    ]
    output = VoiceoverOutput(
        segments=segments, full_audio_path="/tmp/fake_vo_full.mp3", total_duration_seconds=sum(s.duration_seconds for s in segments)
    )
    return {"voiceover_output": output.model_dump(mode="json"), "trace": [log_and_trace("voiceover", "fake_complete")]}


def _make_fake_video_assembly_node(tmp_dir: str):
    def _fake_video_assembly_node(state: dict) -> dict:
        from agents.schemas.audio_render import AssemblyOutput

        voiceover = state["voiceover_output"]
        render_path = os.path.join(tmp_dir, f"render_{state['run_id']}.mp4")
        with open(render_path, "wb") as f:
            f.write(b"\x00")  # dummy bytes -- just needs to exist for os.path.exists() checks

        output = AssemblyOutput(
            render_path=render_path,
            duration_seconds=voiceover["total_duration_seconds"],  # matches voiceover exactly -> no AV-sync drift
            resolution="1920x1080",
            fps=30,
            has_captions=True,
            has_music=True,
        )
        return {"assembly_output": output.model_dump(mode="json"), "trace": [log_and_trace("video_assembly", "fake_complete")]}

    return _fake_video_assembly_node


@pytest.fixture(autouse=True)
def mock_external_apis(monkeypatch, tmp_path):
    monkeypatch.setattr("tools.youtube.most_popular", lambda **kwargs: FAKE_TRENDING_VIDEOS)
    monkeypatch.setattr("tools.trends.related_queries", lambda *a, **k: {"rising": ["x"], "top": ["y", "z"]})

    for module_name in ("graph.nodes.research", "graph.nodes.script", "graph.nodes.visual", "graph.nodes.publish"):
        monkeypatch.setattr(f"{module_name}.call_structured", _fake_call_structured)

    # embeddings: avoid real OpenAI calls + avoid ScriptEmbedding's FK constraint against a
    # synthetic test run_id that has no matching `runs` row in Postgres
    monkeypatch.setattr("tools.embeddings.embed_text", lambda text: FAKE_EMBEDDING)
    monkeypatch.setattr("tools.embeddings.most_similar_transcript", lambda vector, limit=5: [])
    monkeypatch.setattr("tools.embeddings.most_similar_script", lambda vector, exclude_run_id=None, limit=5: [])
    monkeypatch.setattr("tools.embeddings.store_script_embedding", lambda run_id, full_text: None)

    # stock media / image gen: avoid real Pexels/Pixabay/OpenAI-image network calls
    monkeypatch.setattr("tools.stock_media.search_videos", lambda keywords, per_page=5: [])
    monkeypatch.setattr("tools.stock_media.search_images", lambda keywords, per_page=5: [])
    monkeypatch.setattr("tools.image_gen.generate_image", _fake_generate_image)
    # avoid a real OpenAI network call from tools.character_assets._openai_edit regardless of
    # whether OPENAI_API_KEY happens to be set in the developer's local .env
    monkeypatch.setattr("tools.character_assets._openai_edit", _fake_openai_edit)

    # asset_visual_node / thumbnail_node stay real (fast once image_gen is mocked) but still write
    # files via media_path() -- redirect those into pytest's auto-cleaned tmp_path
    fake_media_path = _make_fake_media_path(str(tmp_path))
    monkeypatch.setattr("graph.nodes.visual.media_path", fake_media_path)
    monkeypatch.setattr("graph.nodes.publish.media_path", fake_media_path)
    # tools.character_assets caches the mascot pack on disk keyed only by character name (not
    # run_id), so without this redirect it would write into (and read stale files back from) the
    # real project media_dir across test runs instead of pytest's auto-cleaned tmp_path
    monkeypatch.setattr("tools.character_assets.media_path", fake_media_path)

    # voiceover/video_assembly: replaced outright, see module docstring
    monkeypatch.setattr("graph.builder.voiceover_node", _fake_voiceover_node)
    monkeypatch.setattr("graph.builder.video_assembly_node", _make_fake_video_assembly_node(str(tmp_path)))

    # the fake video_assembly_node above writes a real (dummy) file so critic_compliance_node's
    # real spec check passes -- that also means upload_node's real branch (file exists) now runs
    # in tests, so the actual YouTube OAuth upload call itself must be mocked
    monkeypatch.setattr("tools.youtube.resumable_upload", lambda **kwargs: "fake_yt_id_000")
    monkeypatch.setattr("tools.youtube.get_or_create_playlist", lambda **kwargs: "fake_playlist_id_000")
    monkeypatch.setattr("tools.youtube.add_video_to_playlist", lambda *args, **kwargs: None)
