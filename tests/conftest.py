"""Shared test fixtures. Mocks all external I/O (YouTube Data API, Google Trends, LLM calls) so
the whole suite runs hermetically — no API keys, no network, no Redis required. Graph-wiring
tests (test_graph_skeleton.py) only care about retry/escalation/interrupt routing, not real
trend-research correctness; that gets its own focused tests in test_research_agents.py which
mock the same seams but assert on the actual scoring/synthesis logic.
"""
import pytest

from agents.schemas.common import ContentFormat, ContentType
from graph.nodes.research import _StrategyChoice, _TopicIdea, _TrendSynthesis

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
    raise AssertionError(f"unmocked output_model requested in test: {output_model}")


@pytest.fixture(autouse=True)
def mock_external_apis(monkeypatch):
    monkeypatch.setattr("tools.youtube.most_popular", lambda **kwargs: FAKE_TRENDING_VIDEOS)
    monkeypatch.setattr("tools.trends.related_queries", lambda *a, **k: {"rising": ["x"], "top": ["y", "z"]})
    monkeypatch.setattr("graph.nodes.research.call_structured", _fake_call_structured)
