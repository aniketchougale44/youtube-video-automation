"""Focused tests for the real Trend Research + Strategy Agent logic: objective scoring functions,
LLM-synthesis-to-TopicCandidate assembly, graceful Trends degradation, and Strategy's selection
validation. External calls are mocked (see conftest.py); these tests assert on the actual
business logic, not just that the graph doesn't crash.
"""
import uuid

from agents.schemas.common import ContentFormat, ContentType
from agents.schemas.research import TopicCandidate, TrendResearchOutput
from graph.nodes.research import (
    competition_score,
    composite_score,
    freshness_score,
    search_volume_score,
    strategy_node,
    trend_research_node,
)
from graph.state import initial_state
from tests.conftest import FAKE_SEARCH_VIDEOS, FAKE_TRENDING_VIDEOS, _days_ago


def test_freshness_score_decays_with_age():
    # Relative, not literal dates: freshness_score() floors at 0.0 by ~30 days old, so any pair of
    # hardcoded dates eventually both sit on the floor and the comparison stops testing decay at
    # all (it starts failing outright once both are stale). See conftest._days_ago.
    assert freshness_score([_days_ago(1)]) > freshness_score([_days_ago(10)])
    assert freshness_score([_days_ago(10)]) > freshness_score([_days_ago(20)])


def test_freshness_score_handles_bad_input_gracefully():
    assert freshness_score(["not-a-date", None, ""]) == 5.0
    assert freshness_score([]) == 5.0


def test_competition_score_increases_with_views():
    assert competition_score([10_000_000]) > competition_score([1_000])


def test_competition_score_bounded_0_to_10():
    assert 0.0 <= competition_score([10**12]) <= 10.0
    assert 0.0 <= competition_score([1]) <= 10.0


def test_search_volume_score_falls_back_when_trends_unavailable():
    assert search_volume_score(None, fallback=4.2) == 4.2
    assert search_volume_score({"rising": [], "top": []}, fallback=4.2) == 4.2


def test_search_volume_score_uses_trends_when_available():
    result = search_volume_score({"rising": ["a", "b"], "top": ["c"]}, fallback=0.0)
    assert result == min(10.0, 2 * 1.5 + 1 * 0.5)


def test_composite_score_rewards_low_competition():
    low_competition = composite_score(search_volume=5, freshness=5, competition=1)
    high_competition = composite_score(search_volume=5, freshness=5, competition=9)
    assert low_competition > high_competition


def test_trend_research_node_produces_ranked_candidates():
    state = initial_state(run_id=str(uuid.uuid4()), thread_id="t1")
    result = trend_research_node(state)

    output = TrendResearchOutput.model_validate(result["trend_output"])
    assert len(output.candidates) == 2
    assert output.candidates[0].composite_score >= output.candidates[1].composite_score
    # source_video_ids must trace back to a real retrieved video ID (never fabricated) -- from
    # either endpoint the node pulls, since raw_videos merges most_popular + search_with_stats
    real_ids = {v["video_id"] for v in (*FAKE_TRENDING_VIDEOS, *FAKE_SEARCH_VIDEOS)}
    for candidate in output.candidates:
        assert set(candidate.source_video_ids) <= real_ids


def test_trend_research_node_raises_when_no_videos_available(monkeypatch):
    # Both sources have to be emptied, not just most_popular: the node merges them and only raises
    # when the combined, deduped list is empty. Emptying one alone left search_with_stats live,
    # which is what made this assertion fail while quietly spending real search.list quota.
    monkeypatch.setattr("tools.youtube.most_popular", lambda **kwargs: [])
    monkeypatch.setattr("tools.youtube.search_with_stats", lambda *a, **k: [])
    state = initial_state(run_id=str(uuid.uuid4()), thread_id="t1")
    try:
        trend_research_node(state)
        raise AssertionError("expected RuntimeError when no trending videos are retrieved")
    except RuntimeError as exc:
        assert "no trending videos" in str(exc)


def _topic(title: str, score: float) -> TopicCandidate:
    return TopicCandidate(
        title=title, description="d", search_volume_score=score, competition_score=score,
        freshness_score=score, composite_score=score,
    )


def test_strategy_node_selects_matching_candidate_by_title():
    state = initial_state(run_id=str(uuid.uuid4()), thread_id="t1")
    state["trend_output"] = TrendResearchOutput(
        candidates=[_topic("Fake Topic A", 9.0), _topic("Fake Topic B", 5.0)]
    ).model_dump(mode="json")

    result = strategy_node(state)
    decision = result["strategy_decision"]

    assert decision["selected_topic"]["title"] == "Fake Topic A"
    assert decision["content_type"] == ContentType.EVERGREEN.value
    assert decision["content_format"] == ContentFormat.LONG_FORM.value


def test_strategy_node_falls_back_to_top_candidate_on_title_mismatch(monkeypatch):
    from graph.nodes.research import _StrategyChoice

    monkeypatch.setattr(
        "graph.nodes.research.call_structured",
        lambda prompt, output_model, system=None: _StrategyChoice(
            selected_topic_title="A title the LLM invented that matches nothing",
            content_type=ContentType.EVERGREEN,
            content_format=ContentFormat.SHORT,
            target_length_seconds=45,
            rationale="mismatch test",
        ),
    )

    state = initial_state(run_id=str(uuid.uuid4()), thread_id="t1")
    state["trend_output"] = TrendResearchOutput(
        candidates=[_topic("Real Top Candidate", 9.0), _topic("Real Second Candidate", 5.0)]
    ).model_dump(mode="json")

    result = strategy_node(state)
    decision = result["strategy_decision"]

    assert decision["selected_topic"]["title"] == "Real Top Candidate"  # fell back, didn't crash
