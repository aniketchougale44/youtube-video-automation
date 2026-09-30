"""The feedback loop's numeric adjustments must move the candidate ranking, not just the prompt.

Before this, strategy_weight_adjustments were rendered into the Strategy Agent's prompt as a
sentence and nowhere else, so the entire learning loop depended on an LLM choosing to act on text.
It also meant the deterministic fallback -- candidates[0], used whenever the LLM returns a title
matching no candidate -- ignored the learning completely.
"""
import pytest

from agents.schemas.research import TopicCandidate
from graph.nodes.research import _LEARNED_WEIGHT_SCALE, apply_learned_weights


def _candidate(title: str, freshness: float, composite: float = 5.0) -> TopicCandidate:
    return TopicCandidate(
        title=title, description="d",
        search_volume_score=5.0, competition_score=5.0,
        freshness_score=freshness, composite_score=composite,
    )


def test_no_adjustments_leaves_candidates_untouched():
    cands = [_candidate("a", 9.0), _candidate("b", 1.0)]
    out = apply_learned_weights(cands, {})
    assert [c.title for c in out] == ["a", "b"]
    assert [c.composite_score for c in out] == [5.0, 5.0]


def test_zero_bias_is_treated_as_no_signal():
    out = apply_learned_weights([_candidate("a", 9.0)], {"evergreen_bias": 0.0})
    assert out[0].composite_score == 5.0


def test_positive_bias_promotes_the_evergreen_candidate():
    """evergreen_bias > 0 means timeless content out-performed, so low freshness should rise."""
    trending = _candidate("trending", freshness=10.0)
    evergreen = _candidate("evergreen", freshness=0.0)

    out = apply_learned_weights([trending, evergreen], {"evergreen_bias": 0.2})

    assert out[0].title == "evergreen", "the evergreen candidate should now rank first"
    assert out[0].composite_score > 5.0
    assert out[1].composite_score < 5.0


def test_negative_bias_promotes_the_trending_candidate():
    trending = _candidate("trending", freshness=10.0)
    evergreen = _candidate("evergreen", freshness=0.0)

    out = apply_learned_weights([evergreen, trending], {"evergreen_bias": -0.2})

    assert out[0].title == "trending"


def test_a_mid_freshness_candidate_is_left_alone():
    """freshness 5 sits on the axis' midpoint, so neither direction of bias describes it."""
    out = apply_learned_weights([_candidate("neutral", freshness=5.0)], {"evergreen_bias": 0.2})
    assert out[0].composite_score == 5.0


def test_the_nudge_is_bounded_so_it_cannot_override_a_clear_winner():
    """A saturated bias must not out-vote the objective signals: max shift is 1.0 on a 0-10 scale."""
    max_shift = 0.2 * 1.0 * _LEARNED_WEIGHT_SCALE
    assert max_shift == pytest.approx(1.0)

    strong_but_trendy = _candidate("strong", freshness=10.0, composite=9.0)
    weak_but_evergreen = _candidate("weak", freshness=0.0, composite=6.0)

    out = apply_learned_weights([strong_but_trendy, weak_but_evergreen], {"evergreen_bias": 0.2})
    assert out[0].title == "strong", "a 3-point objective lead should survive the learned nudge"


def test_scores_stay_within_range():
    out = apply_learned_weights(
        [_candidate("hi", freshness=0.0, composite=9.8), _candidate("lo", freshness=10.0, composite=0.1)],
        {"evergreen_bias": 0.2},
    )
    assert all(0.0 <= c.composite_score <= 10.0 for c in out)


def test_inputs_are_not_mutated():
    """The original objective scores must stay auditable after re-weighting."""
    original = _candidate("a", freshness=10.0, composite=5.0)
    apply_learned_weights([original], {"evergreen_bias": 0.2})
    assert original.composite_score == 5.0
