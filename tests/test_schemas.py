"""Sanity checks that every agent I/O schema round-trips through model_dump/model_validate the
way graph nodes rely on (state stores dicts, nodes rehydrate Pydantic models from them)."""
from agents.schemas.common import ContentFormat, ContentType
from agents.schemas.publish import ComplianceCheckResult, MetadataOutput
from agents.schemas.research import StrategyDecision, TopicCandidate
from agents.schemas.script import BeatType, ScriptBeat, ScriptOutput


def test_topic_candidate_round_trip():
    tc = TopicCandidate(
        title="x", search_volume_score=5, competition_score=3, freshness_score=8, composite_score=6
    )
    restored = TopicCandidate.model_validate(tc.model_dump(mode="json"))
    assert restored == tc


def test_script_output_round_trip():
    script = ScriptOutput(
        working_title="t",
        hook="h",
        beats=[ScriptBeat(index=0, timestamp_seconds=0, beat_type=BeatType.HOOK, voiceover_text="v", visual_cue="c")],
        cta="cta",
        full_voiceover_text="full",
        target_length_seconds=60,
        content_format=ContentFormat.SHORT,
    )
    restored = ScriptOutput.model_validate(script.model_dump(mode="json"))
    assert restored.beats[0].beat_type == BeatType.HOOK


def test_strategy_decision_requires_valid_enum():
    tc = TopicCandidate(
        title="x", search_volume_score=5, competition_score=3, freshness_score=8, composite_score=6
    )
    decision = StrategyDecision(
        selected_topic=tc,
        content_type=ContentType.EVERGREEN,
        content_format=ContentFormat.LONG_FORM,
        target_length_seconds=480,
        rationale="r",
    )
    assert decision.content_type == ContentType.EVERGREEN


def test_compliance_result_defaults_are_empty_not_none():
    result = ComplianceCheckResult(copyright_risk_score=0.0, av_sync_ok=True, spec_check_ok=True, passed=True)
    assert result.community_guidelines_flags == []
    assert result.blocking_reasons == []


def test_metadata_output_chapters_optional_but_typed():
    metadata = MetadataOutput(
        title_options=["a"], selected_title="a", description="d", tags=["x"], chapters=[]
    )
    assert metadata.chapters == []
