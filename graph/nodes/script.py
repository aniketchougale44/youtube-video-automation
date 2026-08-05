"""Script Writer Agent, Script QA Critic, and Fact-Check Agent — stub logic.

The Script QA critic is the hard originality/plagiarism gate: real implementation embeds the
draft against agents.schemas transcript_embeddings (pgvector) of top existing videos on the topic
and rejects on similarity above threshold, lifted text, or policy-violating claims.
"""
from agents.schemas.research import StrategyDecision
from agents.schemas.script import (
    BeatType,
    ClaimVerdict,
    ClaimVerification,
    FactCheckOutput,
    ScriptBeat,
    ScriptOutput,
    ScriptQAResult,
)
from graph.nodes._helpers import (
    bumped_retry_counts,
    consume_force_reject,
    current_retry_count,
    log_and_trace,
)
from graph.state import PipelineState

STAGE_SCRIPT = "script_writer"
STAGE_QA = "critic_script_qa"
STAGE_FACT = "fact_check"


def script_writer_node(state: PipelineState) -> dict:
    trace = log_and_trace(
        STAGE_SCRIPT, "start", revision=current_retry_count(state, STAGE_QA),
        feedback=(state.get("script_qa_result") or {}).get("feedback_for_retry"),
    )

    strategy = StrategyDecision.model_validate(state["strategy_decision"])
    revision_number = current_retry_count(state, STAGE_QA)

    # STUB: real logic calls the LLM with strategy + prior critic feedback in the prompt
    output = ScriptOutput(
        working_title=f"[STUB] {strategy.selected_topic.title}",
        hook="[STUB] Placeholder hook line.",
        beats=[
            ScriptBeat(index=0, timestamp_seconds=0, beat_type=BeatType.HOOK, voiceover_text="[STUB] hook", visual_cue="[STUB] cold open"),
            ScriptBeat(index=1, timestamp_seconds=10, beat_type=BeatType.STORY, voiceover_text="[STUB] body", visual_cue="[STUB] b-roll"),
            ScriptBeat(index=2, timestamp_seconds=440, beat_type=BeatType.CTA, voiceover_text="[STUB] cta", visual_cue="[STUB] end card"),
        ],
        cta="[STUB] Subscribe for more.",
        full_voiceover_text="[STUB] Full placeholder voiceover text.",
        target_length_seconds=strategy.target_length_seconds,
        content_format=strategy.content_format,
        revision_number=revision_number,
    )

    return {
        "script_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_SCRIPT, "complete", revision=revision_number)],
    }


def critic_script_qa_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_QA, "start")

    forced_reject = consume_force_reject(state, STAGE_QA)
    retry_count = current_retry_count(state, STAGE_QA)

    # STUB: real logic embeds script beats and compares to transcript_embeddings via pgvector cosine distance
    result = ScriptQAResult(
        originality_score=3.0 if forced_reject else 9.2,
        max_similarity_score=0.9 if forced_reject else 0.12,
        similarity_matches=[],
        flagged_claims=[],
        policy_flags=["[STUB] forced rejection for demo"] if forced_reject else [],
        passed=not forced_reject,
        feedback_for_retry="[STUB] Rewrite to reduce similarity to source material." if forced_reject else None,
        retry_count=retry_count,
    )

    return {
        "script_qa_result": result.model_dump(mode="json"),
        "retry_counts": bumped_retry_counts(state, STAGE_QA) if forced_reject else state.get("retry_counts", {}),
        "debug_force_reject": state.get("debug_force_reject", {}),
        "trace": [trace, log_and_trace(STAGE_QA, "complete", passed=result.passed, score=result.originality_score)],
    }


def fact_check_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_FACT, "start")

    # STUB: real logic calls tools.research (Tavily) per flagged claim from the QA critic
    output = FactCheckOutput(
        verifications=[
            ClaimVerification(
                claim_text="[STUB] placeholder claim",
                beat_index=1,
                verdict=ClaimVerdict.VERIFIED,
                confidence_score=0.9,
                sources=["https://example.com/stub-source"],
            )
        ]
    )

    return {
        "fact_check_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_FACT, "complete", claims=len(output.verifications))],
    }
