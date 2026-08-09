"""Script Writer Agent, Script QA Critic, and Fact-Check Agent — real logic.

The Script QA critic is the hard originality/plagiarism gate: embeds the draft and compares it
against TranscriptEmbedding (top existing videos — currently unpopulated, no ingestion job exists
yet; degrades gracefully to "no matches") and ScriptEmbedding (our own back catalog, populated by
this node on every pass) via pgvector cosine similarity, rejecting above
settings.originality_similarity_threshold. It also extracts checkable factual claims that
fact_check_node then verifies.
"""
from pydantic import BaseModel, Field

from agents.schemas.research import StrategyDecision
from agents.schemas.script import (
    BeatType,
    ClaimFlag,
    ClaimVerdict,
    ClaimVerification,
    FactCheckOutput,
    ScriptBeat,
    ScriptOutput,
    ScriptQAResult,
    SimilarityMatch,
)
from core.llm import call_structured
from core.logging import get_logger
from core.settings import get_settings
from graph.nodes._helpers import (
    bumped_retry_counts,
    consume_force_reject,
    current_retry_count,
    log_and_trace,
)
from graph.state import PipelineState
from tools import embeddings as embeddings_tool
from tools import research as research_tool
from tools.research import ResearchNotConfiguredError

logger = get_logger("graph.nodes.script")

STAGE_SCRIPT = "script_writer"
STAGE_QA = "critic_script_qa"
STAGE_FACT = "fact_check"

WORDS_PER_SECOND = 2.5  # spoken-pace estimate used to derive beat timestamps/durations


# --- Script Writer: LLM drafts beats, code computes timestamps (objective, not LLM-guessed) ---


class _BeatDraft(BaseModel):
    beat_type: BeatType
    voiceover_text: str = Field(description="What the narrator says during this beat")
    visual_cue: str = Field(description="Brief directorial hint for what's on screen — not spoken")


class _ScriptDraft(BaseModel):
    working_title: str
    hook: str = Field(description="Opening line(s) — should match the first beat's voiceover_text")
    beats: list[_BeatDraft]
    cta: str


_SCRIPT_WRITER_SYSTEM_PROMPT = (
    "You are a YouTube scriptwriter. Write a complete, narratively ordered script (hook first, cta "
    "last) for the given topic and format. Write natural spoken narration, not essay prose — short "
    "sentences, direct address to the viewer. Each beat's visual_cue is a brief directorial note "
    "(what's shown on screen), never part of what's spoken."
)


def script_writer_node(state: PipelineState) -> dict:
    revision_number = current_retry_count(state, STAGE_QA)
    feedback = (state.get("script_qa_result") or {}).get("feedback_for_retry")
    trace = log_and_trace(STAGE_SCRIPT, "start", revision=revision_number, feedback=feedback)

    strategy = StrategyDecision.model_validate(state["strategy_decision"])
    target_words = round(strategy.target_length_seconds * WORDS_PER_SECOND)

    prompt = (
        f"Topic: {strategy.selected_topic.title}\n"
        f"Description: {strategy.selected_topic.description}\n"
        f"Content type: {strategy.content_type}, format: {strategy.content_format}\n"
        f"Target length: {strategy.target_length_seconds}s (~{target_words} spoken words total "
        f"across all beats, at ~{WORDS_PER_SECOND} words/sec)\n"
        f"Strategy rationale: {strategy.rationale}"
    )
    if feedback:
        prompt = f"Revise the previous draft to address this critic feedback: {feedback}\n\n{prompt}"

    draft = call_structured(prompt, _ScriptDraft, system=_SCRIPT_WRITER_SYSTEM_PROMPT)

    beats: list[ScriptBeat] = []
    cursor = 0.0
    for index, beat_draft in enumerate(draft.beats):
        word_count = len(beat_draft.voiceover_text.split())
        duration = max(word_count / WORDS_PER_SECOND, 1.0)
        beats.append(
            ScriptBeat(
                index=index,
                timestamp_seconds=round(cursor, 2),
                beat_type=beat_draft.beat_type,
                voiceover_text=beat_draft.voiceover_text,
                visual_cue=beat_draft.visual_cue,
            )
        )
        cursor += duration

    output = ScriptOutput(
        working_title=draft.working_title,
        hook=draft.hook,
        beats=beats,
        cta=draft.cta,
        full_voiceover_text=" ".join(b.voiceover_text for b in beats),
        target_length_seconds=strategy.target_length_seconds,
        content_format=strategy.content_format,
        revision_number=revision_number,
    )

    return {
        "script_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_SCRIPT, "complete", revision=revision_number, beats=len(beats))],
    }


# --- Script QA critic: pgvector originality gate + claim extraction for fact_check_node ---


class _ExtractedClaim(BaseModel):
    beat_index: int
    claim_text: str
    reason: str = Field(description="Why this claim is worth fact-checking")


class _ClaimExtraction(BaseModel):
    claims: list[_ExtractedClaim] = Field(default_factory=list)


_CLAIM_EXTRACTION_SYSTEM_PROMPT = (
    "Identify 0-3 specific, checkable factual claims (statistics, historical facts, scientific "
    "claims) in this script that would benefit from verification. Skip opinions and subjective "
    "statements. If there are no checkable factual claims, return an empty list."
)


def critic_script_qa_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_QA, "start")

    forced_reject = consume_force_reject(state, STAGE_QA)
    retry_count = current_retry_count(state, STAGE_QA)

    if forced_reject:
        result = ScriptQAResult(
            originality_score=3.0,
            max_similarity_score=0.9,
            similarity_matches=[],
            flagged_claims=[],
            policy_flags=["[STUB] forced rejection for demo"],
            passed=False,
            feedback_for_retry="[STUB] Rewrite to reduce similarity to source material.",
            retry_count=retry_count,
        )
    else:
        settings = get_settings()
        script = ScriptOutput.model_validate(state["script_output"])

        vector = embeddings_tool.embed_text(script.full_voiceover_text)
        transcript_matches = embeddings_tool.most_similar_transcript(vector)
        script_matches = embeddings_tool.most_similar_script(vector, exclude_run_id=state["run_id"])
        all_matches = transcript_matches + script_matches
        max_sim = max((score for _source, score in all_matches), default=0.0)
        passed = max_sim <= settings.originality_similarity_threshold

        similarity_matches = [
            SimilarityMatch(source_video_id=source, similarity_score=score)
            for source, score in all_matches
            if score > 0.3  # only surface matches worth a human's attention
        ]

        claim_extraction = call_structured(
            f"Script:\n{script.full_voiceover_text}", _ClaimExtraction, system=_CLAIM_EXTRACTION_SYSTEM_PROMPT
        )
        flagged_claims = [
            ClaimFlag(beat_index=c.beat_index, claim_text=c.claim_text, reason=c.reason)
            for c in claim_extraction.claims
        ]

        if passed:
            embeddings_tool.store_script_embedding(state["run_id"], script.full_voiceover_text)

        result = ScriptQAResult(
            originality_score=round((1 - max_sim) * 10, 2),
            max_similarity_score=round(max_sim, 4),
            similarity_matches=similarity_matches,
            flagged_claims=flagged_claims,
            policy_flags=[],
            passed=passed,
            feedback_for_retry=(
                None
                if passed
                else f"Originality too close to existing content (similarity={max_sim:.2f}); "
                "rewrite the framing, examples, and wording to be more original."
            ),
            retry_count=retry_count,
        )

    return {
        "script_qa_result": result.model_dump(mode="json"),
        # bump on any real failure, not just the demo force-reject hook — route_script_qa's
        # escalation check depends on this actually incrementing, or a genuinely-failing script
        # would loop back to script_writer forever without ever escalating
        "retry_counts": bumped_retry_counts(state, STAGE_QA) if not result.passed else state.get("retry_counts", {}),
        "debug_force_reject": state.get("debug_force_reject", {}),
        "trace": [trace, log_and_trace(STAGE_QA, "complete", passed=result.passed, score=result.originality_score)],
    }


# --- Fact-check: verifies whatever critic_script_qa_node flagged, via tools.research (Tavily) ---


def _interpret_search_results(results: list[dict]) -> tuple[ClaimVerdict, float]:
    """Tavily doesn't return a boolean true/false verdict, so this is a simple heuristic — presence
    of relevant results raises confidence, not a real fact-checking oracle."""
    if not results:
        return ClaimVerdict.UNCERTAIN, 0.2
    return ClaimVerdict.VERIFIED, min(0.85, 0.5 + 0.1 * len(results))


def fact_check_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_FACT, "start")

    flagged_claims = (state.get("script_qa_result") or {}).get("flagged_claims", [])

    verifications = []
    for claim in flagged_claims:
        try:
            results = research_tool.search(claim["claim_text"], max_results=3)
            verdict, confidence = _interpret_search_results(results)
            sources = [r["url"] for r in results]
        except ResearchNotConfiguredError:
            verdict, confidence, sources = ClaimVerdict.UNVERIFIED, 0.0, []

        verifications.append(
            ClaimVerification(
                claim_text=claim["claim_text"],
                beat_index=claim["beat_index"],
                verdict=verdict,
                confidence_score=confidence,
                sources=sources,
            )
        )

    output = FactCheckOutput(verifications=verifications)

    return {
        "fact_check_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_FACT, "complete", claims=len(output.verifications))],
    }
