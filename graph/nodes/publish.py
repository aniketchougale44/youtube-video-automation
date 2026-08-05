"""Metadata/SEO, Thumbnail, Compliance Critic, Human Approval Gateway, and Upload Agent — stub logic.

human_approval_node is the interrupt() point: it pauses the graph and surfaces an ApprovalPacket to
the review dashboard. Resuming requires a Command(resume=ApprovalDecision(...).model_dump()).
"""
from langgraph.types import interrupt

from agents.schemas.publish import (
    ApprovalDecision,
    ApprovalPacket,
    ChapterMarker,
    ComplianceCheckResult,
    MetadataOutput,
    ThumbnailCandidate,
    ThumbnailOutput,
    UploadResult,
    UploadStatus,
    Visibility,
)
from agents.schemas.script import ScriptOutput
from graph.nodes._helpers import (
    bumped_retry_counts,
    consume_force_reject,
    current_retry_count,
    log_and_trace,
)
from graph.state import PipelineState

STAGE_METADATA = "metadata_seo"
STAGE_THUMBNAIL = "thumbnail"
STAGE_COMPLIANCE = "critic_compliance"
STAGE_APPROVAL = "human_approval"
STAGE_UPLOAD = "upload"


def metadata_seo_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_METADATA, "start")

    script = ScriptOutput.model_validate(state["script_output"])

    # STUB: real logic calls the LLM for SEO-tuned title/description/tags/chapters
    output = MetadataOutput(
        title_options=[script.working_title, f"{script.working_title} (2026)"],
        selected_title=script.working_title,
        description="[STUB] placeholder description",
        tags=["stub", "placeholder"],
        chapters=[ChapterMarker(timestamp_seconds=b.timestamp_seconds, title=f"[STUB] {b.beat_type}") for b in script.beats],
        pinned_comment="[STUB] pinned comment",
    )

    return {
        "metadata_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_METADATA, "complete", title=output.selected_title)],
    }


def thumbnail_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_THUMBNAIL, "start")

    # STUB: real logic calls tools.image_gen + overlays text via Pillow, produces 2-3 candidates
    output = ThumbnailOutput(
        candidates=[
            ThumbnailCandidate(image_path="/tmp/stub_thumb_1.png", prompt_used="[STUB] prompt A", text_overlay="[STUB]"),
            ThumbnailCandidate(image_path="/tmp/stub_thumb_2.png", prompt_used="[STUB] prompt B", text_overlay="[STUB]"),
        ]
    )

    return {
        "thumbnail_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_THUMBNAIL, "complete", candidates=len(output.candidates))],
    }


def critic_compliance_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_COMPLIANCE, "start")

    forced_reject = consume_force_reject(state, STAGE_COMPLIANCE)
    retry_count = current_retry_count(state, STAGE_COMPLIANCE)

    # STUB: real logic runs Content-ID risk scan, LLM community-guidelines self-check, AV sync + spec check
    result = ComplianceCheckResult(
        copyright_risk_score=8.0 if forced_reject else 0.5,
        community_guidelines_flags=["[STUB] forced rejection for demo"] if forced_reject else [],
        av_sync_ok=not forced_reject,
        spec_check_ok=True,
        spec_check_notes=[],
        passed=not forced_reject,
        blocking_reasons=["[STUB] simulated copyright risk"] if forced_reject else [],
        retry_count=retry_count,
    )

    return {
        "compliance_result": result.model_dump(mode="json"),
        "retry_counts": bumped_retry_counts(state, STAGE_COMPLIANCE) if forced_reject else state.get("retry_counts", {}),
        "debug_force_reject": state.get("debug_force_reject", {}),
        "trace": [trace, log_and_trace(STAGE_COMPLIANCE, "complete", passed=result.passed)],
    }


def human_approval_node(state: PipelineState) -> dict:
    """Pauses the graph via interrupt(). The FastAPI review dashboard resumes it with
    Command(resume=ApprovalDecision(...).model_dump(mode="json"))."""
    trace = log_and_trace(STAGE_APPROVAL, "awaiting_interrupt")

    script = ScriptOutput.model_validate(state["script_output"])
    metadata = MetadataOutput.model_validate(state["metadata_output"])
    thumbnails = ThumbnailOutput.model_validate(state["thumbnail_output"])
    compliance = ComplianceCheckResult.model_validate(state["compliance_result"])

    packet = ApprovalPacket(
        run_id=state["run_id"],
        script_summary=script.hook,
        thumbnail_options=[c.image_path for c in thumbnails.candidates],
        title=metadata.selected_title,
        description=metadata.description,
        compliance_result=compliance,
    )

    decision_raw = interrupt(packet.model_dump(mode="json"))
    decision = ApprovalDecision.model_validate(decision_raw)

    return {
        "approval_decision": decision.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_APPROVAL, "resumed", approved=decision.approved)],
    }


def auto_approve_node(state: PipelineState) -> dict:
    """Used instead of human_approval_node when REQUIRE_HUMAN_APPROVAL=false."""
    trace = log_and_trace(STAGE_APPROVAL, "auto_approved")
    decision = ApprovalDecision(run_id=state["run_id"], approved=True, reviewer="system:auto_approve")
    return {
        "approval_decision": decision.model_dump(mode="json"),
        "trace": [trace],
    }


def upload_node(state: PipelineState) -> dict:
    metadata = MetadataOutput.model_validate(state["metadata_output"])
    trace = log_and_trace(STAGE_UPLOAD, "start", title=metadata.selected_title)

    # STUB: real logic calls tools.youtube.resumable_upload(file_path, metadata, ...), gated by the quota limiter
    result = UploadResult(
        youtube_video_id="stub_yt_id_000",
        status=UploadStatus.UPLOADED,
        visibility=Visibility.UNLISTED,
        quota_units_used=1600,
    )

    return {
        "upload_result": result.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_UPLOAD, "complete", youtube_video_id=result.youtube_video_id)],
    }
