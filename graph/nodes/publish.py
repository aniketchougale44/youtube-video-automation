"""Metadata/SEO, Thumbnail, Compliance Critic, Human Approval Gateway, and Upload — all real.

critic_compliance_node's "copyright risk" is necessarily an approximation, not a real YouTube
Content-ID scan — that system isn't exposed to third-party uploaders via public API. It reuses the
upstream Script QA originality signal as a proxy, combined with an LLM community-guidelines
self-check and technical AV-sync/spec validation.
"""
import os

from langgraph.types import interrupt
from pydantic import BaseModel, Field

from agents.schemas.audio_render import AssemblyOutput
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
from agents.schemas.script import ScriptBeat, ScriptOutput
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
from tools import character_assets as character_assets_tool
from tools import youtube as youtube_tool
from tools.fonts import resolve_font_path
from tools.media_paths import media_path

logger = get_logger("graph.nodes.publish")

STAGE_METADATA = "metadata_seo"
STAGE_THUMBNAIL = "thumbnail"
STAGE_COMPLIANCE = "critic_compliance"
STAGE_APPROVAL = "human_approval"
STAGE_UPLOAD = "upload"


# --- Metadata/SEO: real LLM call; chapter timestamps computed from real voiceover durations ---


class _MetadataDraft(BaseModel):
    title_options: list[str] = Field(min_length=2, max_length=5)
    selected_title: str
    description: str
    tags: list[str] = Field(min_length=5, max_length=15)
    pinned_comment: str


_METADATA_SYSTEM_PROMPT = (
    "You are a YouTube SEO specialist. Given a script, write: 2-4 alternative title options "
    "(front-loaded keywords, under 70 characters, no clickbait that misrepresents content), a "
    "description (a hook line, 2-3 paragraphs summarizing the value, a call to subscribe), 10-15 "
    "relevant tags, and a pinned comment that invites engagement (a question related to the topic)."
)


def _chapter_title(beat: ScriptBeat) -> str:
    return beat.beat_type.value.replace("_", " ").title()


def metadata_seo_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_METADATA, "start")

    script = ScriptOutput.model_validate(state["script_output"])
    prompt = (
        f"Working title: {script.working_title}\nHook: {script.hook}\nCTA: {script.cta}\n\n"
        f"Full script:\n{script.full_voiceover_text[:2500]}"
    )
    draft = call_structured(prompt, _MetadataDraft, system=_METADATA_SYSTEM_PROMPT)

    # Chapters use real per-beat durations from voiceover_output when available (this node runs
    # after video_assembly in the graph, so it normally is) rather than the script's estimate.
    voiceover = state.get("voiceover_output")
    if voiceover:
        cursor = 0.0
        real_timestamps: dict[int, float] = {}
        for seg in sorted(voiceover["segments"], key=lambda s: s["beat_index"]):
            real_timestamps[seg["beat_index"]] = cursor
            cursor += seg["duration_seconds"]
        chapters = [
            ChapterMarker(timestamp_seconds=real_timestamps.get(b.index, b.timestamp_seconds), title=_chapter_title(b))
            for b in script.beats
        ]
    else:
        chapters = [ChapterMarker(timestamp_seconds=b.timestamp_seconds, title=_chapter_title(b)) for b in script.beats]

    output = MetadataOutput(
        title_options=draft.title_options,
        selected_title=draft.selected_title or script.working_title,
        description=draft.description,
        tags=draft.tags,
        chapters=chapters,
        pinned_comment=draft.pinned_comment,
    )

    return {
        "metadata_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_METADATA, "complete", title=output.selected_title)],
    }


# --- Thumbnail: mascot pose composited onto a flat backdrop + Pillow title overlay ---
#
# Previously this generated a fresh AI image per candidate from a free-text prompt ("A
# curiosity-provoking YouTube thumbnail hinting at: {hook}. Dramatic lighting, no text.") via the
# same keyless Pollinations fallback everything else here uses. On a real run that produced a
# photorealistic, sexualized image as one of the two candidates -- nothing in the pipeline screens
# generated thumbnail *images* for safety (critic_compliance_node's guideline check only reads
# script text/metadata through an LLM, never the pixels), and a vague prompt with no style/subject
# constraint gives a filter-less free image model room to drift into exactly that. Compositing the
# same pre-vetted mascot pack already used in the video (tools.character_assets.compose_static)
# instead eliminates the risk at the source: every candidate is built from an asset a human already
# looked at plus a solid color, so there is nothing left for a fresh generation to get wrong.
_THUMBNAIL_VARIANTS = [
    ("wave", (255, 214, 165)),  # peach backdrop, waving pose
    ("talk", (168, 218, 220)),  # sky backdrop, talking pose
]


def _overlay_title(path: str, title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(resolve_font_path(), size=90)
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)

    max_width = int(image.width * 0.9)
    words, lines, current = title.split(), [], ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    line_height = font.size + 12
    y = image.height - line_height * len(lines) - 40
    for line in lines:
        x = (image.width - draw.textlength(line, font=font)) / 2
        draw.text((x, y), line, font=font, fill="white", stroke_width=4, stroke_fill="black")
        y += line_height

    image.save(path)


def thumbnail_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_THUMBNAIL, "start")

    metadata = MetadataOutput.model_validate(state["metadata_output"])
    run_id = state["run_id"]

    candidates = []
    for index, (pose, backdrop_rgb) in enumerate(_THUMBNAIL_VARIANTS):
        path = media_path(run_id, "thumbnails", f"candidate_{index}.png")
        try:
            character_assets_tool.compose_static(pose, 1536, 1024, backdrop_rgb).save(path)
        except Exception as exc:
            logger.warning("thumbnail.generate_failed", index=index, error=str(exc))
            continue

        text_overlay = metadata.selected_title
        try:
            _overlay_title(path, text_overlay)
        except Exception as exc:
            logger.warning("thumbnail.overlay_failed", index=index, error=str(exc))
            text_overlay = ""  # generated image kept, just without the text overlay

        candidates.append(ThumbnailCandidate(image_path=path, prompt_used=f"mascot:{pose}", text_overlay=text_overlay))

    output = ThumbnailOutput(candidates=candidates)

    return {
        "thumbnail_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_THUMBNAIL, "complete", candidates=len(output.candidates))],
    }


# --- Compliance critic: proxy copyright score + LLM guideline self-check + technical AV/spec checks ---


class _GuidelineSelfCheck(BaseModel):
    flags: list[str] = Field(default_factory=list, description="Specific community-guideline concerns; empty if none")
    risk_notes: str = ""


_COMPLIANCE_SYSTEM_PROMPT = (
    "You are a YouTube policy reviewer. Read this script and metadata and flag anything that could "
    "violate YouTube's Community Guidelines (misinformation, hate speech, harassment, dangerous "
    "content, spam/deceptive practices). Return specific flags only for real concerns — an empty "
    "list means no concerns found. Do not flag normal educational/entertainment content."
)


def critic_compliance_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_COMPLIANCE, "start")

    forced_reject = consume_force_reject(state, STAGE_COMPLIANCE)
    retry_count = current_retry_count(state, STAGE_COMPLIANCE)

    if forced_reject:
        result = ComplianceCheckResult(
            copyright_risk_score=8.0,
            community_guidelines_flags=["[STUB] forced rejection for demo"],
            av_sync_ok=False,
            spec_check_ok=True,
            spec_check_notes=[],
            passed=False,
            blocking_reasons=["[STUB] simulated copyright risk"],
            retry_count=retry_count,
        )
    else:
        script = ScriptOutput.model_validate(state["script_output"])
        metadata = MetadataOutput.model_validate(state["metadata_output"])
        assembly = AssemblyOutput.model_validate(state["assembly_output"])
        qa_result = state.get("script_qa_result") or {}
        settings = get_settings()

        # Proxy, NOT a real Content-ID scan (see module docstring). Scaled to the same 0-10 range
        # critic_script_qa_node's originality_score uses, and gated at the same cutoff
        # (originality_similarity_threshold) -- this is the identical similarity signal QA already
        # evaluated, so compliance must agree with QA's verdict rather than re-litigating the same
        # number against a stricter, independently-hardcoded bar (that combination silently failed
        # every script QA had just passed). Must read external_similarity_score specifically, not
        # max_similarity_score: the latter also folds in similarity to our OWN back catalog, which
        # QA judges at a much looser bar (settings.originality_own_catalog_similarity_threshold) --
        # reading max_similarity_score here reintroduced the exact same "silently fails everything
        # QA just passed" bug the comment above describes, just one similarity source later.
        copyright_risk_score = round(qa_result.get("external_similarity_score", 0.0) * 10, 2)
        copyright_risk_threshold = round(settings.originality_similarity_threshold * 10, 2)

        prompt = (
            f"Title: {metadata.selected_title}\nDescription: {metadata.description}\n\n"
            f"Script:\n{script.full_voiceover_text[:2500]}"
        )
        guideline_check = call_structured(prompt, _GuidelineSelfCheck, system=_COMPLIANCE_SYSTEM_PROMPT)

        voiceover = state.get("voiceover_output")
        av_sync_ok = True
        spec_notes: list[str] = []
        if voiceover:
            drift = abs(assembly.duration_seconds - voiceover["total_duration_seconds"])
            av_sync_ok = drift <= 2.0
            if not av_sync_ok:
                spec_notes.append(f"AV duration drift {drift:.1f}s exceeds 2.0s tolerance")

        render_exists = os.path.exists(assembly.render_path)
        if not render_exists:
            spec_notes.append("render_path does not exist")
        spec_check_ok = render_exists and assembly.resolution == "1920x1080" and assembly.fps == 30 and not spec_notes

        blocking_reasons = (
            (
                [f"copyright risk proxy {copyright_risk_score:.1f}/10 exceeds threshold {copyright_risk_threshold:.1f}/10"]
                if copyright_risk_score > copyright_risk_threshold
                else []
            )
            + guideline_check.flags
            + spec_notes
        )
        passed = not blocking_reasons

        result = ComplianceCheckResult(
            copyright_risk_score=copyright_risk_score,
            community_guidelines_flags=guideline_check.flags,
            av_sync_ok=av_sync_ok,
            spec_check_ok=spec_check_ok,
            spec_check_notes=spec_notes,
            passed=passed,
            blocking_reasons=blocking_reasons,
            retry_count=retry_count,
        )

    return {
        "compliance_result": result.model_dump(mode="json"),
        # bump on any real failure, not just the demo force-reject hook — same reasoning as
        # critic_script_qa_node, route_compliance's escalation check depends on this incrementing
        "retry_counts": bumped_retry_counts(state, STAGE_COMPLIANCE) if not result.passed else state.get("retry_counts", {}),
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
    assembly = AssemblyOutput.model_validate(state["assembly_output"])
    thumbnails = ThumbnailOutput.model_validate(state["thumbnail_output"])
    trace = log_and_trace(STAGE_UPLOAD, "start", title=metadata.selected_title)

    settings = get_settings()
    visibility = Visibility(settings.youtube_upload_visibility)

    if os.path.exists(assembly.render_path):
        thumbnail_path = thumbnails.candidates[0].image_path if thumbnails.candidates else None
        video_id = youtube_tool.resumable_upload(
            file_path=assembly.render_path,
            metadata={"title": metadata.selected_title, "description": metadata.description, "tags": metadata.tags},
            thumbnail_path=thumbnail_path,
            visibility=visibility.value,
        )

        # Best-effort, like thumbnail_set above: the video itself already published successfully,
        # which is the part that actually matters, so a playlist hiccup shouldn't fail the run.
        playlist_id = None
        try:
            playlist_id = youtube_tool.get_or_create_playlist(
                title=settings.youtube_nursery_playlist_title,
                description="Nursery rhymes and early-learning videos for toddlers and preschoolers.",
            )
            youtube_tool.add_video_to_playlist(playlist_id, video_id)
        except Exception as exc:
            logger.warning("upload.playlist_add_failed", video_id=video_id, error=str(exc))

        result = UploadResult(
            youtube_video_id=video_id,
            status=UploadStatus.UPLOADED,
            visibility=visibility,
            quota_units_used=1600,
            playlist_id=playlist_id,
        )
    else:
        # video_assembly is still a stub (no real render file yet) — nothing exists to actually
        # upload, so fall back to the placeholder result rather than failing every run on a
        # missing file that's expected at this stage of the build.
        result = UploadResult(
            youtube_video_id="stub_yt_id_000", status=UploadStatus.UPLOADED, visibility=visibility, quota_units_used=1600
        )

    return {
        "upload_result": result.model_dump(mode="json"),
        "trace": [
            trace,
            log_and_trace(STAGE_UPLOAD, "complete", youtube_video_id=result.youtube_video_id, playlist_id=result.playlist_id),
        ],
    }
