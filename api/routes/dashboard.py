"""Live operations dashboard: every run, its real-time pipeline stage, recent agent activity,
and cost — plus the human-approval review flow (list runs awaiting approval, inspect the
ApprovalPacket surfaced by the human_approval_node interrupt(), approve or reject)."""
import json
import os
import time
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agents.schemas.common import RunStatus
from core.settings import get_settings
from db.base import SessionLocal, get_db
from db.crud import get_run, list_runs
from db.models import AgentLog, Cost, Run, Video
from graph.checkpointer import postgres_checkpointer
from graph.run import get_state
from tools.cache import cache_scan_json
from tools.script_progress import KEY_PREFIX as SCRIPT_RUN_KEY_PREFIX
from worker.tasks import resume_pipeline_task

router = APIRouter()
templates = Jinja2Templates(directory="api/templates")

# Mirrors the node order wired in graph/builder.py — used only to place a run's live position on
# the progress bar, not to drive any control flow.
PIPELINE_NODE_ORDER = [
    "trend_research", "strategy", "script_writer", "critic_script_qa",
    "fact_check", "visual_planning", "asset_visual", "voiceover",
    "video_assembly", "metadata_seo", "thumbnail", "critic_compliance",
    "approval_gate", "upload",
]
_LIVE_STATUSES = (RunStatus.PENDING, RunStatus.RUNNING)
_TERMINAL_LABELS = {
    RunStatus.PUBLISHED: "published",
    RunStatus.REJECTED: "rejected",
    RunStatus.FAILED: "failed",
    RunStatus.ESCALATED: "escalated",
}


def _media_url(fs_path: str | None) -> str | None:
    """Turn a filesystem path a run wrote (a thumbnail candidate, the final render — always under
    settings.media_dir) into a URL the browser can load from the /media static mount in api/main.py.
    Returns None for a missing path or anything that resolves outside media_dir (so a stray absolute
    path in the DB can't be used to read arbitrary files through the dashboard)."""
    if not fs_path:
        return None
    try:
        rel = os.path.relpath(os.path.abspath(fs_path), os.path.abspath(get_settings().media_dir))
    except ValueError:  # e.g. different drive on Windows
        return None
    if rel.startswith(".."):
        return None
    return "/media/" + rel.replace(os.sep, "/")


def _youtube_url(video_id: str | None) -> str | None:
    return f"https://youtube.com/watch?v={video_id}" if video_id else None


def _run_duration_seconds(run: Run) -> float | None:
    """Wall-clock time from run creation to its last update — the elapsed time shown per row. For a
    still-running row updated_at keeps advancing, so this tracks live; for a terminal row it's the
    final total."""
    if not run.created_at or not run.updated_at:
        return None
    return max(0.0, (run.updated_at - run.created_at).total_seconds())


def _checkpoint_snapshot(run):
    """The run's live LangGraph StateSnapshot (channel values + pending interrupts), or None when
    the checkpoint store can't be reached. The checkpoint — not Postgres — is the source of truth
    for anything mid-flight, so both the approval packet and the full script are read from here."""
    try:
        with postgres_checkpointer() as cp:
            return get_state(cp, run.langgraph_thread_id)
    except Exception:
        return None


def _packet_from_snapshot(snapshot) -> dict | None:
    if snapshot is None:
        return None
    for task in snapshot.tasks:
        for pending_interrupt in getattr(task, "interrupts", []) or []:
            return pending_interrupt.value
    return None


def _extract_approval_packet(run) -> dict | None:
    return _packet_from_snapshot(_checkpoint_snapshot(run))


def _script_view(script_output: dict | None) -> dict | None:
    """The full script, flattened for the reviewer's read-along pane next to the video player:
    working title, hook, every beat's narration + visual cue, and the CTA."""
    if not script_output:
        return None
    return {
        "working_title": script_output.get("working_title") or "",
        "hook": script_output.get("hook") or "",
        "cta": script_output.get("cta") or "",
        "full_text": script_output.get("full_voiceover_text") or "",
        "target_length_seconds": script_output.get("target_length_seconds"),
        "beats": [
            {
                "index": b.get("index"),
                "type": (b.get("beat_type") or "").replace("_", " "),
                "timestamp_seconds": b.get("timestamp_seconds"),
                "voiceover_text": b.get("voiceover_text") or "",
                "visual_cue": b.get("visual_cue") or "",
            }
            for b in script_output.get("beats") or []
        ],
    }


def _approval_rows(db: Session, runs) -> list[dict]:
    """One JSON-friendly card per awaiting-approval run for the dashboard's approval queue: the
    finished video (`render_url`), the full script (`script`) for side-by-side review, the
    thumbnail candidates, and flattened compliance. `has_packet` is False when the checkpoint
    can't be reached (worker/Redis down); the card then degrades to an id + 'open full review'
    link instead of vanishing."""
    rows = []
    for run in runs:
        snapshot = _checkpoint_snapshot(run)
        packet = _packet_from_snapshot(snapshot) or {}
        values = getattr(snapshot, "values", None) or {}
        compliance = packet.get("compliance_result") or None

        render_path = (values.get("assembly_output") or {}).get("render_path")
        if not render_path:
            video = db.execute(
                select(Video).where(Video.run_id == run.id).order_by(Video.created_at.desc())
            ).scalars().first()
            render_path = video.render_path if video else None

        rows.append(
            {
                "id": str(run.id),
                "short_id": str(run.id)[:8],
                "requested_at": run.updated_at.isoformat() if run.updated_at else None,
                "has_packet": bool(packet),
                "title": packet.get("title") or (run.selected_topic or {}).get("title") or "Untitled run",
                "description": packet.get("description") or "",
                "script_summary": packet.get("script_summary") or "",
                "script": _script_view(values.get("script_output")),
                "render_url": _media_url(render_path),
                "thumbnails": [
                    {"index": i, "url": _media_url(path)}
                    for i, path in enumerate(packet.get("thumbnail_options") or [])
                ],
                "compliance": {
                    "copyright_risk_score": compliance["copyright_risk_score"],
                    "av_sync_ok": compliance["av_sync_ok"],
                    "spec_check_ok": compliance["spec_check_ok"],
                    "passed": compliance["passed"],
                    "blocking_reasons": compliance.get("blocking_reasons") or [],
                    "community_guidelines_flags": compliance.get("community_guidelines_flags") or [],
                }
                if compliance
                else None,
            }
        )
    return rows


def _live_node(run: Run) -> str | None:
    """Which graph node is executing right now, read straight from the LangGraph checkpoint.
    The checkpointer persists state after every completed superstep (see graph/checkpointer.py),
    so this reflects genuine mid-flight progress for a run the worker is actively executing —
    not just the coarse status column, which only changes at task start/end."""
    if run.status not in _LIVE_STATUSES:
        return None
    try:
        with postgres_checkpointer() as cp:
            snapshot = get_state(cp, run.langgraph_thread_id)
    except Exception:
        return None
    next_nodes = list(snapshot.next or ())
    return next_nodes[0] if next_nodes else None


def _stage_progress(run: Run) -> tuple[str, int]:
    """(human label, completed-stage index) for a run's progress bar."""
    total = len(PIPELINE_NODE_ORDER)
    if run.status in _LIVE_STATUSES:
        node = _live_node(run)
        if node is None:
            return "queued", 0
        idx = PIPELINE_NODE_ORDER.index(node) if node in PIPELINE_NODE_ORDER else 0
        return node.replace("_", " "), idx
    if run.status == RunStatus.AWAITING_APPROVAL:
        return "awaiting approval", PIPELINE_NODE_ORDER.index("approval_gate")
    return _TERMINAL_LABELS.get(run.status, run.status.value), total


def _script_runs() -> list[dict]:
    """Background activity from the standalone scripts/produce_*.py one-offs (see
    tools/script_progress.py) -- these bypass the LangGraph pipeline entirely, so they'd
    otherwise be completely invisible here even while actively generating/rendering/uploading."""
    try:
        rows = cache_scan_json(SCRIPT_RUN_KEY_PREFIX)
    except Exception:
        return []
    rows.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
    return rows


def _build_snapshot(db: Session) -> dict:
    runs = list_runs(db, limit=50)
    total_stages = len(PIPELINE_NODE_ORDER)
    now = datetime.now(UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    status_counts = dict(db.execute(select(Run.status, func.count()).group_by(Run.status)).all())
    cost_today = db.execute(
        select(func.coalesce(func.sum(Cost.total_usd), 0.0)).where(Cost.created_at >= today_start)
    ).scalar_one()
    cost_total = db.execute(select(func.coalesce(func.sum(Cost.total_usd), 0.0))).scalar_one()

    # Per-run spend and produced-video info for the 50 rows on screen, fetched in one query each
    # rather than per row.
    run_ids = [run.id for run in runs]
    cost_by_run = dict(
        db.execute(
            select(Cost.run_id, func.coalesce(func.sum(Cost.total_usd), 0.0))
            .where(Cost.run_id.in_(run_ids))
            .group_by(Cost.run_id)
        ).all()
    ) if run_ids else {}
    video_by_run: dict = {}
    if run_ids:
        for video in db.execute(
            select(Video).where(Video.run_id.in_(run_ids)).order_by(Video.created_at)
        ).scalars():
            video_by_run[video.run_id] = video  # last (newest) wins if a run somehow has several

    run_rows = []
    for run in runs:
        label, idx = _stage_progress(run)
        video = video_by_run.get(run.id)
        run_rows.append(
            {
                "id": str(run.id),
                "short_id": str(run.id)[:8],
                "topic": (video.title if video and video.title else None)
                or (run.selected_topic or {}).get("title")
                or "—",
                "status": run.status.value,
                "stage_label": label,
                "progress": idx,
                "progress_total": total_stages,
                "error": run.error,
                "cost_usd": round(float(cost_by_run.get(run.id, 0.0)), 2),
                "duration_seconds": _run_duration_seconds(run),
                "youtube_url": _youtube_url(video.youtube_video_id if video else None),
                "render_url": _media_url(video.render_path if video else None),
                "thumbnail_url": _media_url(video.thumbnail_path if video else None),
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "updated_at": run.updated_at.isoformat() if run.updated_at else None,
            }
        )

    log_rows = db.execute(
        select(AgentLog, Run.id).join(Run, AgentLog.run_id == Run.id).order_by(AgentLog.created_at.desc()).limit(40)
    ).all()
    log_rows_out = [
        {
            "run_short": str(run_id)[:8],
            "stage": log.stage.value,
            "agent_name": log.agent_name,
            "is_critic": log.is_critic,
            "passed": log.passed,
            "critic_score": log.critic_score,
            "error": log.error,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log, run_id in log_rows
    ]

    return {
        "generated_at": now.isoformat(),
        "stats": {
            "total": sum(status_counts.values()),
            "pending": status_counts.get(RunStatus.PENDING, 0),
            "running": status_counts.get(RunStatus.RUNNING, 0),
            "awaiting_approval": status_counts.get(RunStatus.AWAITING_APPROVAL, 0),
            "published": status_counts.get(RunStatus.PUBLISHED, 0),
            "needs_attention": status_counts.get(RunStatus.FAILED, 0) + status_counts.get(RunStatus.ESCALATED, 0),
            "cost_today_usd": round(float(cost_today), 2),
            "cost_total_usd": round(float(cost_total), 2),
        },
        "approvals": _approval_rows(db, list_runs(db, status=RunStatus.AWAITING_APPROVAL)),
        "runs": run_rows,
        "logs": log_rows_out,
        "script_runs": _script_runs(),
    }


@router.get("/")
def dashboard_index(request: Request, db: Session = Depends(get_db)):
    # The approval queue, run table, and activity feed all render client-side from this one
    # snapshot (embedded for first paint, then refreshed over /stream) so nothing on the page
    # needs a manual reload to stay current.
    return templates.TemplateResponse(request, "index.html", {"snapshot": _build_snapshot(db)})


@router.get("/stream")
def dashboard_stream():
    """Server-Sent Events feed: pushes a fresh snapshot (run statuses, live pipeline position,
    recent agent activity, cost) every 2s so the dashboard updates without polling or a reload."""

    def event_source():
        while True:
            db = SessionLocal()
            try:
                payload = _build_snapshot(db)
                yield f"data: {json.dumps(payload)}\n\n"
            except Exception as exc:
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            finally:
                db.close()
            time.sleep(2)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{run_id}")
def dashboard_review(request: Request, run_id: uuid.UUID, db: Session = Depends(get_db)):
    run = get_run(db, run_id)
    if run is None:
        raise HTTPException(404, "run not found")

    snapshot = _checkpoint_snapshot(run)
    packet = _packet_from_snapshot(snapshot)
    values = getattr(snapshot, "values", None) or {}

    video = db.execute(
        select(Video).where(Video.run_id == run.id).order_by(Video.created_at.desc())
    ).scalars().first()
    cost_usd = db.execute(
        select(func.coalesce(func.sum(Cost.total_usd), 0.0)).where(Cost.run_id == run.id)
    ).scalar_one()

    render_path = (values.get("assembly_output") or {}).get("render_path") or (
        video.render_path if video else None
    )
    thumb_urls = [_media_url(p) for p in (packet or {}).get("thumbnail_options", [])]
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "run": run,
            "packet": packet,
            "script": _script_view(values.get("script_output")),
            "thumb_urls": thumb_urls,
            "video": video,
            "youtube_url": _youtube_url(video.youtube_video_id if video else None),
            "render_url": _media_url(render_path),
            "cost_usd": round(float(cost_usd), 2),
            "duration_seconds": _run_duration_seconds(run),
        },
    )


@router.post("/{run_id}/approve")
def dashboard_approve(
    run_id: uuid.UUID,
    reviewer: str = Form("dashboard"),
    notes: str = Form(""),
    selected_thumbnail_index: int = Form(0),
    db: Session = Depends(get_db),
):
    run = get_run(db, run_id)
    if run is None or run.status != RunStatus.AWAITING_APPROVAL:
        raise HTTPException(409, "run not awaiting approval")
    decision = {
        "run_id": str(run_id),
        "approved": True,
        "selected_thumbnail_index": selected_thumbnail_index,
        "reviewer": reviewer,
        "notes": notes,
    }
    resume_pipeline_task.delay(str(run_id), decision)
    return RedirectResponse(url="/dashboard/", status_code=303)


@router.post("/{run_id}/reject")
def dashboard_reject(run_id: uuid.UUID, reviewer: str = Form("dashboard"), notes: str = Form(""), db: Session = Depends(get_db)):
    run = get_run(db, run_id)
    if run is None or run.status != RunStatus.AWAITING_APPROVAL:
        raise HTTPException(409, "run not awaiting approval")
    decision = {"run_id": str(run_id), "approved": False, "reviewer": reviewer, "notes": notes}
    resume_pipeline_task.delay(str(run_id), decision)
    return RedirectResponse(url="/dashboard/", status_code=303)
