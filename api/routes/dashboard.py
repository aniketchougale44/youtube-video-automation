"""Live operations dashboard: every run, its real-time pipeline stage, recent agent activity,
and cost — plus the human-approval review flow (list runs awaiting approval, inspect the
ApprovalPacket surfaced by the human_approval_node interrupt(), approve or reject)."""
import json
import time
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agents.schemas.common import RunStatus
from db.base import SessionLocal, get_db
from db.crud import get_run, list_runs
from db.models import AgentLog, Cost, Run
from graph.checkpointer import postgres_checkpointer
from graph.run import get_state
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


def _extract_approval_packet(run) -> dict | None:
    """Reads the live ApprovalPacket straight out of the LangGraph checkpoint (the interrupt
    payload) rather than duplicating it into Postgres — the checkpoint is the source of truth
    for anything mid-flight."""
    try:
        with postgres_checkpointer() as cp:
            snapshot = get_state(cp, run.langgraph_thread_id)
    except Exception:
        return None
    for task in snapshot.tasks:
        for pending_interrupt in getattr(task, "interrupts", []) or []:
            return pending_interrupt.value
    return None


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

    run_rows = []
    for run in runs:
        label, idx = _stage_progress(run)
        run_rows.append(
            {
                "id": str(run.id),
                "short_id": str(run.id)[:8],
                "topic": (run.selected_topic or {}).get("title") or "—",
                "status": run.status.value,
                "stage_label": label,
                "progress": idx,
                "progress_total": total_stages,
                "error": run.error,
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
        "runs": run_rows,
        "logs": log_rows_out,
    }


@router.get("/")
def dashboard_index(request: Request, db: Session = Depends(get_db)):
    pending = list_runs(db, status=RunStatus.AWAITING_APPROVAL)
    approvals = [{"run": run, "packet": _extract_approval_packet(run)} for run in pending]
    snapshot = _build_snapshot(db)
    return templates.TemplateResponse(request, "index.html", {"approvals": approvals, "snapshot": snapshot})


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
    packet = _extract_approval_packet(run)
    return templates.TemplateResponse(request, "review.html", {"run": run, "packet": packet})


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
