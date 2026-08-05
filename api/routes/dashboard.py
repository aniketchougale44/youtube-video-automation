"""Minimal human-approval review dashboard: list runs awaiting approval, inspect the
ApprovalPacket surfaced by the human_approval_node interrupt(), approve or reject."""
import uuid

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from agents.schemas.common import RunStatus
from db.base import get_db
from db.crud import get_run, list_runs
from graph.checkpointer import postgres_checkpointer
from graph.run import get_state
from worker.tasks import resume_pipeline_task

router = APIRouter()
templates = Jinja2Templates(directory="api/templates")


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


@router.get("/")
def dashboard_index(request: Request, db: Session = Depends(get_db)):
    pending = list_runs(db, status=RunStatus.AWAITING_APPROVAL)
    recent = list_runs(db, limit=20)
    return templates.TemplateResponse(request, "index.html", {"pending": pending, "recent": recent})


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
