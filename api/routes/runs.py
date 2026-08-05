"""Control API: trigger runs, inspect status, submit approval decisions.

Every mutating endpoint here only writes a Run row and enqueues a Celery task — it never runs
the LangGraph pipeline inline in the request/response cycle (that would tie up an API worker for
the full render duration and defeats resumability).
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agents.schemas.common import RunStatus
from db.base import get_db
from db.crud import create_run, get_run, list_runs
from worker.tasks import resume_pipeline_task, run_pipeline_task

router = APIRouter()


class CreateRunRequest(BaseModel):
    debug_force_reject: dict[str, int] = Field(
        default_factory=dict, description="Testing hook: {'critic_script_qa': 2} forces N rejections before pass."
    )


class RunSummary(BaseModel):
    id: uuid.UUID
    status: RunStatus
    current_stage: str | None = None
    selected_topic: dict | None = None
    error: str | None = None

    model_config = {"from_attributes": True}


class DecisionRequest(BaseModel):
    approved: bool
    selected_thumbnail_index: int = 0
    reviewer: str = "api"
    notes: str = ""


@router.post("", response_model=RunSummary, status_code=202)
def create_pipeline_run(body: CreateRunRequest, db: Session = Depends(get_db)) -> RunSummary:
    run = create_run(db, debug_force_reject=body.debug_force_reject)
    run_pipeline_task.delay(str(run.id))
    return RunSummary.model_validate(run)


@router.get("", response_model=list[RunSummary])
def list_pipeline_runs(status: RunStatus | None = None, db: Session = Depends(get_db)) -> list[RunSummary]:
    return [RunSummary.model_validate(r) for r in list_runs(db, status=status)]


@router.get("/{run_id}", response_model=RunSummary)
def get_pipeline_run(run_id: uuid.UUID, db: Session = Depends(get_db)) -> RunSummary:
    run = get_run(db, run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return RunSummary.model_validate(run)


@router.post("/{run_id}/decision", status_code=202)
def submit_decision(run_id: uuid.UUID, body: DecisionRequest, db: Session = Depends(get_db)) -> dict:
    run = get_run(db, run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    if run.status != RunStatus.AWAITING_APPROVAL:
        raise HTTPException(409, f"run is not awaiting approval (status={run.status})")

    decision = {
        "run_id": str(run_id),
        "approved": body.approved,
        "selected_thumbnail_index": body.selected_thumbnail_index,
        "reviewer": body.reviewer,
        "notes": body.notes,
    }
    resume_pipeline_task.delay(str(run_id), decision)
    return {"run_id": str(run_id), "queued": True}
