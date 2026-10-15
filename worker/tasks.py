"""Celery tasks. The API and scheduler enqueue these instead of ever running the LangGraph
pipeline inline in a request/response cycle."""
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.schemas.common import RunStatus
from core.crypto import decrypt_secret
from core.logging import configure_logging, get_logger
from db.base import SessionLocal
from db.crud import (
    compute_performance_baseline,
    get_run,
    get_video_by_youtube_id,
    latest_strategy_weight_adjustments,
    persist_costs,
    persist_performance_snapshot,
    persist_trace_as_agent_logs,
    successful_upload_for_run,
    summarize_recent_performance,
    update_run_from_graph_result,
)
from db.models import UserCredential
from graph.checkpointer import postgres_checkpointer
from graph.feedback_graph import build_feedback_graph
from graph.run import get_state as graph_get_state
from graph.run import patch_state, resume_run, start_run
from tools import cost as cost_tool
from tools import youtube as youtube_tool
from tools.youtube import YouTubeNotConfiguredError
from worker.celery_app import celery_app

configure_logging()
logger = get_logger("worker.tasks")


def _mark_run_failed(db: Session, run_id: str, error: BaseException) -> None:
    """Surfaces a task-level exception on the Run row itself. Without this, a run that errors
    before update_run_from_graph_result ever runs (e.g. a missing API key, a network failure)
    stays frozen at whatever status it already had — invisible to the dashboard even though
    Celery is retrying (or has given up) in the background."""
    db.rollback()
    run = get_run(db, uuid.UUID(run_id))
    if run is None:
        return
    run.status = RunStatus.FAILED
    run.error = str(error)[:2000]
    db.add(run)
    db.commit()


@contextmanager
def _acting_as_run_owner(db, run) -> Iterator[None]:
    """Runs the block as the channel owner of `run`, or as the deployment itself if it has none.

    The two cases are both real and must not be confused. A run created through the web app
    belongs to a user and has to upload to that user's channel with that user's own Google Cloud
    client. A run created by the scheduler or a scripts/ one-off has no owner and uses the .env
    credentials, exactly as before accounts existed.

    A run whose owner has not connected a channel is failed here, before any work starts, rather
    than twenty minutes and a full render later at the upload node. The credentials are read at
    the last moment and never written into the graph state -- a LangGraph checkpoint is persisted
    to Postgres and replayed on resume, so a refresh token placed in it would be stored in the
    clear and outlive the run.
    """
    if run.user_id is None:
        yield
        return

    credential = db.execute(
        select(UserCredential).where(UserCredential.user_id == run.user_id)
    ).scalars().first()

    client_id = decrypt_secret(credential.google_client_id) if credential else ""
    client_secret = decrypt_secret(credential.google_client_secret) if credential else ""
    refresh_token = decrypt_secret(credential.youtube_refresh_token) if credential else ""

    if not (client_id and client_secret and refresh_token):
        raise YouTubeNotConfiguredError(
            "this account has not connected a YouTube channel yet -- connect one in Settings "
            "before starting a run"
        )

    with youtube_tool.acting_for(
        client_id, client_secret, refresh_token, credential.youtube_channel_id
    ):
        yield


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30, name="worker.run_pipeline_task")
def run_pipeline_task(self, run_id: str) -> dict:
    db = SessionLocal()
    try:
        run = get_run(db, uuid.UUID(run_id))
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        # The idempotency check UploadHistory exists for. A Celery redelivery of a run that already
        # published must not republish: task_reject_on_worker_lost requeues on worker loss, and a
        # failure anywhere after upload_node would otherwise replay it. The checkpointer resume in
        # graph.run already avoids re-entering a completed node, so this is the second line of
        # defence -- cheap (one indexed read) against the cost of a duplicate public video.
        published = successful_upload_for_run(db, run.id)
        if published is not None:
            logger.warning(
                "pipeline.task.already_published",
                run_id=run_id,
                youtube_video_id=published.youtube_video_id,
                attempt=published.attempt_number,
            )
            return {"run_id": run_id, "status": str(run.status), "already_published": published.youtube_video_id}

        debug_force_reject = (run.stage_status or {}).get("debug_force_reject", {})
        user_inputs = (run.stage_status or {}).get("user_inputs") or {}
        past_performance_summary = summarize_recent_performance(db)
        strategy_weight_adjustments = latest_strategy_weight_adjustments(db)
        db.commit()  # release the read transaction — postgres_checkpointer() runs DDL (CREATE
        # INDEX CONCURRENTLY) on a separate connection that would otherwise block waiting on it

        with _acting_as_run_owner(db, run), cost_tool.collecting(), postgres_checkpointer() as checkpointer:
            _thread_id, result = start_run(
                checkpointer,
                run_id=run.langgraph_thread_id,
                debug_force_reject=debug_force_reject,
                past_performance_summary=past_performance_summary,
                strategy_weight_adjustments=strategy_weight_adjustments,
                user_topic=user_inputs.get("topic") or None,
                user_script=user_inputs.get("script") or None,
            )
            # Must be drained *inside* the scope: collecting() resets the ContextVar on exit, so
            # reading it afterwards returns an empty list and every cost row is silently lost.
            cost_entries = cost_tool.collected()

        interrupted = "__interrupt__" in result
        update_run_from_graph_result(db, run, result, interrupted=interrupted)
        persist_trace_as_agent_logs(db, run.id, result.get("trace", []))
        persist_costs(db, run.id, cost_entries)

        logger.info("pipeline.task.complete", run_id=run_id, status=run.status, interrupted=interrupted)
        return {"run_id": run_id, "status": str(run.status), "interrupted": interrupted}
    except Exception as exc:
        logger.error("pipeline.task.failed", run_id=run_id, error=str(exc))
        _mark_run_failed(db, run_id, exc)
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30, name="worker.resume_pipeline_task")
def resume_pipeline_task(self, run_id: str, decision: dict) -> dict:
    db = SessionLocal()
    try:
        run = get_run(db, uuid.UUID(run_id))
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        db.commit()  # release the read transaction before postgres_checkpointer()'s DDL runs

        # The approval resume is where upload_node actually runs, so it needs the owner's
        # credentials just as much as the initial invocation does.
        with _acting_as_run_owner(db, run), postgres_checkpointer() as checkpointer:
            result = resume_run(checkpointer, run.langgraph_thread_id, decision)

        update_run_from_graph_result(db, run, result, interrupted=False)
        persist_trace_as_agent_logs(db, run.id, result.get("trace", []))

        logger.info("pipeline.resume.complete", run_id=run_id, status=run.status)
        return {"run_id": run_id, "status": str(run.status)}
    except Exception as exc:
        logger.error("pipeline.resume.failed", run_id=run_id, error=str(exc))
        _mark_run_failed(db, run_id, exc)
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30, name="worker.revise_metadata_task")
def revise_metadata_task(self, run_id: str, changes: dict) -> dict:
    """Edits the title/description of a run parked at the approval gate.

    UNUSED and not reachable from the chat surface -- see the REVISE branch in api/routes/chat.py.
    Kept because the cheap-edit idea is right (a title change should not cost a 20-minute
    re-render), but the mechanism is not: graph.run.patch_state destroys the pending interrupt,
    and re-entering the gate to rebuild it can consume a queued decision and publish. The edit
    needs to travel in the ApprovalDecision instead, applied on resume, so the gate is passed
    exactly once and only by a real decision.
    """
    db = SessionLocal()
    try:
        run = get_run(db, uuid.UUID(run_id))
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        if run.status != RunStatus.AWAITING_APPROVAL:
            # Not an error worth retrying: the run moved on (approved elsewhere, or failed) and
            # the edit no longer has anything to apply to.
            logger.warning("pipeline.revise.not_awaiting", run_id=run_id, status=str(run.status))
            return {"run_id": run_id, "applied": False, "reason": "run not awaiting approval"}
        thread_id = run.langgraph_thread_id
        db.commit()  # release the read transaction before postgres_checkpointer()'s DDL runs

        with postgres_checkpointer() as checkpointer:
            snapshot = graph_get_state(checkpointer, thread_id)
            metadata = dict((getattr(snapshot, "values", None) or {}).get("metadata_output") or {})
            if not metadata:
                raise ValueError("no metadata_output in checkpoint state")

            title = (changes.get("title") or "").strip()
            description = (changes.get("description") or "").strip()
            if title:
                metadata["selected_title"] = title[:100]  # YouTube's hard title limit
            if description:
                metadata["description"] = description[:5000]
            patch_state(checkpointer, thread_id, {"metadata_output": metadata})

        logger.info(
            "pipeline.revise.applied", run_id=run_id,
            changed=sorted(k for k in ("title", "description") if changes.get(k)),
        )
        return {
            "run_id": run_id,
            "applied": True,
            "title": metadata.get("selected_title"),
            "description": metadata.get("description"),
        }
    except Exception as exc:
        logger.error("pipeline.revise.failed", run_id=run_id, error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60, name="worker.performance_feedback_task")
def performance_feedback_task(self, youtube_video_id: str, window: str) -> dict:
    """Triggered by the scheduler at the 24h/7d mark after a video publishes. Computes the
    historical baseline and persists the resulting snapshot -- this is what makes
    db.crud.summarize_recent_performance() (read by strategy_node on the *next* run) see real
    data instead of "no data yet" forever."""
    db = SessionLocal()
    try:
        video = get_video_by_youtube_id(db, youtube_video_id)
        if video is None:
            logger.error("performance_feedback.video_not_found", youtube_video_id=youtube_video_id)
            return {"youtube_video_id": youtube_video_id, "window": window, "status": "video_not_found"}

        baseline = compute_performance_baseline(db, exclude_video_id=video.id)

        app = build_feedback_graph().compile()
        result = app.invoke({
            "youtube_video_id": youtube_video_id,
            "window": window,
            "baseline": baseline,
            # Anchors the analytics date range, so a snapshot captured late still measures the
            # window it is labelled with. Graph nodes stay DB-free, so it is injected here.
            "published_at": video.published_at.isoformat() if video.published_at else "",
        })

        persist_performance_snapshot(db, video.id, result["performance_snapshot"])
        persist_trace_as_agent_logs(db, video.run_id, result.get("trace", []))

        logger.info(
            "performance_feedback.complete",
            youtube_video_id=youtube_video_id,
            window=window,
            insights=result.get("learning_update", {}).get("insights"),
        )
        return result
    except Exception as exc:
        logger.error("performance_feedback.failed", youtube_video_id=youtube_video_id, error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()
