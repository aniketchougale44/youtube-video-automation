"""Celery tasks. The API and scheduler enqueue these instead of ever running the LangGraph
pipeline inline in a request/response cycle."""
import uuid

from core.logging import configure_logging, get_logger
from db.base import SessionLocal
from db.crud import (
    get_run,
    persist_trace_as_agent_logs,
    summarize_recent_performance,
    update_run_from_graph_result,
)
from graph.checkpointer import postgres_checkpointer
from graph.feedback_graph import build_feedback_graph
from graph.run import resume_run, start_run
from worker.celery_app import celery_app

configure_logging()
logger = get_logger("worker.tasks")


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30, name="worker.run_pipeline_task")
def run_pipeline_task(self, run_id: str) -> dict:
    db = SessionLocal()
    try:
        run = get_run(db, uuid.UUID(run_id))
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        debug_force_reject = (run.stage_status or {}).get("debug_force_reject", {})
        past_performance_summary = summarize_recent_performance(db)

        with postgres_checkpointer() as checkpointer:
            _thread_id, result = start_run(
                checkpointer,
                run_id=run.langgraph_thread_id,
                debug_force_reject=debug_force_reject,
                past_performance_summary=past_performance_summary,
            )

        interrupted = "__interrupt__" in result
        update_run_from_graph_result(db, run, result, interrupted=interrupted)
        persist_trace_as_agent_logs(db, run.id, result.get("trace", []))

        logger.info("pipeline.task.complete", run_id=run_id, status=run.status, interrupted=interrupted)
        return {"run_id": run_id, "status": str(run.status), "interrupted": interrupted}
    except Exception as exc:
        logger.error("pipeline.task.failed", run_id=run_id, error=str(exc))
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

        with postgres_checkpointer() as checkpointer:
            result = resume_run(checkpointer, run.langgraph_thread_id, decision)

        update_run_from_graph_result(db, run, result, interrupted=False)
        persist_trace_as_agent_logs(db, run.id, result.get("trace", []))

        logger.info("pipeline.resume.complete", run_id=run_id, status=run.status)
        return {"run_id": run_id, "status": str(run.status)}
    except Exception as exc:
        logger.error("pipeline.resume.failed", run_id=run_id, error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60, name="worker.performance_feedback_task")
def performance_feedback_task(self, youtube_video_id: str, window: str) -> dict:
    """Triggered by the scheduler at the 24h/7d mark after a video publishes."""
    try:
        app = build_feedback_graph().compile()
        result = app.invoke({"youtube_video_id": youtube_video_id, "window": window})
        logger.info("performance_feedback.complete", youtube_video_id=youtube_video_id, window=window)
        return result
    except Exception as exc:
        logger.error("performance_feedback.failed", youtube_video_id=youtube_video_id, error=str(exc))
        raise self.retry(exc=exc)
