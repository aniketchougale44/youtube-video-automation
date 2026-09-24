"""Thin helpers for invoking/resuming the publish graph. The API layer (api/routes/runs.py) and
scheduler (scheduler/jobs.py) both go through these instead of touching StateGraph directly."""
import uuid

from langgraph.types import Command

from core.logging import get_logger
from graph.builder import build_publish_graph
from graph.state import initial_state

logger = get_logger("graph.run")


def start_run(
    checkpointer,
    run_id: str | None = None,
    debug_force_reject: dict | None = None,
    past_performance_summary: str | None = None,
    strategy_weight_adjustments: dict[str, float] | None = None,
) -> tuple[str, dict]:
    """Compiles the graph and invokes it from START. Returns (thread_id, final_or_interrupted_state).
    `past_performance_summary` and `strategy_weight_adjustments` are pulled from Postgres by the
    caller (worker/tasks.py) — graph nodes never touch the DB directly, so they have to arrive as
    part of the initial state."""
    run_id = run_id or str(uuid.uuid4())
    thread_id = run_id
    settings_state = initial_state(
        run_id=run_id, thread_id=thread_id, debug_force_reject=debug_force_reject,
        past_performance_summary=past_performance_summary,
        strategy_weight_adjustments=strategy_weight_adjustments,
    )

    app = build_publish_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    logger.info("run.start", run_id=run_id)
    result = app.invoke(settings_state, config=config)
    return thread_id, result


def resume_run(checkpointer, thread_id: str, resume_value: dict) -> dict:
    """Resumes a graph paused at an interrupt() (e.g. human_approval_node) with the reviewer's decision."""
    app = build_publish_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    logger.info("run.resume", thread_id=thread_id)
    result = app.invoke(Command(resume=resume_value), config=config)
    return result


def get_state(checkpointer, thread_id: str):
    app = build_publish_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    return app.get_state(config)
