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
    user_topic: str | None = None,
    user_script: str | None = None,
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
        user_topic=user_topic,
        user_script=user_script,
    )

    app = build_publish_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    # Resume rather than restart when this thread already has work in flight.
    #
    # Passing a fresh initial_state re-seeds every key and re-enters at START, so a Celery retry
    # (worker.run_pipeline_task retries 3x) re-ran the entire pipeline from trend_research after a
    # transient failure late in the graph -- observed costing ~500 YouTube quota units and ~25min
    # of LLM/render work per retry, for a single flaky TTS call during voiceover. Invoking with
    # None is LangGraph's "continue from the checkpoint" form; `next` is non-empty exactly when a
    # previous attempt left tasks pending, and empty for both a fresh thread and a finished one.
    #
    # This is also what stops a retry re-uploading: a failure after upload_node would otherwise
    # replay the whole graph, upload included.
    pending = app.get_state(config).next
    if pending:
        logger.info("run.resume_pending", run_id=run_id, pending=list(pending))
        result = app.invoke(None, config=config)
    else:
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


def patch_state(checkpointer, thread_id: str, values: dict) -> None:
    """Writes `values` into a paused thread's channel state.

    NOT currently safe to use on a thread parked at the approval gate, and nothing calls it on one.
    Two observed problems, neither yet solved:

    1. update_state() writes a new checkpoint and the pending interrupt does not survive it. The
       task stays scheduled but its payload is gone, so the review packet reads back empty and the
       dashboard card degrades to "compliance unreadable" with no title or thumbnails.
    2. human_approval_node builds the review packet from state when it runs, so an edit applied
       this way is not reflected in the card the operator is looking at.

    DO NOT "fix" the lost packet by re-invoking the thread afterwards. invoke(None) resumes
    whatever pending work the thread has, and if a decision is already waiting for that thread --
    a queued resume_pipeline_task redelivered after a worker restart, for one -- the gate consumes
    it and the run proceeds to upload. That is a publish nobody asked for in that moment. An
    approval gate may only ever be passed by an explicit decision, never by a repair routine.

    Only safe for state the still-pending nodes read. Patching something an already-executed node
    produced (the render path, the voiceover) changes the record without changing the artifact.
    """
    app = build_publish_graph().compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    app.update_state(config, values)
    logger.info("run.patch_state", thread_id=thread_id, keys=sorted(values))
