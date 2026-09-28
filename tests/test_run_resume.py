"""start_run must resume a half-finished thread, not replay it from START.

Regression test for a real Stage-2 run: a transient edge-tts NoAudioReceived during `voiceover`
(beat 8 of 17, after its 3 resilience retries) propagated out of the graph. Celery retried
run_pipeline_task, start_run re-seeded a fresh initial_state, and the whole pipeline re-ran from
trend_research -- re-spending ~500 YouTube quota units and ~25 minutes of LLM and render work to
redo seven nodes that had already succeeded and were sitting in the Postgres checkpointer.

The checkpointer docstring calls out the other half of this: a failure *after* upload_node would
replay the graph including the upload.
"""
import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver

from graph.run import start_run

pytestmark = pytest.mark.usefixtures("mock_external_apis")


class _Boom(RuntimeError):
    pass


def test_retry_resumes_from_the_checkpoint_instead_of_replaying(monkeypatch):
    """First attempt dies inside voiceover; the second must not re-run the research nodes."""
    calls: list[str] = []

    def _counting(stage: str, real):
        def node(state):
            calls.append(stage)
            return real(state)

        return node

    from graph import builder

    for stage in ("trend_research", "strategy", "script_writer"):
        monkeypatch.setattr(builder, f"{stage}_node", _counting(stage, getattr(builder, f"{stage}_node")))

    # voiceover explodes on the first call only, exactly like a flaky TTS request
    state = {"failed_once": False}
    real_voiceover = builder.voiceover_node

    def flaky_voiceover(s):
        calls.append("voiceover")
        if not state["failed_once"]:
            state["failed_once"] = True
            raise _Boom("No audio was received. Please verify that your parameters are correct.")
        return real_voiceover(s)

    monkeypatch.setattr(builder, "voiceover_node", flaky_voiceover)

    checkpointer = MemorySaver()
    run_id = str(uuid.uuid4())

    with pytest.raises(_Boom):
        start_run(checkpointer, run_id=run_id)

    first_pass = list(calls)
    assert "trend_research" in first_pass
    assert first_pass.count("voiceover") == 1

    calls.clear()
    start_run(checkpointer, run_id=run_id)  # the Celery retry

    assert "trend_research" not in calls, (
        f"retry replayed the research nodes instead of resuming: {calls}"
    )
    assert "strategy" not in calls and "script_writer" not in calls, calls
    assert "voiceover" in calls, f"retry did not resume at the failed node: {calls}"


def test_a_fresh_thread_still_starts_from_the_beginning(monkeypatch):
    """The resume branch must not swallow normal first runs."""
    from graph import builder

    calls: list[str] = []
    real = builder.trend_research_node

    def counting(state):
        calls.append("trend_research")
        return real(state)

    monkeypatch.setattr(builder, "trend_research_node", counting)
    start_run(MemorySaver(), run_id=str(uuid.uuid4()))

    assert calls.count("trend_research") == 1, calls
