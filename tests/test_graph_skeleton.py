"""Validates the publish graph's wiring: happy path through the interrupt, dashboard rejection,
critic retry loops (script QA + compliance), and escalation after max_retries — all against stub
node logic. Real agent-logic tests land alongside each stage's real implementation."""
import uuid

import pytest
from langgraph.checkpoint.memory import MemorySaver

from core.settings import get_settings
from graph.run import resume_run, start_run


@pytest.fixture(autouse=True)
def _force_human_approval(monkeypatch):
    """This module validates the human-approval interrupt wiring, so pin REQUIRE_HUMAN_APPROVAL on
    regardless of what the dev .env sets it to (build_publish_graph reads it at build time)."""
    monkeypatch.setattr(get_settings(), "require_human_approval", True)


def _new_cp() -> MemorySaver:
    return MemorySaver()


def test_happy_path_reaches_interrupt_and_uploads_on_approval():
    cp = _new_cp()
    thread_id, result = start_run(cp, run_id=str(uuid.uuid4()))

    assert "__interrupt__" in result
    packet = result["__interrupt__"][0].value
    assert packet["title"]
    assert packet["thumbnail_options"]

    resumed = resume_run(cp, thread_id, {"run_id": thread_id, "approved": True, "reviewer": "test"})

    assert resumed["upload_result"]["status"] == "uploaded"
    assert resumed["upload_result"]["youtube_video_id"]


def test_dashboard_rejection_does_not_upload():
    cp = _new_cp()
    thread_id, result = start_run(cp, run_id=str(uuid.uuid4()))
    assert "__interrupt__" in result

    resumed = resume_run(cp, thread_id, {"run_id": thread_id, "approved": False, "reviewer": "test"})

    assert resumed.get("upload_result") is None
    assert resumed["approval_decision"]["approved"] is False


def test_script_qa_critic_retries_then_passes():
    cp = _new_cp()
    _thread_id, result = start_run(cp, run_id=str(uuid.uuid4()), debug_force_reject={"critic_script_qa": 2})

    assert result["script_qa_result"]["passed"] is True
    assert result["retry_counts"]["critic_script_qa"] == 2
    assert "__interrupt__" in result  # still reached the approval gate after recovering


def test_compliance_critic_retries_then_passes():
    cp = _new_cp()
    _thread_id, result = start_run(cp, run_id=str(uuid.uuid4()), debug_force_reject={"critic_compliance": 1})

    assert result["compliance_result"]["passed"] is True
    assert result["retry_counts"]["critic_compliance"] == 1
    assert "__interrupt__" in result


def test_script_qa_escalates_after_max_retries():
    cp = _new_cp()
    _thread_id, result = start_run(cp, run_id=str(uuid.uuid4()), debug_force_reject={"critic_script_qa": 99})

    assert result["escalated"] is True
    assert "critic_script_qa" in result["escalation_reason"]
    assert "upload_result" not in result
    assert "__interrupt__" not in result


def test_compliance_escalates_after_max_retries():
    cp = _new_cp()
    _thread_id, result = start_run(cp, run_id=str(uuid.uuid4()), debug_force_reject={"critic_compliance": 99})

    assert result["escalated"] is True
    assert "critic_compliance" in result["escalation_reason"]
    assert "upload_result" not in result
