"""Shared fallback-alert node: reached when a critic rejects max_retries times in a row.
Terminal node for that branch — routes to END after notifying a human."""
from agents.schemas.feedback import NotificationEvent, NotificationSeverity
from graph.nodes._helpers import log_and_trace
from graph.state import PipelineState
from tools.notify import send_notification


def escalate_node(state: PipelineState) -> dict:
    reason = state.get("escalation_reason") or "unknown_stage_exceeded_max_retries"
    trace = log_and_trace("escalation", "triggered", reason=reason, run_id=state.get("run_id"))

    event = NotificationEvent(
        run_id=state.get("run_id"),
        severity=NotificationSeverity.CRITICAL,
        message=f"Run {state.get('run_id')} escalated: {reason}",
        context={"retry_counts": state.get("retry_counts", {})},
    )
    send_notification(event)

    return {
        "escalated": True,
        "trace": [trace, log_and_trace("escalation", "notified", notification_message=event.message)],
    }
