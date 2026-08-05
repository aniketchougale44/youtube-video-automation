"""Shared helpers used by every node stub: structured trace events + the debug force-reject hook
that lets us exercise critic retry loops without real critic logic wired up yet."""
from datetime import UTC, datetime
from typing import Any

from core.logging import get_logger

logger = get_logger("graph")


def trace_event(stage: str, event: str, **fields: Any) -> dict:
    return {
        "stage": stage,
        "event": event,
        "timestamp": datetime.now(UTC).isoformat(),
        **fields,
    }


def log_and_trace(stage: str, event: str, **fields: Any) -> dict:
    logger.info(f"{stage}.{event}", **fields)
    return trace_event(stage, event, **fields)


def consume_force_reject(state: dict, stage: str) -> bool:
    """Pop one unit of forced-rejection budget for `stage` from state.debug_force_reject.
    Used only to demonstrate/exercise the critic retry loop before real critic logic exists."""
    budget = dict(state.get("debug_force_reject") or {})
    remaining = budget.get(stage, 0)
    if remaining > 0:
        budget[stage] = remaining - 1
        state["debug_force_reject"] = budget  # mutate the in-flight dict so subsequent reads see it
        return True
    return False


def current_retry_count(state: dict, stage: str) -> int:
    return (state.get("retry_counts") or {}).get(stage, 0)


def bumped_retry_counts(state: dict, stage: str) -> dict[str, int]:
    counts = dict(state.get("retry_counts") or {})
    counts[stage] = counts.get(stage, 0) + 1
    return counts
