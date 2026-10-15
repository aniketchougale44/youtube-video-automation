"""Per-account limits on what is still shared after per-user YouTube credentials.

The scarce thing is the render worker: minutes of CPU per video, a small pool, and fifty queued
runs from one account is a queue every other account waits behind. The deployment's own LLM keys
are shared too, unlike the YouTube credentials.

Counts come from Postgres rather than a Redis counter, and these tests assert on the queries so
that stays true. A counter drifts -- it survives a rollback the run did not, and it resets when
Redis is flushed, which is precisely when nobody notices the limit has gone.
"""
import types
import uuid
from unittest.mock import MagicMock

import pytest

from agents.schemas.common import RunStatus
from core import limits


def _settings(concurrent=2, daily=10):
    return types.SimpleNamespace(
        max_concurrent_runs_per_user=concurrent, max_runs_per_user_per_day=daily
    )


@pytest.fixture
def counts(monkeypatch):
    """Drives the two counts directly; the SQL behind them is asserted separately."""
    box = {"in_flight": 0, "today": 0}
    monkeypatch.setattr(limits, "runs_in_flight", lambda db, uid: box["in_flight"])
    monkeypatch.setattr(limits, "runs_started_today", lambda db, uid: box["today"])
    monkeypatch.setattr(limits, "get_settings", _settings)
    return box


def test_a_first_run_is_allowed(counts):
    limits.check_can_start_run(None, uuid.uuid4())  # does not raise


def test_too_many_at_once_is_refused_with_a_reason_worth_reading(counts):
    counts["in_flight"] = 2
    with pytest.raises(limits.RateLimited) as caught:
        limits.check_can_start_run(None, uuid.uuid4())
    message = str(caught.value)
    assert "2" in message
    # The operator is told what to do next, not just that they were refused.
    assert "what's happening" in message


def test_the_daily_cap_is_refused_separately_from_concurrency(counts):
    counts["in_flight"] = 0
    counts["today"] = 10
    with pytest.raises(limits.RateLimited) as caught:
        limits.check_can_start_run(None, uuid.uuid4())
    assert "daily limit" in str(caught.value)
    assert "midnight UTC" in str(caught.value)


def test_a_run_with_no_owner_is_exempt(counts):
    """The scheduler and the scripts/ one-offs are the operator's own cron, not a tenant
    competing for the pool."""
    counts["in_flight"] = 99
    counts["today"] = 99
    limits.check_can_start_run(None, None)  # does not raise


def test_a_video_waiting_for_review_does_not_block_making_the_next_one():
    """AWAITING_APPROVAL is excluded from the in-flight count on purpose: that run has finished
    its work and is waiting on a person. Counting it would mean forgetting to review one video
    stops you making another -- a limit that punishes the wrong thing."""
    assert RunStatus.AWAITING_APPROVAL not in limits._IN_FLIGHT
    assert set(limits._IN_FLIGHT) == {RunStatus.PENDING, RunStatus.RUNNING}


def _sql(db: MagicMock) -> str:
    stmt = db.execute.call_args[0][0]
    return str(stmt.compile(compile_kwargs={"literal_binds": True}))


def test_the_in_flight_count_is_asked_of_the_database_for_one_user():
    db = MagicMock()
    user_id = uuid.uuid4()
    limits.runs_in_flight(db, user_id)
    sql = _sql(db)
    assert "count" in sql.lower()
    # SQLAlchemy renders a UUID literal without dashes, so match the hex form.
    assert "runs.user_id" in sql and user_id.hex in sql
    assert "awaiting_approval" not in sql


def test_the_daily_count_is_filtered_by_user_and_by_today():
    db = MagicMock()
    user_id = uuid.uuid4()
    limits.runs_started_today(db, user_id)
    sql = _sql(db)
    assert "runs.user_id" in sql and user_id.hex in sql
    assert "created_at" in sql
