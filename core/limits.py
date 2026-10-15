"""Per-account limits on the things one user can take from everyone else.

The YouTube quota is already per-account, because each user brings their own Google project. These
limits cover what is still genuinely shared, and the sharing is not obvious from the outside:

- The worker. Rendering is minutes of CPU per video and the pool is small, so fifty queued runs
  from one account is a queue every other account waits behind. This is the real scarcity.
- The LLM keys. Unlike YouTube credentials, the groq/gemini keys belong to the deployment, so one
  account's runs spend the operator's free tier and can exhaust it for everybody.

Counts come from Postgres rather than a Redis counter. A counter drifts: it survives a rollback
that the run did not, and it resets if Redis is flushed, which is exactly when someone would
notice the limit is gone. Counting rows asks the question directly -- "how many runs does this
account actually have in flight" -- and cannot disagree with reality.

Scheduler and script runs have no owner and are deliberately exempt: they are the operator's own
cron, not a tenant competing for the pool.
"""
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agents.schemas.common import RunStatus
from core.logging import get_logger
from core.settings import get_settings
from db.models import Run

logger = get_logger("core.limits")

# A run occupying the worker right now. AWAITING_APPROVAL is deliberately excluded: that run has
# finished its work and is waiting on a person, so counting it would mean forgetting to review
# one video blocks making the next -- a limit that punishes the wrong thing.
_IN_FLIGHT = (RunStatus.PENDING, RunStatus.RUNNING)


class RateLimited(RuntimeError):
    """Raised with a message meant to be shown to the operator verbatim."""


def _start_of_day() -> datetime:
    return datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)


def runs_in_flight(db: Session, user_id) -> int:
    return db.execute(
        select(func.count())
        .select_from(Run)
        .where(Run.user_id == user_id, Run.status.in_(_IN_FLIGHT))
    ).scalar_one()


def runs_started_today(db: Session, user_id) -> int:
    return db.execute(
        select(func.count())
        .select_from(Run)
        .where(Run.user_id == user_id, Run.created_at >= _start_of_day())
    ).scalar_one()


def check_can_start_run(db: Session, user_id) -> None:
    """Raises RateLimited if this account may not start another run right now.

    Checked at the moment of starting, not when the run is picked up: the point is to refuse
    before anything is queued, so the operator gets an answer in the chat instead of a run that
    sits pending and then dies somewhere they are not looking.
    """
    if user_id is None:  # scheduler / operator cron
        return

    settings = get_settings()

    in_flight = runs_in_flight(db, user_id)
    if in_flight >= settings.max_concurrent_runs_per_user:
        raise RateLimited(
            f"You already have **{in_flight}** video"
            f"{'s' if in_flight != 1 else ''} being made. "
            "Each one takes around twenty minutes of rendering, and they share the same worker, "
            "so I'll start the next when one finishes. Ask me what's happening to follow them."
        )

    today = runs_started_today(db, user_id)
    if today >= settings.max_runs_per_user_per_day:
        raise RateLimited(
            f"That's **{today}** videos started today, which is this account's daily limit. "
            "It resets at midnight UTC. If you need more, raise "
            "MAX_RUNS_PER_USER_PER_DAY in the server's settings."
        )

    logger.info(
        "limits.run_allowed", user_id=str(user_id), in_flight=in_flight, today=today,
    )
