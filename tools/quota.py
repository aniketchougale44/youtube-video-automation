"""YouTube Data API v3 quota tracker — a Redis-backed daily token bucket (10,000 units/day
default). Every tools.youtube call consumes from this before firing; callers must catch
QuotaExceededError and defer/queue rather than fail the whole run."""
import functools
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

import redis

from core.logging import get_logger
from core.settings import get_settings

logger = get_logger("tools.quota")

# Published per-operation costs from Google's YouTube Data API v3 quota table.
UNIT_COSTS: dict[str, int] = {
    "videos.list": 1,
    "search.list": 100,
    "channels.list": 1,
    "videos.insert": 1600,
    "videos.update": 50,
    "thumbnails.set": 50,
    "playlists.list": 1,
    "playlists.insert": 50,
    "playlistItems.insert": 50,
}


class QuotaExceededError(RuntimeError):
    pass


_redis_client: redis.Redis | None = None


def _redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        # protocol=2 (RESP2): stays compatible with Redis <6, which doesn't support the RESP3
        # HELLO handshake redis-py otherwise negotiates by default.
        _redis_client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True, protocol=2)
    return _redis_client


def _today_key() -> str:
    return f"youtube_quota:{datetime.now(UTC):%Y-%m-%d}"


def remaining_units() -> int:
    used = int(_redis().get(_today_key()) or 0)
    return max(0, get_settings().youtube_daily_quota_units - used)


def can_afford(operation: str) -> bool:
    return remaining_units() >= UNIT_COSTS.get(operation, 1)


def consume(operation: str) -> int:
    """Reserves quota for `operation`. Raises QuotaExceededError if the daily budget is
    exhausted — callers (tools.youtube.*) should let this propagate so the calling agent node
    can defer the operation rather than silently proceeding over budget."""
    cost = UNIT_COSTS.get(operation, 1)
    if not can_afford(operation):
        raise QuotaExceededError(
            f"insufficient YouTube quota for '{operation}' (cost={cost}, remaining={remaining_units()})"
        )
    key = _today_key()
    pipe = _redis().pipeline()
    pipe.incrby(key, cost)
    pipe.expire(key, 60 * 60 * 26)  # a bit past the UTC day boundary so it always covers today
    pipe.execute()
    logger.info("quota.consumed", operation=operation, cost=cost, remaining=remaining_units())
    return cost


def refund(operation: str) -> int:
    """Gives back quota reserved for a call that never reached Google.

    consume() debits before the request, which is the safe order -- it cannot overspend. But
    without a refund a failing call is charged anyway, and retries multiply it. Observed: a
    videos.insert against an expired OAuth token failed five times (three resilience attempts plus
    two Celery redeliveries) and burned 5 x 1600 = 8000 of the 10000 daily units while uploading
    precisely nothing.

    Deliberately clamped at zero rather than allowed to go negative: an over-refund would hand out
    quota Google never gave back, which is worse than losing a few units. A failed call that DID
    reach Google (a 500 after the video was accepted, say) is the case this slightly over-refunds,
    and that is the safer direction to be wrong in than never refunding at all.
    """
    cost = UNIT_COSTS.get(operation, 1)
    key = _today_key()
    used = int(_redis().get(key) or 0)
    give_back = min(cost, used)
    if give_back:
        _redis().decrby(key, give_back)
    logger.info("quota.refunded", operation=operation, refunded=give_back, remaining=remaining_units())
    return give_back


@contextmanager
def reserve(operation: str) -> Iterator[None]:
    """consume() on enter, refund() if the block raises.

    For functions that make more than one billed call, or make one conditionally -- those cannot
    use @metered, since a decorator charges unconditionally on entry. _get_or_create_playlist_call
    is the cautionary case: it only inserts when the playlist is missing, so charging
    playlists.insert up front over-bills 50 units on every run that reuses an existing playlist.
    """
    consume(operation)
    try:
        yield
    except Exception:
        refund(operation)
        raise


def metered(operation: str):
    """Decorator: reserve quota for `operation`, and refund it if the wrapped call raises.

    Applied *below* @with_resilience so it runs per attempt -- each real attempt should be charged,
    and each failed attempt refunded, rather than one charge covering a whole retry series.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            consume(operation)
            try:
                return fn(*args, **kwargs)
            except Exception:
                refund(operation)
                raise

        return wrapper

    return decorator
