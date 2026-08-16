"""YouTube Data API v3 quota tracker — a Redis-backed daily token bucket (10,000 units/day
default). Every tools.youtube call consumes from this before firing; callers must catch
QuotaExceededError and defer/queue rather than fail the whole run."""
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
