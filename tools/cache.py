"""Redis-backed JSON cache. Used to cache trend/search results for 24h so repeated runs don't
burn YouTube search quota (100 units/call) or re-hit third-party research APIs needlessly."""
import json
from typing import Any

import redis

from core.settings import get_settings

DEFAULT_TTL_SECONDS = 60 * 60 * 24

_redis_client: redis.Redis | None = None


def _redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        # protocol=2 (RESP2): stays compatible with Redis <6, which doesn't support the RESP3
        # HELLO handshake redis-py otherwise negotiates by default.
        _redis_client = redis.Redis.from_url(get_settings().redis_url, decode_responses=True, protocol=2)
    return _redis_client


def cache_get_json(key: str) -> Any | None:
    raw = _redis().get(key)
    return json.loads(raw) if raw is not None else None


def cache_set_json(key: str, value: Any, ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
    _redis().set(key, json.dumps(value), ex=ttl_seconds)


def cache_scan_json(prefix: str) -> list[Any]:
    """Returns the parsed JSON value of every key starting with prefix -- for a caller that needs
    to enumerate a whole key family (e.g. tools.script_progress's per-script run status) rather
    than fetch one known key. SCAN-based (not KEYS): non-blocking, safe against a large keyspace."""
    client = _redis()
    values = []
    for key in client.scan_iter(match=f"{prefix}*"):
        raw = client.get(key)
        if raw is not None:
            values.append(json.loads(raw))
    return values
