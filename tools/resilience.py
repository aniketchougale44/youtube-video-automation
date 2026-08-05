"""Shared retry + circuit-breaker wrapper for every outbound external API call (YouTube, TTS,
stock media, image-gen, research). Nothing in tools/*.py calls an external API directly without
going through `with_resilience`.
"""
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from core.logging import get_logger

logger = get_logger("tools.resilience")

P = ParamSpec("P")
T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised instead of calling the wrapped function while its circuit breaker is open."""


class CircuitBreaker:
    """Trips after `failure_threshold` consecutive failures, then rejects calls for
    `cooldown_seconds` before allowing a single trial call through (half-open)."""

    def __init__(self, name: str, failure_threshold: int = 5, cooldown_seconds: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    def _is_open(self) -> bool:
        if self._opened_at is None:
            return False
        # cooldown elapsed -> half-open, allow a trial call through
        return time.monotonic() - self._opened_at < self.cooldown_seconds

    def before_call(self) -> None:
        with self._lock:
            if self._is_open():
                raise CircuitOpenError(f"circuit '{self.name}' open - {self.failure_threshold} consecutive failures")

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()
                logger.error("circuit_breaker.opened", name=self.name, failures=self._failures)


_breakers: dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_circuit_breaker(name: str, **kwargs) -> CircuitBreaker:
    with _breakers_lock:
        if name not in _breakers:
            _breakers[name] = CircuitBreaker(name, **kwargs)
        return _breakers[name]


def with_resilience(
    provider: str,
    max_attempts: int = 3,
    retry_on: tuple[type[Exception], ...] = (Exception,),
    failure_threshold: int = 5,
    cooldown_seconds: float = 60.0,
):
    """Decorator: exponential-backoff retry (tenacity) wrapped in a per-provider circuit breaker.
    On the circuit tripping, callers should catch CircuitOpenError and route to tools.notify."""

    breaker = get_circuit_breaker(provider, failure_threshold=failure_threshold, cooldown_seconds=cooldown_seconds)

    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        retrying = retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential_jitter(initial=1, max=20),
            retry=retry_if_exception_type(retry_on),
            reraise=True,
        )(fn)

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            breaker.before_call()
            try:
                result = retrying(*args, **kwargs)
            except Exception:
                breaker.record_failure()
                raise
            breaker.record_success()
            return result

        return wrapper

    return decorator
