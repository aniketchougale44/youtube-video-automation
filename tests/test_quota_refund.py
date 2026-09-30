"""A failed YouTube call must not be billed, and a dead credential must not be retried.

Regression test for the Stage-4 run that spent 8000 of 10000 daily quota units uploading nothing.
YOUTUBE_REFRESH_TOKEN had expired, so videos.insert raised RefreshError('invalid_grant') on every
attempt. quota.consume() debits before the request and nothing refunded it, and with_resilience's
default retry_on=(Exception,) retried the permanent failure three times per node. Celery then
redelivered the task twice more:

    upload.start
      videos.insert 1600  10000 -> 7995   RefreshError
      videos.insert 1600         -> 6395   RefreshError
      videos.insert 1600         -> 4795   RefreshError
    (Celery redelivery)
      videos.insert 1600         -> 3195   RefreshError
      videos.insert 1600         -> 1595   RefreshError
    QuotaExceededError: insufficient quota

Five charges, zero uploads, and the day's budget gone.
"""
import pytest
from google.auth.exceptions import RefreshError

from tools import quota
from tools.resilience import with_resilience


@pytest.fixture
def fake_bucket(monkeypatch):
    """In-memory stand-in for the Redis token bucket."""
    state = {"used": 0}

    class _FakeRedis:
        def get(self, _key):
            return state["used"]

        def decrby(self, _key, amount):
            state["used"] -= amount

        def incrby(self, _key, amount):
            state["used"] += amount

        def expire(self, _key, _ttl):
            pass

        def pipeline(self):
            return self

        def execute(self):
            pass

    monkeypatch.setattr(quota, "_redis", lambda: _FakeRedis())
    return state


def test_a_failed_call_is_refunded(fake_bucket):
    @quota.metered("videos.insert")
    def _boom():
        raise RuntimeError("upload exploded")

    before = quota.remaining_units()
    with pytest.raises(RuntimeError):
        _boom()
    assert quota.remaining_units() == before, "a failed call was still billed"


def test_a_successful_call_is_billed(fake_bucket):
    @quota.metered("videos.insert")
    def _ok():
        return "video-id"

    before = quota.remaining_units()
    assert _ok() == "video-id"
    assert quota.remaining_units() == before - 1600


def test_refund_never_hands_back_more_than_was_spent(fake_bucket):
    """Refunding below zero would invent quota Google never returned."""
    quota.refund("videos.insert")
    assert quota.remaining_units() == quota.get_settings().youtube_daily_quota_units


def test_invalid_grant_is_not_retried(fake_bucket):
    """The whole 8000-unit incident in one assertion."""
    attempts = []

    @with_resilience(never_retry=(RefreshError,), provider="test_upload_never_retry")
    @quota.metered("videos.insert")
    def _dead_token():
        attempts.append(1)
        raise RefreshError("invalid_grant: Bad Request")

    before = quota.remaining_units()
    with pytest.raises(RefreshError):
        _dead_token()

    assert len(attempts) == 1, f"a permanent auth failure was retried {len(attempts)} times"
    assert quota.remaining_units() == before, "quota was spent on an upload that never happened"


def test_transient_errors_are_still_retried(fake_bucket):
    """never_retry must not disable retrying generally -- only the listed types."""
    attempts = []

    @with_resilience(never_retry=(RefreshError,), provider="test_upload_still_retries", max_attempts=3)
    @quota.metered("videos.list")
    def _flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("transient blip")
        return "ok"

    before = quota.remaining_units()
    assert _flaky() == "ok"
    assert len(attempts) == 3
    # two failed attempts refunded, the successful one billed
    assert quota.remaining_units() == before - 1
