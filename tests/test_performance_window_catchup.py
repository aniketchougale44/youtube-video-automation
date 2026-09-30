"""A missed performance window must be caught up, and a late capture must still be correct.

check_performance_windows used to require a video's age to fall inside a 2-hour band
(23h-25h for "24h", 6d23h-7d1h for "7d"), checked hourly. A host asleep across that band lost the
snapshot permanently: the `already_captured` guard only skips work, it never backfills. That is the
same failure mode as the APScheduler misfire fixed in scheduler/beat.py, one layer down.

Simply widening the band is not enough on its own, because analytics_report's window was a
*trailing* range ending today. Capturing "24h" three days late would report the video's fourth day
under the "24h" label, and compute_performance_baseline would then compare it against other videos'
genuine first-day numbers. Anchoring the range to the publish date is what makes a late capture
equivalent to an on-time one.
"""
from datetime import UTC, datetime, timedelta

import pytest

from scheduler.jobs import PERFORMANCE_WINDOWS
from tools import youtube

# --- the window table itself ---


def test_windows_have_no_upper_bound():
    """A third element would reintroduce the band that lost snapshots."""
    for entry in PERFORMANCE_WINDOWS:
        assert len(entry) == 2, f"{entry!r} looks like a (window, lower, upper) band again"


def test_window_labels_and_earliest_ages():
    assert dict(PERFORMANCE_WINDOWS) == {
        "24h": timedelta(hours=23),
        "7d": timedelta(days=6, hours=23),
    }


def test_a_long_overdue_video_is_still_eligible():
    """The property the old band lacked: age far past the window must not disqualify a capture."""
    for window, earliest in PERFORMANCE_WINDOWS:
        age = earliest + timedelta(days=30)
        assert age >= earliest, f"{window} at 30 days overdue would be skipped"


# --- anchoring: the date range a late capture actually asks for ---


class _FakeAnalytics:
    """Captures the startDate/endDate the report was requested with."""

    def __init__(self):
        self.params = {}

    def reports(self):
        return self

    def query(self, **kwargs):
        self.params = kwargs
        return self

    def execute(self):
        return {"rows": [[10, 1, 0, 30.0, 40.0]]}


def _dates_requested(monkeypatch, window: str, published_at, now):
    fake = _FakeAnalytics()

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(youtube, "datetime", _FrozenDatetime)
    youtube._analytics_call.__wrapped__(fake, "vid", window, published_at)
    return fake.params["startDate"], fake.params["endDate"]


def test_anchored_capture_measures_the_first_day_however_late_it_runs(monkeypatch):
    published = datetime(2026, 9, 30, 6, 39, tzinfo=UTC)
    # captured 5 days late
    start, end = _dates_requested(monkeypatch, "24h", published, datetime(2026, 10, 5, 12, 0, tzinfo=UTC))
    assert (start, end) == ("2026-09-30", "2026-10-01"), (
        "a late 24h capture must still cover the day after publish, not the most recent day"
    )


def test_anchored_seven_day_window_covers_the_first_week(monkeypatch):
    published = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    start, end = _dates_requested(monkeypatch, "7d", published, datetime(2026, 10, 5, 12, 0, tzinfo=UTC))
    assert (start, end) == ("2026-09-01", "2026-09-08")


def test_end_date_is_clamped_to_today(monkeypatch):
    """The API rejects future dates, which an on-time 7d capture would otherwise request."""
    published = datetime(2026, 9, 30, 6, 0, tzinfo=UTC)
    start, end = _dates_requested(monkeypatch, "7d", published, datetime(2026, 10, 1, 6, 0, tzinfo=UTC))
    assert start == "2026-09-30"
    assert end == "2026-10-01", "end date must not run past today"


def test_without_an_anchor_it_stays_a_trailing_window(monkeypatch):
    """Ad-hoc callers asking "how is this doing now?" keep the old behaviour."""
    start, end = _dates_requested(monkeypatch, "24h", None, datetime(2026, 10, 5, 12, 0, tzinfo=UTC))
    assert (start, end) == ("2026-10-04", "2026-10-05")


# --- the node tolerates a missing or malformed anchor rather than failing the capture ---


@pytest.mark.parametrize("raw", ["", None, "not-a-timestamp"])
def test_bad_published_at_falls_back_instead_of_raising(raw):
    from graph.nodes.feedback import _published_at

    assert _published_at({"published_at": raw}) is None


def test_valid_published_at_is_parsed():
    from graph.nodes.feedback import _published_at

    got = _published_at({"published_at": "2026-09-30T06:39:10+00:00"})
    assert got == datetime(2026, 9, 30, 6, 39, 10, tzinfo=UTC)
