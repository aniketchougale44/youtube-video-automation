"""Focused tests for the real Performance Monitor + Learning Agent logic (graph/nodes/feedback.py).
External calls (tools.youtube.analytics_report) are mocked here explicitly -- the feedback graph
isn't part of the main publish graph so conftest.py's autouse fixture doesn't touch it."""
from graph.nodes.feedback import learning_node, performance_monitor_node
from tools.youtube import YouTubeNotConfiguredError


def test_performance_monitor_node_builds_snapshot_from_real_call(monkeypatch):
    monkeypatch.setattr(
        "graph.nodes.feedback.youtube_tool.analytics_report",
        lambda video_id, window: {
            "video_id": video_id, "window": window, "views": 1234, "impressions": 0, "ctr": 0.0,
            "avg_view_duration_seconds": 88.5, "retention_pct": 42.0, "likes": 50, "comments": 4,
        },
    )
    result = performance_monitor_node({"youtube_video_id": "abc123", "window": "24h"})
    snapshot = result["performance_snapshot"]

    assert snapshot["youtube_video_id"] == "abc123"
    assert snapshot["window"] == "24h"
    assert snapshot["views"] == 1234
    assert snapshot["retention_pct"] == 42.0
    assert result["trace"][-1]["event"] == "complete"


def test_performance_monitor_node_degrades_gracefully_without_oauth(monkeypatch):
    def _raise(*args, **kwargs):
        raise YouTubeNotConfiguredError("OAuth not configured, missing: YOUTUBE_REFRESH_TOKEN")

    monkeypatch.setattr("graph.nodes.feedback.youtube_tool.analytics_report", _raise)
    result = performance_monitor_node({"youtube_video_id": "abc123", "window": "7d"})
    snapshot = result["performance_snapshot"]

    assert snapshot["views"] == 0
    assert snapshot["youtube_video_id"] == "abc123"
    assert result["trace"][-1]["event"] == "analytics_not_configured"


def test_learning_node_reports_insufficient_data_with_no_baseline():
    result = learning_node({"performance_snapshot": {"youtube_video_id": "abc123", "views": 500, "retention_pct": 40.0}})
    update = result["learning_update"]

    assert update["strategy_weight_adjustments"] == {}
    assert "not enough prior" in update["insights"][0].lower()


def test_learning_node_flags_retention_above_baseline():
    result = learning_node({
        "performance_snapshot": {"youtube_video_id": "abc123", "views": 1000, "retention_pct": 55.0},
        "baseline": {"sample_size": 10, "avg_views": 1000.0, "avg_retention_pct": 40.0},
    })
    update = result["learning_update"]

    assert update["strategy_weight_adjustments"]["evergreen_bias"] > 0
    assert any("above" in insight for insight in update["insights"])


def test_learning_node_flags_retention_below_baseline():
    result = learning_node({
        "performance_snapshot": {"youtube_video_id": "abc123", "views": 1000, "retention_pct": 20.0},
        "baseline": {"sample_size": 10, "avg_views": 1000.0, "avg_retention_pct": 40.0},
    })
    update = result["learning_update"]

    assert update["strategy_weight_adjustments"]["evergreen_bias"] < 0
    assert any("below" in insight for insight in update["insights"])


def test_learning_node_flags_low_views_relative_to_baseline():
    result = learning_node({
        "performance_snapshot": {"youtube_video_id": "abc123", "views": 100, "retention_pct": 40.0},
        "baseline": {"sample_size": 10, "avg_views": 1000.0, "avg_retention_pct": 40.0},
    })
    update = result["learning_update"]

    assert any("well below" in insight for insight in update["insights"])


def test_learning_node_reports_no_adjustment_when_in_line_with_baseline():
    result = learning_node({
        "performance_snapshot": {"youtube_video_id": "abc123", "views": 1000, "retention_pct": 40.0},
        "baseline": {"sample_size": 10, "avg_views": 1000.0, "avg_retention_pct": 40.0},
    })
    update = result["learning_update"]

    assert update["strategy_weight_adjustments"] == {}
    assert "in line" in update["insights"][0]
