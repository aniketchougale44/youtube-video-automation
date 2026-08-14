"""Performance Monitor Agent + Learning Agent — real logic.

These run in a *separate* scheduled graph (graph/feedback_graph.py), not the main publish
pipeline, because they fire on a delay (24h/7d) after a video goes live, decoupled from the
run that published it.

Like every other node, these stay DB-free: worker/tasks.py::performance_feedback_task computes
the historical `baseline` (average views/retention across prior published videos) from Postgres
and injects it into the graph's initial state, the same way run_pipeline_task injects
past_performance_summary into the publish graph.
"""
from agents.schemas.feedback import LearningUpdate, PerformanceSnapshot, PerformanceWindow
from graph.nodes._helpers import log_and_trace
from tools import youtube as youtube_tool
from tools.youtube import YouTubeNotConfiguredError

STAGE_PERF = "performance_monitor"
STAGE_LEARN = "learning"

# A retention/view swing smaller than this is treated as noise, not a real signal worth an insight.
_RETENTION_DELTA_THRESHOLD_PP = 5.0
_VIEWS_RATIO_LOW = 0.5
_VIEWS_RATIO_HIGH = 1.5


def performance_monitor_node(state: dict) -> dict:
    video_id = state.get("youtube_video_id", "")
    window = state.get("window", "24h")
    trace = log_and_trace(STAGE_PERF, "start", youtube_video_id=video_id, window=window)

    try:
        report = youtube_tool.analytics_report(video_id, window)
        snapshot = PerformanceSnapshot(
            youtube_video_id=video_id,
            window=PerformanceWindow(window),
            views=report.get("views", 0),
            impressions=report.get("impressions", 0),
            ctr=report.get("ctr", 0.0),
            avg_view_duration_seconds=report.get("avg_view_duration_seconds", 0.0),
            retention_pct=report.get("retention_pct", 0.0),
            likes=report.get("likes", 0),
            comments=report.get("comments", 0),
        )
        complete_trace = log_and_trace(
            STAGE_PERF, "complete", views=snapshot.views, retention_pct=snapshot.retention_pct
        )
    except YouTubeNotConfiguredError as exc:
        # YouTube OAuth (YOUTUBE_CLIENT_ID/SECRET/REFRESH_TOKEN) isn't set up yet -- don't fail the
        # whole feedback graph over it, since this fires on an unattended schedule. Record a
        # zeroed snapshot so the run completes and the gap is visible in the trace/AgentLog instead
        # of silently retrying forever.
        snapshot = PerformanceSnapshot(youtube_video_id=video_id, window=PerformanceWindow(window))
        complete_trace = log_and_trace(STAGE_PERF, "analytics_not_configured", error=str(exc))

    return {
        "performance_snapshot": snapshot.model_dump(mode="json"),
        "trace": [trace, complete_trace],
    }


def learning_node(state: dict) -> dict:
    trace = log_and_trace(STAGE_LEARN, "start")

    snapshot = state.get("performance_snapshot") or {}
    baseline = state.get("baseline") or {}
    sample_size = baseline.get("sample_size", 0)

    insights: list[str] = []
    adjustments: dict[str, float] = {}

    if sample_size == 0:
        insights.append("Not enough prior published videos yet to compare against — no adjustment made.")
    else:
        views = snapshot.get("views", 0)
        retention = snapshot.get("retention_pct", 0.0)
        avg_views = baseline.get("avg_views", 0.0)
        avg_retention = baseline.get("avg_retention_pct", 0.0)

        if avg_retention > 0:
            retention_delta = retention - avg_retention
            if abs(retention_delta) >= _RETENTION_DELTA_THRESHOLD_PP:
                direction = "above" if retention_delta > 0 else "below"
                insights.append(
                    f"Retention {retention:.1f}% is {abs(retention_delta):.1f}pp {direction} the "
                    f"{sample_size}-video average ({avg_retention:.1f}%)."
                )
                # Nudge the evergreen/trending balance in the direction retention moved; kept small
                # and bounded so one outlier video can't swing strategy on its own.
                adjustments["evergreen_bias"] = round(max(-0.2, min(0.2, retention_delta / 100)), 3)

        if avg_views > 0:
            views_ratio = views / avg_views
            if views_ratio <= _VIEWS_RATIO_LOW or views_ratio >= _VIEWS_RATIO_HIGH:
                comparison = "well below" if views_ratio < 1 else "well above"
                insights.append(f"Views ({views}) are {comparison} the recent average ({avg_views:.0f}).")

        if not insights:
            insights.append("Performance is in line with the recent average — no strategy adjustment warranted.")

    update = LearningUpdate(
        based_on_video_ids=[snapshot.get("youtube_video_id", "")],
        insights=insights,
        strategy_weight_adjustments=adjustments,
    )

    return {
        "learning_update": update.model_dump(mode="json"),
        "trace": [
            trace,
            log_and_trace(STAGE_LEARN, "complete", insights=update.insights, adjustments=update.strategy_weight_adjustments),
        ],
    }
