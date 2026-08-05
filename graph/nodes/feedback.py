"""Performance Monitor Agent + Learning Agent — stub logic.

These run in a *separate* scheduled graph (graph/feedback_graph.py), not the main publish
pipeline, because they fire on a delay (24h/7d) after a video goes live, decoupled from the
run that published it.
"""
from agents.schemas.feedback import LearningUpdate, PerformanceSnapshot, PerformanceWindow
from graph.nodes._helpers import log_and_trace

STAGE_PERF = "performance_monitor"
STAGE_LEARN = "learning"


def performance_monitor_node(state: dict) -> dict:
    trace = log_and_trace(STAGE_PERF, "start", youtube_video_id=state.get("youtube_video_id"))

    # STUB: real logic calls tools.youtube.analytics_report() for the given video_id/window
    snapshot = PerformanceSnapshot(
        youtube_video_id=state.get("youtube_video_id", "stub_yt_id_000"),
        window=PerformanceWindow(state.get("window", "24h")),
        views=100,
        impressions=1000,
        ctr=0.1,
        avg_view_duration_seconds=120.0,
        retention_pct=45.0,
        likes=10,
        comments=2,
    )

    return {
        "performance_snapshot": snapshot.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_PERF, "complete", views=snapshot.views)],
    }


def learning_node(state: dict) -> dict:
    trace = log_and_trace(STAGE_LEARN, "start")

    # STUB: real logic aggregates PerformanceSnapshot rows across recent runs and adjusts
    # Strategy Agent prompt weights (e.g. favor formats/topics with higher retention)
    update = LearningUpdate(
        based_on_video_ids=[state.get("youtube_video_id", "stub_yt_id_000")],
        insights=["[STUB] placeholder insight"],
        strategy_weight_adjustments={"evergreen_bias": 0.0},
    )

    return {
        "learning_update": update.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_LEARN, "complete", insights=len(update.insights))],
    }
