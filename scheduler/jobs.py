"""Job bodies invoked by APScheduler (scheduler/beat.py). Each just creates/updates DB rows and
enqueues a Celery task — no pipeline or API logic lives here."""
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from core.logging import get_logger
from db.base import SessionLocal
from db.crud import create_run
from db.models import PerformanceSnapshot, Video

logger = get_logger("scheduler.jobs")

# (window label, lower bound, upper bound) — a video's age must fall in this range for its
# snapshot to be due. Bounds are wide enough to tolerate the job's own polling interval.
PERFORMANCE_WINDOWS = (
    ("24h", timedelta(hours=23), timedelta(hours=25)),
    ("7d", timedelta(days=6, hours=23), timedelta(days=7, hours=1)),
)


def trigger_new_pipeline_run() -> None:
    """Fired on the configured cadence (PIPELINE_CRON_SCHEDULE, e.g. 3x/week)."""
    from worker.tasks import run_pipeline_task  # deferred: avoids import cycle at module load

    db = SessionLocal()
    try:
        run = create_run(db)
        run_pipeline_task.delay(str(run.id))
        logger.info("scheduler.triggered_run", run_id=str(run.id))
    finally:
        db.close()


def renew_youtube_websub_subscription() -> None:
    """Re-subscribes to the channel's PubSubHubbub feed before the hub's lease expires (leases run
    <=10 days; we renew daily). No-op unless the callback URL + channel id are configured."""
    from core.settings import get_settings
    from tools import websub

    settings = get_settings()
    if not (settings.youtube_channel_id and settings.youtube_websub_callback_url):
        return
    try:
        websub.subscribe()
        logger.info("scheduler.websub_renewed")
    except Exception as exc:
        logger.error("scheduler.websub_renew_failed", error=str(exc))


def check_performance_windows() -> None:
    """Runs hourly. Finds published videos that just crossed the 24h/7d mark and haven't had
    that window captured yet, and enqueues the performance-feedback graph for each."""
    from worker.tasks import (
        performance_feedback_task,  # deferred: avoids import cycle at module load
    )

    db = SessionLocal()
    try:
        now = datetime.now(UTC)
        videos = db.execute(
            select(Video).where(Video.published_at.is_not(None), Video.youtube_video_id.is_not(None))
        ).scalars()

        for video in videos:
            age = now - video.published_at
            for window, lower, upper in PERFORMANCE_WINDOWS:
                if not (lower <= age <= upper):
                    continue
                already_captured = db.execute(
                    select(PerformanceSnapshot).where(
                        PerformanceSnapshot.video_id == video.id, PerformanceSnapshot.window == window
                    )
                ).first()
                if already_captured:
                    continue
                performance_feedback_task.delay(video.youtube_video_id, window)
                logger.info("scheduler.triggered_performance_check", video_id=str(video.id), window=window)
    finally:
        db.close()
