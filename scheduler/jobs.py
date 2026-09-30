"""Job bodies invoked by APScheduler (scheduler/beat.py). Each just creates/updates DB rows and
enqueues a Celery task — no pipeline or API logic lives here."""
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from core.logging import get_logger
from db.base import SessionLocal
from db.crud import create_run
from db.models import PerformanceSnapshot, Video

logger = get_logger("scheduler.jobs")

# (window, earliest age at which it may be captured). There is deliberately no upper bound: the
# pair used to be (23h, 25h) and (6d23h, 7d1h), a 2-hour band checked hourly, so a host asleep
# across that band lost the snapshot permanently -- the `already_captured` check only skips work,
# it never backfills. Since tools.youtube.analytics_report anchors the date range to the publish
# date, a late capture still measures the window it claims to, which makes catching up correct
# rather than merely better than nothing.
PERFORMANCE_WINDOWS = (
    ("24h", timedelta(hours=23)),
    ("7d", timedelta(days=6, hours=23)),
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
            for window, earliest in PERFORMANCE_WINDOWS:
                if age < earliest:
                    continue  # too soon; a later run of this job will pick it up
                already_captured = db.execute(
                    select(PerformanceSnapshot).where(
                        PerformanceSnapshot.video_id == video.id, PerformanceSnapshot.window == window
                    )
                ).first()
                if already_captured:
                    continue
                performance_feedback_task.delay(video.youtube_video_id, window)
                logger.info(
                    "scheduler.triggered_performance_check",
                    video_id=str(video.id),
                    window=window,
                    age_hours=round(age.total_seconds() / 3600, 1),
                    late=age > earliest + timedelta(hours=2),
                )
    finally:
        db.close()
