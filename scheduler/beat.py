"""Standalone scheduler process — entrypoint: `python -m scheduler.beat`.

Deliberately its own process/container (see docker-compose.yml's `scheduler` service), not an
APScheduler instance started inside the FastAPI app, so a slow API deploy/restart can never skip
a cron firing and a stuck request can never block the schedule.
"""
from datetime import UTC, datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from core.logging import configure_logging, get_logger
from core.settings import get_settings
from scheduler.jobs import (
    check_performance_windows,
    renew_youtube_websub_subscription,
    trigger_new_pipeline_run,
)

configure_logging()
logger = get_logger("scheduler.beat")


def build_scheduler() -> BlockingScheduler:
    settings = get_settings()
    scheduler = BlockingScheduler(timezone="UTC")

    scheduler.add_job(
        trigger_new_pipeline_run,
        CronTrigger.from_crontab(settings.pipeline_cron_schedule),
        id="trigger_new_pipeline_run",
        replace_existing=True,
        misfire_grace_time=3600,
    )
    scheduler.add_job(
        check_performance_windows,
        "interval",
        hours=1,
        id="check_performance_windows",
        replace_existing=True,
        misfire_grace_time=600,
    )
    scheduler.add_job(
        renew_youtube_websub_subscription,
        "interval",
        hours=24,
        id="renew_youtube_websub_subscription",
        replace_existing=True,
        misfire_grace_time=3600,
        next_run_time=datetime.now(UTC),  # also subscribe immediately on scheduler startup
    )
    return scheduler


def main() -> None:
    settings = get_settings()
    scheduler = build_scheduler()
    logger.info("scheduler.starting", pipeline_cron=settings.pipeline_cron_schedule, jobs=[j.id for j in scheduler.get_jobs()])
    scheduler.start()


if __name__ == "__main__":
    main()
