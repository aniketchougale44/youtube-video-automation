"""Celery app instance. Run with: celery -A worker.celery_app worker --loglevel=info
(the `worker` service in docker-compose.yml does exactly this)."""
from celery import Celery

from core.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "autotube",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_acks_late=True,          # redeliver if a worker dies mid-task instead of silently dropping it
    worker_prefetch_multiplier=1,  # don't hoard long-running render/upload tasks across workers
    result_expires=60 * 60 * 24 * 7,
)
