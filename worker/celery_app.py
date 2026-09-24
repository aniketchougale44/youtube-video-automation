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
    # Redis has no real per-message ack, so kombu fakes it with a visibility timeout: any task
    # still unacked after it expires is handed to another worker. Default is 1 hour, and with
    # task_acks_late=True the ack only lands when the task *finishes* -- so a publish run longer
    # than an hour gets redelivered and the same run executes twice, concurrently. That's routine
    # now that AI_VIDEO beats are uncapped: a free-Colab clip is ~6-12 min, so a dozen-beat script
    # is a ~2 hour task. 12h matches Colab's own hard session cap, past which the run is dead
    # anyway (its endpoint is gone) and redelivery would just repeat a doomed render.
    broker_transport_options={"visibility_timeout": 60 * 60 * 12},
)
