"""Celery app instance. Run with: celery -A worker.celery_app worker --loglevel=info
(the `worker` service in docker-compose.yml does exactly this)."""
from celery import Celery

from core.logging import code_version, get_logger
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
    task_acks_late=True,          # ack on completion, not on receipt
    # task_acks_late alone does NOT survive a killed worker, despite what it sounds like. Celery
    # acknowledges the message whenever the child process executing a task is terminated by a
    # signal -- deliberately, to stop a task that segfaults the worker from being redelivered
    # forever. So `acks_late` protects against a task *raising*, not against the worker *dying*.
    #
    # Verified here: the worker was SIGQUIT'd during asset_visual, six nodes in. Afterwards the
    # broker db held only _kombu.binding.* keys -- no queued message, no `unacked` hash -- and the
    # run sat at status='pending' forever with 8 checkpoints of finished work and nothing left to
    # resume it. Exactly the silent drop this line used to claim it prevented.
    #
    # The tradeoff this accepts: a task that reliably kills its worker will now be redelivered
    # rather than dropped. run_pipeline_task's own max_retries=3 does not bound that, since a
    # worker-loss requeue is not a task retry. A genuinely poisonous run is the lesser problem --
    # it is visible and fixable, whereas a dropped run is neither.
    task_reject_on_worker_lost=True,
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


# Announce which commit this process actually imported, and publish it where the dashboard can read
# it. A bind-mounted source directory means the files on disk can be several commits ahead of the
# running interpreter, and nothing about the container's state hints at that.
_logger = get_logger("worker.celery_app")
WORKER_VERSION = code_version()
_logger.info("worker.starting", code_version=WORKER_VERSION)

try:
    import redis as _redis

    _redis.Redis.from_url(settings.redis_url, decode_responses=True, protocol=2).set(
        "worker:code_version", WORKER_VERSION
    )
except Exception as _exc:  # Redis down must never stop the worker booting
    _logger.warning("worker.version_publish_failed", error=str(_exc))
