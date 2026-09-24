"""Lightweight progress reporting for the standalone scripts/produce_*.py one-offs.

These scripts deliberately bypass the LangGraph pipeline (graph/builder.py) -- see their own
docstrings -- so they never create a db.models.Run row and are invisible to the dashboard's run
tracking (api/routes/dashboard.py) by default: someone watching the dashboard while one of these
runs on the host has no way to see it's even happening. This module gives them a minimal way to
report "what's actually happening" -- current step, progress, errors -- through the same Redis
instance tools/cache.py already uses for caching. Redis is the right shared store here (rather
than a status file under media/): media/ is a host path for these host-run scripts but a separate
Docker named volume inside the api container (see docker-compose.yml), so a file written on the
host is invisible to the dashboard process -- whereas Redis is exposed on the host (see .env's
REDIS_URL / docker-compose.override.yml's port remap) and reachable from inside the containers on
the same instance, so both sides see the same state. api/routes/dashboard.py's _script_runs reads
this back to render a "Background script activity" section alongside pipeline runs.

Not for concurrent runs of the same script -- a second run of the same script name overwrites the
first's status, same tradeoff scripts/produce_*.py already accept by reusing one OUTPUT_DIR.
"""
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime

from tools.cache import cache_set_json

KEY_PREFIX = "script_run:"
TTL_SECONDS = 60 * 60 * 6  # stale entries (crashed interpreter, killed process, forgotten test
# run) age out on their own instead of cluttering the dashboard forever


class ScriptRun:
    def __init__(self, name: str, total_steps: int):
        self.name = name
        self.total_steps = total_steps
        self.step_index = 0
        self.step_label = "starting"
        self.status = "running"
        self.output_path: str | None = None
        self.error: str | None = None
        self.started_at = datetime.now(UTC).isoformat()
        self._save()

    def _save(self) -> None:
        cache_set_json(
            f"{KEY_PREFIX}{self.name}",
            {
                "name": self.name,
                "status": self.status,
                "step_index": self.step_index,
                "total_steps": self.total_steps,
                "step_label": self.step_label,
                "output_path": self.output_path,
                "error": self.error,
                "started_at": self.started_at,
                "updated_at": datetime.now(UTC).isoformat(),
            },
            ttl_seconds=TTL_SECONDS,
        )

    def step(self, label: str) -> None:
        self.step_index += 1
        self.step_label = label
        self._save()

    def done(self, output_path: str | None = None) -> None:
        self.status = "done"
        self.step_label = "done"
        self.step_index = self.total_steps
        self.output_path = output_path
        self._save()

    def fail(self, error: str) -> None:
        self.status = "failed"
        self.step_label = "failed"
        self.error = error
        self._save()


@contextmanager
def track(name: str, total_steps: int):
    """Usage:
        with script_progress.track("produce_mimi_rainbow_story", total_steps=20) as run:
            run.step("generating backdrop 1/9")
            ...
            run.done(output_path=final_path)

    Auto-marks the run failed (capturing the traceback) if the block raises, and auto-marks it
    done if the block exits normally without an explicit run.done()/run.fail() call.
    """
    run = ScriptRun(name, total_steps)
    try:
        yield run
    except Exception:
        run.fail(traceback.format_exc(limit=5))
        raise
    else:
        if run.status == "running":
            run.done()
