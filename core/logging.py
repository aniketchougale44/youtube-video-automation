"""structlog configuration. Every agent/tool call logs through this, one structured event per call."""
import logging
import sys

import structlog

from core.settings import get_settings


def _force_utf8_streams() -> None:
    """Make stdout/stderr encode anything, so a log call can never kill a run.

    structlog's PrintLoggerFactory writes with plain print(), and on Windows stdout defaults to
    cp1252. Every logged string here is potentially LLM-generated or non-Latin -- this pipeline
    produces Hindi and trilingual scripts -- so a title containing something as ordinary as a
    non-breaking hyphen (U+2011) raises UnicodeEncodeError from inside the logger.

    That failure mode is vicious rather than merely annoying: the `except` block that tries to
    report it logs the same offending string, raises again, and nothing catches the second one. A
    whole pipeline run dies inside its own error reporting, four Celery retries deep, with the real
    cause ("a hyphen") nowhere near the traceback.

    backslashreplace over 'replace' so a mangled character stays diagnosable in the log rather
    than becoming an anonymous '?'.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError, ValueError):
            # Not a reconfigurable TextIOWrapper (pytest capture, a pipe someone replaced with a
            # StringIO). Nothing to do, and not worth failing startup over.
            pass


def configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    _force_utf8_streams()
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
            if settings.app_env == "production"
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
