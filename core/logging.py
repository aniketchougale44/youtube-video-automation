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


def code_version() -> str:
    """A fingerprint of the Python source this process would import, as "<git sha>+<hash>" or just
    the hash.

    Answers one question: is the code a running process loaded still the code on disk? The
    containers bind-mount the source, so editing a file changes the disk and not the already-running
    interpreter, and nothing about the container's state hints at the difference. That cost this
    project a feature -- a run was queued carrying a user_topic the running worker had no code to
    read, so it silently did the full trend research the feature exists to skip.

    Hashing the tree rather than asking git, because .git is not mounted into the containers (git
    there returns "unknown", which is exactly as useless as no check at all) and because the mounted
    files, not the commit, are what actually gets imported. Size and mtime rather than contents: it
    is ~10x cheaper on a few hundred files and cannot miss an edit, since writing a file changes
    both.
    """
    import hashlib
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    packages = ("agents", "api", "core", "db", "graph", "scheduler", "tools", "worker")

    digest = hashlib.sha256()
    for package in packages:
        base = os.path.join(root, package)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for name in sorted(f for f in filenames if f.endswith(".py")):
                path = os.path.join(dirpath, name)
                try:
                    stat = os.stat(path)
                except OSError:
                    continue
                digest.update(os.path.relpath(path, root).replace(os.sep, "/").encode())
                digest.update(f"{stat.st_size}:{int(stat.st_mtime)}".encode())
    return digest.hexdigest()[:12]
