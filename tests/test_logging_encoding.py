"""Logging must never be the thing that kills a run.

Regression test for a real pipeline failure: the Strategy Agent proposed the title
"Cross‑Play Chronicles: When Game Worlds Collide" (a non-breaking hyphen), tools/trends.py
logged it, structlog's PrintLogger wrote it to a cp1252 stdout, and the resulting
UnicodeEncodeError propagated out of trend_research_node. The `except` block then logged the same
title, raised again with nothing to catch it, and the run failed through all four Celery attempts.

This matters beyond one hyphen: the pipeline produces Hindi and trilingual scripts, so Devanagari
would land in log lines on every such run.
"""
import os
import subprocess
import sys

# U+2011 is what actually broke it; the rest cover the multilingual content this project produces.
HOSTILE_STRINGS = [
    "Cross‑Play Chronicles: When Game Worlds Collide",  # non-breaking hyphen
    "हिंदी कहानी",  # Hindi (Devanagari)
    "café naïve — em–dashes",
    "emoji \U0001f984 outside the BMP",
]


def _run_in_cp1252(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a subprocess whose stdout really is cp1252.

    A plain in-process assertion proves nothing here: pytest replaces sys.stdout with a UTF-8
    capture object, so the original bug is invisible under the test runner. Forcing the encoding
    via PYTHONIOENCODING in a child is the only way to reproduce the Windows console condition on
    any host.
    """
    # Inherit the real environment and override only the encoding. A hand-built minimal env drops
    # SYSTEMROOT/PATH, and importing asyncio on Windows then dies with WinError 10106 before the
    # test reaches the thing it is actually checking.
    env = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    env.pop("PYTHONUTF8", None)  # would override PYTHONIOENCODING and hide the condition
    return subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


def test_logging_survives_non_cp1252_text_on_a_cp1252_stream():
    body = (
        "from core.logging import configure_logging, get_logger\n"
        "configure_logging()\n"
        "log = get_logger('test')\n"
        f"for s in {HOSTILE_STRINGS!r}:\n"
        "    log.info('hostile', value=s)\n"
        "print('SURVIVED')\n"
    )
    proc = _run_in_cp1252(body)
    assert "UnicodeEncodeError" not in proc.stderr, proc.stderr[-800:]
    assert "SURVIVED" in proc.stdout, f"stdout={proc.stdout[-400:]!r} stderr={proc.stderr[-800:]!r}"


def test_the_error_handler_path_also_survives():
    """The original failure was the `except` block re-raising while reporting. Logging an
    exception whose message carries the offending text must be safe too."""
    body = (
        "from core.logging import configure_logging, get_logger\n"
        "configure_logging()\n"
        "log = get_logger('test')\n"
        "title = 'Cross\\u2011Play Chronicles'\n"
        "try:\n"
        "    raise RuntimeError(f'lookup failed for {title}')\n"
        "except Exception as exc:\n"
        "    log.warning('lookup_failed', title=title, error=str(exc))\n"
        "print('SURVIVED')\n"
    )
    proc = _run_in_cp1252(body)
    assert "UnicodeEncodeError" not in proc.stderr, proc.stderr[-800:]
    assert "SURVIVED" in proc.stdout, f"stdout={proc.stdout[-400:]!r} stderr={proc.stderr[-800:]!r}"


def test_without_the_fix_a_cp1252_stream_really_does_fail():
    """Guards the guard: if this ever stops failing, the two tests above have stopped testing
    anything and the reconfigure() call could be deleted unnoticed."""
    body = (
        "import sys\n"
        "print(sys.stdout.encoding)\n"
        "sys.stdout.write('Cross\\u2011Play')\n"
    )
    proc = _run_in_cp1252(body)
    assert "cp1252" in proc.stdout.lower(), f"child stdout was not cp1252: {proc.stdout!r}"
    assert "UnicodeEncodeError" in proc.stderr, (
        "a raw cp1252 write of U+2011 no longer raises, so these tests prove nothing: "
        f"{proc.stderr[-400:]!r}"
    )
