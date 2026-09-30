"""Per-run spend capture, so cost-per-video is measured rather than estimated.

`db.models.Cost` and `agents.schemas.common.StageCost` have existed since the initial schema with
nothing writing to them, which is why the README's cost table was a price-list guess.

The awkward part is attribution: the code that knows the *price* (core/llm.py, tools/tts.py,
tools/image_gen.py) is several frames below the code that knows the *stage*, and threading a
collector through every call site would touch every node signature. So a ContextVar holds both the
current stage and the collector for the run in flight:

  * graph/nodes/_helpers.py::log_and_trace sets the stage. Every node calls it on entry, so that is
    one hook rather than sixteen.
  * worker/tasks.py opens a collection scope per task and drains it into Postgres afterwards.

Nothing here can fail a run. An unpriced model, a provider that reports no token usage, a missing
collection scope -- all degrade to recording less, never to raising. A cost ledger that breaks
video production would be a bad trade.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from core.logging import get_logger

logger = get_logger("tools.cost")

_entries: ContextVar[list[dict] | None] = ContextVar("cost_entries", default=None)
_stage: ContextVar[str] = ContextVar("cost_stage", default="")

# USD per 1M tokens (input, output), from each provider's public list price. Substring-matched
# against the model id, so "claude-sonnet-4-5-20250929" matches "claude-sonnet". Unpriced models
# record units with a zero rate rather than being dropped -- token counts are still useful, and a
# visible 0.0 invites a fix where a silently missing row does not.
LLM_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (0.80, 4.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-3": (0.30, 2.50),
    "llama-3.3-70b": (0.59, 0.79),      # groq
    "openai/gpt-oss-120b": (0.15, 0.75),  # groq
}

# Flat per-unit prices for the non-LLM providers.
TTS_PRICES_PER_MCHAR = {"openai": 15.0}   # tts-1, $15 / 1M characters
IMAGE_PRICES_PER_IMAGE = {"openai": 0.04}  # gpt-image-1 / dall-e-3 standard 1024px


@contextmanager
def collecting():
    """Opens a collection scope. Entries recorded inside it are returned by collected()."""
    token = _entries.set([])
    try:
        yield
    finally:
        _entries.reset(token)


def collected() -> list[dict]:
    return list(_entries.get() or [])


def set_stage(stage: str) -> None:
    _stage.set(stage)


def current_stage() -> str:
    return _stage.get()


def record(provider: str, units: float, unit_cost_usd: float, stage: str | None = None, **metadata: Any) -> None:
    """Appends one spend entry. Silently does nothing outside a collecting() scope, so importing
    this module never changes behaviour for scripts or tests that don't opt in."""
    bucket = _entries.get()
    if bucket is None:
        return
    entry = {
        "stage": stage or _stage.get() or "",
        "provider": provider,
        "units": float(units),
        "unit_cost_usd": float(unit_cost_usd),
        "total_usd": round(float(units) * float(unit_cost_usd), 6),
        "metadata": metadata,
    }
    bucket.append(entry)
    logger.info("cost.recorded", **{k: v for k, v in entry.items() if k != "metadata"})


def _price_for_model(model: str) -> tuple[float, float] | None:
    name = (model or "").lower()
    for key, prices in LLM_PRICES_PER_MTOK.items():
        if key.lower() in name:
            return prices
    return None


def record_llm(provider: str, model: str, input_tokens: int, output_tokens: int) -> None:
    """Records one LLM call. Input and output are separate entries because their rates differ by up
    to 5x -- collapsing them into an average would make the totals wrong in a way nobody could
    later untangle from the stored row."""
    prices = _price_for_model(model)
    in_rate, out_rate = prices if prices else (0.0, 0.0)
    if prices is None:
        logger.warning("cost.unpriced_model", provider=provider, model=model)

    if input_tokens:
        record(provider, input_tokens / 1_000_000, in_rate, model=model, kind="input_tokens")
    if output_tokens:
        record(provider, output_tokens / 1_000_000, out_rate, model=model, kind="output_tokens")


def record_tts(provider: str, characters: int) -> None:
    rate = TTS_PRICES_PER_MCHAR.get(provider, 0.0)  # edge-tts is free -> 0.0, still worth counting
    record(provider, characters / 1_000_000, rate, kind="tts_characters")


def record_image(provider: str, count: int = 1) -> None:
    rate = IMAGE_PRICES_PER_IMAGE.get(provider, 0.0)  # pollinations is 0.0 (when it works)
    record(provider, count, rate, kind="images")
