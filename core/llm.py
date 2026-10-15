"""Shared LLM client: tries each configured provider in order and falls back on failure. Every
agent node that needs an LLM call goes through `call_structured()` so provider fallback routing
lives in exactly one place instead of being reimplemented per agent.

Provider order: Anthropic (paid) -> OpenAI (paid) -> Groq (free tier, no card) -> Gemini (free
tier, no card). A deployment with none of the paid keys set still runs end-to-end on Groq and/or
Gemini alone — each factory below just returns None when its key isn't set, so the chain skips
straight past it.
"""
import time
from typing import TypeVar

from pydantic import BaseModel

from core.logging import get_logger
from core.settings import get_settings
from tools import cost as cost_tool

logger = get_logger("core.llm")

T = TypeVar("T", bound=BaseModel)


class NoLLMProviderConfigured(RuntimeError):
    """Raised when no provider key is set, or every configured provider failed."""


def _anthropic_llm():
    settings = get_settings()
    if not settings.anthropic_api_key:
        return None
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=settings.llm_primary_model, api_key=settings.anthropic_api_key, timeout=60, max_retries=2)


def _openai_llm():
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.llm_fallback_model, api_key=settings.openai_api_key, timeout=60, max_retries=2)


def _groq_llm():
    """Free tier, no card required: https://console.groq.com/keys"""
    settings = get_settings()
    if not settings.groq_api_key:
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(model=settings.llm_groq_model, api_key=settings.groq_api_key, timeout=60, max_retries=2)


def _deepseek_llm():
    """DeepSeek via its OpenAI-compatible endpoint, so ChatOpenAI serves it with no new dependency.

    Pinned to deepseek-chat (V3) rather than deepseek-reasoner (R1) on purpose. Every agent node
    reaches the model through call_structured() -> with_structured_output(), which is built on
    function calling, and the reasoner does not support function calling or structured output. It
    would not "reason better" here, it would fail the only calling convention this pipeline has.

    The reasoning that matters for video decisions -- what to make, whether a script passes the
    critics -- lives in the graph, which is where it stays inspectable and testable.
    """
    settings = get_settings()
    if not settings.deepseek_api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url="https://api.deepseek.com",
        timeout=60,
        max_retries=2,
    )


def _gemini_llm():
    """Free tier, no card required: https://aistudio.google.com/apikey"""
    settings = get_settings()
    if not settings.google_api_key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=settings.llm_gemini_model, api_key=settings.google_api_key, timeout=60, max_retries=2)


# Order is fallback order, not preference: the first provider with a key configured handles the
# call, and the rest exist for when it fails. DeepSeek sits ahead of the free tiers because it is
# the one deliberately chosen for judgement-heavy nodes; groq and gemini remain underneath as the
# no-card safety net, which is what keeps the pipeline running when a paid key lapses.
_PROVIDERS = (
    ("anthropic", _anthropic_llm),
    ("openai", _openai_llm),
    ("deepseek", _deepseek_llm),
    ("groq", _groq_llm),
    ("gemini", _gemini_llm),
)


# A key that is present but unusable -- unpaid balance, revoked, out of credit -- fails on every
# single call, and a provider ahead of the working ones turns that into a tax on the whole
# pipeline: ~1s per call, fourteen stages a run, plus the chat the operator is waiting on.
#
# It is specifically the billing/auth failures that are worth remembering. A timeout or a 500 is
# worth retrying next call; "Insufficient Balance" will be just as true a second later, and a
# human has to go and fix it. Those get the provider skipped for a while instead.
_PROVIDER_COOLDOWN_SECONDS = 600

# Only unambiguous, human-must-intervene failures. This list also held "quota" and the bare
# strings "401"/"402"/"403", which was a trap twice over:
#
#   - "quota" matches a free tier's RATE LIMIT ("RESOURCE_EXHAUSTED ... quota"), which clears in
#     seconds. Benching on that takes a working provider out of the chain for ten minutes.
#   - a bare "402" matches anywhere in the text, including inside a request id, so an unrelated
#     error could bench a healthy provider by coincidence.
#
# Everything else -- 429, 503, timeouts, a malformed tool call -- is transient, and is retried.
_UNUSABLE_SIGNS = (
    "insufficient balance",
    "insufficient_quota",      # OpenAI's billing error; distinct from its rate limit
    "invalid api key",
    "incorrect api key",
    "api key not valid",
    "payment required",
    "billing",
)
_unusable_until: dict[str, float] = {}

# Sentinel for "the model declined to call a tool", distinct from a provider failure.
_NO_TOOL = object()

# How many times to walk the whole chain before giving up, and the pause between passes. groq
# (400 "failed to parse tool call") and gemini (503 "model is overloaded") were observed failing
# in the same moment; with a single pass that is a dead run, twenty minutes of work in, for
# something that clears by itself. A second pass seconds later costs almost nothing.
_CHAIN_ATTEMPTS = 3
_CHAIN_BACKOFF_SECONDS = 2.0


def _mark_unusable_if_terminal(provider_name: str, exc: Exception) -> None:
    text = str(exc).lower()
    if any(sign in text for sign in _UNUSABLE_SIGNS):
        _unusable_until[provider_name] = time.monotonic() + _PROVIDER_COOLDOWN_SECONDS
        logger.warning(
            "llm.provider_benched", provider=provider_name,
            seconds=_PROVIDER_COOLDOWN_SECONDS, reason=str(exc)[:160],
        )



def _usable_providers():
    """Providers worth trying right now, in order.

    If every configured provider is benched, the benches are ignored for this call rather than
    reporting "nothing configured". A provider that probably cannot pay is still a better bet
    than failing a run outright, and the alternative is that one bad minute takes the whole
    pipeline down for ten.
    """
    live = [(name, factory) for name, factory in _PROVIDERS if factory() is not None]
    now = time.monotonic()
    ready = [(n, f) for n, f in live if not (_unusable_until.get(n, 0) > now)]
    if ready:
        return ready
    if live:
        logger.warning("llm.all_benched_trying_anyway", providers=[n for n, _ in live])
    return live


def _run_chain(do_call, what: str):
    """Walks the provider chain, and walks it again if the whole pass failed.

    One pass used to be the whole policy, and that is not enough when the providers themselves
    have a bad moment: groq answering 400 "failed to parse tool call" and gemini answering 503
    "model is overloaded" within the same second produced "no usable LLM provider" and killed a
    run that was already twenty minutes deep. Neither failure needed a human; both needed a few
    seconds.

    `do_call(provider_name, llm)` returns the result, or raises to move on to the next provider.
    """
    last_error: Exception | None = None
    for attempt in range(_CHAIN_ATTEMPTS):
        if attempt:
            time.sleep(_CHAIN_BACKOFF_SECONDS * attempt)
            logger.info("llm.chain_retry", attempt=attempt + 1, call=what)
        for provider_name, factory in _usable_providers():
            llm = factory()
            if llm is None:
                continue
            try:
                return do_call(provider_name, llm)
            except Exception as exc:
                logger.error(
                    "llm.call.failed", provider=provider_name,
                    model=_model_name(llm), error=str(exc), call=what,
                )
                _mark_unusable_if_terminal(provider_name, exc)
                last_error = exc

    raise NoLLMProviderConfigured(
        "no usable LLM provider: set one of ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY "
        "(platform.deepseek.com), GROQ_API_KEY (free: console.groq.com/keys), GOOGLE_API_KEY "
        "(free: aistudio.google.com/apikey)"
    ) from last_error


def call_structured(prompt: str, output_model: type[T], system: str | None = None) -> T:
    """Invokes each configured provider in order with structured output (`output_model`) until
    one succeeds. Raises NoLLMProviderConfigured if none are configured or all fail."""
    messages = ([("system", system)] if system else []) + [("human", prompt)]

    def do_call(provider_name: str, llm):
        # include_raw=True so token usage survives. with_structured_output() otherwise returns
        # only the parsed model and the AIMessage carrying usage_metadata is discarded, which is
        # why per-run spend could never be measured. Falls back to the plain call if a provider
        # does not support it -- losing the cost row is acceptable, losing the video is not.
        try:
            wrapped = llm.with_structured_output(output_model, include_raw=True)
            envelope = wrapped.invoke(messages)
            result = envelope["parsed"] if isinstance(envelope, dict) else envelope
            _record_usage(provider_name, _model_name(llm), envelope)
        except TypeError:
            result = llm.with_structured_output(output_model).invoke(messages)
        if result is None:
            raise ValueError("structured output returned no parsed value")
        logger.info("llm.call.success", provider=provider_name, model=_model_name(llm))
        return result

    return _run_chain(do_call, "structured")


def _record_usage(provider: str, model: str, envelope) -> None:
    """Pulls usage_metadata off the raw AIMessage and hands it to tools.cost. Best-effort: a
    provider that reports no usage simply records nothing.

    Takes either shape: with_structured_output(include_raw=True) returns a dict with the message
    under "raw", while bind_tools() returns the AIMessage itself. This used to return early on
    anything that was not a dict, so every tool-selection call was free as far as the cost table
    was concerned -- and an unmeasured call is indistinguishable, on the spend panel, from one
    that never happened.
    """
    raw = envelope.get("raw") if isinstance(envelope, dict) else envelope
    usage = getattr(raw, "usage_metadata", None) or {}
    try:
        cost_tool.record_llm(
            provider, model,
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
        )
    except Exception as exc:  # cost accounting must never fail an LLM call
        logger.warning("llm.usage_record_failed", provider=provider, error=str(exc))


def _model_name(llm) -> str:
    return getattr(llm, "model", None) or getattr(llm, "model_name", None) or type(llm).__name__


def choose_tool(
    prompt: str, tools: list[type[BaseModel]], system: str | None = None
) -> tuple[str, dict] | None:
    """Asks the first working provider which tool to call, and with what arguments.

    Returns (tool_name, args), or None when the model declined to call anything -- which the
    caller should treat as "I did not understand", not as an error.

    The same provider chain and benching as call_structured(): a provider that cannot pay is
    skipped rather than retried on every message.

    Deliberately one step, not a loop. Each tool here answers from the database and returns its
    own finished text, so there is nothing for a second round to refine -- and an agent that can
    keep calling tools until it decides to stop is an agent that can keep spending until it
    decides to stop. Routing is the model's job; the facts are not.
    """
    messages = ([("system", system)] if system else []) + [("human", prompt)]

    def do_call(provider_name: str, llm):
        response = llm.bind_tools(tools).invoke(messages)
        _record_usage(provider_name, _model_name(llm), response)
        calls = getattr(response, "tool_calls", None) or []
        logger.info(
            "llm.tool_choice", provider=provider_name, model=_model_name(llm),
            tool=calls[0]["name"] if calls else None,
        )
        if not calls:
            return _NO_TOOL
        # Only the first call is honoured. A model asked "what's happening and make me a video"
        # can emit both, and silently doing the second half of a compound request is worse than
        # answering the first and letting the operator ask again.
        return calls[0]["name"], dict(calls[0].get("args") or {})

    chosen = _run_chain(do_call, "tool_choice")
    # _run_chain treats a raised exception as "try the next provider", so "the model chose not to
    # call anything" has to come back as a value, not as None -- None would be indistinguishable
    # from a provider returning nothing and would send the chain round again for no reason.
    return None if chosen is _NO_TOOL else chosen
