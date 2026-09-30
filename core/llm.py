"""Shared LLM client: tries each configured provider in order and falls back on failure. Every
agent node that needs an LLM call goes through `call_structured()` so provider fallback routing
lives in exactly one place instead of being reimplemented per agent.

Provider order: Anthropic (paid) -> OpenAI (paid) -> Groq (free tier, no card) -> Gemini (free
tier, no card). A deployment with none of the paid keys set still runs end-to-end on Groq and/or
Gemini alone — each factory below just returns None when its key isn't set, so the chain skips
straight past it.
"""
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


def _gemini_llm():
    """Free tier, no card required: https://aistudio.google.com/apikey"""
    settings = get_settings()
    if not settings.google_api_key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=settings.llm_gemini_model, api_key=settings.google_api_key, timeout=60, max_retries=2)


_PROVIDERS = (
    ("anthropic", _anthropic_llm),
    ("openai", _openai_llm),
    ("groq", _groq_llm),
    ("gemini", _gemini_llm),
)


def call_structured(prompt: str, output_model: type[T], system: str | None = None) -> T:
    """Invokes each configured provider in order with structured output (`output_model`) until
    one succeeds. Raises NoLLMProviderConfigured if none are configured or all fail."""
    messages = ([("system", system)] if system else []) + [("human", prompt)]

    last_error: Exception | None = None
    for provider_name, factory in _PROVIDERS:
        llm = factory()
        if llm is None:
            continue
        try:
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
        except Exception as exc:
            logger.error("llm.call.failed", provider=provider_name, model=_model_name(llm), error=str(exc))
            last_error = exc

    raise NoLLMProviderConfigured(
        "no usable LLM provider: set one of ANTHROPIC_API_KEY, OPENAI_API_KEY, GROQ_API_KEY "
        "(free: console.groq.com/keys), GOOGLE_API_KEY (free: aistudio.google.com/apikey)"
    ) from last_error


def _record_usage(provider: str, model: str, envelope) -> None:
    """Pulls usage_metadata off the raw AIMessage and hands it to tools.cost. Best-effort: a
    provider that reports no usage simply records nothing."""
    if not isinstance(envelope, dict):
        return
    raw = envelope.get("raw")
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
