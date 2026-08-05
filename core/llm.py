"""Shared LLM client: Claude Sonnet primary, GPT-4o-mini fallback. Every agent node that needs an
LLM call goes through `call_structured()` so provider fallback routing lives in exactly one place
instead of being reimplemented per agent."""
from typing import TypeVar

from pydantic import BaseModel

from core.logging import get_logger
from core.settings import get_settings

logger = get_logger("core.llm")

T = TypeVar("T", bound=BaseModel)


class NoLLMProviderConfigured(RuntimeError):
    """Raised when neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set, or both providers failed."""


def _primary_llm():
    settings = get_settings()
    if not settings.anthropic_api_key:
        return None
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model=settings.llm_primary_model, api_key=settings.anthropic_api_key, timeout=60, max_retries=2)


def _fallback_llm():
    settings = get_settings()
    if not settings.openai_api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.llm_fallback_model, api_key=settings.openai_api_key, timeout=60, max_retries=2)


def call_structured(prompt: str, output_model: type[T], system: str | None = None) -> T:
    """Invokes the primary LLM with structured output (`output_model`); falls back to the
    secondary provider on any failure (auth, rate limit, timeout, provider outage). Raises
    NoLLMProviderConfigured if neither provider is configured or both fail."""
    messages = ([("system", system)] if system else []) + [("human", prompt)]

    last_error: Exception | None = None
    for provider_name, factory in (("primary", _primary_llm), ("fallback", _fallback_llm)):
        llm = factory()
        if llm is None:
            continue
        try:
            structured_llm = llm.with_structured_output(output_model)
            result = structured_llm.invoke(messages)
            logger.info("llm.call.success", provider=provider_name, model=_model_name(llm))
            return result
        except Exception as exc:
            logger.error("llm.call.failed", provider=provider_name, model=_model_name(llm), error=str(exc))
            last_error = exc

    raise NoLLMProviderConfigured(
        "no usable LLM provider: set ANTHROPIC_API_KEY and/or OPENAI_API_KEY"
    ) from last_error


def _model_name(llm) -> str:
    return getattr(llm, "model", None) or getattr(llm, "model_name", None) or type(llm).__name__
