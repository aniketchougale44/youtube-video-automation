"""Tests for the free-tier provider fallback logic added to core/llm.py, tools/image_gen.py, and
tools/tts.py. No real network calls: each provider's actual API call is monkeypatched, these tests
only assert on *which* provider gets picked and that failures degrade to the free option instead
of raising.

Uses a plain SimpleNamespace stand-in for Settings rather than constructing the real pydantic
Settings class: the `langsmith` pytest plugin loads this repo's real .env into the process
environment as a side effect of collection, and since pydantic-settings reads os.environ as a
source independent of (and secondary only to explicit kwargs -- but above the dotenv file), that
leaks the developer's real API keys into what's supposed to be an isolated per-test config. A
SimpleNamespace has no relationship to os.environ at all, so it can't leak.

Both test_image_gen_falls_back_to_pollinations_when_openai_fails and the embed_text tests import
the real function by name at collection time (before any per-test fixture runs) rather than going
through the module attribute: conftest.py's autouse mock_external_apis fixture replaces
tools.image_gen.generate_image and tools.embeddings.embed_text wholesale with fakes for every
other test in the suite (so pipeline tests never hit the real network), which would otherwise
shadow the exact fallback logic these tests need to exercise."""
import types

from tools.embeddings import embed_text as _real_embed_text
from tools.image_gen import generate_image as _real_generate_image


def _settings(**overrides) -> types.SimpleNamespace:
    defaults = {
        "anthropic_api_key": "", "llm_primary_model": "claude-sonnet-4-5",
        "openai_api_key": "", "llm_fallback_model": "gpt-4o-mini",
        "groq_api_key": "", "llm_groq_model": "llama-3.3-70b-versatile",
        "deepseek_api_key": "", "llm_deepseek_model": "deepseek-chat",
        "google_api_key": "", "llm_gemini_model": "gemini-2.0-flash",
        "gemini_embedding_model": "models/gemini-embedding-001",
        "image_gen_provider": "pollinations", "openai_image_model": "gpt-image-1",
        "tts_provider": "edge", "openai_tts_model": "tts-1", "openai_tts_voice": "alloy",
        "edge_tts_voice": "en-US-AndrewNeural",
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def test_llm_provider_chain_skips_unconfigured_providers(monkeypatch):
    from core import llm

    monkeypatch.setattr("core.llm.get_settings", lambda: _settings(groq_api_key="fake-groq-key"))

    assert llm._anthropic_llm() is None
    assert llm._openai_llm() is None
    assert llm._gemini_llm() is None
    assert llm._groq_llm() is not None


def test_llm_call_structured_falls_through_to_first_configured_provider(monkeypatch):
    from pydantic import BaseModel

    from core import llm

    class _Out(BaseModel):
        value: str

    class _FakeStructuredLLM:
        def invoke(self, messages):
            return _Out(value="from gemini")

    class _FakeGeminiLLM:
        model = "gemini-2.0-flash"

        def with_structured_output(self, output_model):
            return _FakeStructuredLLM()

    # _PROVIDERS binds each factory function directly (not a name looked up at call time), so
    # patching core.llm._gemini_llm alone wouldn't affect the copy already captured in this tuple
    # -- replace the whole chain instead of trying to patch through it.
    monkeypatch.setattr("core.llm._PROVIDERS", (("gemini", lambda: _FakeGeminiLLM()),))

    result = llm.call_structured("prompt", _Out)
    assert result.value == "from gemini"


def test_llm_call_structured_raises_when_nothing_configured(monkeypatch):
    from pydantic import BaseModel

    from core import llm

    class _Out(BaseModel):
        value: str

    monkeypatch.setattr("core.llm.get_settings", lambda: _settings())
    try:
        llm.call_structured("prompt", _Out)
        raise AssertionError("expected NoLLMProviderConfigured")
    except llm.NoLLMProviderConfigured as exc:
        assert "GROQ_API_KEY" in str(exc)


def test_image_gen_uses_pollinations_when_no_openai_key(monkeypatch):
    monkeypatch.setattr("tools.image_gen.get_settings", lambda: _settings(image_gen_provider="pollinations"))

    calls = {}

    def _pollinations(prompt, output_path, size):
        calls["used"] = True
        return output_path

    monkeypatch.setattr("tools.image_gen._pollinations_call", _pollinations)

    result = _real_generate_image("a cat", "out.png")
    assert result == "out.png"
    assert calls.get("used") is True


def test_image_gen_falls_back_to_pollinations_when_openai_fails(monkeypatch):
    monkeypatch.setattr(
        "tools.image_gen.get_settings",
        lambda: _settings(image_gen_provider="openai", openai_api_key="fake-key"),
    )

    def _fail(*args, **kwargs):
        raise RuntimeError("429 insufficient_quota")

    calls = {}

    def _pollinations(prompt, output_path, size):
        calls["used"] = True
        return output_path

    monkeypatch.setattr("tools.image_gen._openai_call", _fail)
    monkeypatch.setattr("tools.image_gen._pollinations_call", _pollinations)

    result = _real_generate_image("a cat", "out.png")
    assert result == "out.png"
    assert calls.get("used") is True


def test_tts_uses_edge_when_no_openai_key(monkeypatch):
    from tools import tts

    monkeypatch.setattr("tools.tts.get_settings", lambda: _settings(tts_provider="edge"))
    monkeypatch.setattr("tools.tts._edge_call", lambda text, output_path, voice, rate="+0%": output_path)

    result = tts.synthesize("hello", "out.mp3")
    assert result == "out.mp3"


def test_tts_falls_back_to_edge_when_openai_fails(monkeypatch):
    from tools import tts

    monkeypatch.setattr(
        "tools.tts.get_settings", lambda: _settings(tts_provider="openai", openai_api_key="fake-key")
    )

    def _fail(*args, **kwargs):
        raise RuntimeError("429 insufficient_quota")

    calls = {}

    def _edge(text, output_path, voice, rate="+0%"):
        calls["used"] = True
        return output_path

    monkeypatch.setattr("tools.tts._openai_call", _fail)
    monkeypatch.setattr("tools.tts._edge_call", _edge)

    result = tts.synthesize("hello", "out.mp3")
    assert result == "out.mp3"
    assert calls.get("used") is True


def test_embed_text_uses_gemini_when_no_openai_key(monkeypatch):
    monkeypatch.setattr("tools.embeddings.get_settings", lambda: _settings(google_api_key="fake-key"))
    monkeypatch.setattr("tools.embeddings._gemini_embed_call", lambda text: [0.1, 0.2, 0.3])

    result = _real_embed_text("some script text")
    assert result == [0.1, 0.2, 0.3]


def test_embed_text_falls_back_to_gemini_when_openai_fails(monkeypatch):
    monkeypatch.setattr(
        "tools.embeddings.get_settings",
        lambda: _settings(openai_api_key="fake-key", google_api_key="fake-key"),
    )

    def _fail(text):
        raise RuntimeError("429 insufficient_quota")

    calls = {}

    def _gemini(text):
        calls["used"] = True
        return [0.4, 0.5, 0.6]

    monkeypatch.setattr("tools.embeddings._openai_embed_call", _fail)
    monkeypatch.setattr("tools.embeddings._gemini_embed_call", _gemini)

    result = _real_embed_text("some script text")
    assert result == [0.4, 0.5, 0.6]
    assert calls.get("used") is True


def test_embed_text_raises_when_nothing_configured(monkeypatch):
    from tools.embeddings import EmbeddingsNotConfiguredError

    monkeypatch.setattr("tools.embeddings.get_settings", lambda: _settings())
    try:
        _real_embed_text("some script text")
        raise AssertionError("expected EmbeddingsNotConfiguredError")
    except EmbeddingsNotConfiguredError:
        pass


def test_a_provider_with_no_balance_is_benched_instead_of_retried(monkeypatch):
    """A key that is present but unpaid fails identically on every call.

    Left in the chain it costs a round trip per call -- fourteen stages a run, plus the chat an
    operator is waiting on -- to learn what the previous call already proved. Terminal billing and
    auth failures bench the provider; a timeout or a 500 would not, because those are worth
    retrying.
    """
    from pydantic import BaseModel

    from core import llm

    class _Out(BaseModel):
        value: str

    calls = {"deepseek": 0, "groq": 0}

    class _Dead:
        def with_structured_output(self, *_a, **_k):
            calls["deepseek"] += 1
            raise RuntimeError("Error code: 402 - Insufficient Balance")

    class _Live:
        def with_structured_output(self, *_a, **_k):
            calls["groq"] += 1
            return self

        def invoke(self, _messages):
            return {"parsed": _Out(value="ok"), "raw": None}

    monkeypatch.setattr("core.llm.get_settings", lambda: _settings())
    monkeypatch.setattr(llm, "_PROVIDERS", (("deepseek", lambda: _Dead()), ("groq", lambda: _Live())))
    monkeypatch.setattr(llm, "_unusable_until", {})

    assert llm.call_structured("one", _Out).value == "ok"
    assert llm.call_structured("two", _Out).value == "ok"

    assert calls["deepseek"] == 1, "the unpaid provider should be tried once, then benched"
    assert calls["groq"] == 2, "every call still gets served by the working provider"
