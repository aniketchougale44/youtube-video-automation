"""Text-to-speech client wrapper. Wired to OpenAI TTS (audio.speech) — no ElevenLabs/Azure key
available; settings.openai_tts_model / settings.openai_tts_voice pick the model/voice."""
from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import with_resilience

logger = get_logger("tools.tts")


class TTSNotConfiguredError(RuntimeError):
    """Raised when OPENAI_API_KEY isn't set."""


def synthesize(text: str, output_path: str, voice: str | None = None) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        raise TTSNotConfiguredError("OPENAI_API_KEY is not set")
    return _synthesize_call(text, output_path, voice or settings.openai_tts_voice)


@with_resilience(provider="openai_tts")
def _synthesize_call(text: str, output_path: str, voice: str) -> str:
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.audio.speech.create(model=settings.openai_tts_model, voice=voice, input=text)
    response.write_to_file(output_path)
    logger.info("tts.synthesize", chars=len(text), output_path=output_path, voice=voice)
    return output_path
