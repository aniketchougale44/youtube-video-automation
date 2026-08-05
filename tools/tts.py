"""Text-to-speech client wrapper — stub implementation.
Real logic: ElevenLabs client .generate(text=..., voice=voice_id) if ELEVENLABS_API_KEY is set,
else Azure Cognitive Services SpeechSynthesizer as fallback."""
from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import with_resilience

logger = get_logger("tools.tts")


@with_resilience(provider="tts")
def synthesize(text: str, voice_id: str | None = None, output_path: str = "/tmp/stub_tts.mp3") -> str:
    settings = get_settings()
    provider = "elevenlabs" if settings.elevenlabs_api_key else "azure"
    logger.info("tts.synthesize.stub", provider=provider, chars=len(text), output_path=output_path)
    return output_path
