"""Text-to-speech client wrapper.

Provider order: settings.tts_provider == "openai" (paid, needs OPENAI_API_KEY) -> edge-tts
(https://github.com/rany2/edge-tts, free, keyless, Microsoft Edge's neural voices) as the
always-available fallback, so voiceover never blocks a run just because no paid TTS key is
configured or its billing is exhausted. ElevenLabs/Azure Speech are reserved for a future
provider, not yet implemented.
"""
import asyncio

from core.logging import get_logger
from core.settings import get_settings
from tools import cost as cost_tool
from tools.resilience import with_resilience

logger = get_logger("tools.tts")


def synthesize(text: str, output_path: str, voice: str | None = None, rate: str = "+0%") -> str:
    """rate is an edge-tts-style relative speed offset (e.g. "-8%") -- a touch slower than the
    default reads as a more natural, unhurried storyteller cadence instead of a rushed TTS voice.
    Ignored on the openai path (its API has no matching per-call parameter)."""
    settings = get_settings()

    if settings.tts_provider == "openai" and settings.openai_api_key:
        try:
            return _openai_call(text, output_path, voice or settings.openai_tts_voice)
        except Exception as exc:
            # e.g. no billing credits, rate-limited, circuit open -- edge-tts is free and keyless,
            # so a broken/unfunded paid key should degrade gracefully, not fail the run.
            logger.warning("tts.openai_failed_falling_back", error=str(exc))

    return _edge_call(text, output_path, voice or settings.edge_tts_voice, rate)


@with_resilience(provider="openai_tts")
def _openai_call(text: str, output_path: str, voice: str) -> str:
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.audio.speech.create(model=settings.openai_tts_model, voice=voice, input=text)
    response.write_to_file(output_path)
    cost_tool.record_tts("openai", len(text))
    logger.info("tts.synthesize", provider="openai", chars=len(text), output_path=output_path, voice=voice)
    return output_path


# max_attempts=5 rather than the default 3: edge-tts is a free, unauthenticated endpoint that
# intermittently answers with NoAudioReceived, and a long script makes one call per beat (17 on a
# real run), so the chance of at least one transient failure per run is high. Three attempts span
# only ~15s of jittered backoff, which a real outage observed here outlasted; five spans ~60s.
# A failure here is no longer catastrophic -- graph.run resumes from the checkpoint rather than
# replaying the pipeline -- but the cheapest fix is still not to fail.
@with_resilience(provider="edge_tts", max_attempts=5)
def _edge_call(text: str, output_path: str, voice: str, rate: str = "+0%") -> str:
    import edge_tts

    async def _run() -> None:
        await edge_tts.Communicate(text, voice, rate=rate).save(output_path)

    asyncio.run(_run())
    cost_tool.record_tts("edge", len(text))  # free, but the character count is still worth having
    logger.info("tts.synthesize", provider="edge", chars=len(text), output_path=output_path, voice=voice, rate=rate)
    return output_path
