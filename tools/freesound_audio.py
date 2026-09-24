"""Background-sound sourcing for short vertical clips (scripts/produce_reel_clip.py only -- the
long-form pipeline has no music, see graph/nodes/audio_render.py's has_music=False).

Freesound.org's search API (free key, freesound.org/apiv2/apply) is searched by mood/vibe keyword
for a CC-licensed clip. When no key is set, or the search/download fails, falls back to a short
locally synthesized ambient pad (stdlib `wave`, no extra dependency) so a reel clip is never
rendered silent -- same graceful-degradation contract as tools.tts (openai -> edge-tts) and
tools.image_gen (openai -> pollinations).
"""
import wave as wave_module

import httpx
import numpy as np

from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import with_resilience

logger = get_logger("tools.freesound_audio")

API_BASE = "https://freesound.org/apiv2"


def get_sound(mood: str, duration_seconds: float, output_path: str) -> str:
    """Downloads a mood-matched CC sound to output_path (its native length -- callers loop/trim to
    the target clip duration when compositing). Falls back to a synthesized ambient pad matching
    duration_seconds on any failure, so callers never have to handle a "no sound" case themselves.
    """
    settings = get_settings()
    if settings.freesound_api_key:
        try:
            url = _search(mood)
            if url:
                _download(url, output_path)
                logger.info("freesound_audio.sourced", mood=mood, output_path=output_path)
                return output_path
        except Exception as exc:
            logger.warning("freesound_audio.search_failed_falling_back", mood=mood, error=str(exc))

    _synthesize_ambient(duration_seconds, output_path)
    logger.info("freesound_audio.synthesized_fallback", mood=mood, output_path=output_path)
    return output_path


@with_resilience(provider="freesound_search")
def _search(mood: str) -> str | None:
    settings = get_settings()
    response = httpx.get(
        f"{API_BASE}/search/text/",
        params={
            "query": mood,
            "token": settings.freesound_api_key,
            "fields": "previews,duration",
            "filter": "duration:[3 TO 30]",
            "sort": "rating_desc",
            "page_size": 1,
        },
        timeout=30,
    )
    response.raise_for_status()
    results = response.json().get("results") or []
    if not results:
        return None
    previews = results[0].get("previews") or {}
    return previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")


@with_resilience(provider="freesound_download")
def _download(url: str, output_path: str) -> None:
    with httpx.stream("GET", url, timeout=60, follow_redirects=True) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.writelines(response.iter_bytes())


def _synthesize_ambient(duration_seconds: float, output_path: str) -> None:
    """Keyless fallback bed: a soft two-tone sine pad with a gentle fade in/out, written as a plain
    PCM .wav via the stdlib `wave` module -- not meant to compete with a real track, just to avoid
    ever shipping a silent clip."""
    sample_rate = 44100
    n_samples = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, n_samples, endpoint=False)
    base_freq = 220.0  # A3 -- a calm, low pad tone
    signal = 0.15 * np.sin(2 * np.pi * base_freq * t) + 0.08 * np.sin(2 * np.pi * base_freq * 1.5 * t)

    fade_len = int(sample_rate * min(1.0, duration_seconds / 4))
    if fade_len > 0:
        fade = np.ones_like(signal)
        fade[:fade_len] = np.linspace(0, 1, fade_len)
        fade[-fade_len:] = np.linspace(1, 0, fade_len)
        signal *= fade

    pcm = (signal * 32767).astype(np.int16)
    with wave_module.open(output_path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())
