"""AI image-generation client wrapper (thumbnails, illustrations).

Provider order: settings.image_gen_provider ("openai") if OPENAI_API_KEY is set -> Pollinations.ai
(https://pollinations.ai, free, keyless, no signup) as the always-available fallback, so a run
never has to skip visuals/thumbnails just because no paid image-gen key is configured. stability/
replicate aren't wired (neither has a free tier worth carrying alongside Pollinations); if either
is set as the provider we log once and use the Pollinations fallback rather than crashing an
otherwise-fine render on a stale env var.
"""
import base64
from urllib.parse import quote

import httpx

from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import with_resilience

logger = get_logger("tools.image_gen")


def generate_image(prompt: str, output_path: str, size: str = "1024x1024") -> str:
    settings = get_settings()

    if settings.image_gen_provider in ("stability", "replicate"):
        logger.warning(
            "image_gen.provider_not_wired_using_pollinations", provider=settings.image_gen_provider
        )
        return _pollinations_call(prompt, output_path, size)

    if settings.image_gen_provider == "openai" and settings.openai_api_key:
        try:
            return _openai_call(prompt, output_path, size)
        except Exception as exc:
            # e.g. no billing credits, rate-limited, circuit open -- Pollinations is free and
            # keyless, so a broken/unfunded paid key should degrade gracefully, not fail the run.
            logger.warning("image_gen.openai_failed_falling_back", error=str(exc))

    return _pollinations_call(prompt, output_path, size)


@with_resilience(provider="image_gen_openai")
def _openai_call(prompt: str, output_path: str, size: str) -> str:
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.images.generate(model=settings.openai_image_model, prompt=prompt, size=size, n=1)
    image = response.data[0]

    if getattr(image, "b64_json", None):
        raw = base64.b64decode(image.b64_json)
    elif getattr(image, "url", None):
        raw = httpx.get(image.url, timeout=60).content
    else:
        raise RuntimeError("image_gen: response had neither b64_json nor url")

    with open(output_path, "wb") as f:
        f.write(raw)

    logger.info("image_gen.generate", provider="openai", prompt=prompt[:80], output_path=output_path)
    return output_path


@with_resilience(provider="image_gen_pollinations")
def _pollinations_call(prompt: str, output_path: str, size: str) -> str:
    width, height = _parse_size(size)
    response = httpx.get(
        f"https://image.pollinations.ai/prompt/{quote(prompt)}",
        params={"width": width, "height": height, "nologo": "true"},
        timeout=60,
    )
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    logger.info("image_gen.generate", provider="pollinations", prompt=prompt[:80], output_path=output_path)
    return output_path


def _parse_size(size: str) -> tuple[int, int]:
    try:
        width_str, height_str = size.lower().split("x")
        return int(width_str), int(height_str)
    except (ValueError, AttributeError):
        return 1024, 1024
