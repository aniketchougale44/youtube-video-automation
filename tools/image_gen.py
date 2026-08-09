"""AI image-generation client wrapper (thumbnails, illustrations). Wired to OpenAI images.generate
(settings.image_gen_provider == "openai"); other providers (stability/replicate) are out of scope
for this pass and fail loudly rather than silently no-op."""
import base64

import httpx

from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import with_resilience

logger = get_logger("tools.image_gen")


class ImageGenNotConfiguredError(RuntimeError):
    """Raised when the configured provider's API key isn't set."""


def generate_image(prompt: str, output_path: str, size: str = "1024x1024") -> str:
    settings = get_settings()
    if settings.image_gen_provider != "openai":
        raise NotImplementedError(f"image_gen_provider={settings.image_gen_provider!r} is not implemented")
    if not settings.openai_api_key:
        raise ImageGenNotConfiguredError("OPENAI_API_KEY is not set")
    return _generate_call(prompt, output_path, size)


@with_resilience(provider="image_gen")
def _generate_call(prompt: str, output_path: str, size: str) -> str:
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

    logger.info("image_gen.generate", provider=settings.image_gen_provider, prompt=prompt[:80], output_path=output_path)
    return output_path
