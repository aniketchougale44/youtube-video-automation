"""AI image-generation client wrapper (thumbnails, illustrations) — stub implementation.
Real logic: OpenAI images.generate(model=settings.openai_image_model, prompt=prompt, size=size),
or Stability/Replicate equivalents selected by settings.image_gen_provider."""
from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import with_resilience

logger = get_logger("tools.image_gen")


@with_resilience(provider="image_gen")
def generate_image(prompt: str, output_path: str = "/tmp/stub_image.png", size: str = "1024x1024") -> str:
    settings = get_settings()
    logger.info("image_gen.generate.stub", provider=settings.image_gen_provider, prompt=prompt[:80], output_path=output_path)
    return output_path
