"""Text-to-video via Hugging Face Inference Providers -- the pipeline's primary AI_VIDEO
generator (see graph/nodes/visual.py's _try_ai_video). Routed through a hosted backend
(settings.hf_video_provider, default fal.ai) rather than running any model locally: this is what
replaced both the offline Wan2.1 GPU path (too slow/heavy for this machine) and Veo (paid-tier
only, no free tier at all -- see tools/veo_video.py) as the day-to-day generator.

Model choice: Wan-AI/Wan2.2-TI2V-5B -- Apache-2.0 (unrestricted commercial use, unlike e.g.
LTX-Video's revenue-capped community license), confirmed pure text-to-video (no reference image
required), and Hugging Face's own documented example model for the text-to-video task.

Not free, just cheap: a free Hugging Face account gets a small monthly Inference Providers credit
(~$0.10 as of writing) that doesn't cover even one clip (~$0.15 on fal.ai) -- sustained use needs
occasional prepaid top-ups at huggingface.co/settings/billing. There is no metered/paid-tier gate
like Veo's though: any funded HF account (free tier + top-up) works, no separate Cloud Billing
account or project-level enablement required.
"""
from huggingface_hub import InferenceClient

from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import with_resilience

logger = get_logger("tools.hf_video")


class HFVideoNotConfiguredError(RuntimeError):
    """Raised when HUGGINGFACE_API_KEY is not set."""


def generate_video(prompt: str, output_path: str, negative_prompt: str = "") -> str:
    """Generates a short clip via Hugging Face Inference Providers and saves it to output_path.

    Raises HFVideoNotConfiguredError if no key is set; any other failure (provider outage, model
    cold-start timeout, exhausted credits) propagates as-is -- callers should treat this the same
    as tools.veo_video.generate_video: catch broadly and fall back, don't let it crash a run.
    """
    settings = get_settings()
    if not settings.huggingface_api_key:
        raise HFVideoNotConfiguredError("HUGGINGFACE_API_KEY is not set")

    video_bytes = _generate(prompt, negative_prompt)
    with open(output_path, "wb") as f:
        f.write(video_bytes)

    logger.info("hf_video.generated", prompt=prompt[:80], output_path=output_path)
    return output_path


@with_resilience(provider="hf_video_generate")
def _generate(prompt: str, negative_prompt: str) -> bytes:
    settings = get_settings()
    client = InferenceClient(provider=settings.hf_video_provider, api_key=settings.huggingface_api_key)
    return client.text_to_video(
        prompt,
        model=settings.hf_video_model,
        negative_prompt=[negative_prompt] if negative_prompt else None,
    )
