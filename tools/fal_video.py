"""Text-to-video directly against fal.ai's hosted API -- a fourth AI_VIDEO generator alongside
tools/hf_video.py (which also happens to route through fal.ai, but indirectly via Hugging Face
Inference Providers and billed against HUGGINGFACE_API_KEY), tools/nvidia_video.py, and
tools/veo_video.py. This module calls fal.ai directly with a native FAL_API_KEY instead, which is
simpler and avoids the HF credit ceiling. See graph/nodes/visual.py's _try_ai_video for call order.

Model: fal-ai/wan-t2v (Wan-2.1 text-to-video) -- pay-per-use, ~$0.40/video at 720p, no monthly
credit ceiling to run into like tools/hf_video.py's HF Inference Providers free tier.
"""
import fal_client

from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import with_resilience

logger = get_logger("tools.fal_video")


class FalVideoNotConfiguredError(RuntimeError):
    """Raised when FAL_API_KEY is not set."""


def generate_video(prompt: str, output_path: str, negative_prompt: str = "", aspect_ratio: str | None = None) -> str:
    """Generates a short clip via fal.ai's hosted API and saves it to output_path.

    aspect_ratio overrides settings.fal_video_aspect_ratio for this call only (e.g. "9:16" for a
    vertical Reel/Shorts-style clip in scripts/produce_reel_clip.py) -- the setting itself stays
    "16:9" since the long-form pipeline's landscape beats are the default caller.

    Raises FalVideoNotConfiguredError if no key is set; any other failure (rate limit, exhausted
    balance, content-policy rejection, timeout) propagates as-is -- callers should catch broadly
    and fall back, the same as tools.hf_video/tools.nvidia_video/tools.veo_video.
    """
    settings = get_settings()
    if not settings.fal_api_key:
        raise FalVideoNotConfiguredError("FAL_API_KEY is not set")

    video_url = _generate(prompt, negative_prompt, aspect_ratio or settings.fal_video_aspect_ratio)
    _download(video_url, output_path)

    logger.info("fal_video.generated", prompt=prompt[:80], output_path=output_path)
    return output_path


@with_resilience(provider="fal_video_generate")
def _generate(prompt: str, negative_prompt: str, aspect_ratio: str) -> str:
    settings = get_settings()
    client = fal_client.SyncClient(key=settings.fal_api_key)
    arguments: dict = {
        "prompt": prompt,
        "resolution": settings.fal_video_resolution,
        "aspect_ratio": aspect_ratio,
    }
    if negative_prompt:
        arguments["negative_prompt"] = negative_prompt

    result = client.subscribe(settings.fal_video_model, arguments=arguments)
    # The SDK returns the endpoint's `data` payload directly, but some endpoints/versions nest it
    # under a "data" key -- handle both rather than trusting one undocumented shape.
    data = result.get("data", result) if isinstance(result, dict) else result
    video = data.get("video") if isinstance(data, dict) else None
    url = video.get("url") if isinstance(video, dict) else None
    if not url:
        raise RuntimeError(f"fal_video: no video.url in response (keys={list(data.keys()) if isinstance(data, dict) else type(data)})")
    return url


@with_resilience(provider="fal_video_download")
def _download(url: str, output_path: str) -> None:
    import httpx

    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as response:
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.writelines(response.iter_bytes())
