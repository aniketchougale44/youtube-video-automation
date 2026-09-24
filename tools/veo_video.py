"""Veo 3.1 video generation client wrapper -- text-to-video, image-to-video, and first/last-frame
interpolation.

Unlike tools/image_gen.py and tools/tts.py, there is no free-tier fallback here: Veo is a paid,
metered model with no keyless equivalent (Pollinations/edge-tts have no video analog), so a
missing GOOGLE_API_KEY fails loudly (VeoNotConfiguredError) instead of degrading gracefully --
treat this as an optional, explicitly-invoked capability rather than a step in the default render
path.

Generation genuinely takes minutes, so this blocks synchronously in a poll loop the same way the
official google-genai example does -- callers running this inside a Celery task should expect a
long-running task, not a fire-and-forget one.
"""
import time

from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import with_resilience

logger = get_logger("tools.veo_video")

POLL_INTERVAL_SECONDS = 10


class VeoNotConfiguredError(RuntimeError):
    """Raised when GOOGLE_API_KEY is not set."""


def generate_video(
    prompt: str,
    output_path: str,
    first_frame_path: str | None = None,
    last_frame_path: str | None = None,
    aspect_ratio: str | None = None,
) -> str:
    """Generates a video with Veo 3.1 and saves it to output_path.

    Text-to-video if neither frame is given; image-to-video anchored to first_frame_path if only
    that's given; first/last-frame interpolation (the clip morphs from one frame to the other) if
    both are given -- last_frame_path is a Veo 3.1 constraint that only makes sense alongside a
    starting frame, so it's rejected on its own rather than silently ignored.

    aspect_ratio (e.g. "9:16" for a vertical Reel/Shorts-style clip in
    scripts/produce_reel_clip.py) overrides Veo's landscape default for this call only.

    Blocks until the video is ready (typically 1-6 minutes) or raises on failure -- see module
    docstring for why there's no fallback path.
    """
    settings = get_settings()
    if not settings.google_api_key:
        raise VeoNotConfiguredError("GOOGLE_API_KEY is not set -- Veo has no free-tier fallback")
    if last_frame_path and not first_frame_path:
        raise ValueError("last_frame_path requires first_frame_path to also be set")

    from google import genai

    client = genai.Client(api_key=settings.google_api_key)

    operation = _submit(client, settings.veo_model, prompt, first_frame_path, last_frame_path, aspect_ratio)
    while not operation.done:
        time.sleep(POLL_INTERVAL_SECONDS)
        operation = _poll(client, operation)

    if getattr(operation, "error", None):
        raise RuntimeError(f"veo_video: generation failed: {operation.error}")

    generated_video = operation.response.generated_videos[0]
    _download(client, generated_video.video, output_path)

    logger.info("veo_video.generated", prompt=prompt[:80], output_path=output_path)
    return output_path


@with_resilience(provider="veo_video_submit")
def _submit(client, model: str, prompt: str, first_frame_path: str | None, last_frame_path: str | None, aspect_ratio: str | None):
    from google.genai import types

    kwargs = {}
    if first_frame_path:
        kwargs["image"] = types.Image.from_file(location=first_frame_path)
    config_kwargs = {"duration_seconds": get_settings().veo_video_duration_seconds}
    if last_frame_path:
        config_kwargs["last_frame"] = types.Image.from_file(location=last_frame_path)
    if aspect_ratio:
        config_kwargs["aspect_ratio"] = aspect_ratio
    config = types.GenerateVideosConfig(**config_kwargs)

    return client.models.generate_videos(model=model, prompt=prompt, config=config, **kwargs)


@with_resilience(provider="veo_video_poll")
def _poll(client, operation):
    return client.operations.get(operation)


@with_resilience(provider="veo_video_download")
def _download(client, video_file, output_path: str) -> None:
    client.files.download(file=video_file)
    video_file.save(output_path)
