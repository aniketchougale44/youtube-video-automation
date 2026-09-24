"""Text-to-video via a self-hosted Colab endpoint (see colab/wan_video_colab.ipynb). Used by
scripts/produce_reel_clip.py and -- when settings.ai_video_provider is "colab" or "auto" -- by the
main pipeline's AI_VIDEO beats (graph/nodes/visual.py::_try_ai_video).

Model: Wan-AI/Wan2.1-T2V-1.3B (Apache-2.0), the smallest real video-diffusion transformer that
still gives genuine motion. The notebook keeps its UMT5-XXL text encoder on CPU
(enable_model_cpu_offload) so peak VRAM stays ~8-9GB and it runs on a free-tier T4 -- earlier
attempts (ModelScope-1.7b too soft; CogVideoX-2B/5B's ~9GB T5-XXL encoder OOM'd or ran 30-45
min/clip; SD1.5/DreamShaper+AnimateDiff had flat retrofitted motion) are in git history and the
notebook's own markdown. Native 832x480 @ 16fps, ~10-20 min/clip on a T4.

This has no stable/managed endpoint: colab_video_url must be a live Gradio public URL from a
currently-running Colab session (the notebook prints it on its last cell) -- it changes every
notebook restart, and the session dies after Colab's idle/max-runtime limits. So a run with
ai_video_provider="colab" and no live URL simply falls the AI_VIDEO beats back to the mascot;
nothing crashes.

Talks to the notebook's gr.Interface via gradio_client rather than hand-rolled HTTP, since Gradio
already exposes a typed predict endpoint and handles the file download for the video output.
"""
import shutil
import time

from gradio_client import Client

from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import with_resilience

logger = get_logger("tools.colab_video")


class ColabVideoNotConfiguredError(RuntimeError):
    """Raised when COLAB_VIDEO_URL is not set."""


def generate_video(prompt: str, output_path: str, negative_prompt: str = "") -> str:
    """Generates a short clip via the Colab-hosted ModelScope endpoint and saves it to output_path.

    Raises ColabVideoNotConfiguredError if no URL is set; any other failure (notebook not running,
    stale/expired URL, model still loading, OOM on the free GPU) propagates as-is -- same contract
    as tools.fal_video/tools.hf_video.
    """
    settings = get_settings()
    if not settings.colab_video_url:
        raise ColabVideoNotConfiguredError(
            "COLAB_VIDEO_URL is not set -- run colab/wan_video_colab.ipynb in Google Colab and "
            "paste the Gradio public URL it prints into .env"
        )

    logger.info(
        "colab_video.request",
        prompt=prompt[:120],
        negative_prompt=negative_prompt[:80],
        url=settings.colab_video_url,
        num_frames=settings.colab_video_num_frames,
        height=settings.colab_video_height,
        width=settings.colab_video_width,
        guidance_scale=settings.colab_video_guidance_scale,
    )
    started = time.monotonic()
    try:
        result_path = _generate(prompt, negative_prompt)
    except Exception as exc:
        logger.error(
            "colab_video.failed",
            error=str(exc),
            error_type=type(exc).__name__,
            elapsed_seconds=round(time.monotonic() - started, 1),
        )
        raise

    shutil.copyfile(result_path, output_path)
    logger.info(
        "colab_video.generated",
        prompt=prompt[:80],
        output_path=output_path,
        elapsed_seconds=round(time.monotonic() - started, 1),
    )
    return output_path


# gr.Interface's endpoint name is version-dependent: Gradio <=5.x hardcodes it to "predict",
# while Gradio 6.x names it after the wrapped function ("generate" here). The notebook pip-installs
# gradio unpinned, so whichever Colab resolves that day decides which one is live -- and calling
# the wrong one fails with "Cannot find a function with api_name ...", which reads exactly like a
# dead endpoint and silently drops every AI_VIDEO beat back to the mascot. Try both rather than
# betting on a version. The notebook also pins api_name="generate" explicitly, so "/generate" wins
# on the first try for any session started from the current notebook.
_API_NAME_CANDIDATES = ("/generate", "/predict")


def _is_unknown_endpoint(exc: Exception) -> bool:
    return "cannot find a function with api_name" in str(exc).lower()


def _predict(client: Client, args: tuple):
    """client.predict(*args) against the first api_name the endpoint actually exposes. Only an
    unknown-endpoint error advances to the next candidate -- a real generation failure (OOM, bad
    args) is re-raised immediately rather than burning another multi-minute attempt on it."""
    last_exc: Exception | None = None
    for api_name in _API_NAME_CANDIDATES:
        try:
            return client.predict(*args, api_name=api_name)
        except Exception as exc:
            if not _is_unknown_endpoint(exc):
                raise
            logger.warning("colab_video.api_name_not_found", api_name=api_name, error=str(exc))
            last_exc = exc
    raise RuntimeError(
        f"colab_video: the Colab endpoint exposes none of {_API_NAME_CANDIDATES} -- "
        f"is colab/wan_video_colab.ipynb the running notebook? (last error: {last_exc})"
    )


@with_resilience(provider="colab_video_generate", max_attempts=1)
# max_attempts=1 (no retry): unlike a paid API's transient rate-limit/network blip, a failed
# generation on the free Colab endpoint is usually a real error (OOM, bad args) that retrying
# won't fix -- and repeated back-to-back attempts on the same session have been observed to pile
# up GPU memory across calls and eventually crash the whole Colab kernel outright.
def _generate(prompt: str, negative_prompt: str) -> str:
    settings = get_settings()

    logger.info("colab_video.connect", url=settings.colab_video_url)
    connect_started = time.monotonic()
    # gradio_client's default httpx timeout is 5s -- gradio.live's first (cold) TLS handshake
    # routinely takes longer than that, which surfaced as ConnectTimeout on the very first call
    # of a session. Generous timeouts here; the model itself takes minutes anyway.
    client = Client(settings.colab_video_url, httpx_kwargs={"timeout": 60.0})
    logger.info("colab_video.connected", elapsed_seconds=round(time.monotonic() - connect_started, 1))

    # Positional, matching colab/wan_video_colab.ipynb's gr.Interface `inputs` order exactly --
    # gradio_client.predict() maps by position, not by the wrapped function's parameter names.
    logger.info("colab_video.predict.start")
    predict_started = time.monotonic()
    result = _predict(
        client,
        (
            prompt,
            negative_prompt,
            settings.colab_video_num_frames,
            settings.colab_video_height,
            settings.colab_video_width,
            settings.colab_video_guidance_scale,
        ),
    )
    logger.info(
        "colab_video.predict.done",
        elapsed_seconds=round(time.monotonic() - predict_started, 1),
        result_type=type(result).__name__,
    )

    # A gr.Video() output normally comes back as a local temp filepath (gradio_client already
    # downloaded it), but newer Gradio versions sometimes wrap it in a dict -- handle both.
    if isinstance(result, dict):
        result = result.get("video") or result.get("path") or result.get("name")
    if not result:
        raise RuntimeError("colab_video: no output file returned by the Colab endpoint")
    logger.info("colab_video.output_file", path=str(result))
    return result
