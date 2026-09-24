"""Text-to-video via NVIDIA's hosted Cosmos3 Nano Preview API (build.nvidia.com) -- a third
AI_VIDEO generator alongside tools/hf_video.py (primary) and tools/veo_video.py (secondary/
legacy). See graph/nodes/visual.py's _try_ai_video for call order.

Model: nvidia/cosmos3-nano -- a world foundation model that generates physics-aware video from a
text prompt. Unlike the rest of the Cosmos family (Predict/Transfer, and Cosmos3-Super), which
build.nvidia.com only distributes as self-hosted NIM containers requiring a GPU, Cosmos3 Nano has
an actual hosted "Preview API" -- but confirmed by capturing the live browser network request
(DevTools) on build.nvidia.com/nvidia/cosmos3-nano's Experience tab, that hosted preview is only
reachable at https://buildapi.ngc.nvidia.com/v2/predict/models/{namespace}/cosmos3-nano using a
short-lived session JWT tied to the logged-in browser session (confirmed: a plain NVIDIA_API_KEY
personal key sent as Bearer gets a 401 "Jwt is not in the form of Header.Payload.Signature..."),
and the model card exposes no separate curl/Bearer-key example anywhere on the page. There is
currently no known way to call this model with just NVIDIA_API_KEY from server-side/pipeline code
-- this module is kept in place (dormant in practice) in case NVIDIA later exposes a real
Bearer-key endpoint for it; until then _generate will reliably fail and asset_visual_node will
fall through to Veo/mascot, same as any other sourcing miss.

Request/response schema per NVIDIA's NIM 3.0.0 Cosmos WFM API reference (docs.nvidia.com/nim/
cosmos/3.0.0/api-reference.html), also confirmed via the captured request payload: prompt/
negative_prompt/seed/guidance_scale/steps/resolution/num_output_frames/fps in, `{"b64_video":
"<base64-encoded mp4>"}` out. DEFAULT_INVOKE_URL below is the best-known guess (NVIDIA's standard
genai/{org}/{model} pattern) but is UNVERIFIED to work with a personal API key -- override via
settings.nvidia_video_invoke_url if/when a working Bearer-key endpoint is found.
"""
import base64
import time

import httpx

from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import with_resilience

logger = get_logger("tools.nvidia_video")

DEFAULT_INVOKE_URL = "https://ai.api.nvidia.com/v1/genai/nvidia/cosmos3-nano"
STATUS_URL = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/{request_id}"
POLL_INTERVAL_SECONDS = 5
POLL_TIMEOUT_SECONDS = 600


class NvidiaVideoNotConfiguredError(RuntimeError):
    """Raised when NVIDIA_API_KEY is not set."""


def generate_video(prompt: str, output_path: str, negative_prompt: str = "") -> str:
    """Generates a short clip via the hosted Cosmos3 Nano Preview API and saves it to output_path.

    Raises NvidiaVideoNotConfiguredError if no key is set; any other failure (rate limit, quota,
    timeout, outage) propagates as-is -- callers should catch broadly and fall back, the same as
    tools.hf_video and tools.veo_video."""
    settings = get_settings()
    if not settings.nvidia_api_key:
        raise NvidiaVideoNotConfiguredError("NVIDIA_API_KEY is not set")

    video_bytes = _generate(prompt, negative_prompt)
    with open(output_path, "wb") as f:
        f.write(video_bytes)

    logger.info("nvidia_video.generated", prompt=prompt[:80], output_path=output_path)
    return output_path


@with_resilience(provider="nvidia_video_generate")
def _generate(prompt: str, negative_prompt: str) -> bytes:
    settings = get_settings()
    headers = {
        "Authorization": f"Bearer {settings.nvidia_api_key}",
        "Accept": "application/json",
    }
    payload: dict = {
        "prompt": prompt,
        "resolution": settings.nvidia_video_resolution,
        "num_output_frames": settings.nvidia_video_num_frames,
        "fps": settings.nvidia_video_fps,
    }
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    invoke_url = settings.nvidia_video_invoke_url or DEFAULT_INVOKE_URL

    with httpx.Client(timeout=60) as client:
        response = client.post(invoke_url, headers=headers, json=payload)

        elapsed = 0.0
        while response.status_code == 202:
            request_id = response.headers.get("NVCF-REQID")
            if not request_id:
                response.raise_for_status()
                break
            if elapsed >= POLL_TIMEOUT_SECONDS:
                raise TimeoutError(f"nvidia_video: generation timed out after {elapsed:.0f}s")
            time.sleep(POLL_INTERVAL_SECONDS)
            elapsed += POLL_INTERVAL_SECONDS
            response = client.get(STATUS_URL.format(request_id=request_id), headers=headers)

        response.raise_for_status()

    return _extract_video_b64(response.json())


def _extract_video_b64(result: dict) -> bytes:
    video_b64 = result.get("b64_video")
    if not video_b64:
        raise RuntimeError(f"nvidia_video: no b64_video in response (keys={list(result.keys())})")
    return base64.b64decode(video_b64)
