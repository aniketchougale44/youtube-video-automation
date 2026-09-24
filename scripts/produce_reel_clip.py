"""One-off standalone script: produces a single short, vertical (9:16), Reel/Shorts-style AI video
clip via tools.colab_video (Colab-hosted CogVideoX-5B -- free, the only generator this script
uses), with a mood-matched sound bed via tools.freesound_audio mixed underneath, then upscales to
4K/60fps for the final deliverable -- modeled on the reels the user dropped into
media/reference/reels/ (all 720x1280, 24-30fps, 8-14s, first-person POV drifting through
hyper-saturated bioluminescent fantasy landscapes -- glowing forest bike path, galaxy/moonlit
beach, glowing ocean at night -- no dialogue, ambient/dreamy vibe carrying the whole thing).

Deliberately bypasses the full LangGraph pipeline (graph/builder.py) and its long-form,
landscape, narrated-beat assembly (graph/nodes/audio_render.py) -- same rationale as
scripts/produce_ai_video_story.py, but for a short vertical clip instead of a multi-beat story.

Note on "4K/60fps": CogVideoX-5B outputs at its own native resolution/frame rate (well below 4K,
well below 60fps -- there's no video-gen model that outputs native 4K60 today). _upscale_to_4k60
below gets there via ffmpeg: motion interpolation (minterpolate) for the frame-rate bump, then a
lanczos spatial upscale for resolution -- real 4K60 *pixels*, not re-generated native detail.

Usage:
    python scripts/produce_reel_clip.py --prompt "..." --mood "upbeat energetic"
    python scripts/produce_reel_clip.py --prompt "..." --upload   # also upload as an unlisted
                                                                     # YouTube Short
"""
import argparse
import os
import re
import subprocess
import time

from moviepy import AudioFileClip, VideoFileClip, afx, vfx

from core.logging import get_logger
from tools import colab_video as colab_video_tool
from tools import freesound_audio as freesound_audio_tool
from tools import quota as quota_tool
from tools import script_progress
from tools import youtube as youtube_tool

logger = get_logger("scripts.produce_reel_clip")

REFERENCE_DIR = "media/reference/reels"
OUTPUT_ROOT = "media/reels"

UPSCALE_WIDTH, UPSCALE_HEIGHT = 2160, 3840  # 9:16 "4K" (matches 3840x2160 UHD pixel count, rotated)
UPSCALE_FPS = 60
TARGET_DURATION_SECONDS = 10.0  # Colab's native clip length is short (as little as ~3s) --
# normalized here (loop if short, trim if long) so the final output always actually runs the
# requested ~10s.

DEFAULT_PROMPT = (
    "First-person POV gliding slowly forward through a magical bioluminescent fantasy forest at "
    "night, glowing pink and turquoise cherry blossom trees overhead, thousands of sparkling "
    "fireflies and warm floating lanterns drifting through the air, a glowing golden path underfoot "
    "reflecting in shallow water, hyper-saturated dreamlike color grading, cinematic immersive "
    "camera motion, ultra detailed fantasy landscape"
)
DEFAULT_MOOD = "dreamy ethereal ambient cinematic"
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, low detail, distorted, deformed, watermark, text, logo, oversaturated "
    "noise, static image, flat lighting, ugly, jpeg artifacts"
)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:40] or "reel"


def _list_reference_clips() -> list[str]:
    if not os.path.isdir(REFERENCE_DIR):
        return []
    return sorted(
        os.path.join(REFERENCE_DIR, name)
        for name in os.listdir(REFERENCE_DIR)
        if name.lower().endswith((".mp4", ".mov", ".webm"))
    )


def _upscale_to_4k60(input_path: str, output_path: str) -> str:
    """ffmpeg post-process: motion-interpolate up to UPSCALE_FPS, then lanczos-scale up to
    UPSCALE_WIDTH x UPSCALE_HEIGHT. Interpolating before scaling (at the smaller native resolution)
    is far cheaper than the reverse -- minterpolate's optical-flow cost scales with pixel count.

    Scale-to-fit + pad (rather than a plain scale=W:H) so a source clip whose aspect ratio isn't
    exactly 9:16 -- Colab's generator has no aspect-ratio control -- gets letterboxed instead of
    squished/stretched."""
    filter_chain = (
        f"minterpolate=fps={UPSCALE_FPS}:mi_mode=mci:mc_mode=aobmc,"
        f"scale={UPSCALE_WIDTH}:{UPSCALE_HEIGHT}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={UPSCALE_WIDTH}:{UPSCALE_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-vf", filter_chain,
            "-c:v", "libx264", "-preset", "medium", "-crf", "16",
            "-c:a", "aac", "-b:a", "192k",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    return output_path


def produce(prompt: str, mood: str, out_dir: str, negative_prompt: str = "", progress: script_progress.ScriptRun | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    run_started = time.monotonic()
    logger.info("reel.produce.start", out_dir=out_dir, prompt=prompt[:120], mood=mood)

    if progress:
        progress.step("generating clip (colab)")
    logger.info("reel.stage.start", stage="generate_clip")
    stage_started = time.monotonic()
    clip_path = os.path.join(out_dir, "clip.mp4")
    # tools/colab_video.py -- requires COLAB_VIDEO_URL to point at a currently-running
    # colab/wan_video_colab.ipynb session; raises ColabVideoNotConfiguredError with setup
    # instructions otherwise. An empty negative_prompt here would override the notebook's own
    # default (the Gradio interface's default value only applies to its web UI, not API calls --
    # every arg must be sent explicitly), so a real one matters here.
    # No aspect-ratio control, hence _upscale_to_4k60's letterbox/pad handling below rather than a
    # plain stretch-to-fit scale.
    colab_video_tool.generate_video(prompt, output_path=clip_path, negative_prompt=negative_prompt)
    logger.info(
        "reel.clip_ready",
        path=clip_path,
        size_bytes=os.path.getsize(clip_path) if os.path.exists(clip_path) else None,
        stage_seconds=round(time.monotonic() - stage_started, 1),
    )

    if progress:
        progress.step("normalizing duration")
    logger.info("reel.stage.start", stage="normalize_duration")
    stage_started = time.monotonic()
    raw = VideoFileClip(clip_path)
    if raw.duration >= TARGET_DURATION_SECONDS:
        video = raw.subclipped(0, TARGET_DURATION_SECONDS)
    else:
        # Loop (not freeze-frame hold) a short native clip up to the target length -- same
        # technique graph/nodes/audio_render.py's _clip_for_beat uses for AI_VIDEO beats shorter
        # than their voiceover-driven duration.
        video = raw.with_effects([vfx.Loop(duration=TARGET_DURATION_SECONDS)])
    video = video.with_duration(TARGET_DURATION_SECONDS)
    logger.info(
        "reel.duration_normalized",
        native_duration=round(raw.duration, 2),
        target_duration=TARGET_DURATION_SECONDS,
        looped=raw.duration < TARGET_DURATION_SECONDS,
        stage_seconds=round(time.monotonic() - stage_started, 1),
    )

    if progress:
        progress.step("sourcing background sound")
    logger.info("reel.stage.start", stage="source_sound", mood=mood)
    stage_started = time.monotonic()
    sound_path = os.path.join(out_dir, "sound.wav")
    freesound_audio_tool.get_sound(mood, duration_seconds=video.duration, output_path=sound_path)
    logger.info("reel.sound_ready", path=sound_path, stage_seconds=round(time.monotonic() - stage_started, 1))

    sound = AudioFileClip(sound_path)
    sound = sound.with_effects(
        [afx.AudioLoop(duration=video.duration), afx.AudioFadeIn(0.4), afx.AudioFadeOut(0.6)]
    ).with_duration(video.duration)
    final = video.with_audio(sound)

    if progress:
        progress.step("rendering native clip")
    logger.info("reel.stage.start", stage="render_native", fps=video.fps or 24)
    stage_started = time.monotonic()
    native_path = os.path.join(out_dir, "final_native.mp4")
    native_fps = video.fps or 24
    try:
        final.write_videofile(native_path, fps=native_fps, codec="libx264", audio_codec="aac", logger=None)
    finally:
        sound.close()
        video.close()
        raw.close()
        final.close()
    logger.info(
        "reel.native_render_complete",
        output_path=native_path,
        duration=round(video.duration, 2),
        size_bytes=os.path.getsize(native_path) if os.path.exists(native_path) else None,
        stage_seconds=round(time.monotonic() - stage_started, 1),
    )

    if progress:
        progress.step("upscaling to 4K/60fps")
    logger.info("reel.stage.start", stage="upscale_4k60", target=f"{UPSCALE_WIDTH}x{UPSCALE_HEIGHT}@{UPSCALE_FPS}")
    stage_started = time.monotonic()
    output_path = os.path.join(out_dir, "final.mp4")
    try:
        _upscale_to_4k60(native_path, output_path)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "reel.upscale_failed",
            error=exc.stderr.decode(errors="replace")[-2000:],
            stage_seconds=round(time.monotonic() - stage_started, 1),
        )
        raise
    logger.info(
        "reel.upscale_complete",
        output_path=output_path,
        resolution=f"{UPSCALE_WIDTH}x{UPSCALE_HEIGHT}",
        fps=UPSCALE_FPS,
        size_bytes=os.path.getsize(output_path) if os.path.exists(output_path) else None,
        stage_seconds=round(time.monotonic() - stage_started, 1),
    )
    logger.info("reel.produce.complete", output_path=output_path, total_seconds=round(time.monotonic() - run_started, 1))
    return output_path


def upload(render_path: str, title: str, description: str) -> str | None:
    logger.info(
        "reel.upload.start",
        render_path=render_path,
        size_bytes=os.path.getsize(render_path) if os.path.exists(render_path) else None,
        title=title,
    )
    if not quota_tool.can_afford("videos.insert"):
        logger.warning("reel.upload_skipped_quota", remaining=quota_tool.remaining_units())
        print(f"Skipped upload: insufficient YouTube quota today (remaining={quota_tool.remaining_units()}). Rerun --upload after it resets.")
        return None

    started = time.monotonic()
    try:
        video_id = youtube_tool.resumable_upload(
            file_path=render_path,
            metadata={"title": title, "description": f"{description}\n\n#Shorts", "tags": ["shorts", "ai video"]},
            visibility="unlisted",
        )
    except Exception as exc:
        logger.error("reel.upload.failed", error=str(exc), error_type=type(exc).__name__, elapsed_seconds=round(time.monotonic() - started, 1))
        raise
    logger.info("reel.uploaded", youtube_video_id=video_id, elapsed_seconds=round(time.monotonic() - started, 1))
    return video_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="video generation prompt (subject/action/style)")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT, help="what to steer the generator away from")
    parser.add_argument("--mood", default=DEFAULT_MOOD, help="keywords describing the background sound's vibe")
    parser.add_argument("--title", default="A Dream You Never Want to Wake Up From | AI Fantasy Journey")
    parser.add_argument("--description", default="An AI-generated fantasy dreamscape.")
    parser.add_argument("--upload", action="store_true", help="also upload the render to YouTube as an unlisted Short")
    args = parser.parse_args()

    references = _list_reference_clips()
    if references:
        print(f"Found {len(references)} reference clip(s) in {REFERENCE_DIR}: {', '.join(os.path.basename(r) for r in references)}")
        print("(Reference only -- watch them yourself and adjust --prompt/--mood to match; this script can't watch video.)")
    else:
        print(f"No reference clips found in {REFERENCE_DIR} yet -- drop the downloaded reels there any time.")

    out_dir = os.path.join(OUTPUT_ROOT, _slugify(args.prompt))
    total_steps = 5 + (1 if args.upload else 0)
    with script_progress.track("produce_reel_clip", total_steps=total_steps) as progress:
        path = produce(args.prompt, args.mood, out_dir, negative_prompt=args.negative_prompt, progress=progress)
        print(f"Rendered: {path}")

        if args.upload:
            progress.step("uploading to YouTube")
            video_id = upload(path, args.title, args.description)
            if video_id:
                print(f"Uploaded: https://youtube.com/watch?v={video_id}")
                progress.done(output_path=f"https://youtube.com/watch?v={video_id}")
            else:
                progress.done(output_path=path)
        else:
            progress.done(output_path=path)
