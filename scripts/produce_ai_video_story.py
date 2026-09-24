"""One-off standalone script: produces "Mila the Firefly and the Lonely Moon" (a short bedtime
fable, user-authored -- not AI-scripted) with real AI-generated animated video clips for every beat
(via tools.fal_video, fal.ai's fal-ai/wan-t2v) instead of a Ken-Burns pan over a static image --
same rationale as scripts/produce_bilingual_story.py, but exercising the AI_VIDEO path end to end
rather than AI_IMAGE/CHARACTER_ANIMATION.

Deliberately bypasses the full LangGraph pipeline (graph/builder.py): the script is user-provided,
not something for script_writer_node/critics to draft or gate, so this skips straight to
rendering.

Usage:
    python scripts/produce_ai_video_story.py            # render only
    python scripts/produce_ai_video_story.py --upload    # also upload (real, unlisted) -- will
                                                            # raise tools.quota.QuotaExceededError
                                                            # and skip upload if today's YouTube
                                                            # quota can't cover it; rerun with
                                                            # --upload once it resets.
"""
import argparse
import os

from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
)

from core.logging import get_logger
from graph.nodes.audio_render import TARGET_FPS, TARGET_H, TARGET_W, _resize_to_cover
from tools import fal_video as fal_video_tool
from tools import quota as quota_tool
from tools import tts as tts_tool
from tools import youtube as youtube_tool
from tools.fonts import resolve_font_path

logger = get_logger("scripts.produce_ai_video_story")

OUTPUT_DIR = "media/firefly_moon_story"
FADE_SECONDS = 0.35
VOICE = "en-US-AndrewNeural"

TITLE = "Mila the Firefly and the Lonely Moon | AI-Animated Kids' Story"
DESCRIPTION = (
    "A heartwarming AI-animated bedtime story about Mila the firefly, who discovers that even the "
    "biggest light in the sky sometimes needs a friend. Every scene in this video was generated "
    "with AI video generation."
)
TAGS = ["kids story", "bedtime story", "ai animated story", "moral story for kids", "firefly story"]

# One AI-generated video clip per beat -- video_prompt drives fal.ai's fal-ai/wan-t2v, voiceover_text
# drives the narration track. Kept short/simple per beat since each is a real, billed generation.
BEATS = [
    {
        "video_prompt": (
            "A tiny glowing firefly with warm golden light drifting through a dark quiet forest at "
            "night, other fireflies twinkling softly among the leaves, dreamy children's storybook "
            "animation style, gentle magical atmosphere"
        ),
        "voiceover_text": (
            "Deep in a quiet forest, a small firefly named Mila glowed brighter than any of her "
            "friends. Every night she danced through the trees, lighting up the dark with her warm "
            "golden light."
        ),
    },
    {
        "video_prompt": (
            "A small firefly looking up at a sad, dim, lonely-looking moon in a starry night sky "
            "above a forest, the moon's glow flickering weakly, wistful children's storybook "
            "animation style"
        ),
        "voiceover_text": (
            "One night, Mila noticed the Moon looked pale and tired, its glow flickering weakly. "
            "\"Why do you look so sad?\" Mila asked. The Moon sighed, \"I shine every night, but I "
            "am always alone up here.\""
        ),
    },
    {
        "video_prompt": (
            "Many tiny fireflies rising together from a forest into the night sky, forming a swirling "
            "spiral of warm golden light reaching up toward the moon, magical uplifting children's "
            "storybook animation style"
        ),
        "voiceover_text": (
            "Mila had an idea. She called every firefly in the forest, and together they rose into "
            "the sky in a swirling spiral of golden light, wrapping the Moon in a warm, glowing hug."
        ),
    },
    {
        "video_prompt": (
            "A bright, happy, glowing moon shining warmly over a peaceful forest at night, fireflies "
            "twinkling around it like stars, cozy heartwarming children's storybook animation style, "
            "warm golden colors"
        ),
        "voiceover_text": (
            "The Moon glowed brighter than ever before, warmed by so many little friends. From that "
            "night on, Mila and her fireflies visited every evening -- because even the biggest "
            "light in the sky shines warmer with a friend beside it."
        ),
    },
]


def _generate_clips() -> list[str]:
    paths = []
    for i, beat in enumerate(BEATS):
        path = os.path.join(OUTPUT_DIR, f"beat_{i}_clip.mp4")
        if not os.path.exists(path):
            fal_video_tool.generate_video(beat["video_prompt"], output_path=path)
            logger.info("story.clip_ready", beat=i, path=path)
        paths.append(path)
    return paths


def _clip_for_duration(clip_path: str, duration: float):
    raw = VideoFileClip(clip_path)
    source = raw.subclipped(0, duration) if raw.duration >= duration else raw.with_effects([vfx.Loop(duration=duration)])
    return _resize_to_cover(source).with_duration(duration), raw


def render(clip_paths: list[str]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font_path = resolve_font_path("en")

    audio_paths, durations = [], []
    for i, beat in enumerate(BEATS):
        audio_path = os.path.join(OUTPUT_DIR, f"beat_{i}_audio.mp3")
        tts_tool.synthesize(beat["voiceover_text"], output_path=audio_path, voice=VOICE)
        clip = AudioFileClip(audio_path)
        durations.append(clip.duration)
        clip.close()
        audio_paths.append(audio_path)
        logger.info("story.narration_ready", beat=i, duration=durations[-1])

    beat_clips, raw_handles = [], []
    for path, duration in zip(clip_paths, durations, strict=True):
        clip, raw = _clip_for_duration(path, duration)
        beat_clips.append(clip)
        raw_handles.append(raw)

    for i, clip in enumerate(beat_clips):
        half = clip.duration / 2
        effects = []
        if i > 0:
            effects.append(vfx.FadeIn(min(FADE_SECONDS, half)))
        if i < len(beat_clips) - 1:
            effects.append(vfx.FadeOut(min(FADE_SECONDS, half)))
        if effects:
            beat_clips[i] = clip.with_effects(effects)

    video_no_audio = concatenate_videoclips(beat_clips, method="compose")
    audio_clips = [AudioFileClip(p) for p in audio_paths]
    full_audio = concatenate_audioclips(audio_clips)
    video = video_no_audio.with_audio(full_audio)

    caption_clips = []
    t_cursor = 0.0
    for beat, duration in zip(BEATS, durations, strict=True):
        caption = TextClip(
            font=font_path,
            text=beat["voiceover_text"],
            font_size=40,
            color="white",
            stroke_color="black",
            stroke_width=2,
            method="caption",
            size=(int(TARGET_W * 0.85), None),
        )
        caption = caption.with_duration(duration).with_start(t_cursor).with_position(("center", "bottom"))
        caption_clips.append(caption)
        t_cursor += duration
    video = CompositeVideoClip([video, *caption_clips], size=(TARGET_W, TARGET_H))

    output_path = os.path.join(OUTPUT_DIR, "final.mp4")
    try:
        video.write_videofile(output_path, fps=TARGET_FPS, codec="libx264", audio_codec="aac", logger=None)
    finally:
        full_audio.close()
        for c in audio_clips:
            c.close()
        for raw in raw_handles:
            raw.close()
        video.close()

    logger.info("story.render_complete", output_path=output_path, duration=sum(durations))
    return output_path


def upload(render_path: str) -> str | None:
    if not quota_tool.can_afford("videos.insert"):
        logger.warning("story.upload_skipped_quota", remaining=quota_tool.remaining_units())
        print(f"Skipped upload: insufficient YouTube quota today (remaining={quota_tool.remaining_units()}). Rerun --upload after it resets.")
        return None

    video_id = youtube_tool.resumable_upload(
        file_path=render_path,
        metadata={"title": TITLE, "description": DESCRIPTION, "tags": TAGS},
        visibility="unlisted",
    )
    logger.info("story.uploaded", youtube_video_id=video_id)
    return video_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="also upload the render to YouTube (real, unlisted)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clips = _generate_clips()
    path = render(clips)
    print(f"Rendered: {path}")

    if args.upload:
        video_id = upload(path)
        if video_id:
            print(f"Uploaded: https://youtube.com/watch?v={video_id}")
