"""One-off standalone script: produces "Animal Sounds Song for Toddlers | Learn Animal Names" --
a call-and-response nursery-rhyme-style video (user-authored -- not AI-scripted), same production
approach as scripts/produce_ai_video_story.py: one real AI-generated video clip per beat (via
tools.fal_video, fal.ai's fal-ai/wan-t2v) instead of a Ken-Burns pan over a static image.

Deliberately bypasses the full LangGraph pipeline (graph/builder.py) the same way
scripts/produce_ai_video_story.py and scripts/produce_bilingual_story.py do.

Usage:
    python scripts/produce_animal_song.py            # render only
    python scripts/produce_animal_song.py --upload    # also upload (real, unlisted) -- will
                                                         # raise tools.quota.QuotaExceededError and
                                                         # skip upload if today's YouTube quota
                                                         # can't cover it; rerun with --upload once
                                                         # it resets.
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

logger = get_logger("scripts.produce_animal_song")

OUTPUT_DIR = "media/animal_sounds_song"
FADE_SECONDS = 0.35

# Each beat gets its own edge-tts voice so the animals sound like distinct characters talking to
# the viewer rather than one narrator reading a list -- warm adult narrator for intro/outro, then a
# different age/gender/accent per animal for contrast.
NARRATOR_VOICE = "en-US-AndrewNeural"
COW_VOICE = "en-US-GuyNeural"        # calm, deep
DOG_VOICE = "en-US-AnaNeural"        # playful, child-like energy
CAT_VOICE = "en-GB-SoniaNeural"      # soft, refined British accent
DUCK_VOICE = "en-AU-NatashaNeural"   # quirky, distinct accent
LION_VOICE = "en-US-BrianNeural"     # bold, confident

TITLE = "Animal Sounds Song for Toddlers | Learn Animal Names | AI-Animated"
DESCRIPTION = (
    "A fun sing-along animal sounds song for toddlers and preschoolers -- learn animal names and "
    "the sounds they make! Every scene in this video was generated with AI video generation."
)
TAGS = [
    "animal sounds for toddlers",
    "learn animal names",
    "nursery rhymes",
    "kids song",
    "preschool learning",
    "toddler songs",
]

# Call-and-response beats: each names an animal and its sound, then invites the child to repeat it
# -- same cadence as classic animal-sounds nursery rhymes, original wording. One AI-generated video
# clip per beat via fal.ai's fal-ai/wan-t2v.
BEATS = [
    {
        "video_prompt": (
            "A cheerful sunny farm meadow with a cow, a dog, a cat, and a duck all standing together "
            "happily, simple flat 2D children's storybook animation style, bright cheerful colors, "
            "inviting and playful"
        ),
        "voiceover_text": (
            "Let's learn some animal sounds today! Come along and sing with me!"
        ),
        "voice": NARRATOR_VOICE,
    },
    {
        "video_prompt": (
            "A happy round brown and white cow standing in a green meadow mooing with its mouth open, "
            "simple flat 2D children's storybook animation style, bright cheerful colors"
        ),
        "voiceover_text": "The cow says moo! Moo, moo, moo! Can you say moo like the cow?",
        "voice": COW_VOICE,
    },
    {
        "video_prompt": (
            "A happy fluffy brown puppy sitting in a garden barking playfully with its tail wagging, "
            "simple flat 2D children's storybook animation style, bright cheerful colors"
        ),
        "voiceover_text": "The dog says woof! Woof, woof, woof! Can you say woof like the dog?",
        "voice": DOG_VOICE,
    },
    {
        "video_prompt": (
            "A cute orange striped kitten sitting on a windowsill meowing with its mouth open, simple "
            "flat 2D children's storybook animation style, bright cheerful colors"
        ),
        "voiceover_text": "The cat says meow! Meow, meow, meow! Can you say meow like the cat?",
        "voice": CAT_VOICE,
    },
    {
        "video_prompt": (
            "A happy yellow duckling swimming and splashing in a pond quacking, simple flat 2D "
            "children's storybook animation style, bright cheerful colors"
        ),
        "voiceover_text": "The duck says quack! Quack, quack, quack! Can you say quack like the duck?",
        "voice": DUCK_VOICE,
    },
    {
        "video_prompt": (
            "A friendly cartoon lion cub standing on a sunny savanna roaring playfully with its mane "
            "flowing, simple flat 2D children's storybook animation style, bright cheerful colors, "
            "not scary"
        ),
        "voiceover_text": "The lion says roar! Roar, roar, roar! Can you say roar like the lion?",
        "voice": LION_VOICE,
    },
    {
        "video_prompt": (
            "All the happy farm and jungle animals -- cow, dog, cat, duck, and lion cub -- dancing "
            "together joyfully in a sunny meadow, simple flat 2D children's storybook animation "
            "style, bright cheerful colors, celebratory"
        ),
        "voiceover_text": "Great job singing along! See you next time for more animal fun!",
        "voice": NARRATOR_VOICE,
    },
]


def _generate_clips() -> list[str]:
    paths = []
    for i, beat in enumerate(BEATS):
        path = os.path.join(OUTPUT_DIR, f"beat_{i}_clip.mp4")
        if not os.path.exists(path):
            fal_video_tool.generate_video(beat["video_prompt"], output_path=path)
            logger.info("song.clip_ready", beat=i, path=path)
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
        tts_tool.synthesize(beat["voiceover_text"], output_path=audio_path, voice=beat["voice"])
        clip = AudioFileClip(audio_path)
        durations.append(clip.duration)
        clip.close()
        audio_paths.append(audio_path)
        logger.info("song.narration_ready", beat=i, duration=durations[-1])

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
            font_size=44,
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

    logger.info("song.render_complete", output_path=output_path, duration=sum(durations))
    return output_path


def upload(render_path: str) -> str | None:
    if not quota_tool.can_afford("videos.insert"):
        logger.warning("song.upload_skipped_quota", remaining=quota_tool.remaining_units())
        print(f"Skipped upload: insufficient YouTube quota today (remaining={quota_tool.remaining_units()}). Rerun --upload after it resets.")
        return None

    video_id = youtube_tool.resumable_upload(
        file_path=render_path,
        metadata={"title": TITLE, "description": DESCRIPTION, "tags": TAGS},
        visibility="unlisted",
    )
    logger.info("song.uploaded", youtube_video_id=video_id)
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
