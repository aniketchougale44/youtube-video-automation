"""One-off standalone script: produces "ABCD Alphabet Song | Learn English, Hindi & Marathi" --
same content and structure as scripts/produce_trilingual_abc.py, but using the project's free,
keyless mascot-animation pipeline (tools.character_assets + graph.nodes.audio_render's
_character_clip_for_beat: the shared Mimi the Fox pose pack animated over a per-beat AI-image
backdrop) instead of a paid fal.ai video clip per letter -- switched to this after the configured
FAL_API_KEY's account balance was exhausted mid-run. Each letter is taught with its English example
word, then that word's Hindi and Marathi translation, each spoken in a distinct voice for that
language.

Deliberately bypasses the full LangGraph pipeline (graph/builder.py) the same way the other
scripts/produce_*.py one-offs do.

Usage:
    python scripts/produce_trilingual_abc_mascot.py            # render only
    python scripts/produce_trilingual_abc_mascot.py --upload    # also upload (real, unlisted) --
                                                                    # will raise
                                                                    # tools.quota.QuotaExceededError
                                                                    # and skip upload if today's
                                                                    # YouTube quota can't cover it;
                                                                    # rerun with --upload once it
                                                                    # resets.
"""
import argparse
import os

from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
)

from core.logging import get_logger
from graph.nodes.audio_render import TARGET_FPS, TARGET_H, TARGET_W, _character_clip_for_beat
from tools import character_assets as character_assets_tool
from tools import image_gen as image_gen_tool
from tools import quota as quota_tool
from tools import tts as tts_tool
from tools import youtube as youtube_tool
from tools.fonts import resolve_font_path

logger = get_logger("scripts.produce_trilingual_abc_mascot")

OUTPUT_DIR = "media/trilingual_abc"
FADE_SECONDS = 0.35

EN_VOICE = "en-US-EmmaNeural"
HI_VOICE = "hi-IN-SwaraNeural"
MR_VOICE = "mr-IN-AarohiNeural"

TITLE = "ABCD Alphabet Song | Learn English, Hindi & Marathi | Mimi the Fox"
DESCRIPTION = (
    "Learn the English alphabet A to Z along with each word's Hindi and Marathi translation -- a "
    "fun trilingual alphabet video for toddlers and preschoolers, with Mimi the Fox!"
)
TAGS = [
    "abcd for kids",
    "learn alphabet",
    "trilingual kids video",
    "hindi marathi english",
    "nursery rhymes",
    "preschool learning",
]

# (letter, English word, image description, Hindi word, Marathi word)
LETTERS = [
    ("A", "Apple", "a big shiny red apple", "सेब", "सफरचंद"),
    ("B", "Ball", "a bouncy red and blue ball", "गेंद", "चेंडू"),
    ("C", "Cat", "a cute orange cat", "बिल्ली", "मांजर"),
    ("D", "Dog", "a happy brown puppy", "कुत्ता", "कुत्रा"),
    ("E", "Elephant", "a friendly grey elephant", "हाथी", "हत्ती"),
    ("F", "Fish", "a colorful swimming fish", "मछली", "मासा"),
    ("G", "Goat", "a playful white goat", "बकरी", "बकरी"),
    ("H", "Hat", "a colorful pointy party hat", "टोपी", "टोपी"),
    ("I", "Ice cream", "a scoop of pink ice cream in a cone", "आइसक्रीम", "आइस्क्रीम"),
    ("J", "Jug", "a blue water jug", "जग", "जग"),
    ("K", "Kite", "a colorful flying kite", "पतंग", "पतंग"),
    ("L", "Lion", "a friendly cartoon lion cub with a big mane", "शेर", "सिंह"),
    ("M", "Mango", "a ripe yellow mango", "आम", "आंबा"),
    ("N", "Nest", "a cozy bird's nest with little eggs", "घोंसला", "घरटे"),
    ("O", "Orange", "a juicy orange fruit", "संतरा", "संत्रे"),
    ("P", "Parrot", "a colorful green parrot", "तोता", "पोपट"),
    ("Q", "Queen", "a cheerful cartoon queen with a golden crown", "रानी", "राणी"),
    ("R", "Rabbit", "a fluffy white rabbit", "खरगोश", "ससा"),
    ("S", "Sun", "a bright smiling sun", "सूरज", "सूर्य"),
    ("T", "Tiger", "a friendly cartoon tiger cub", "बाघ", "वाघ"),
    ("U", "Umbrella", "a colorful open umbrella", "छाता", "छत्री"),
    ("V", "Van", "a cheerful cartoon van", "वैन", "व्हॅन"),
    ("W", "Watermelon", "a juicy sliced watermelon", "तरबूज", "कलिंगड"),
    ("X", "Xylophone", "a colorful toy xylophone", "जाइलोफोन", "झायलोफोन"),
    ("Y", "Yak", "a fluffy brown yak", "याक", "याक"),
    ("Z", "Zebra", "a striped black and white zebra", "ज़ेबरा", "झेब्रा"),
]


def _build_beats() -> list[dict]:
    beats = [
        {
            "backdrop_prompt": (
                "A cheerful classroom scene with colorful floating ABC alphabet letters and a "
                "blackboard, bright and inviting, simple flat 2D children's storybook illustration"
            ),
            "segments": [("Let's learn the alphabet in English, Hindi, and Marathi! Here we go!", EN_VOICE)],
            "caption": "Let's learn the ABCs in three languages!",
        }
    ]
    for letter, word, image_desc, hi_word, mr_word in LETTERS:
        beats.append(
            {
                "backdrop_prompt": (
                    f"A colorful alphabet flashcard background showing the big bold letter "
                    f"'{letter}' next to {image_desc}, simple flat 2D children's alphabet book "
                    "illustration style, bright cheerful colors"
                ),
                "segments": [
                    (f"{letter} is for {word}!", EN_VOICE),
                    (f"हिंदी में: {hi_word}", HI_VOICE),
                    (f"मराठीत: {mr_word}", MR_VOICE),
                ],
                "caption": f"{letter} — {word}\nहिंदी: {hi_word}   मराठी: {mr_word}",
            }
        )
    beats.append(
        {
            "backdrop_prompt": (
                "All 26 colorful alphabet letters dancing and celebrating together on a sunny "
                "background, joyful children's storybook illustration style, bright cheerful colors"
            ),
            "segments": [("Great job learning your ABCs in three languages! See you next time!", EN_VOICE)],
            "caption": "Great job! See you next time!",
        }
    )
    return beats


BEATS = _build_beats()


def _generate_backdrops() -> list[str]:
    paths = []
    for i, beat in enumerate(BEATS):
        path = os.path.join(OUTPUT_DIR, f"beat_{i}_backdrop.jpg")
        if not os.path.exists(path):
            image_gen_tool.generate_image(beat["backdrop_prompt"], output_path=path, size="1920x1080")
            logger.info("abc.backdrop_ready", beat=i, path=path)
        paths.append(path)
    return paths


def render(backdrop_paths: list[str]) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font_path = resolve_font_path("hi")  # Nirmala/Mangal covers both Devanagari and Latin glyphs
    character_assets_tool.get_character_pack()  # ensure the mascot pose pack exists (cached after first call)

    audio_paths, durations = [], []
    for i, beat in enumerate(BEATS):
        segment_paths = []
        for j, (text, voice) in enumerate(beat["segments"]):
            seg_path = os.path.join(OUTPUT_DIR, f"beat_{i}_seg_{j}.mp3")
            tts_tool.synthesize(text, output_path=seg_path, voice=voice)
            segment_paths.append(seg_path)

        audio_path = os.path.join(OUTPUT_DIR, f"beat_{i}_audio.mp3")
        if len(segment_paths) == 1:
            import shutil

            shutil.copyfile(segment_paths[0], audio_path)
        else:
            seg_clips = [AudioFileClip(p) for p in segment_paths]
            combined = concatenate_audioclips(seg_clips)
            combined.write_audiofile(audio_path, logger=None)
            combined.close()
            for c in seg_clips:
                c.close()

        clip = AudioFileClip(audio_path)
        durations.append(clip.duration)
        clip.close()
        audio_paths.append(audio_path)
        logger.info("abc.narration_ready", beat=i, duration=durations[-1])

    beat_clips = []
    for i, (backdrop_path, duration) in enumerate(zip(backdrop_paths, durations, strict=True)):
        beat_clips.append(_character_clip_for_beat(i, duration, backdrop_image_path=backdrop_path))

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
            text=beat["caption"],
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
        video.close()

    logger.info("abc.render_complete", output_path=output_path, duration=sum(durations))
    return output_path


def upload(render_path: str) -> str | None:
    if not quota_tool.can_afford("videos.insert"):
        logger.warning("abc.upload_skipped_quota", remaining=quota_tool.remaining_units())
        print(f"Skipped upload: insufficient YouTube quota today (remaining={quota_tool.remaining_units()}). Rerun --upload after it resets.")
        return None

    video_id = youtube_tool.resumable_upload(
        file_path=render_path,
        metadata={"title": TITLE, "description": DESCRIPTION, "tags": TAGS},
        visibility="unlisted",
    )
    logger.info("abc.uploaded", youtube_video_id=video_id)
    return video_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="also upload the render to YouTube (real, unlisted)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    backdrops = _generate_backdrops()
    path = render(backdrops)
    print(f"Rendered: {path}")

    if args.upload:
        video_id = upload(path)
        if video_id:
            print(f"Uploaded: https://youtube.com/watch?v={video_id}")
