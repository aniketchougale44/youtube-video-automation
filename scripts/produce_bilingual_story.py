"""One-off standalone script: produces "The Brave Little Sparrow and the Giant Elephant" (a jungle
animal moral fable, user-authored -- not AI-scripted) as two videos, English and Hindi, sharing
the same four story illustrations (generated once, free/keyless via tools.image_gen's Pollinations
fallback) but with per-language narration (edge-tts) and captions (Devanagari needs a different
font than Latin -- see tools.fonts).

Deliberately bypasses the full LangGraph pipeline (graph/builder.py): the script is user-provided,
not something for script_writer_node/critics to draft or gate, so this skips straight to
rendering -- same rationale the old scripts/generate_puppy_story.py used for its offline-model
validation runs.

Usage:
    python scripts/produce_bilingual_story.py            # render both languages only
    python scripts/produce_bilingual_story.py --upload   # also upload (real, unlisted) -- will
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
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
)

from core.logging import get_logger
from graph.nodes.audio_render import TARGET_FPS, TARGET_H, TARGET_W, _ken_burns_image_clip
from tools import image_gen as image_gen_tool
from tools import quota as quota_tool
from tools import tts as tts_tool
from tools import youtube as youtube_tool
from tools.fonts import resolve_font_path

logger = get_logger("scripts.produce_bilingual_story")

OUTPUT_DIR = "media/sparrow_elephant_story"
FADE_SECONDS = 0.35

# One image per beat, reused for both language versions -- the scene is identical either way, and
# Pollinations understands English prompts best regardless of narration language.
BEATS = [
    {
        "image_prompt": (
            "A lush green jungle at midday, a tiny colorful sparrow perched on a branch singing, "
            "and a huge gentle grey elephant playfully splashing water by a river nearby, simple "
            "flat 2D children's book illustration style, warm cheerful colors"
        ),
        "en": (
            "Deep inside the whispering green jungle lived a tiny sparrow named Chiku and a "
            "massive, gentle elephant named Gajju. Gajju spent his afternoons splashing cool "
            "water by the river, while Chiku fluttered from branch to branch, singing sweet "
            "morning tunes. Because Gajju was so enormous, he often thought tiny creatures could "
            "never accomplish big things. \"You are so small, little Chiku,\" Gajju would chuckle "
            "softly, \"even a playful breeze could blow you away!\""
        ),
        "hi": (
            "एक सुंदर और घने जंगल में चीकू नाम की एक नन्हीं गौरैया और गज्जू नाम का एक विशाल हाथी "
            "रहते थे। गज्जू दिनभर नदी किनारे ठंडे पानी से फव्वारे बनाता था, जबकि चीकू डाल-डाल "
            "फुदककर मीठे गीत गाती थी। गज्जू शरीर से बहुत बड़ा था, इसलिए वह अक्सर सोचता था कि छोटे "
            "जीव बड़े काम नहीं कर सकते। वह मुस्कुराते हुए कहता, \"चीकू, तुम इतनी छोटी हो कि हवा का "
            "एक तेज़ झोंका भी तुम्हें उड़ा ले जाएगा!\""
        ),
    },
    {
        "image_prompt": (
            "A jungle brushfire with thick grey smoke rising at the edge of a dry meadow, a large "
            "worried elephant with its back leg stuck in a hidden ditch between two fallen logs, "
            "trumpeting in distress, simple flat 2D children's book illustration style, dramatic "
            "but not scary, warm colors"
        ),
        "en": (
            "One sunny afternoon, a sudden brushfire sparked at the edge of the dry meadow. Thick "
            "grey smoke rose into the sky, frightening the woodland animals. Gajju rushed forward "
            "to stamp it out with his heavy feet, but his back leg accidentally slipped into a "
            "deep, hidden ditch between two fallen logs. He was completely stuck! As the smoke "
            "crept closer, Gajju trumpeted in distress."
        ),
        "hi": (
            "एक दोपहर, सूखी घास में अचानक आग की लपटें उठने लगीं। धुआं उठता देख जंगल के सभी जानवर "
            "घबरा गए। गज्जू तुरंत आग बुझाने के लिए दौड़ा, लेकिन उसका भारी पैर दो पुराने लट्ठों के "
            "बीच एक गहरे गड्ढे में फंस गया। गज्जू ने बहुत ज़ोर लगाया, पर वह बाहर नहीं निकल सका। "
            "धुआं पास आ रहा था और गज्जू घबराहट में चिंघाड़ने लगा।"
        ),
    },
    {
        "image_prompt": (
            "A tiny brave sparrow flying fast through the jungle canopy alerting a troop of "
            "monkeys and a herd of deer, monkeys pulling sturdy vines to free a stuck elephant's "
            "leg while other animals help, teamwork scene, simple flat 2D children's book "
            "illustration style, warm cheerful colors"
        ),
        "en": (
            "Seeing his friend in danger, Chiku did not hesitate. Knowing she was too small to "
            "pull him out alone, she used her quick wings to dart across the canopy. She chirped "
            "at the top of her lungs, alerting a troop of monkeys and a herd of deer. The monkeys "
            "quickly brought sturdy jungle vines to create a pulling rope, while the deer and "
            "forest animals pulled together, freeing Gajju's leg just in time. Free at last, "
            "Gajju filled his trunk with river water and sprayed a massive shower over the grass, "
            "extinguishing the flames."
        ),
        "hi": (
            "चीकू ने अपने दोस्त को संकट में देखा और बिना एक पल गंवाए मदद के लिए उड़ पड़ी। वह जानती "
            "थी कि वह अकेले गज्जू को नहीं खींच सकती, इसलिए उसने तुरंत अपनी तेज़ चहचहाहट से बंदरों "
            "और हिरणों की टोली को आवाज़ दी। सभी जानवर फौरन मदद के लिए दौड़े आए। बंदरों ने मजबूत "
            "लताएं फेंकी और सबने मिलकर गज्जू को गड्ढे से बाहर खींच लिया। आज़ाद होते ही गज्जू ने "
            "नदी से सूंड़ भरकर पानी फेंका और आग को पूरी तरह बुझा दिया।"
        ),
    },
    {
        "image_prompt": (
            "A happy elephant spraying a big shower of water from its trunk over green grass "
            "putting out flames, sparrow and elephant and forest animals smiling together at "
            "sunset, heartwarming teamwork scene, simple flat 2D children's book illustration "
            "style, warm golden colors"
        ),
        "en": (
            "From that day on, Gajju knew that true strength is not measured by the size of your "
            "body, but by the courage in your heart and the power of teamwork."
        ),
        "hi": (
            "उस दिन के बाद गज्जू समझ गया कि असली ताकत शरीर के बड़े आकार में नहीं, बल्कि मन की "
            "हिम्मत, समझदारी और सच्ची दोस्ती में होती है।"
        ),
    },
]

TITLES = {
    "en": "The Brave Little Sparrow and the Giant Elephant | A Kids' Story",
    "hi": "नन्हीं चिड़िया और दयालु हाथी | बच्चों की कहानी",
}
DESCRIPTIONS = {
    "en": (
        "A heartwarming animated story about Chiku the sparrow and Gajju the elephant, and how "
        "true strength comes from courage and teamwork, not size."
    ),
    "hi": (
        "चीकू नाम की गौरैया और गज्जू नाम के हाथी की एक प्यारी कहानी -- जो सिखाती है कि असली ताकत "
        "शरीर के आकार में नहीं, बल्कि हिम्मत और दोस्ती में होती है।"
    ),
}
VOICES = {"en": "en-US-AndrewNeural", "hi": "hi-IN-MadhurNeural"}


def _generate_backdrops() -> list[str]:
    paths = []
    for i, beat in enumerate(BEATS):
        path = os.path.join(OUTPUT_DIR, f"beat_{i}_backdrop.jpg")
        if not os.path.exists(path):
            image_gen_tool.generate_image(beat["image_prompt"], output_path=path, size="1920x1080")
            logger.info("story.backdrop_ready", beat=i, path=path)
        paths.append(path)
    return paths


def _render_language(language: str, backdrop_paths: list[str]) -> str:
    lang_dir = os.path.join(OUTPUT_DIR, language)
    os.makedirs(lang_dir, exist_ok=True)
    font_path = resolve_font_path(language)
    voice = VOICES[language]

    audio_paths, durations = [], []
    for i, beat in enumerate(BEATS):
        audio_path = os.path.join(lang_dir, f"beat_{i}_audio.mp3")
        tts_tool.synthesize(beat[language], output_path=audio_path, voice=voice)
        clip = AudioFileClip(audio_path)
        durations.append(clip.duration)
        clip.close()
        audio_paths.append(audio_path)
        logger.info("story.narration_ready", language=language, beat=i, duration=durations[-1])

    beat_clips = []
    for i, (path, duration) in enumerate(zip(backdrop_paths, durations, strict=True)):
        beat_clips.append(_ken_burns_image_clip(path, duration, zoom_in=i % 2 == 0))

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
    for beat, clip, duration in zip(BEATS, beat_clips, durations, strict=True):
        caption = TextClip(
            font=font_path,
            text=beat[language],
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

    output_path = os.path.join(lang_dir, "final.mp4")
    try:
        video.write_videofile(output_path, fps=TARGET_FPS, codec="libx264", audio_codec="aac", logger=None)
    finally:
        full_audio.close()
        for c in audio_clips:
            c.close()
        video.close()

    logger.info("story.render_complete", language=language, output_path=output_path, duration=sum(durations))
    return output_path


def upload(language: str, render_path: str) -> str | None:
    if not quota_tool.can_afford("videos.insert"):
        logger.warning("story.upload_skipped_quota", language=language, remaining=quota_tool.remaining_units())
        print(f"[{language}] Skipped upload: insufficient YouTube quota today (remaining={quota_tool.remaining_units()}). Rerun --upload after it resets.")
        return None

    video_id = youtube_tool.resumable_upload(
        file_path=render_path,
        metadata={
            "title": TITLES[language],
            "description": DESCRIPTIONS[language],
            "tags": ["kids story", "moral story for kids", "bedtime story", "animal story", "hindi kahani" if language == "hi" else "english story"],
        },
        visibility="unlisted",
    )
    logger.info("story.uploaded", language=language, youtube_video_id=video_id)
    return video_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="also upload both renders to YouTube (real, unlisted)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    backdrops = _generate_backdrops()

    for lang in ("en", "hi"):
        path = _render_language(lang, backdrops)
        print(f"[{lang}] Rendered: {path}")
        if args.upload:
            video_id = upload(lang, path)
            if video_id:
                print(f"[{lang}] Uploaded: https://youtube.com/watch?v={video_id}")
