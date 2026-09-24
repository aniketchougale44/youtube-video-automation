"""One-off standalone script: produces "Mimi the Fox and the Rainbow Berry Hunt" -- an original,
user-facing bedtime/adventure story starring the channel's shared mascot (see
core/settings.py's mascot_character_name/mascot_character_prompt). Uses the same free, keyless
mascot-animation pipeline as scripts/produce_trilingual_abc_mascot.py (tools.character_assets +
graph.nodes.audio_render's _character_clip_for_beat: Mimi's pose pack animated over a per-beat
AI-image backdrop) rather than a paid AI-video clip per beat -- reliable, no billing risk, and
visually consistent with the channel's other Mimi videos.

Two things this script does beyond the ABCD-mascot template to make it read as an actual bedtime
story rather than a flashcard drill:
  - tools.tts.synthesize is called with rate="-8%" -- a touch slower than default reading speed,
    which is what makes a TTS voice land as an unhurried storyteller instead of a rushed narrator.
  - Each beat's voiceover_text is written with natural storytelling cadence (varied sentence
    length, dialogue, a rhetorical question, exclamations) rather than flat declarative sentences,
    since prosody alone can't fix text that reads like a spec sheet.
  - A soft ambient music bed (tools.freesound_audio, same tool scripts/produce_reel_clip.py uses)
    plays under the narration at low volume for the whole runtime, faded in/out -- purely a
    production-polish addition; falls back to a locally synthesized ambient pad if unset/failed,
    so the video is never silent-but-narration.

Deliberately bypasses the full LangGraph pipeline (graph/builder.py) the same way the other
scripts/produce_*.py one-offs do.

Usage:
    python scripts/produce_mimi_rainbow_story.py            # render only
    python scripts/produce_mimi_rainbow_story.py --upload    # also upload (real, unlisted) --
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
    CompositeAudioClip,
    CompositeVideoClip,
    TextClip,
    afx,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
)

from core.logging import get_logger
from graph.nodes.audio_render import TARGET_FPS, TARGET_H, TARGET_W, _character_clip_for_beat
from tools import character_assets as character_assets_tool
from tools import freesound_audio as freesound_audio_tool
from tools import image_gen as image_gen_tool
from tools import quota as quota_tool
from tools import script_progress
from tools import tts as tts_tool
from tools import youtube as youtube_tool
from tools.fonts import resolve_font_path

logger = get_logger("scripts.produce_mimi_rainbow_story")

OUTPUT_DIR = "media/mimi_rainbow_story"
FADE_SECONDS = 0.35
VOICE = "en-US-EmmaNeural"  # same brand voice as scripts/produce_trilingual_abc_mascot.py
NARRATION_RATE = "-8%"  # slightly slower than default -- unhurried storyteller cadence
MUSIC_VOLUME = 0.14  # low enough narration always stays clearly out front
MUSIC_MOOD = "gentle warm whimsical storybook ambient"

TITLE = "Mimi the Fox and the Rainbow Berry Hunt | AI-Animated Kids' Bedtime Story"
DESCRIPTION = (
    "Join Mimi the Fox on a colorful adventure! After a rainy morning, Mimi follows a trail of "
    "glowing rainbow berries and meets two new friends along the way -- but the real treasure "
    "she finds isn't the berries at all. A gentle, animated bedtime story about teamwork, "
    "courage, and sharing, for toddlers and preschoolers."
)
TAGS = [
    "kids story",
    "bedtime story",
    "ai animated story",
    "mimi the fox",
    "moral story for kids",
    "story for toddlers",
    "kids adventure story",
]

BEATS = [
    {
        "backdrop_prompt": (
            "A cozy fox den entrance in a sunlit forest clearing just after rain, dew drops "
            "glistening on green leaves, a soft colorful rainbow arching across the sky, warm "
            "morning light, simple flat 2D children's storybook illustration, bright cheerful "
            "colors, no animals in the scene"
        ),
        "voiceover_text": (
            "One sunny morning, right after a soft rain, Mimi the Fox poked her nose out of her "
            "cozy den. And there, stretched all the way across the sky, was the biggest, "
            "brightest rainbow she had ever seen!"
        ),
    },
    {
        "backdrop_prompt": (
            "A cluster of small glowing berries in rainbow colors -- red, orange, yellow, green, "
            "blue, purple -- scattered on a mossy forest floor where a rainbow touches the "
            "ground, magical sparkles, simple flat 2D children's storybook illustration, bright "
            "cheerful colors"
        ),
        "voiceover_text": (
            "Right where the rainbow touched the ground, Mimi spotted something amazing: tiny "
            "berries, each one glowing a different color of the rainbow! \"I have to find where "
            "they lead,\" Mimi giggled, and off she scampered."
        ),
    },
    {
        "backdrop_prompt": (
            "A wide gentle river with lily pads, a small green turtle peeking shyly from behind "
            "a smooth rock at the water's edge, sunny forest background, simple flat 2D "
            "children's storybook illustration, bright cheerful colors"
        ),
        "voiceover_text": (
            "Before long, Mimi reached a wide, splashing river. Sitting by the water was Ollie "
            "the Turtle, looking a little worried. \"I want to see the rainbow berries too,\" "
            "Ollie said softly, \"but I'm scared of the current.\""
        ),
    },
    {
        "backdrop_prompt": (
            "A cheerful fox and a small turtle riding together across a river on a big green "
            "lily pad raft, gentle splashes, sunny sky, forest on both riverbanks, simple flat "
            "2D children's storybook illustration, bright cheerful colors"
        ),
        "voiceover_text": (
            "\"We can do it together!\" said Mimi with a big smile. She hopped onto a lily pad, "
            "and Ollie climbed up right behind her. Paddle by paddle, they crossed the river "
            "side by side, laughing the whole way."
        ),
    },
    {
        "backdrop_prompt": (
            "A tall oak tree with autumn-colored leaves, a small squirrel stretching and jumping "
            "to reach a single glowing rainbow berry just out of reach on a high branch, simple "
            "flat 2D children's storybook illustration, bright cheerful colors"
        ),
        "voiceover_text": (
            "On the other side, they found Rosie the Squirrel, stretching and jumping, trying to "
            "reach a glowing berry high in a tall oak tree. \"It's just a little too high!\" "
            "Rosie huffed."
        ),
    },
    {
        "backdrop_prompt": (
            "A joyful scene of a small turtle tucked into its shell as a stepping stone, a fox "
            "gently boosting a squirrel upward toward a glowing berry on a tree branch, warm "
            "afternoon light, simple flat 2D children's storybook illustration, bright cheerful "
            "colors"
        ),
        "voiceover_text": (
            "Mimi had an idea. Ollie tucked into his shell like a little stepping stone, and "
            "Mimi gave Rosie one gentle boost. Up, up, up Rosie climbed, and plucked the "
            "sparkling berry with a cheer!"
        ),
    },
    {
        "backdrop_prompt": (
            "A magical meadow bursting with glowing rainbow-colored berries under a bright "
            "rainbow's arch, golden sparkling light, wildflowers, simple flat 2D children's "
            "storybook illustration, bright cheerful colors, no animals in the scene"
        ),
        "voiceover_text": (
            "Together, the three friends followed the last sparkling trail of berries, and "
            "gasped. There, glowing under the rainbow's arch, was a whole meadow bursting with "
            "every color imaginable!"
        ),
    },
    {
        "backdrop_prompt": (
            "A fox, a turtle, and a squirrel having a cheerful picnic together sharing colorful "
            "glowing berries in a sunny meadow, warm golden light, simple flat 2D children's "
            "storybook illustration, bright cheerful colors"
        ),
        "voiceover_text": (
            "Now, Mimi could have kept the whole meadow to herself. But instead, she shared "
            "every last berry with Ollie and Rosie. \"Yummy adventures,\" Mimi said, \"are "
            "always better with friends.\""
        ),
    },
    {
        "backdrop_prompt": (
            "A peaceful forest path at sunset, warm pink and gold sky, first stars twinkling "
            "overhead, cozy and calm, simple flat 2D children's storybook illustration, warm "
            "soft colors, no animals in the scene"
        ),
        "voiceover_text": (
            "As the sun dipped low and painted the sky pink and gold, Mimi walked home with a "
            "happy heart. She had found something even more colorful than the rainbow berries: "
            "true friendship. Goodnight, little adventurers."
        ),
    },
]


def _generate_backdrops(progress: script_progress.ScriptRun | None = None) -> list[str]:
    paths = []
    for i, beat in enumerate(BEATS):
        if progress:
            progress.step(f"generating backdrop {i + 1}/{len(BEATS)}")
        path = os.path.join(OUTPUT_DIR, f"beat_{i}_backdrop.jpg")
        if not os.path.exists(path):
            image_gen_tool.generate_image(beat["backdrop_prompt"], output_path=path, size="1920x1080")
            logger.info("mimi_rainbow.backdrop_ready", beat=i, path=path)
        paths.append(path)
    return paths


def _background_music(total_duration: float) -> AudioFileClip:
    music_path = os.path.join(OUTPUT_DIR, "music.wav")
    if not os.path.exists(music_path):
        freesound_audio_tool.get_sound(MUSIC_MOOD, duration_seconds=total_duration, output_path=music_path)
    music = AudioFileClip(music_path)
    return music.with_effects(
        [
            afx.AudioLoop(duration=total_duration),
            afx.MultiplyVolume(MUSIC_VOLUME),
            afx.AudioFadeIn(1.0),
            afx.AudioFadeOut(1.5),
        ]
    ).with_duration(total_duration)


def render(backdrop_paths: list[str], progress: script_progress.ScriptRun | None = None) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font_path = resolve_font_path("en")
    character_assets_tool.get_character_pack()  # ensure the mascot pose pack exists (cached after first call)

    narration_paths, durations = [], []
    for i, beat in enumerate(BEATS):
        if progress:
            progress.step(f"synthesizing narration {i + 1}/{len(BEATS)}")
        narration_path = os.path.join(OUTPUT_DIR, f"beat_{i}_narration.mp3")
        if not os.path.exists(narration_path):
            tts_tool.synthesize(beat["voiceover_text"], output_path=narration_path, voice=VOICE, rate=NARRATION_RATE)
        clip = AudioFileClip(narration_path)
        durations.append(clip.duration)
        clip.close()
        narration_paths.append(narration_path)
        logger.info("mimi_rainbow.narration_ready", beat=i, duration=durations[-1])

    if progress:
        progress.step("rendering video")

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
    narration_clips = [AudioFileClip(p) for p in narration_paths]
    full_narration = concatenate_audioclips(narration_clips)
    music = _background_music(full_narration.duration)
    full_audio = CompositeAudioClip([music, full_narration])
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
            size=(int(TARGET_W * 0.82), None),
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
        music.close()
        full_narration.close()
        for c in narration_clips:
            c.close()
        video.close()

    logger.info("mimi_rainbow.render_complete", output_path=output_path, duration=sum(durations))
    return output_path


def upload(render_path: str) -> str | None:
    if not quota_tool.can_afford("videos.insert"):
        logger.warning("mimi_rainbow.upload_skipped_quota", remaining=quota_tool.remaining_units())
        print(f"Skipped upload: insufficient YouTube quota today (remaining={quota_tool.remaining_units()}). Rerun --upload after it resets.")
        return None

    video_id = youtube_tool.resumable_upload(
        file_path=render_path,
        metadata={"title": TITLE, "description": DESCRIPTION, "tags": TAGS},
        visibility="unlisted",
    )
    logger.info("mimi_rainbow.uploaded", youtube_video_id=video_id)
    return video_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="also upload the render to YouTube (real, unlisted)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_steps = len(BEATS) * 2 + 1 + (1 if args.upload else 0)  # backdrops + narration + render (+ upload)
    with script_progress.track("produce_mimi_rainbow_story", total_steps=total_steps) as progress:
        backdrops = _generate_backdrops(progress)
        path = render(backdrops, progress)
        print(f"Rendered: {path}")

        if args.upload:
            progress.step("uploading to YouTube")
            video_id = upload(path)
            if video_id:
                print(f"Uploaded: https://youtube.com/watch?v={video_id}")
                progress.done(output_path=f"https://youtube.com/watch?v={video_id}")
            else:
                progress.done(output_path=path)
        else:
            progress.done(output_path=path)
