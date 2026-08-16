"""Voiceover Agent + Video Assembly Agent — real logic.

video_assembly_node is the highest-complexity node in the pipeline and takes genuine wall-clock
time (a multi-minute long-form video plausibly takes 1-5 min locally to encode) — this is
qualitatively different from every other node here, which is just an API round trip.

CHARACTER_ANIMATION beats (the AI_IMAGE fallback -- see graph/nodes/visual.py) get the shared
mascot pose pack composited over a flat backdrop with real per-frame motion (bounce, mouth-flap,
blink, an opening wave) via `_character_clip_for_beat`/`_character_frame_factory` -- this is what
replaced panning a single AI-generated still.

STOCK_IMAGE beats (the rare case where a stock-footage search only turns up photos, not video)
still get a Ken Burns pan/zoom rather than a static hold: a time-varying resize composed directly
with vfx.Crop doesn't track a growing frame's center correctly (Crop only takes a static
x_center/y_center, not a callable). Instead, the resized (growing/shrinking) image is placed on a
fixed-size CompositeVideoClip canvas at the target resolution -- the canvas itself clips every
frame to its bounds, so no dynamic crop is needed at all.
"""
import math
import os

import numpy as np
from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoClip,
    VideoFileClip,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
)
from PIL import Image

from agents.schemas.audio_render import AssemblyOutput, VoiceoverOutput, VoiceoverSegment
from agents.schemas.script import ScriptOutput
from agents.schemas.visual import AssetOutput, AssetType, SourcedAsset
from core.logging import get_logger
from graph.nodes._helpers import log_and_trace
from graph.state import PipelineState
from tools import character_assets as character_assets_tool
from tools import tts as tts_tool
from tools.fonts import resolve_font_path
from tools.media_paths import media_path

logger = get_logger("graph.nodes.audio_render")

STAGE_VOICE = "voiceover"
STAGE_ASSEMBLY = "video_assembly"

TARGET_W, TARGET_H = 1920, 1080
TARGET_FPS = 30


def voiceover_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_VOICE, "start")

    script = ScriptOutput.model_validate(state["script_output"])
    run_id = state["run_id"]

    segments: list[VoiceoverSegment] = []
    for beat in script.beats:
        path = media_path(run_id, "voiceover", f"beat_{beat.index}.mp3")
        tts_tool.synthesize(beat.voiceover_text, output_path=path)

        audio_clip = AudioFileClip(path)
        duration = audio_clip.duration
        audio_clip.close()

        segments.append(VoiceoverSegment(beat_index=beat.index, audio_path=path, duration_seconds=duration, text=beat.voiceover_text))

    full_path = media_path(run_id, "voiceover", "full.mp3")
    segment_clips = [AudioFileClip(s.audio_path) for s in segments]
    full_audio = concatenate_audioclips(segment_clips)
    full_audio.write_audiofile(full_path, logger=None)
    full_audio.close()
    for clip in segment_clips:
        clip.close()

    output = VoiceoverOutput(
        segments=segments,
        full_audio_path=full_path,
        total_duration_seconds=sum(s.duration_seconds for s in segments),
    )

    return {
        "voiceover_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_VOICE, "complete", segments=len(segments))],
    }


def _resize_to_cover(clip, width: int = TARGET_W, height: int = TARGET_H):
    """Scales `clip` up to fully cover a width x height frame (never letterboxed), then
    center-crops down to exactly width x height."""
    scale = max(width / clip.w, height / clip.h)
    covered = clip.resized(scale)
    return covered.with_effects([vfx.Crop(x_center=covered.w / 2, y_center=covered.h / 2, width=width, height=height)])


KEN_BURNS_RATE_PER_SECOND = 0.035  # ~3.5%/s scale change -- a 6s beat zooms ~20%, gentle not jarring
KEN_BURNS_MAX_ZOOM = 1.3


def _ken_burns_image_clip(path: str, duration: float, zoom_in: bool, width: int = TARGET_W, height: int = TARGET_H):
    """Still images hold camera motion instead of a static frame: resized larger/smaller than the
    canvas over time while pinned to center, on a fixed-size CompositeVideoClip canvas that clips
    every frame to (width, height) -- see module docstring for why this replaces vfx.Crop here."""
    base = ImageClip(path)
    cover_scale = max(width / base.w, height / base.h)
    zoom_span = min(KEN_BURNS_MAX_ZOOM - 1.0, KEN_BURNS_RATE_PER_SECOND * duration)
    start_zoom, end_zoom = (1.0, 1.0 + zoom_span) if zoom_in else (1.0 + zoom_span, 1.0)

    def scale_at(t: float) -> float:
        progress = (t / duration) if duration > 0 else 0.0
        return cover_scale * (start_zoom + (end_zoom - start_zoom) * progress)

    moving = base.resized(scale_at).with_position("center").with_duration(duration)
    return CompositeVideoClip([moving], size=(width, height)).with_duration(duration)


# Pastel backdrop palette the mascot sits on, cycled by beat index -- gives beats visual variety
# without falling back to a per-beat AI-generated backdrop image.
_BACKDROP_PALETTE = [
    (255, 214, 165),  # peach
    (168, 218, 220),  # sky
    (255, 209, 220),  # blush
    (202, 240, 187),  # mint
    (255, 241, 168),  # butter
]

_TALK_FLAP_PERIOD = 0.22  # seconds per mouth open/closed toggle -- fast enough to read as talking
_BOUNCE_PERIOD = 1.6
_BOUNCE_AMPLITUDE_PX = 18
_BLINK_EVERY = 3.2
_BLINK_DURATION = 0.15
_WAVE_DURATION = 1.0  # mascot waves hello for the first second of each beat
_SPRITE_HEIGHT_FRACTION = 0.6


def _character_frame_factory(pack: dict[str, str], width: int, height: int, backdrop_rgb: tuple[int, int, int]):
    """Builds a moviepy make_frame(t) closure: a small state machine picks a pose per t (idle /
    talk / blink / wave), composited with a vertical bounce onto a flat-color backdrop. This is
    genuine per-frame motion, not a moving crop of a single image."""
    poses = {}
    sprite_h = int(height * _SPRITE_HEIGHT_FRACTION)
    for name, path in pack.items():
        img = Image.open(path).convert("RGBA")
        scale = sprite_h / img.height
        poses[name] = img.resize((max(1, int(img.width * scale)), sprite_h))

    backdrop = Image.new("RGBA", (width, height), (*backdrop_rgb, 255))

    def make_frame(t: float) -> np.ndarray:
        if t < _WAVE_DURATION:
            pose = "wave"
        elif (t % _BLINK_EVERY) < _BLINK_DURATION:
            pose = "blink"
        else:
            pose = "talk" if int(t / _TALK_FLAP_PERIOD) % 2 == 0 else "base"

        sprite = poses[pose]
        bounce = int(_BOUNCE_AMPLITUDE_PX * abs(math.sin(math.pi * t / _BOUNCE_PERIOD)))
        x = (width - sprite.width) // 2
        y = height - sprite.height - int(height * 0.05) - bounce

        frame = backdrop.copy()
        frame.alpha_composite(sprite, (x, y))
        return np.array(frame.convert("RGB"))

    return make_frame


def _character_clip_for_beat(beat_index: int, duration: float, width: int = TARGET_W, height: int = TARGET_H):
    pack = character_assets_tool.get_character_pack()
    backdrop_rgb = _BACKDROP_PALETTE[beat_index % len(_BACKDROP_PALETTE)]
    make_frame = _character_frame_factory(pack, width, height, backdrop_rgb)
    return VideoClip(make_frame, duration=duration)


def _clip_for_beat(asset: SourcedAsset | None, duration: float):
    """Returns (clip, raw_video_handle_or_None) — the raw handle is tracked separately for
    cleanup since resize/crop wrap it in a derived clip that no longer exposes .close()."""
    if asset is not None and asset.asset_type == AssetType.CHARACTER_ANIMATION:
        return _character_clip_for_beat(asset.beat_index, duration), None

    if asset is None or not asset.local_path or not os.path.exists(asset.local_path):
        return ColorClip(size=(TARGET_W, TARGET_H), color=(20, 20, 20)).with_duration(duration), None

    if asset.asset_type == AssetType.STOCK_VIDEO:
        raw = VideoFileClip(asset.local_path)
        source = raw.subclipped(0, duration) if raw.duration >= duration else raw.with_effects([vfx.Loop(duration=duration)])
        return _resize_to_cover(source).with_duration(duration), raw

    # alternate zoom direction per beat so consecutive stills don't all push in the same way
    return _ken_burns_image_clip(asset.local_path, duration, zoom_in=asset.beat_index % 2 == 0), None


def video_assembly_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_ASSEMBLY, "start")

    script = ScriptOutput.model_validate(state["script_output"])
    voiceover = VoiceoverOutput.model_validate(state["voiceover_output"])
    asset_output = AssetOutput.model_validate(state["asset_output"])
    run_id = state["run_id"]

    segments_by_beat = {s.beat_index: s for s in voiceover.segments}
    assets_by_beat = {a.beat_index: a for a in asset_output.assets}

    beat_clips = []
    video_handles = []
    for beat in script.beats:
        duration = segments_by_beat[beat.index].duration_seconds if beat.index in segments_by_beat else 3.0
        clip, raw_handle = _clip_for_beat(assets_by_beat.get(beat.index), duration)
        beat_clips.append(clip)
        if raw_handle is not None:
            video_handles.append(raw_handle)

    video_no_audio = concatenate_videoclips(beat_clips, method="compose")
    full_audio = AudioFileClip(voiceover.full_audio_path)
    video = video_no_audio.with_audio(full_audio)

    has_captions = False
    try:
        font_path = resolve_font_path()
        caption_clips = []
        t_cursor = 0.0
        for beat, clip in zip(script.beats, beat_clips, strict=False):
            caption = TextClip(
                font=font_path,
                text=beat.voiceover_text,
                font_size=48,
                color="white",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(int(TARGET_W * 0.85), None),
            )
            caption = caption.with_duration(clip.duration).with_start(t_cursor).with_position(("center", "bottom"))
            caption_clips.append(caption)
            t_cursor += clip.duration
        if caption_clips:
            video = CompositeVideoClip([video, *caption_clips])
            has_captions = True
    except Exception as exc:
        logger.warning("video_assembly.captions_failed", error=str(exc))

    total_duration = video.duration
    render_path = media_path(run_id, "render", "final.mp4")
    try:
        video.write_videofile(render_path, fps=TARGET_FPS, codec="libx264", audio_codec="aac", logger=None)
    finally:
        full_audio.close()
        for handle in video_handles:
            handle.close()
        video.close()

    output = AssemblyOutput(
        render_path=render_path,
        duration_seconds=total_duration,
        resolution=f"{TARGET_W}x{TARGET_H}",
        fps=TARGET_FPS,
        has_captions=has_captions,
        has_music=False,  # background-music sourcing not implemented
    )

    return {
        "assembly_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_ASSEMBLY, "complete", render_path=output.render_path, duration=output.duration_seconds)],
    }
