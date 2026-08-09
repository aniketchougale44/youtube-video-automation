"""Voiceover Agent + Video Assembly Agent — real logic.

video_assembly_node is the highest-complexity node in the pipeline and takes genuine wall-clock
time (a multi-minute long-form video plausibly takes 1-5 min locally to encode) — this is
qualitatively different from every other node here, which is just an API round trip.

Per-beat clips use a static "cover resize" (scale to fill the 1920x1080 frame, center-crop) rather
than a Ken Burns pan/zoom: a time-varying resize composed with a fixed-position crop was tested
and doesn't track a growing frame's center correctly with this moviepy version's Crop API (static
x_center/y_center, not callables) — a static held image per beat is simple and reliably correct,
a pan/zoom effect is a reasonable future addition once that's worth re-deriving carefully.
"""
import os

from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoFileClip,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
)

from agents.schemas.audio_render import AssemblyOutput, VoiceoverOutput, VoiceoverSegment
from agents.schemas.script import ScriptOutput
from agents.schemas.visual import AssetOutput, AssetType, SourcedAsset
from core.logging import get_logger
from graph.nodes._helpers import log_and_trace
from graph.state import PipelineState
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


def _clip_for_beat(asset: SourcedAsset | None, duration: float):
    """Returns (clip, raw_video_handle_or_None) — the raw handle is tracked separately for
    cleanup since resize/crop wrap it in a derived clip that no longer exposes .close()."""
    if asset is None or not asset.local_path or not os.path.exists(asset.local_path):
        return ColorClip(size=(TARGET_W, TARGET_H), color=(20, 20, 20)).with_duration(duration), None

    if asset.asset_type == AssetType.STOCK_VIDEO:
        raw = VideoFileClip(asset.local_path)
        source = raw.subclipped(0, duration) if raw.duration >= duration else raw.with_effects([vfx.Loop(duration=duration)])
        return _resize_to_cover(source).with_duration(duration), raw

    return _resize_to_cover(ImageClip(asset.local_path)).with_duration(duration), None


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
