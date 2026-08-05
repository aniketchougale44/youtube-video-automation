"""Voiceover Agent + Video Assembly Agent — stub logic."""
from agents.schemas.audio_render import AssemblyOutput, VoiceoverOutput, VoiceoverSegment
from agents.schemas.script import ScriptOutput
from graph.nodes._helpers import log_and_trace
from graph.state import PipelineState

STAGE_VOICE = "voiceover"
STAGE_ASSEMBLY = "video_assembly"


def voiceover_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_VOICE, "start")

    script = ScriptOutput.model_validate(state["script_output"])

    # STUB: real logic calls tools.tts (ElevenLabs/Azure) per beat and concatenates
    segments = [
        VoiceoverSegment(beat_index=b.index, audio_path=f"/tmp/stub_vo_{b.index}.mp3", duration_seconds=5.0, text=b.voiceover_text)
        for b in script.beats
    ]
    output = VoiceoverOutput(
        segments=segments,
        full_audio_path="/tmp/stub_vo_full.mp3",
        total_duration_seconds=sum(s.duration_seconds for s in segments),
    )

    return {
        "voiceover_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_VOICE, "complete", segments=len(segments))],
    }


def video_assembly_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_ASSEMBLY, "start")

    script = ScriptOutput.model_validate(state["script_output"])

    # STUB: real logic stitches asset_output + voiceover_output via MoviePy/FFmpeg with caption burn-in
    output = AssemblyOutput(
        render_path="/tmp/stub_render.mp4",
        duration_seconds=float(script.target_length_seconds),
        resolution="1920x1080",
        fps=30,
        has_captions=True,
        has_music=True,
    )

    return {
        "assembly_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_ASSEMBLY, "complete", render_path=output.render_path)],
    }
