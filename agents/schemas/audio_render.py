"""Voiceover Agent and Video Assembly Agent I/O."""
from datetime import datetime

from pydantic import BaseModel, Field

from agents.schemas.common import utcnow
from agents.schemas.script import ScriptOutput
from agents.schemas.visual import AssetOutput


class VoiceoverInput(BaseModel):
    script: ScriptOutput


class VoiceoverSegment(BaseModel):
    beat_index: int
    audio_path: str
    duration_seconds: float
    text: str


class VoiceoverOutput(BaseModel):
    segments: list[VoiceoverSegment]
    full_audio_path: str
    total_duration_seconds: float
    generated_at: datetime = Field(default_factory=utcnow)


class VideoAssemblyInput(BaseModel):
    script: ScriptOutput
    assets: AssetOutput
    voiceover: VoiceoverOutput
    background_music_path: str | None = None
    burn_in_captions: bool = True


class AssemblyOutput(BaseModel):
    render_path: str
    duration_seconds: float
    resolution: str
    fps: int
    has_captions: bool
    has_music: bool
    assembled_at: datetime = Field(default_factory=utcnow)
