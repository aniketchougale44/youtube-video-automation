"""Metadata/SEO, Thumbnail, Compliance Critic, Human Approval, and Upload Agent I/O."""
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from agents.schemas.audio_render import AssemblyOutput
from agents.schemas.common import PipelineStage, utcnow
from agents.schemas.script import ScriptOutput


class ChapterMarker(BaseModel):
    timestamp_seconds: float
    title: str


class MetadataInput(BaseModel):
    script: ScriptOutput


class MetadataOutput(BaseModel):
    title_options: list[str]
    selected_title: str
    description: str
    tags: list[str]
    chapters: list[ChapterMarker]
    pinned_comment: str = ""
    generated_at: datetime = Field(default_factory=utcnow)


class ThumbnailInput(BaseModel):
    script: ScriptOutput
    metadata: MetadataOutput


class ThumbnailCandidate(BaseModel):
    image_path: str
    prompt_used: str
    text_overlay: str = ""


class ThumbnailOutput(BaseModel):
    candidates: list[ThumbnailCandidate]
    generated_at: datetime = Field(default_factory=utcnow)


class ComplianceCheckInput(BaseModel):
    script: ScriptOutput
    assembly: AssemblyOutput
    metadata: MetadataOutput
    thumbnails: ThumbnailOutput


class ComplianceCheckResult(BaseModel):
    stage: PipelineStage = PipelineStage.CRITIC_COMPLIANCE
    copyright_risk_score: float = Field(ge=0, le=10, description="0 = no risk, 10 = certain infringement")
    community_guidelines_flags: list[str] = Field(default_factory=list)
    av_sync_ok: bool
    spec_check_ok: bool
    spec_check_notes: list[str] = Field(default_factory=list)
    passed: bool
    blocking_reasons: list[str] = Field(default_factory=list)
    retry_count: int = 0
    evaluated_at: datetime = Field(default_factory=utcnow)


class ApprovalPacket(BaseModel):
    run_id: str
    script_summary: str
    thumbnail_options: list[str] = Field(description="paths/urls")
    title: str
    description: str
    compliance_result: ComplianceCheckResult
    requested_at: datetime = Field(default_factory=utcnow)


class ApprovalDecision(BaseModel):
    run_id: str
    approved: bool
    selected_thumbnail_index: int = 0
    reviewer: str = ""
    notes: str = ""
    decided_at: datetime = Field(default_factory=utcnow)


class Visibility(StrEnum):
    PRIVATE = "private"
    UNLISTED = "unlisted"
    PUBLIC = "public"


class UploadInput(BaseModel):
    assembly: AssemblyOutput
    metadata: MetadataOutput
    selected_thumbnail_path: str
    visibility: Visibility = Visibility.UNLISTED


class UploadStatus(StrEnum):
    UPLOADED = "uploaded"
    FAILED = "failed"
    QUOTA_DEFERRED = "quota_deferred"


class UploadResult(BaseModel):
    youtube_video_id: str | None = None
    status: UploadStatus
    visibility: Visibility
    quota_units_used: int = 0
    error: str | None = None
    uploaded_at: datetime = Field(default_factory=utcnow)
