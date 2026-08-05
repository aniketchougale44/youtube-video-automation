"""Shared enums and base types used across every agent's Pydantic I/O models."""
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class ContentFormat(StrEnum):
    SHORT = "short"          # <= 60s vertical
    LONG_FORM = "long_form"


class ContentType(StrEnum):
    TRENDING_NEWS = "trending_news"
    EVERGREEN = "evergreen"


class PipelineStage(StrEnum):
    TREND_RESEARCH = "trend_research"
    STRATEGY = "strategy"
    SCRIPT_WRITER = "script_writer"
    CRITIC_SCRIPT_QA = "critic_script_qa"
    FACT_CHECK = "fact_check"
    VISUAL_PLANNING = "visual_planning"
    ASSET_VISUAL = "asset_visual"
    VOICEOVER = "voiceover"
    VIDEO_ASSEMBLY = "video_assembly"
    METADATA_SEO = "metadata_seo"
    THUMBNAIL = "thumbnail"
    CRITIC_COMPLIANCE = "critic_compliance"
    HUMAN_APPROVAL = "human_approval"
    UPLOAD = "upload"
    PERFORMANCE_MONITOR = "performance_monitor"
    LEARNING = "learning"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    FAILED = "failed"
    ESCALATED = "escalated"     # exceeded max critic retries, needs human intervention


class CriticVerdict(BaseModel):
    """Common shape every critic node returns before deciding whether to route forward or back."""

    stage: PipelineStage
    score: float = Field(ge=0, le=10, description="0-10 quality score")
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    feedback_for_retry: str | None = Field(
        default=None, description="Actionable feedback sent back to the worker node on rejection"
    )
    retry_count: int = 0
    evaluated_at: datetime = Field(default_factory=utcnow)


class StageCost(BaseModel):
    stage: PipelineStage
    provider: str
    unit_cost_usd: float = 0.0
    units: float = 0.0
    total_usd: float = 0.0
    metadata: dict = Field(default_factory=dict)
