"""Performance Monitor, Learning Agent, and Notification Agent I/O."""
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from agents.schemas.common import PipelineStage, utcnow


class PerformanceWindow(StrEnum):
    HOUR_24 = "24h"
    DAY_7 = "7d"


class PerformanceSnapshot(BaseModel):
    youtube_video_id: str
    window: PerformanceWindow
    views: int = 0
    impressions: int = 0
    ctr: float = 0.0
    avg_view_duration_seconds: float = 0.0
    retention_pct: float = 0.0
    likes: int = 0
    comments: int = 0
    captured_at: datetime = Field(default_factory=utcnow)


class LearningUpdate(BaseModel):
    based_on_video_ids: list[str]
    insights: list[str]
    strategy_weight_adjustments: dict[str, float] = Field(default_factory=dict)
    prompt_adjustments: dict[str, str] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utcnow)


class NotificationSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationEvent(BaseModel):
    run_id: str | None = None
    stage: PipelineStage | None = None
    severity: NotificationSeverity
    message: str
    context: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
