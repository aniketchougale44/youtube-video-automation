"""Trend Research Agent and Strategy Agent I/O."""
from datetime import datetime

from pydantic import BaseModel, Field

from agents.schemas.common import ContentFormat, ContentType, utcnow


class TopicCandidate(BaseModel):
    title: str
    description: str = ""
    category: str = ""
    search_volume_score: float = Field(ge=0, le=10)
    competition_score: float = Field(ge=0, le=10, description="Higher = more saturated")
    freshness_score: float = Field(ge=0, le=10)
    composite_score: float = Field(ge=0, le=10)
    source_video_ids: list[str] = Field(default_factory=list, description="Existing videos used only as trend signal, never as source material")


class TrendResearchInput(BaseModel):
    channel_id: str
    categories: list[str] = Field(default_factory=list)
    region_code: str = "US"
    max_candidates: int = 10


class TrendResearchOutput(BaseModel):
    candidates: list[TopicCandidate]
    researched_at: datetime = Field(default_factory=utcnow)


class StrategyInput(BaseModel):
    trend_output: TrendResearchOutput
    channel_goals: str = ""
    past_performance_summary: str = ""


class StrategyDecision(BaseModel):
    selected_topic: TopicCandidate
    content_type: ContentType
    content_format: ContentFormat
    target_length_seconds: int
    rationale: str
    decided_at: datetime = Field(default_factory=utcnow)
