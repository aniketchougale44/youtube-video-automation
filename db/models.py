"""
Postgres schema for AutoTube AI.

Core tables (per spec): Run, Video, AgentLog, UploadHistory, Cost.
Supporting tables: PerformanceSnapshot (closes the analytics -> strategy feedback loop),
ScriptEmbedding / TranscriptEmbedding (pgvector originality memory for the Script QA critic
and the Learning Agent).
"""
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agents.schemas.common import ContentFormat, ContentType, PipelineStage, RunStatus
from agents.schemas.publish import Visibility
from db.base import Base

EMBEDDING_DIM = 1536  # matches text-embedding-3-small / voyage-3-lite; adjust if the embedding model changes


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Run(Base):
    """One row per pipeline execution. The single source of truth for resumability."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    status: Mapped[RunStatus] = mapped_column(SAEnum(RunStatus, name="run_status"), default=RunStatus.PENDING, index=True)
    current_stage: Mapped[PipelineStage | None] = mapped_column(SAEnum(PipelineStage, name="pipeline_stage"), nullable=True)
    stage_status: Mapped[dict] = mapped_column(JSON, default=dict, doc="stage_name -> {status, retry_count, updated_at}")

    content_type: Mapped[ContentType | None] = mapped_column(SAEnum(ContentType, name="content_type"), nullable=True)
    content_format: Mapped[ContentFormat | None] = mapped_column(SAEnum(ContentFormat, name="content_format"), nullable=True)
    selected_topic: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    strategy_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)

    langgraph_thread_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()", onupdate="now()")

    videos: Mapped[list["Video"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    agent_logs: Mapped[list["AgentLog"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    uploads: Mapped[list["UploadHistory"]] = relationship(back_populates="run", cascade="all, delete-orphan")
    costs: Mapped[list["Cost"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Video(Base):
    """The produced artifact for a run: script, render, thumbnail, metadata."""

    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)

    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)
    script_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    render_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    youtube_video_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    visibility: Mapped[Visibility | None] = mapped_column(SAEnum(Visibility, name="visibility"), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")

    run: Mapped["Run"] = relationship(back_populates="videos")
    performance_snapshots: Mapped[list["PerformanceSnapshot"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )


class AgentLog(Base):
    """One row per agent/critic invocation. The observability trace: what ran, what it returned, why a critic rejected it."""

    __tablename__ = "agent_logs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)

    stage: Mapped[PipelineStage] = mapped_column(SAEnum(PipelineStage, name="pipeline_stage_log"))
    agent_name: Mapped[str] = mapped_column(String(100))
    input_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    is_critic: Mapped[bool] = mapped_column(Boolean, default=False)
    critic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()", index=True)

    run: Mapped["Run"] = relationship(back_populates="agent_logs")


class UploadHistory(Base):
    """Every upload attempt (including quota-deferred and failed ones) — the idempotency ledger that prevents double-publish."""

    __tablename__ = "upload_history"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)

    attempt_number: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(32))
    youtube_video_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    quota_units_used: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")

    run: Mapped["Run"] = relationship(back_populates="uploads")


class Cost(Base):
    """Per-stage spend so cost-per-video is always knowable."""

    __tablename__ = "costs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)

    stage: Mapped[PipelineStage] = mapped_column(SAEnum(PipelineStage, name="pipeline_stage_cost"))
    provider: Mapped[str] = mapped_column(String(50))
    unit_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    units: Mapped[float] = mapped_column(Float, default=0.0)
    total_usd: Mapped[float] = mapped_column(Float, default=0.0)
    cost_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")

    run: Mapped["Run"] = relationship(back_populates="costs")


class PerformanceSnapshot(Base):
    """24h/7d Analytics API pulls, feeding the Strategy Agent's future decisions."""

    __tablename__ = "performance_snapshots"

    id: Mapped[uuid.UUID] = _uuid_pk()
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)

    window: Mapped[str] = mapped_column(String(8))  # "24h" | "7d"
    views: Mapped[int] = mapped_column(Integer, default=0)
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    ctr: Mapped[float] = mapped_column(Float, default=0.0)
    avg_view_duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    retention_pct: Mapped[float] = mapped_column(Float, default=0.0)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    comments: Mapped[int] = mapped_column(Integer, default=0)

    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")

    video: Mapped["Video"] = relationship(back_populates="performance_snapshots")


class TranscriptEmbedding(Base):
    """Embeddings of transcripts from existing top-ranking videos, pulled purely as an originality
    reference corpus for the Script QA critic — never used as generation source material."""

    __tablename__ = "transcript_embeddings"

    id: Mapped[uuid.UUID] = _uuid_pk()
    source_video_id: Mapped[str] = mapped_column(String(32), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    text_chunk: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")


class ScriptEmbedding(Base):
    """Embeddings of our own published/generated scripts — self-originality check across our own
    back catalog, and feedstock for the Learning Agent."""

    __tablename__ = "script_embeddings"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    text_chunk: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default="now()")
