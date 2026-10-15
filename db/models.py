"""
Postgres schema for AutoTube AI.

Accounts: User, UserCredential (each user brings their own Google Cloud OAuth client).
Core tables (per spec): Run, Video, AgentLog, UploadHistory, Cost.
Supporting tables: PerformanceSnapshot (closes the analytics -> strategy feedback loop),
ScriptEmbedding / TranscriptEmbedding (pgvector originality memory for the Script QA critic
and the Learning Agent).
"""
import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from agents.schemas.common import ContentFormat, ContentType, PipelineStage, RunStatus
from agents.schemas.publish import Visibility
from db.base import Base

EMBEDDING_DIM = 768  # matches gemini-embedding-001 (free tier) / text-embedding-3-small truncated
# via `dimensions=768` -- see tools/embeddings.py and migration 0004 for why both providers write
# into this same shape.


def _str_enum(enum_cls, name: str) -> SAEnum:
    """SAEnum defaults to binding a Python Enum member's .name (e.g. "AWAITING_APPROVAL"); our
    schemas.common enums are StrEnum, so the DB column (and the enum type created in the Alembic
    migration) uses lowercase .value strings ("awaiting_approval") instead — values_callable
    keeps the two in sync."""
    return SAEnum(enum_cls, name=name, values_callable=lambda obj: [e.value for e in obj])


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class User(Base):
    """An account. Email is the login identity and is stored lowercased to keep it unique in
    practice as well as in the constraint -- people type their own address inconsistently."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    credential: Mapped["UserCredential | None"] = relationship(
        back_populates="user", cascade="all, delete-orphan", uselist=False
    )
    runs: Mapped[list["Run"]] = relationship(back_populates="user")


class UserCredential(Base):
    """One user's own Google Cloud OAuth client and the YouTube channel they authorised with it.

    Users bring their own credentials rather than authorising a central app. That decision is
    what keeps this deployable: YouTube's 10,000 units/day quota is allocated per Google Cloud
    project, so a shared app would give every user combined about four uploads a day, and
    uploading to someone else's channel is a restricted scope that caps an unverified app at 100
    users for the lifetime of the project. Per-user credentials sidestep both, and mean a breach
    of this table does not hand over one central token that reaches every channel.

    Every secret column holds Fernet ciphertext from core.crypto -- never plaintext. The columns
    are sized for ciphertext, which is substantially longer than the value it wraps.
    """

    __tablename__ = "user_credentials"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True
    )
    google_client_id: Mapped[str] = mapped_column(Text, default="")        # encrypted
    google_client_secret: Mapped[str] = mapped_column(Text, default="")    # encrypted
    youtube_refresh_token: Mapped[str] = mapped_column(Text, default="")   # encrypted
    youtube_channel_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    youtube_channel_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # Set when a refresh last failed, so the UI can say "reconnect your channel" before a run
    # spends twenty minutes rendering a video it will not be able to upload.
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    connected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="credential")


class ChatMessage(Base):
    """One turn of the studio conversation, so it survives a page change.

    The chat was browser-memory only: clicking through to Ops and back lost everything, including
    the review card for a finished video waiting on a decision. Stored per account rather than in
    localStorage because the conversation belongs to the account, not to the browser -- two people
    sharing a machine must not read each other's, and the same account on a phone should see the
    same history.

    `review` holds the card payload that accompanied an assistant turn, so a reloaded page can
    re-render the player and the approve/change buttons instead of a bare sentence about a video
    the operator can no longer see.
    """

    __tablename__ = "chat_messages"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(32), nullable=True)
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
    )
    review: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Run(Base):
    """One row per pipeline execution. The single source of truth for resumability."""

    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    # Nullable on purpose: runs created before accounts existed have no owner, and inventing one
    # for them would misreport whose channel they went to. Code that lists runs for a user filters
    # on this, so an ownerless row is simply invisible to every account rather than visible to all.
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[RunStatus] = mapped_column(_str_enum(RunStatus, "run_status"), default=RunStatus.PENDING, index=True)
    current_stage: Mapped[PipelineStage | None] = mapped_column(_str_enum(PipelineStage, "pipeline_stage"), nullable=True)
    stage_status: Mapped[dict] = mapped_column(JSON, default=dict, doc="stage_name -> {status, retry_count, updated_at}")

    content_type: Mapped[ContentType | None] = mapped_column(_str_enum(ContentType, "content_type"), nullable=True)
    content_format: Mapped[ContentFormat | None] = mapped_column(_str_enum(ContentFormat, "content_format"), nullable=True)
    selected_topic: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    strategy_rationale: Mapped[str | None] = mapped_column(Text, nullable=True)

    langgraph_thread_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped["User | None"] = relationship(back_populates="runs")
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
    visibility: Mapped[Visibility | None] = mapped_column(_str_enum(Visibility, "visibility"), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped["Run"] = relationship(back_populates="videos")
    performance_snapshots: Mapped[list["PerformanceSnapshot"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )


class AgentLog(Base):
    """One row per agent/critic invocation. The observability trace: what ran, what it returned, why a critic rejected it."""

    __tablename__ = "agent_logs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)

    stage: Mapped[PipelineStage] = mapped_column(_str_enum(PipelineStage, "pipeline_stage_log"))
    agent_name: Mapped[str] = mapped_column(String(100))
    input_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    output_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    is_critic: Mapped[bool] = mapped_column(Boolean, default=False)
    critic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)

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

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped["Run"] = relationship(back_populates="uploads")


class Cost(Base):
    """Per-stage spend so cost-per-video is always knowable."""

    __tablename__ = "costs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)

    stage: Mapped[PipelineStage] = mapped_column(_str_enum(PipelineStage, "pipeline_stage_cost"))
    provider: Mapped[str] = mapped_column(String(50))
    unit_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    units: Mapped[float] = mapped_column(Float, default=0.0)
    total_usd: Mapped[float] = mapped_column(Float, default=0.0)
    cost_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

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

    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ScriptEmbedding(Base):
    """Embeddings of our own published/generated scripts — self-originality check across our own
    back catalog, and feedstock for the Learning Agent.

    run_id here is the LangGraph thread_id (Run.langgraph_thread_id), not Run.id — graph nodes
    (tools/embeddings.py, called from critic_script_qa_node) only ever see PipelineState["run_id"],
    which is populated from the thread_id, never the real Run.id primary key (see graph/run.py's
    run_id/thread_id conflation). FK targets Run.langgraph_thread_id (unique-indexed) accordingly.
    """

    __tablename__ = "script_embeddings"

    id: Mapped[uuid.UUID] = _uuid_pk()
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.langgraph_thread_id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    text_chunk: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
