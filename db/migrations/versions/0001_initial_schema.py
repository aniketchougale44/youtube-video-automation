"""initial schema: runs, videos, agent_logs, upload_history, costs, performance_snapshots, embeddings

Revision ID: 0001
Revises:
Create Date: 2026-08-05

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql as pg

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIM = 1536

RUN_STATUS = sa.Enum(
    "pending", "running", "awaiting_approval", "approved", "rejected",
    "published", "failed", "escalated", name="run_status",
)
PIPELINE_STAGE = sa.Enum(
    "trend_research", "strategy", "script_writer", "critic_script_qa", "fact_check",
    "visual_planning", "asset_visual", "voiceover", "video_assembly", "metadata_seo",
    "thumbnail", "critic_compliance", "human_approval", "upload", "performance_monitor",
    "learning", name="pipeline_stage",
)
PIPELINE_STAGE_LOG = sa.Enum(
    "trend_research", "strategy", "script_writer", "critic_script_qa", "fact_check",
    "visual_planning", "asset_visual", "voiceover", "video_assembly", "metadata_seo",
    "thumbnail", "critic_compliance", "human_approval", "upload", "performance_monitor",
    "learning", name="pipeline_stage_log",
)
PIPELINE_STAGE_COST = sa.Enum(
    "trend_research", "strategy", "script_writer", "critic_script_qa", "fact_check",
    "visual_planning", "asset_visual", "voiceover", "video_assembly", "metadata_seo",
    "thumbnail", "critic_compliance", "human_approval", "upload", "performance_monitor",
    "learning", name="pipeline_stage_cost",
)
CONTENT_TYPE = sa.Enum("trending_news", "evergreen", name="content_type")
CONTENT_FORMAT = sa.Enum("short", "long_form", name="content_format")
VISIBILITY = sa.Enum("private", "unlisted", "public", name="visibility")


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

    # Enum types are intentionally NOT pre-created here: each sa.Enum column below auto-creates
    # its Postgres ENUM type as part of that table's DDL (with its own checkfirst). Pre-creating
    # them separately causes a duplicate CREATE TYPE when create_table() creates it again.

    op.create_table(
        "runs",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", RUN_STATUS, nullable=False, server_default="pending"),
        sa.Column("current_stage", PIPELINE_STAGE, nullable=True),
        sa.Column("stage_status", pg.JSONB, nullable=False, server_default="{}"),
        sa.Column("content_type", CONTENT_TYPE, nullable=True),
        sa.Column("content_format", CONTENT_FORMAT, nullable=True),
        sa.Column("selected_topic", pg.JSONB, nullable=True),
        sa.Column("strategy_rationale", sa.Text, nullable=True),
        sa.Column("langgraph_thread_id", sa.String(64), nullable=False),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_runs_status", "runs", ["status"])
    op.create_index("ix_runs_langgraph_thread_id", "runs", ["langgraph_thread_id"], unique=True)

    op.create_table(
        "videos",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", pg.UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("tags", pg.JSONB, nullable=True),
        sa.Column("script_json", pg.JSONB, nullable=True),
        sa.Column("render_path", sa.String(500), nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=True),
        sa.Column("thumbnail_path", sa.String(500), nullable=True),
        sa.Column("youtube_video_id", sa.String(32), nullable=True),
        sa.Column("visibility", VISIBILITY, nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_videos_run_id", "videos", ["run_id"])
    op.create_index("ix_videos_youtube_video_id", "videos", ["youtube_video_id"])

    op.create_table(
        "agent_logs",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", pg.UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage", PIPELINE_STAGE_LOG, nullable=False),
        sa.Column("agent_name", sa.String(100), nullable=False),
        sa.Column("input_json", pg.JSONB, nullable=True),
        sa.Column("output_json", pg.JSONB, nullable=True),
        sa.Column("is_critic", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("critic_score", sa.Float, nullable=True),
        sa.Column("passed", sa.Boolean, nullable=True),
        sa.Column("retry_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_agent_logs_run_id", "agent_logs", ["run_id"])
    op.create_index("ix_agent_logs_created_at", "agent_logs", ["created_at"])

    op.create_table(
        "upload_history",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", pg.UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("video_id", pg.UUID(as_uuid=True), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("attempt_number", sa.Integer, nullable=False, server_default="1"),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("youtube_video_id", sa.String(32), nullable=True),
        sa.Column("quota_units_used", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_upload_history_run_id", "upload_history", ["run_id"])
    op.create_index("ix_upload_history_video_id", "upload_history", ["video_id"])

    op.create_table(
        "costs",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", pg.UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage", PIPELINE_STAGE_COST, nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("unit_cost_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("units", sa.Float, nullable=False, server_default="0"),
        sa.Column("total_usd", sa.Float, nullable=False, server_default="0"),
        sa.Column("cost_metadata", pg.JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_costs_run_id", "costs", ["run_id"])

    op.create_table(
        "performance_snapshots",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("video_id", pg.UUID(as_uuid=True), sa.ForeignKey("videos.id", ondelete="CASCADE"), nullable=False),
        sa.Column("window", sa.String(8), nullable=False),
        sa.Column("views", sa.Integer, nullable=False, server_default="0"),
        sa.Column("impressions", sa.Integer, nullable=False, server_default="0"),
        sa.Column("ctr", sa.Float, nullable=False, server_default="0"),
        sa.Column("avg_view_duration_seconds", sa.Float, nullable=False, server_default="0"),
        sa.Column("retention_pct", sa.Float, nullable=False, server_default="0"),
        sa.Column("likes", sa.Integer, nullable=False, server_default="0"),
        sa.Column("comments", sa.Integer, nullable=False, server_default="0"),
        sa.Column("captured_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_performance_snapshots_video_id", "performance_snapshots", ["video_id"])

    op.create_table(
        "transcript_embeddings",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_video_id", sa.String(32), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False, server_default="0"),
        sa.Column("text_chunk", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_transcript_embeddings_source_video_id", "transcript_embeddings", ["source_video_id"])
    op.execute(
        "CREATE INDEX ix_transcript_embeddings_embedding ON transcript_embeddings "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    op.create_table(
        "script_embeddings",
        sa.Column("id", pg.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", pg.UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer, nullable=False, server_default="0"),
        sa.Column("text_chunk", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_script_embeddings_run_id", "script_embeddings", ["run_id"])
    op.execute(
        "CREATE INDEX ix_script_embeddings_embedding ON script_embeddings "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )


def downgrade() -> None:
    op.drop_table("script_embeddings")
    op.drop_table("transcript_embeddings")
    op.drop_table("performance_snapshots")
    op.drop_table("costs")
    op.drop_table("upload_history")
    op.drop_table("agent_logs")
    op.drop_table("videos")
    op.drop_table("runs")

    bind = op.get_bind()
    for enum in (VISIBILITY, CONTENT_FORMAT, CONTENT_TYPE, PIPELINE_STAGE_COST, PIPELINE_STAGE_LOG, PIPELINE_STAGE, RUN_STATUS):
        enum.drop(bind, checkfirst=True)
