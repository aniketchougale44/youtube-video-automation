"""Add accounts and per-user Google credentials, and give runs an owner

Turns a single-operator deployment into a multi-user one. Two tables and one column:

- users: email/password accounts.
- user_credentials: each user's OWN Google Cloud OAuth client and the YouTube channel they
  authorised with it, every secret stored as Fernet ciphertext from core.crypto. Users bring
  their own credentials because YouTube's 10,000 units/day quota is granted per Google Cloud
  project -- one shared app would give the whole userbase about four uploads a day between them
  -- and because uploading to another person's channel is a restricted scope that caps an
  unverified app at 100 users for the lifetime of the project.
- runs.user_id, nullable.

runs.user_id is deliberately left NULL for existing rows rather than backfilled. The runs already
in this database were started from the .env credentials before accounts existed; assigning them
to whichever account is created first would misreport whose channel they published to. Listing
for a user filters on user_id, so these rows become invisible to every account instead of
visible to all of them -- the safe direction. ON DELETE SET NULL for the same reason: deleting an
account must not delete the cost and performance history the pipeline learns from.

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-15

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(120), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "user_credentials",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        # Text, not String(n): these hold Fernet ciphertext, which is markedly longer than the
        # value inside it, and a too-short column would truncate a secret into silent corruption.
        sa.Column("google_client_id", sa.Text(), nullable=False, server_default=""),
        sa.Column("google_client_secret", sa.Text(), nullable=False, server_default=""),
        sa.Column("youtube_refresh_token", sa.Text(), nullable=False, server_default=""),
        sa.Column("youtube_channel_id", sa.String(64), nullable=True),
        sa.Column("youtube_channel_title", sa.String(255), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("connected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_user_credentials_user_id", "user_credentials", ["user_id"], unique=True)

    op.add_column("runs", sa.Column("user_id", UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        "fk_runs_user_id_users", "runs", "users", ["user_id"], ["id"], ondelete="SET NULL"
    )
    op.create_index("ix_runs_user_id", "runs", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_runs_user_id", table_name="runs")
    op.drop_constraint("fk_runs_user_id_users", "runs", type_="foreignkey")
    op.drop_column("runs", "user_id")
    op.drop_index("ix_user_credentials_user_id", table_name="user_credentials")
    op.drop_table("user_credentials")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")
