"""Persist the studio conversation per account

The chat lived only in the browser's memory, so moving between Studio, Ops and Runs started it
from nothing -- including losing the review card for a finished video that was waiting on a
decision, which is the one message in the conversation that there is an action to take on.

Stored server-side rather than in localStorage: the conversation belongs to the account, not to
the browser. Two people sharing a machine must not read each other's, and the same account on
another device should see the same history.

review is JSON rather than a foreign key to anything: it is the packet as it was shown at the
time, so a reloaded page re-renders exactly what the operator was looking at, even if the run has
moved on since.

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-15

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "chat_messages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(32), nullable=True),
        # SET NULL, not CASCADE: deleting a run should not silently delete the conversation that
        # discussed it, which is often the only record of why it was asked for.
        sa.Column("run_id", UUID(as_uuid=True), sa.ForeignKey("runs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("review", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    # The only read this table serves is "this user's messages, oldest first".
    op.create_index("ix_chat_messages_user_created", "chat_messages", ["user_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_chat_messages_user_created", table_name="chat_messages")
    op.drop_table("chat_messages")
