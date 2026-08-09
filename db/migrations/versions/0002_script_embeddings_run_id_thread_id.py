"""script_embeddings.run_id actually holds the LangGraph thread_id, not runs.id

Graph nodes only ever see PipelineState["run_id"], which worker/tasks.py populates from
run.langgraph_thread_id (see graph/run.py::start_run's run_id/thread_id conflation) — they never
have access to the real runs.id primary key, which is only known outside the graph (worker/tasks.py,
db/crud.py). script_embeddings.run_id was originally typed as a UUID FK to runs.id, which any real
write from critic_script_qa_node violates (its run_id value is a thread_id string, not a runs.id).
Retargets the FK at runs.langgraph_thread_id (unique-indexed, a valid FK target) instead.

Revision ID: 0002
Revises: 0001
Create Date: 2026-08-09

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("script_embeddings_run_id_fkey", "script_embeddings", type_="foreignkey")
    op.alter_column(
        "script_embeddings",
        "run_id",
        type_=sa.String(64),
        postgresql_using="run_id::text",
    )
    op.create_foreign_key(
        "script_embeddings_run_id_fkey",
        "script_embeddings",
        "runs",
        ["run_id"],
        ["langgraph_thread_id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("script_embeddings_run_id_fkey", "script_embeddings", type_="foreignkey")
    op.alter_column(
        "script_embeddings",
        "run_id",
        type_=sa.dialects.postgresql.UUID(as_uuid=True),
        postgresql_using="run_id::uuid",
    )
    op.create_foreign_key(
        "script_embeddings_run_id_fkey", "script_embeddings", "runs", ["run_id"], ["id"], ondelete="CASCADE"
    )
