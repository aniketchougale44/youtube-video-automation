"""Shrink embedding columns from vector(1536) to vector(768)

tools/embeddings.py now falls back to Google's free-tier gemini-embedding-001 API whenever
OPENAI_API_KEY isn't set or its billing is exhausted (mirroring the free-tier fallback already
added for the LLM, TTS, and image-gen providers). Gemini's embedding model outputs 768 dimensions;
OpenAI's text-embedding-3-small is now requested at `dimensions=768` too (its Matryoshka
truncation support) so both providers write into the same column shape and stay comparable via
cosine distance regardless of which one produced a given row.

Safe to run against this project's actual data: both embedding tables were only ever populated by
the OpenAI-only code path, which was blocked (no billing credits) for this deployment's entire
history, so there are no live rows to reconcile. If you're running this migration somewhere with
real 1536-dim rows already stored, you'll need to re-embed them at 768 dims first -- pgvector
can't reinterpret existing vector bytes at a new dimensionality.

Revision ID: 0004
Revises: 0003
Create Date: 2026-08-14

"""
from collections.abc import Sequence

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE script_embeddings ALTER COLUMN embedding TYPE vector(768)")
    op.execute("ALTER TABLE transcript_embeddings ALTER COLUMN embedding TYPE vector(768)")


def downgrade() -> None:
    op.execute("ALTER TABLE script_embeddings ALTER COLUMN embedding TYPE vector(1536)")
    op.execute("ALTER TABLE transcript_embeddings ALTER COLUMN embedding TYPE vector(1536)")
