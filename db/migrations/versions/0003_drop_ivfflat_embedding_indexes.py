"""Drop the ivfflat indexes on script_embeddings/transcript_embeddings.embedding

ivfflat is an approximate-nearest-neighbor index whose clustering quality depends on having
substantial data present at CREATE INDEX time. Both indexes were created in 0001 against empty
tables, producing a degenerate index that returns zero matches on ORDER BY ... LIMIT queries even
once real rows exist (confirmed live: Postgres itself warns "ivfflat index created with little
data ... This will cause low recall ... Drop the index until the table has more data" — exactly
what happened when the Script QA originality check went live and reported no similarity matches
against clearly-present rows).

At this project's actual scale (one channel's own script/transcript corpus — realistically low
thousands of rows at most), an exact sequential-scan cosine distance is both simpler and always
correct; ivfflat's approximate speedup only pays for itself at a scale this project isn't at.
Revisit (re-add an ivfflat or hnsw index, sized for the data actually present) if the corpus ever
grows large enough for a full scan to matter.

Revision ID: 0003
Revises: 0002
Create Date: 2026-08-09

"""
from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_script_embeddings_embedding", table_name="script_embeddings")
    op.drop_index("ix_transcript_embeddings_embedding", table_name="transcript_embeddings")


def downgrade() -> None:
    op.execute(
        "CREATE INDEX ix_script_embeddings_embedding ON script_embeddings "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
    op.execute(
        "CREATE INDEX ix_transcript_embeddings_embedding ON transcript_embeddings "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
