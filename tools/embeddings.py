"""Embedding + pgvector similarity helpers for the Script QA critic's originality gate.

Deliberate, narrow exception to the "graph nodes never touch the DB" convention described in
db/crud.py's docstring — that rule is about Run/AgentLog/Video *bookkeeping*, owned by
worker/tasks.py. ScriptEmbedding/TranscriptEmbedding are content data the critic needs to query as
part of its own decision logic, functionally an external tool call rather than run bookkeeping —
so this module owns its own SessionLocal session, the same self-contained-client pattern
tools/quota.py uses for its own Redis client.

TranscriptEmbedding (top-existing-video corpus) has no ingestion path yet — nothing populates it,
so most_similar_transcript() will return [] until that's built. This is a known, accepted gap:
originality checks degrade gracefully to self-only comparison (against our own ScriptEmbedding
back catalog) rather than erroring.

ScriptEmbedding.run_id stores the LangGraph thread_id (state["run_id"], the only run identifier
graph nodes ever see), not the Postgres Run.id primary key — see db/models.py's ScriptEmbedding
docstring. Treat it as an opaque string, not a uuid.UUID.
"""
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.logging import get_logger
from core.settings import get_settings
from db.base import SessionLocal
from db.models import EMBEDDING_DIM, ScriptEmbedding, TranscriptEmbedding
from tools.resilience import with_resilience

logger = get_logger("tools.embeddings")

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_CHARS = 4000


class EmbeddingsNotConfiguredError(RuntimeError):
    """Raised when neither OPENAI_API_KEY nor GOOGLE_API_KEY is set."""


def embed_text(text: str) -> list[float]:
    """OpenAI (paid, needs OPENAI_API_KEY) if configured -> Gemini's free-tier embedding model as
    the fallback (also used directly when no OpenAI key is set). Both are requested at
    EMBEDDING_DIM (768) so rows from either provider land in the same pgvector column shape and
    stay cosine-comparable regardless of which one produced them -- see migration 0004."""
    settings = get_settings()

    if settings.openai_api_key:
        try:
            return _openai_embed_call(text)
        except Exception as exc:
            # e.g. no billing credits, rate-limited, circuit open -- Gemini's embedding model has
            # a free tier, so a broken/unfunded paid key should degrade gracefully, not block the
            # Script QA originality gate entirely.
            logger.warning("embeddings.openai_failed_falling_back", error=str(exc))

    if not settings.google_api_key:
        raise EmbeddingsNotConfiguredError("neither OPENAI_API_KEY nor GOOGLE_API_KEY is set")
    return _gemini_embed_call(text)


@with_resilience(provider="openai_embeddings")
def _openai_embed_call(text: str) -> list[float]:
    from openai import OpenAI

    client = OpenAI(api_key=get_settings().openai_api_key)
    response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=text, dimensions=EMBEDDING_DIM)
    return response.data[0].embedding


@with_resilience(provider="gemini_embeddings")
def _gemini_embed_call(text: str) -> list[float]:
    from google import genai
    from google.genai import types

    settings = get_settings()
    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.embed_content(
        model=settings.gemini_embedding_model,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return response.embeddings[0].values


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    """Splits on paragraph boundaries, keeping chunks under max_chars. Typical script lengths
    (~1000-3000 chars) fit in a single chunk; this only matters for unusually long scripts."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()] or [text]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > max_chars:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


@contextmanager
def _session() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def most_similar_transcript(vector: list[float], limit: int = 5) -> list[tuple[str, float]]:
    """Returns [(source_video_id, similarity_score)], highest similarity first. [] if the corpus
    is empty — never raises on an empty table."""
    with _session() as db:
        stmt = (
            select(TranscriptEmbedding.source_video_id, TranscriptEmbedding.embedding.cosine_distance(vector))
            .order_by(TranscriptEmbedding.embedding.cosine_distance(vector))
            .limit(limit)
        )
        rows = db.execute(stmt).all()
    return [(source_id, round(1 - distance, 4)) for source_id, distance in rows]


def most_similar_script(vector: list[float], exclude_run_id: str | None = None, limit: int = 5) -> list[tuple[str, float]]:
    """Returns [(run_id, similarity_score)] against our own back catalog, highest similarity
    first. Excludes exclude_run_id so a run doesn't get compared against its own not-yet-final
    embedding on a re-check."""
    with _session() as db:
        stmt = select(ScriptEmbedding.run_id, ScriptEmbedding.embedding.cosine_distance(vector))
        if exclude_run_id:
            stmt = stmt.where(ScriptEmbedding.run_id != exclude_run_id)
        stmt = stmt.order_by(ScriptEmbedding.embedding.cosine_distance(vector)).limit(limit)
        rows = db.execute(stmt).all()
    return [(str(run_id), round(1 - distance, 4)) for run_id, distance in rows]


def store_script_embedding(run_id: str, full_text: str) -> None:
    """Called only after a script passes QA — builds the self-originality corpus future runs get
    checked against."""
    with _session() as db:
        for index, chunk in enumerate(chunk_text(full_text)):
            db.add(ScriptEmbedding(run_id=run_id, chunk_index=index, text_chunk=chunk, embedding=embed_text(chunk)))
        db.commit()
    logger.info("embeddings.script_stored", run_id=run_id)
