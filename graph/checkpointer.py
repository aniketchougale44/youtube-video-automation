"""Checkpointer factory. Postgres-backed in real runs (so a crashed worker resumes from the last
completed node instead of restarting or double-uploading); in-memory for quick local/unit tests."""
from collections.abc import Iterator
from contextlib import contextmanager

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from core.settings import get_settings


def memory_checkpointer() -> MemorySaver:
    return MemorySaver()


@contextmanager
def postgres_checkpointer() -> Iterator[PostgresSaver]:
    settings = get_settings()
    conn_string = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
    with PostgresSaver.from_conn_string(conn_string) as saver:
        saver.setup()  # idempotent; creates langgraph's own checkpoint tables if missing
        yield saver
