"""SQLAlchemy engine/session setup. One engine for the whole app (API, worker, scheduler)."""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from core.settings import get_settings


class Base(DeclarativeBase):
    pass


_settings = get_settings()
engine = create_engine(_settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency / general-purpose context manager for a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
