"""UploadHistory must actually be written, and must actually prevent a double publish.

The table, the model and the migration have existed since the initial schema with no code inserting
a row. That was worse than a missing feature: the table reads 0 rows whether or not a video
published, so anyone using it as evidence — including, during this project's first real publish,
the tooling checking whether the upload had worked — draws exactly the wrong conclusion. Its own
docstring calls it "the idempotency ledger that prevents double-publish", which it could not do
while empty.

These tests use an in-memory SQLite database rather than the real Postgres, so they run in CI
without infrastructure.
"""
import uuid

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from agents.schemas.common import RunStatus
from db.crud import record_upload_attempt, successful_upload_for_run
from db.models import Base, Run, UploadHistory, Video


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


@pytest.fixture
def run_and_video(db):
    run = Run(id=uuid.uuid4(), status=RunStatus.PENDING, langgraph_thread_id=str(uuid.uuid4()))
    db.add(run)
    db.commit()
    video = Video(id=uuid.uuid4(), run_id=run.id, title="t")
    db.add(video)
    db.commit()
    return run, video


def test_a_successful_upload_is_recorded(db, run_and_video):
    run, video = run_and_video
    entry = record_upload_attempt(
        db, run, video,
        {"youtube_video_id": "b7W3o5CtC-4", "status": "uploaded", "quota_units_used": 1600},
    )
    assert entry.attempt_number == 1
    assert entry.youtube_video_id == "b7W3o5CtC-4"
    assert db.execute(select(UploadHistory)).scalars().all()  # the table is no longer always empty


def test_a_failed_upload_is_recorded_too(db, run_and_video):
    """Failures are the rows that matter most -- they are the audit trail for a run that spent
    quota and produced nothing."""
    run, video = run_and_video
    entry = record_upload_attempt(
        db, run, video,
        {"youtube_video_id": None, "status": "failed", "quota_units_used": 0,
         "error": "render file missing"},
    )
    assert entry.status == "failed"
    assert entry.youtube_video_id is None
    assert "render file missing" in entry.error


def test_attempts_are_numbered(db, run_and_video):
    run, video = run_and_video
    for _ in range(3):
        record_upload_attempt(db, run, video, {"status": "failed", "quota_units_used": 0})
    numbers = sorted(e.attempt_number for e in db.execute(select(UploadHistory)).scalars())
    assert numbers == [1, 2, 3]


def test_replaying_the_same_success_does_not_duplicate(db, run_and_video):
    """A finished graph replays its terminal state on checkpointer resume, so the same successful
    upload_result reaches this function more than once without a second video being published."""
    run, video = run_and_video
    payload = {"youtube_video_id": "b7W3o5CtC-4", "status": "uploaded", "quota_units_used": 1600}

    first = record_upload_attempt(db, run, video, payload)
    second = record_upload_attempt(db, run, video, payload)

    assert first.id == second.id
    assert len(db.execute(select(UploadHistory)).scalars().all()) == 1


def test_successful_upload_lookup_is_what_blocks_a_republish(db, run_and_video):
    run, video = run_and_video
    assert successful_upload_for_run(db, run.id) is None, "nothing published yet"

    record_upload_attempt(db, run, video, {"status": "failed", "quota_units_used": 0, "error": "boom"})
    assert successful_upload_for_run(db, run.id) is None, (
        "a failed attempt must not block a retry -- that would strand the run permanently"
    )

    record_upload_attempt(
        db, run, video,
        {"youtube_video_id": "b7W3o5CtC-4", "status": "uploaded", "quota_units_used": 1600},
    )
    found = successful_upload_for_run(db, run.id)
    assert found is not None and found.youtube_video_id == "b7W3o5CtC-4"


def test_a_different_run_is_not_blocked(db, run_and_video):
    """The guard is per run, so publishing one video must not stop the next one."""
    run, video = run_and_video
    record_upload_attempt(
        db, run, video,
        {"youtube_video_id": "b7W3o5CtC-4", "status": "uploaded", "quota_units_used": 1600},
    )
    other = Run(id=uuid.uuid4(), status=RunStatus.PENDING, langgraph_thread_id=str(uuid.uuid4()))
    db.add(other)
    db.commit()
    assert successful_upload_for_run(db, other.id) is None
