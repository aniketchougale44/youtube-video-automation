"""Thin persistence helpers shared by the API layer and the Celery worker tasks. Keeps
Run/AgentLog/Video bookkeeping out of the graph nodes themselves (the graph only knows about
PipelineState; SQLAlchemy rows are an outer concern owned by worker/tasks.py)."""
import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agents.schemas.common import PipelineStage, RunStatus
from core.logging import get_logger
from db.models import AgentLog, Cost, PerformanceSnapshot, Run, UploadHistory, Video

logger = get_logger("db.crud")

CRITIC_STAGE_PREFIXES = ("critic_",)


def create_run(db: Session, debug_force_reject: dict | None = None) -> Run:
    run = Run(
        id=uuid.uuid4(),
        status=RunStatus.PENDING,
        langgraph_thread_id=str(uuid.uuid4()),
        stage_status={"debug_force_reject": debug_force_reject or {}},
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_run(db: Session, run_id: uuid.UUID) -> Run | None:
    return db.get(Run, run_id)


def list_runs(db: Session, status: RunStatus | None = None, limit: int = 50) -> list[Run]:
    stmt = select(Run).order_by(Run.created_at.desc()).limit(limit)
    if status is not None:
        stmt = stmt.where(Run.status == status)
    return list(db.execute(stmt).scalars())


def get_video_by_youtube_id(db: Session, youtube_video_id: str) -> Video | None:
    return db.execute(select(Video).where(Video.youtube_video_id == youtube_video_id)).scalar_one_or_none()


def compute_performance_baseline(db: Session, exclude_video_id: uuid.UUID | None = None) -> dict:
    """Averages views/retention across every existing PerformanceSnapshot (any window), excluding
    the video currently being scored -- this is the "baseline" the Learning Agent compares a new
    snapshot against. Returns sample_size=0 until at least one prior snapshot exists."""
    stmt = select(PerformanceSnapshot)
    if exclude_video_id is not None:
        stmt = stmt.where(PerformanceSnapshot.video_id != exclude_video_id)
    snapshots = list(db.execute(stmt).scalars())

    if not snapshots:
        return {"sample_size": 0, "avg_views": 0.0, "avg_retention_pct": 0.0}

    return {
        "sample_size": len(snapshots),
        "avg_views": sum(s.views for s in snapshots) / len(snapshots),
        "avg_retention_pct": sum(s.retention_pct for s in snapshots) / len(snapshots),
    }


def persist_performance_snapshot(db: Session, video_id: uuid.UUID, snapshot: dict) -> PerformanceSnapshot:
    row = PerformanceSnapshot(
        video_id=video_id,
        window=snapshot.get("window", "24h"),
        views=snapshot.get("views", 0),
        impressions=snapshot.get("impressions", 0),
        ctr=snapshot.get("ctr", 0.0),
        avg_view_duration_seconds=snapshot.get("avg_view_duration_seconds", 0.0),
        retention_pct=snapshot.get("retention_pct", 0.0),
        likes=snapshot.get("likes", 0),
        comments=snapshot.get("comments", 0),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def summarize_recent_performance(db: Session, limit: int = 5) -> str:
    """Human-readable digest of the most recent performance snapshots, fed into the Strategy
    Agent's prompt so it can weigh past results rather than deciding blind every run."""
    stmt = (
        select(PerformanceSnapshot, Video)
        .join(Video, PerformanceSnapshot.video_id == Video.id)
        .order_by(PerformanceSnapshot.captured_at.desc())
        .limit(limit)
    )
    rows = db.execute(stmt).all()
    if not rows:
        return "No prior performance data available yet."

    lines = [
        f"- \"{video.title}\" ({snapshot.window}): {snapshot.views} views, "
        f"{snapshot.ctr:.1%} CTR, {snapshot.retention_pct:.0f}% retention"
        for snapshot, video in rows
    ]
    return "Recent video performance:\n" + "\n".join(lines)


def latest_strategy_weight_adjustments(db: Session, lookback: int = 20) -> dict[str, float]:
    """The most recent non-empty `strategy_weight_adjustments` the Learning Agent produced
    (graph/nodes/feedback.py::learning_node), read back out of the AgentLog rows its trace is
    persisted to. Fed into strategy_node's prompt on the next run as an explicit numeric signal
    alongside the qualitative `past_performance_summary` text. Returns {} when the feedback graph
    has never run, or every recent run found performance in line with baseline (adjustments={})."""
    stmt = (
        select(AgentLog.output_json)
        .where(AgentLog.stage == PipelineStage.LEARNING)
        .order_by(AgentLog.created_at.desc())
        .limit(lookback)
    )
    for (payload,) in db.execute(stmt).all():
        if not isinstance(payload, dict) or payload.get("event") != "complete":
            continue
        adjustments = payload.get("adjustments") or {}
        if adjustments:
            return {k: float(v) for k, v in adjustments.items()}
    return {}


def persist_trace_as_agent_logs(db: Session, run_id: uuid.UUID, trace: list[dict]) -> None:
    """Best-effort mapping of graph trace events -> AgentLog rows. One row per event; real
    input/output payloads get attached once each node's real logic (not the stub) is wired in."""
    for evt in trace:
        stage_name = evt.get("stage", "unknown")
        try:
            stage_enum = PipelineStage(stage_name)
        except ValueError:
            continue  # non-pipeline trace events (e.g. "escalation", "human_approval" sub-events)
        db.add(
            AgentLog(
                run_id=run_id,
                stage=stage_enum,
                agent_name=stage_name,
                output_json=evt,
                is_critic=stage_name.startswith(CRITIC_STAGE_PREFIXES),
                created_at=datetime.now(UTC),
            )
        )
    db.commit()


def update_run_from_graph_result(db: Session, run: Run, result: dict, interrupted: bool) -> Run:
    run.stage_status = {**(run.stage_status or {}), "retry_counts": result.get("retry_counts", {})}
    run.error = None  # a graph result means this attempt got past whatever tripped a prior failed
    # attempt (worker/tasks.py marks run.error on exception) — clear it so the dashboard doesn't
    # show a stale error for a run that's actually healthy now; the escalated branch below re-sets it.

    if interrupted:
        run.status = RunStatus.AWAITING_APPROVAL
        run.current_stage = PipelineStage.HUMAN_APPROVAL
    elif result.get("escalated"):
        run.status = RunStatus.ESCALATED
        run.error = result.get("escalation_reason")
    elif result.get("upload_result"):
        run.status = RunStatus.PUBLISHED
        run.current_stage = PipelineStage.UPLOAD
    elif result.get("approval_decision") and not result["approval_decision"].get("approved"):
        run.status = RunStatus.REJECTED
    else:
        run.status = RunStatus.RUNNING

    if result.get("strategy_decision"):
        run.selected_topic = result["strategy_decision"].get("selected_topic")
        run.content_type = result["strategy_decision"].get("content_type")
        run.content_format = result["strategy_decision"].get("content_format")
        run.strategy_rationale = result["strategy_decision"].get("rationale")

    db.add(run)
    db.commit()
    db.refresh(run)

    if result.get("upload_result") or result.get("assembly_output"):
        video = _upsert_video(db, run, result)
        if result.get("upload_result"):
            record_upload_attempt(db, run, video, result["upload_result"])

    return run


def successful_upload_for_run(db: Session, run_id: uuid.UUID) -> UploadHistory | None:
    """The ledger row proving this run already published, if it did.

    This is what makes UploadHistory an idempotency ledger rather than an audit log: a caller can
    ask "did this run already put a video on the channel?" before spending 1600 quota units finding
    out. Graph nodes stay DB-free, so the check belongs to worker/tasks.py at the task boundary.
    """
    return db.execute(
        select(UploadHistory)
        .where(UploadHistory.run_id == run_id, UploadHistory.youtube_video_id.is_not(None))
        .order_by(UploadHistory.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()


def record_upload_attempt(db: Session, run: Run, video: Video, upload_result: dict) -> UploadHistory | None:
    """Appends one row per upload attempt -- successes, failures and quota deferrals alike.

    The table and model have existed since the initial migration with nothing writing to them,
    which was worse than merely missing: it reads 0 rows whether or not a video published, so it
    silently invited exactly the wrong conclusion from anyone using it as evidence.

    Re-running a completed run is a no-op rather than a duplicate row: the LangGraph checkpointer
    replays a finished graph's terminal state on resume, so the same successful upload_result can
    arrive here more than once without a second video ever being published.
    """
    youtube_video_id = upload_result.get("youtube_video_id")
    if youtube_video_id:
        already = db.execute(
            select(UploadHistory).where(
                UploadHistory.run_id == run.id,
                UploadHistory.youtube_video_id == youtube_video_id,
            )
        ).scalar_one_or_none()
        if already is not None:
            return already

    attempts = db.execute(
        select(func.count()).select_from(UploadHistory).where(UploadHistory.run_id == run.id)
    ).scalar() or 0

    entry = UploadHistory(
        id=uuid.uuid4(),
        run_id=run.id,
        video_id=video.id,
        attempt_number=attempts + 1,
        status=str(upload_result.get("status", "")),
        youtube_video_id=youtube_video_id,
        quota_units_used=int(upload_result.get("quota_units_used") or 0),
        error=(upload_result.get("error") or None),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    logger.info(
        "crud.upload_attempt_recorded",
        run_id=str(run.id),
        attempt=entry.attempt_number,
        status=entry.status,
        youtube_video_id=entry.youtube_video_id,
    )
    return entry


def _upsert_video(db: Session, run: Run, result: dict) -> Video:
    video = db.execute(select(Video).where(Video.run_id == run.id)).scalar_one_or_none()
    if video is None:
        video = Video(id=uuid.uuid4(), run_id=run.id)

    metadata = result.get("metadata_output") or {}
    assembly = result.get("assembly_output") or {}
    upload = result.get("upload_result") or {}
    thumbnails = result.get("thumbnail_output") or {}

    video.title = metadata.get("selected_title", video.title)
    video.description = metadata.get("description", video.description)
    video.tags = metadata.get("tags", video.tags)
    video.script_json = result.get("script_output", video.script_json)
    video.render_path = assembly.get("render_path", video.render_path)
    video.duration_seconds = assembly.get("duration_seconds", video.duration_seconds)
    if thumbnails.get("candidates"):
        video.thumbnail_path = thumbnails["candidates"][0].get("image_path")
    if upload.get("youtube_video_id"):
        video.youtube_video_id = upload["youtube_video_id"]
        video.visibility = upload.get("visibility")
        video.published_at = datetime.now(UTC)

    db.add(video)
    db.commit()
    db.refresh(video)
    return video


def persist_costs(db: Session, run_id: uuid.UUID, entries: list[dict]) -> int:
    """Writes tools.cost's collected entries into db.models.Cost. Returns the row count.

    Unknown stage strings are dropped rather than raising: `stage` is a PipelineStage enum column,
    and a cost recorded outside any node (a warm-up LLM call, a standalone script) has no valid
    stage. Losing one such row is a better outcome than a ValueError taking down the task that just
    finished producing a video.
    """
    valid = {p.value for p in PipelineStage}
    written = 0
    for e in entries:
        stage = e.get("stage")
        if stage not in valid:
            logger.warning("crud.cost_unknown_stage", stage=stage, provider=e.get("provider"))
            continue
        db.add(
            Cost(
                id=uuid.uuid4(),
                run_id=run_id,
                stage=PipelineStage(stage),
                provider=str(e.get("provider", ""))[:50],
                unit_cost_usd=float(e.get("unit_cost_usd") or 0.0),
                units=float(e.get("units") or 0.0),
                total_usd=float(e.get("total_usd") or 0.0),
                cost_metadata=e.get("metadata") or None,
            )
        )
        written += 1
    if written:
        db.commit()
    logger.info("crud.costs_persisted", run_id=str(run_id), rows=written,
                total_usd=round(sum(float(e.get("total_usd") or 0) for e in entries), 6))
    return written
