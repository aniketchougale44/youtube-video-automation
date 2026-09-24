"""Thin persistence helpers shared by the API layer and the Celery worker tasks. Keeps
Run/AgentLog/Video bookkeeping out of the graph nodes themselves (the graph only knows about
PipelineState; SQLAlchemy rows are an outer concern owned by worker/tasks.py)."""
import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.schemas.common import PipelineStage, RunStatus
from db.models import AgentLog, PerformanceSnapshot, Run, Video

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
        _upsert_video(db, run, result)

    return run


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
