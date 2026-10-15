"""Live operations dashboard: every run, its real-time pipeline stage, recent agent activity,
and cost — plus the human-approval review flow (list runs awaiting approval, inspect the
ApprovalPacket surfaced by the human_approval_node interrupt(), approve or reject)."""
import itertools
import json
import os
import time
import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy import true as sa_true
from sqlalchemy.orm import Session

from agents.schemas.common import RunStatus
from api.routes.auth import current_user, require_user
from core.logging import code_version, get_logger
from core.settings import get_settings
from db.base import SessionLocal, get_db
from db.crud import get_run, latest_strategy_weight_adjustments, list_runs
from db.models import AgentLog, Cost, PerformanceSnapshot, Run, Video
from graph.checkpointer import postgres_checkpointer
from graph.run import get_state
from tools import youtube as youtube_tool
from tools.cache import cache_get_json, cache_scan_json, cache_set_json
from tools.script_progress import KEY_PREFIX as SCRIPT_RUN_KEY_PREFIX
from worker.tasks import resume_pipeline_task

router = APIRouter()
templates = Jinja2Templates(directory="api/templates")

logger = get_logger("api.dashboard")
_CHANNEL_CACHE_KEY = "dashboard:channel_uploads"
_AUTH_CACHE_KEY = "dashboard:youtube_auth"

# Mirrors the node order wired in graph/builder.py — used only to place a run's live position on
# the progress bar, not to drive any control flow.
PIPELINE_NODE_ORDER = [
    "trend_research", "strategy", "script_writer", "critic_script_qa",
    "fact_check", "visual_planning", "asset_visual", "voiceover",
    "video_assembly", "metadata_seo", "thumbnail", "critic_compliance",
    "approval_gate", "upload",
]
_LIVE_STATUSES = (RunStatus.PENDING, RunStatus.RUNNING)
_TERMINAL_LABELS = {
    RunStatus.PUBLISHED: "published",
    RunStatus.REJECTED: "rejected",
    RunStatus.FAILED: "failed",
    RunStatus.ESCALATED: "escalated",
}


def _media_url(fs_path: str | None) -> str | None:
    """Turn a filesystem path a run wrote (a thumbnail candidate, the final render — always under
    settings.media_dir) into a URL the browser can load from the /media static mount in api/main.py.
    Returns None for a missing path or anything that resolves outside media_dir (so a stray absolute
    path in the DB can't be used to read arbitrary files through the dashboard)."""
    if not fs_path:
        return None
    try:
        rel = os.path.relpath(os.path.abspath(fs_path), os.path.abspath(get_settings().media_dir))
    except ValueError:  # e.g. different drive on Windows
        return None
    if rel.startswith(".."):
        return None
    return "/media/" + rel.replace(os.sep, "/")


def _youtube_url(video_id: str | None) -> str | None:
    return f"https://youtube.com/watch?v={video_id}" if video_id else None


def _run_duration_seconds(run: Run) -> float | None:
    """Wall-clock time from run creation to its last update — the elapsed time shown per row. For a
    still-running row updated_at keeps advancing, so this tracks live; for a terminal row it's the
    final total."""
    if not run.created_at or not run.updated_at:
        return None
    return max(0.0, (run.updated_at - run.created_at).total_seconds())


def _checkpoint_snapshot(run):
    """The run's live LangGraph StateSnapshot (channel values + pending interrupts), or None when
    the checkpoint store can't be reached. The checkpoint — not Postgres — is the source of truth
    for anything mid-flight, so both the approval packet and the full script are read from here."""
    try:
        with postgres_checkpointer() as cp:
            return get_state(cp, run.langgraph_thread_id)
    except Exception:
        return None


def _packet_from_snapshot(snapshot) -> dict | None:
    if snapshot is None:
        return None
    for task in snapshot.tasks:
        for pending_interrupt in getattr(task, "interrupts", []) or []:
            return pending_interrupt.value
    return None


def _extract_approval_packet(run) -> dict | None:
    return _packet_from_snapshot(_checkpoint_snapshot(run))


def _script_view(script_output: dict | None) -> dict | None:
    """The full script, flattened for the reviewer's read-along pane next to the video player:
    working title, hook, every beat's narration + visual cue, and the CTA."""
    if not script_output:
        return None
    return {
        "working_title": script_output.get("working_title") or "",
        "hook": script_output.get("hook") or "",
        "cta": script_output.get("cta") or "",
        "full_text": script_output.get("full_voiceover_text") or "",
        "target_length_seconds": script_output.get("target_length_seconds"),
        "beats": [
            {
                "index": b.get("index"),
                "type": (b.get("beat_type") or "").replace("_", " "),
                "timestamp_seconds": b.get("timestamp_seconds"),
                "voiceover_text": b.get("voiceover_text") or "",
                "visual_cue": b.get("visual_cue") or "",
            }
            for b in script_output.get("beats") or []
        ],
    }


def _approval_rows(db: Session, runs) -> list[dict]:
    """One JSON-friendly card per awaiting-approval run for the dashboard's approval queue: the
    finished video (`render_url`), the full script (`script`) for side-by-side review, the
    thumbnail candidates, and flattened compliance. `has_packet` is False when the checkpoint
    can't be reached (worker/Redis down); the card then degrades to an id + 'open full review'
    link instead of vanishing."""
    rows = []
    for run in runs:
        snapshot = _checkpoint_snapshot(run)
        packet = _packet_from_snapshot(snapshot) or {}
        values = getattr(snapshot, "values", None) or {}
        compliance = packet.get("compliance_result") or None

        render_path = (values.get("assembly_output") or {}).get("render_path")
        if not render_path:
            video = db.execute(
                select(Video).where(Video.run_id == run.id).order_by(Video.created_at.desc())
            ).scalars().first()
            render_path = video.render_path if video else None

        rows.append(
            {
                "id": str(run.id),
                "short_id": str(run.id)[:8],
                "requested_at": run.updated_at.isoformat() if run.updated_at else None,
                "has_packet": bool(packet),
                "title": packet.get("title") or (run.selected_topic or {}).get("title") or "Untitled run",
                "description": packet.get("description") or "",
                "script_summary": packet.get("script_summary") or "",
                "script": _script_view(values.get("script_output")),
                "render_url": _media_url(render_path),
                "thumbnails": [
                    {"index": i, "url": _media_url(path)}
                    for i, path in enumerate(packet.get("thumbnail_options") or [])
                ],
                "compliance": {
                    "copyright_risk_score": compliance["copyright_risk_score"],
                    "av_sync_ok": compliance["av_sync_ok"],
                    "spec_check_ok": compliance["spec_check_ok"],
                    "passed": compliance["passed"],
                    "blocking_reasons": compliance.get("blocking_reasons") or [],
                    "community_guidelines_flags": compliance.get("community_guidelines_flags") or [],
                }
                if compliance
                else None,
            }
        )
    return rows


def _live_node(run: Run) -> str | None:
    """Which graph node is executing right now, read straight from the LangGraph checkpoint.
    The checkpointer persists state after every completed superstep (see graph/checkpointer.py),
    so this reflects genuine mid-flight progress for a run the worker is actively executing —
    not just the coarse status column, which only changes at task start/end."""
    if run.status not in _LIVE_STATUSES:
        return None
    try:
        with postgres_checkpointer() as cp:
            snapshot = get_state(cp, run.langgraph_thread_id)
    except Exception:
        return None
    next_nodes = list(snapshot.next or ())
    return next_nodes[0] if next_nodes else None


def _stage_progress(run: Run) -> tuple[str, int]:
    """(human label, completed-stage index) for a run's progress bar."""
    total = len(PIPELINE_NODE_ORDER)
    if run.status in _LIVE_STATUSES:
        node = _live_node(run)
        if node is None:
            return "queued", 0
        idx = PIPELINE_NODE_ORDER.index(node) if node in PIPELINE_NODE_ORDER else 0
        return node.replace("_", " "), idx
    if run.status == RunStatus.AWAITING_APPROVAL:
        return "awaiting approval", PIPELINE_NODE_ORDER.index("approval_gate")
    return _TERMINAL_LABELS.get(run.status, run.status.value), total


def _script_runs() -> list[dict]:
    """Background activity from the standalone scripts/produce_*.py one-offs (see
    tools/script_progress.py) -- these bypass the LangGraph pipeline entirely, so they'd
    otherwise be completely invisible here even while actively generating/rendering/uploading."""
    try:
        rows = cache_scan_json(SCRIPT_RUN_KEY_PREFIX)
    except Exception:
        return []
    rows.sort(key=lambda r: r.get("updated_at") or "", reverse=True)
    return rows


def _worker_version() -> str | None:
    """Whatever commit the worker announced when it booted; None if it has not, or Redis is down."""
    try:
        from tools.cache import _redis

        return _redis().get("worker:code_version")
    except Exception:
        return None


def _build_snapshot(db: Session, user_id=None) -> dict:
    """`user_id=None` builds the whole-deployment view, which is what the operator dashboard and
    the scheduler want. Anything serving a logged-in web request passes the caller's id: the
    counts and the approval queue are filtered with it, so one account never sees -- or approves,
    which publishes -- another account's run."""
    runs = list_runs(db, limit=50, user_id=user_id)
    total_stages = len(PIPELINE_NODE_ORDER)
    now = datetime.now(UTC)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    status_q = select(Run.status, func.count()).group_by(Run.status)
    cost_today_q = select(func.coalesce(func.sum(Cost.total_usd), 0.0)).where(Cost.created_at >= today_start)
    cost_total_q = select(func.coalesce(func.sum(Cost.total_usd), 0.0))
    if user_id is not None:
        status_q = status_q.where(Run.user_id == user_id)
        # Costs join through the run, so a user's spend is the spend of their own runs.
        owned = select(Run.id).where(Run.user_id == user_id).scalar_subquery()
        cost_today_q = cost_today_q.where(Cost.run_id.in_(owned))
        cost_total_q = cost_total_q.where(Cost.run_id.in_(owned))

    status_counts = dict(db.execute(status_q).all())
    cost_today = db.execute(cost_today_q).scalar_one()
    cost_total = db.execute(cost_total_q).scalar_one()

    # Per-run spend and produced-video info for the 50 rows on screen, fetched in one query each
    # rather than per row.
    run_ids = [run.id for run in runs]
    cost_by_run = dict(
        db.execute(
            select(Cost.run_id, func.coalesce(func.sum(Cost.total_usd), 0.0))
            .where(Cost.run_id.in_(run_ids))
            .group_by(Cost.run_id)
        ).all()
    ) if run_ids else {}
    video_by_run: dict = {}
    if run_ids:
        for video in db.execute(
            select(Video).where(Video.run_id.in_(run_ids)).order_by(Video.created_at)
        ).scalars():
            video_by_run[video.run_id] = video  # last (newest) wins if a run somehow has several

    run_rows = []
    for run in runs:
        label, idx = _stage_progress(run)
        video = video_by_run.get(run.id)
        run_rows.append(
            {
                "id": str(run.id),
                "short_id": str(run.id)[:8],
                # Falls back to what the operator actually asked for. selected_topic is only
                # filled once strategy_node has run, so a run started from the chat showed "—"
                # for its first few minutes -- the stretch where someone is most likely to be
                # watching to confirm their request landed at all.
                "topic": (video.title if video and video.title else None)
                or (run.selected_topic or {}).get("title")
                or ((run.stage_status or {}).get("user_inputs") or {}).get("topic")
                or "—",
                "status": run.status.value,
                "stage_label": label,
                "progress": idx,
                "progress_total": total_stages,
                "error": run.error,
                "cost_usd": round(float(cost_by_run.get(run.id, 0.0)), 2),
                "duration_seconds": _run_duration_seconds(run),
                "youtube_url": _youtube_url(video.youtube_video_id if video else None),
                "render_url": _media_url(video.render_path if video else None),
                "thumbnail_url": _media_url(video.thumbnail_path if video else None),
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "updated_at": run.updated_at.isoformat() if run.updated_at else None,
            }
        )

    log_rows = db.execute(
        select(AgentLog, Run.id).join(Run, AgentLog.run_id == Run.id).order_by(AgentLog.created_at.desc()).limit(40)
    ).all()
    log_rows_out = [
        {
            "run_short": str(run_id)[:8],
            "stage": log.stage.value,
            "agent_name": log.agent_name,
            "is_critic": log.is_critic,
            "passed": log.passed,
            "critic_score": log.critic_score,
            "error": log.error,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log, run_id in log_rows
    ]

    return {
        "generated_at": now.isoformat(),
        # So a stale worker is visible rather than merely logged. The source is bind-mounted, so a
        # worker can run code several commits behind the files on disk -- which silently cost this
        # project a feature (a queued user_topic the running worker had no code to read).
        "versions": {"api": code_version(), "worker": _worker_version()},
        "stats": {
            "total": sum(status_counts.values()),
            "pending": status_counts.get(RunStatus.PENDING, 0),
            "running": status_counts.get(RunStatus.RUNNING, 0),
            "awaiting_approval": status_counts.get(RunStatus.AWAITING_APPROVAL, 0),
            "published": status_counts.get(RunStatus.PUBLISHED, 0),
            "needs_attention": status_counts.get(RunStatus.FAILED, 0) + status_counts.get(RunStatus.ESCALATED, 0),
            "cost_today_usd": round(float(cost_today), 2),
            "cost_total_usd": round(float(cost_total), 2),
        },
        "approvals": _approval_rows(db, list_runs(db, status=RunStatus.AWAITING_APPROVAL, user_id=user_id)),
        "runs": run_rows,
        "logs": log_rows_out,
        "script_runs": _script_runs(),
    }


# --------------------------------------------------------------------------- ops panels
#
# Everything below feeds /dashboard/ops. Each panel is assembled from rows that already exist --
# costs, performance_snapshots, agent_logs, the Redis quota bucket, the OAuth token. Nothing is
# estimated or extrapolated: a panel with no data says so rather than drawing a flat line, because
# an operations screen that invents a number is worse than one that admits a gap.


def _owned_runs(user_id):
    """Cost rows belong to a run, so a user's spend is the spend of the runs they own."""
    return Cost.run_id.in_(select(Run.id).where(Run.user_id == user_id).scalar_subquery())


def _quota_panel() -> dict:
    """Today's YouTube API budget. A full publish is ~2,050 units of the daily 10,000."""
    try:
        from tools import quota

        limit = get_settings().youtube_daily_quota_units
        remaining = quota.remaining_units()
        return {
            "limit": limit,
            "remaining": remaining,
            "used": max(0, limit - remaining),
            "publishes_left": remaining // 2050,
            "error": None,
        }
    except Exception as exc:
        return {"limit": None, "remaining": None, "used": None, "publishes_left": None,
                "error": f"{type(exc).__name__} -- is Redis running?"}


def _youtube_auth_panel() -> dict:
    """Whether the stored refresh token still works.

    Worth its own panel: an expired token fails only at the upload node, ~20 minutes and a full
    render into a run, and the error it raises there ("invalid_grant") does not obviously mean
    "re-authorize". Checking costs one token refresh and no quota.
    """
    settings = get_settings()
    if not settings.youtube_refresh_token:
        return {"ok": False, "detail": "No YOUTUBE_REFRESH_TOKEN set."}

    # Cached for 5 minutes. A failing refresh takes seconds to come back, and this panel sits on
    # the first paint of the page -- without the cache the whole dashboard waits on Google.
    cached = cache_get_json(_AUTH_CACHE_KEY)
    if cached is not None:
        return cached

    try:
        from google.auth.transport.requests import Request as GoogleRequest
        from google.oauth2.credentials import Credentials

        creds = Credentials(
            None,
            refresh_token=settings.youtube_refresh_token,
            client_id=settings.youtube_client_id,
            client_secret=settings.youtube_client_secret,
            token_uri="https://oauth2.googleapis.com/token",
        )
        creds.refresh(GoogleRequest())
        result = {"ok": True, "detail": "Refresh token valid."}
    except Exception as exc:
        detail = str(exc)
        if "invalid_grant" in detail:
            detail = "Token expired or revoked -- run scripts/youtube_oauth_setup.py."
        result = {"ok": False, "detail": detail[:160]}
    cache_set_json(_AUTH_CACHE_KEY, result, ttl_seconds=300)
    return result


def _spend_series(db: Session, days: int = 14, user_id=None) -> list[dict]:
    """Daily measured spend. Only days that actually have cost rows are returned."""
    rows = db.execute(
        select(
            func.date(Cost.created_at).label("day"),
            func.sum(Cost.total_usd).label("total"),
        )
        .where(Cost.created_at >= datetime.now(UTC) - timedelta(days=days))
        .where(_owned_runs(user_id) if user_id is not None else sa_true())
        .group_by(func.date(Cost.created_at))
        .order_by(func.date(Cost.created_at))
    ).all()
    return [{"day": str(r.day), "usd": round(float(r.total or 0), 4)} for r in rows]


def _spend_by_provider(db: Session, user_id=None) -> list[dict]:
    """Where the money goes, all-time, biggest first. Grouped by provider rather than stage:
    the actionable question is which vendor to swap, not which node called it."""
    rows = db.execute(
        select(Cost.provider, func.sum(Cost.total_usd).label("total"))
        .where(_owned_runs(user_id) if user_id is not None else sa_true())
        .group_by(Cost.provider)
        .order_by(func.sum(Cost.total_usd).desc())
    ).all()
    return [{"provider": r.provider or "unknown", "usd": round(float(r.total or 0), 4)} for r in rows]


def _stage_timings(db: Session, limit_runs: int = 20, user_id=None) -> list[dict]:
    """Mean wall-clock seconds per pipeline stage, from consecutive agent_log timestamps.

    agent_logs record when a stage logged, not how long it took, so a stage's duration is inferred
    as the gap to the next log in the same run. The last stage of a run has no successor and is
    dropped rather than guessed, which is why `samples` is reported -- a stage with 1 sample is an
    anecdote, not a timing.
    """
    recent_q = select(Run).order_by(Run.created_at.desc()).limit(limit_runs)
    if user_id is not None:
        recent_q = recent_q.where(Run.user_id == user_id)
    recent_run_ids = [r.id for r in db.execute(recent_q).scalars()]
    if not recent_run_ids:
        return []

    logs = db.execute(
        select(AgentLog)
        .where(AgentLog.run_id.in_(recent_run_ids))
        .order_by(AgentLog.run_id, AgentLog.created_at)
    ).scalars().all()

    by_run: dict = {}
    for log in logs:
        by_run.setdefault(log.run_id, []).append(log)

    totals: dict[str, list[float]] = {}
    for entries in by_run.values():
        for current, nxt in itertools.pairwise(entries):
            if not current.stage or not current.created_at or not nxt.created_at:
                continue
            seconds = (nxt.created_at - current.created_at).total_seconds()
            if 0 <= seconds < 3600:  # a gap longer than an hour is a parked run, not a stage
                totals.setdefault(current.stage, []).append(seconds)

    out = [
        {"stage": stage, "mean_seconds": round(sum(v) / len(v), 2), "samples": len(v)}
        for stage, v in totals.items()
    ]
    out.sort(key=lambda r: r["mean_seconds"], reverse=True)
    return out


def _performance_rows(db: Session, limit: int = 12, user_id=None) -> list[dict]:
    """Newest performance snapshot per window, joined to the video it describes."""
    stmt = (
        select(PerformanceSnapshot, Video)
        .join(Video, Video.id == PerformanceSnapshot.video_id)
        .order_by(PerformanceSnapshot.captured_at.desc())
        .limit(limit)
    )
    if user_id is not None:
        # A video belongs to a run, and a run to an account.
        stmt = stmt.where(
            Video.run_id.in_(select(Run.id).where(Run.user_id == user_id).scalar_subquery())
        )
    rows = db.execute(stmt).all()
    return [
        {
            "title": v.title or "(untitled)",
            "youtube_video_id": v.youtube_video_id,
            "window": s.window,
            "views": s.views,
            "impressions": s.impressions,
            "ctr": round(float(s.ctr or 0), 2),
            "retention_pct": round(float(s.retention_pct or 0), 1),
            "likes": s.likes,
            "comments": s.comments,
            "captured_at": s.captured_at.isoformat() if s.captured_at else None,
        }
        for s, v in rows
    ]


def _build_ops(db: Session, user_id=None) -> dict:
    """Per-account where the data is per-account. Spend and performance are the user's own; quota
    and the YouTube auth check read whatever credentials are acting, which for a signed-in user
    is their own connected channel."""
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "quota": _quota_panel(),
        "youtube_auth": _youtube_auth_panel(),
        "approval_gate_on": get_settings().require_human_approval,
        "upload_visibility": get_settings().youtube_upload_visibility,
        "spend_series": _spend_series(db, user_id=user_id),
        "spend_by_provider": _spend_by_provider(db, user_id=user_id),
        "stage_timings": _stage_timings(db, user_id=user_id),
        "performance": _performance_rows(db, user_id=user_id),
        "learned_weights": latest_strategy_weight_adjustments(db),
    }


def _owned_run_or_404(db: Session, run_id: uuid.UUID, user) -> Run:
    """The run, if it belongs to this account. 404 otherwise -- including for a run that exists
    but belongs to someone else, because a 403 would confirm it exists.

    Approving is the operation that publishes a video to a channel, so "which run is this" and
    "may this account act on it" have to be answered together, in one place, rather than left to
    each endpoint to remember.
    """
    run = get_run(db, run_id)
    if run is None or run.user_id is None or run.user_id != user.id:
        raise HTTPException(404, "run not found")
    return run


@router.get("/")
def dashboard_index(request: Request, db: Session = Depends(get_db), user=Depends(current_user)):
    # The approval queue, run table, and activity feed all render client-side from this one
    # snapshot (embedded for first paint, then refreshed over /stream) so nothing on the page
    # needs a manual reload to stay current.
    return templates.TemplateResponse(request, "index.html", {"snapshot": _build_snapshot(db)})


@router.get("/stream")
def dashboard_stream(user=Depends(require_user)):
    """Server-Sent Events feed: pushes a fresh snapshot (run statuses, live pipeline position,
    recent agent activity, cost) every 2s so the dashboard updates without polling or a reload."""

    def event_source():
        while True:
            db = SessionLocal()
            try:
                payload = _build_snapshot(db)
                yield f"data: {json.dumps(payload)}\n\n"
            except Exception as exc:
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
            finally:
                db.close()
            time.sleep(2)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/studio")
def studio(request: Request, user=Depends(current_user), db: Session = Depends(get_db)):
    """The chat-led control surface. Renders empty and fills itself from /dashboard/snapshot, so a
    slow database never blocks first paint.

    An account with no connected channel is sent to onboarding instead: it could start a run from
    here, but the run would render for twenty minutes and then fail at the upload with nowhere to
    publish to.
    """
    from db.models import UserCredential

    credential = db.execute(
        select(UserCredential).where(UserCredential.user_id == user.id)
    ).scalars().first()
    from core.crypto import decrypt_secret

    if not (credential and decrypt_secret(credential.youtube_refresh_token)):
        return RedirectResponse(url="/onboarding", status_code=303)
    return templates.TemplateResponse(request, "studio.html", {"email": user.email})


@router.get("/channel")
def dashboard_channel(user=Depends(require_user)):
    """The channel's real uploads, for the studio's Published panel.

    Not read from the `videos` table: that only records what the pipeline published, so one-off
    scripts/produce_*.py uploads are invisible to it -- 8 of 12 videos on this channel at the time
    this was added. Cached for 5 minutes because the studio polls every 5 seconds and this costs 2
    quota units per call; the list changes at most a few times a day.
    """
    cached = cache_get_json(_CHANNEL_CACHE_KEY)
    if cached is not None:
        return cached
    try:
        videos = youtube_tool.list_channel_uploads()
    except Exception as exc:
        logger.warning("dashboard.channel_uploads_failed", error=str(exc))
        return {"videos": [], "error": type(exc).__name__}
    payload = {"videos": videos}
    cache_set_json(_CHANNEL_CACHE_KEY, payload, ttl_seconds=300)
    return payload


@router.get("/snapshot")
def dashboard_snapshot(db: Session = Depends(get_db), user=Depends(require_user)):
    """The same payload /stream pushes, as a plain GET.

    The studio polls this instead of holding an SSE connection: its panes update every few seconds,
    and a long-lived stream per open tab is a poor trade for that. Both read _build_snapshot, so the
    two surfaces can never disagree about run state.
    """
    return _build_snapshot(db, user_id=user.id)


# Registered after the literal paths above on purpose: run_id is typed uuid.UUID, so a request for
# /dashboard/studio would match this route's shape and fail validation with a 422 rather than
# falling through to it.
@router.get("/ops")
def ops(request: Request, user=Depends(current_user)):
    return templates.TemplateResponse(request, "ops.html", {})


@router.get("/ops/snapshot")
def ops_snapshot(db: Session = Depends(get_db), user=Depends(require_user)):
    """Separate from /dashboard/snapshot on purpose: this one makes a live token-refresh call to
    Google, so it is polled far less often than the 5s run/stage poll."""
    return _build_ops(db, user_id=user.id)


@router.get("/{run_id}")
def dashboard_review(request: Request, run_id: uuid.UUID, db: Session = Depends(get_db), user=Depends(current_user)):
    run = get_run(db, run_id)
    if run is None:
        raise HTTPException(404, "run not found")

    snapshot = _checkpoint_snapshot(run)
    packet = _packet_from_snapshot(snapshot)
    values = getattr(snapshot, "values", None) or {}

    video = db.execute(
        select(Video).where(Video.run_id == run.id).order_by(Video.created_at.desc())
    ).scalars().first()
    cost_usd = db.execute(
        select(func.coalesce(func.sum(Cost.total_usd), 0.0)).where(Cost.run_id == run.id)
    ).scalar_one()

    render_path = (values.get("assembly_output") or {}).get("render_path") or (
        video.render_path if video else None
    )
    thumb_urls = [_media_url(p) for p in (packet or {}).get("thumbnail_options", [])]
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "run": run,
            "packet": packet,
            "script": _script_view(values.get("script_output")),
            "thumb_urls": thumb_urls,
            "video": video,
            "youtube_url": _youtube_url(video.youtube_video_id if video else None),
            "render_url": _media_url(render_path),
            "cost_usd": round(float(cost_usd), 2),
            "duration_seconds": _run_duration_seconds(run),
        },
    )


@router.post("/{run_id}/approve")
def dashboard_approve(
    run_id: uuid.UUID,
    reviewer: str = Form("dashboard"),
    notes: str = Form(""),
    selected_thumbnail_index: int = Form(0),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    run = _owned_run_or_404(db, run_id, user)
    if run.status != RunStatus.AWAITING_APPROVAL:
        raise HTTPException(409, "run not awaiting approval")
    decision = {
        "run_id": str(run_id),
        "approved": True,
        "selected_thumbnail_index": selected_thumbnail_index,
        "reviewer": reviewer,
        "notes": notes,
    }
    resume_pipeline_task.delay(str(run_id), decision)
    return RedirectResponse(url="/dashboard/", status_code=303)


@router.post("/{run_id}/reject")
def dashboard_reject(
    run_id: uuid.UUID,
    reviewer: str = Form("dashboard"),
    notes: str = Form(""),
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    run = _owned_run_or_404(db, run_id, user)
    if run.status != RunStatus.AWAITING_APPROVAL:
        raise HTTPException(409, "run not awaiting approval")
    decision = {"run_id": str(run_id), "approved": False, "reviewer": reviewer, "notes": notes}
    resume_pipeline_task.delay(str(run_id), decision)
    return RedirectResponse(url="/dashboard/", status_code=303)
