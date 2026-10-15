"""Serves a run's rendered files to the account that owns the run, and to nobody else.

This replaces a public StaticFiles mount. That mount served every render, voiceover and thumbnail
to anyone who asked, with no session and no ownership check -- an unlisted video is only unlisted
on YouTube, and before it is uploaded at all the file here is the only copy. A run id in a URL is
not a secret: it appears in the dashboard, in a shared screenshot, in a pasted link, in any proxy
log along the way.

Ownership is read from the run, not from the path, so the existing media/<run_id>/... layout is
kept as it is. Moving everyone's files into per-user directories would mean rewriting render
paths already stored in Postgres and in LangGraph checkpoints, and would make the directory
layout, rather than the database, the thing that decides who may watch a video. The run row is
the authority on that.

Runs with no owner -- the ones that predate accounts -- are served to nobody. They are already
absent from every per-user view, so nothing in the UI links to them, and "visible to any signed-in
stranger" is the wrong reading of "has no owner".
"""
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.routes.auth import require_user
from core.logging import get_logger
from core.settings import get_settings
from db.base import get_db
from db.models import Run

router = APIRouter()
logger = get_logger("api.media")


def _resolve(run_id: uuid.UUID, subpath: str) -> str:
    """Absolute path to the requested file, or raises 404.

    Guards traversal twice over: the resolved path must sit inside this run's own directory, and
    that directory must sit inside media_dir. Starlette's StaticFiles used to do the second check;
    serving files by hand means doing it here, and "../" in a URL path segment is the oldest way
    to read a file that was never meant to be served.
    """
    settings = get_settings()
    media_root = os.path.abspath(settings.media_dir)
    run_root = os.path.abspath(os.path.join(media_root, str(run_id)))
    target = os.path.abspath(os.path.join(run_root, subpath))

    if os.path.commonpath([run_root, target]) != run_root:
        raise HTTPException(404, "not found")
    if os.path.commonpath([media_root, run_root]) != media_root:
        raise HTTPException(404, "not found")
    if not os.path.isfile(target):
        raise HTTPException(404, "not found")
    return target


@router.get("/{run_id}/{subpath:path}")
def serve_media(
    run_id: uuid.UUID,
    subpath: str,
    request: Request,
    db: Session = Depends(get_db),
    user=Depends(require_user),
):
    run = db.get(Run, run_id)
    # 404 rather than 403 for someone else's run: a 403 confirms the run exists, which is itself
    # worth not telling a stranger who is trying ids.
    if run is None or run.user_id is None or run.user_id != user.id:
        logger.info(
            "media.denied", run_id=str(run_id), user_id=str(user.id),
            reason="missing" if run is None else ("unowned" if run.user_id is None else "other_user"),
        )
        raise HTTPException(404, "not found")

    path = _resolve(run_id, subpath)
    # FileResponse handles conditional and range requests, which is what lets the review card's
    # <video> element seek instead of re-downloading the file to scrub.
    return FileResponse(path)
