"""Inbound webhooks.

`POST/GET /api/webhooks/youtube` is a YouTube PubSubHubbub (WebSub) subscriber endpoint: once
subscribed (see tools/websub.py), Google's hub calls it whenever a video on
settings.youtube_channel_id is added, updated, or removed -- near-real-time, versus the hourly
scheduler.check_performance_windows poll.

What we do with a notification:
  * new/updated entry that matches one of our Video rows -> backfill Video.published_at from the
    feed's <published> timestamp if we never recorded it, so the 24h/7d performance windows are
    measured from the real go-live time rather than whenever upload_node happened to commit.
  * deleted entry that matches one of our Video rows -> fire a CRITICAL notification (one of our
    published videos vanishing -- takedown, strike, manual removal -- is exactly the kind of
    event worth knowing about immediately).

`POST /api/webhooks/generic` is retained as a catch-all logging sink for ad-hoc integrations.
"""
import hashlib
import hmac
from datetime import datetime
from xml.etree import ElementTree

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy.orm import Session

from agents.schemas.feedback import NotificationEvent, NotificationSeverity
from core.logging import get_logger
from core.settings import get_settings
from db.base import get_db
from db.crud import get_video_by_youtube_id
from tools.notify import send_notification

router = APIRouter()
logger = get_logger("api.webhooks")

_ATOM_NS = "{http://www.w3.org/2005/Atom}"
_YT_NS = "{http://www.youtube.com/xml/schemas/2015}"
_TOMBSTONE_NS = "{http://purl.org/atompub/tombstones/1.0}"


@router.get("/youtube")
async def youtube_websub_verify(request: Request) -> Response:
    """WebSub subscription handshake: the hub GETs this with hub.challenge and expects it echoed
    back verbatim as text/plain to confirm the subscription (or its removal)."""
    params = request.query_params
    mode = params.get("hub.mode")
    challenge = params.get("hub.challenge")
    expected_token = get_settings().youtube_websub_verify_token

    if mode not in ("subscribe", "unsubscribe") or not challenge:
        return Response("missing hub.mode/hub.challenge", status_code=400)
    if expected_token and params.get("hub.verify_token") != expected_token:
        logger.warning("webhook.youtube.verify_token_mismatch", mode=mode)
        return Response("verify_token mismatch", status_code=404)

    logger.info("webhook.youtube.verified", mode=mode, topic=params.get("hub.topic"),
                lease_seconds=params.get("hub.lease_seconds"))
    return Response(challenge, media_type="text/plain")


@router.post("/youtube")
async def youtube_websub_notify(request: Request, db: Session = Depends(get_db)) -> Response:
    body = await request.body()
    settings = get_settings()

    if settings.youtube_websub_secret and not _signature_ok(body, request.headers.get("X-Hub-Signature", "")):
        logger.warning("webhook.youtube.bad_signature")
        return Response("bad signature", status_code=202)  # 2xx so the hub doesn't retry a spoof

    try:
        feed = ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        logger.warning("webhook.youtube.unparseable", error=str(exc))
        return Response(status_code=202)

    for event in _iter_events(feed):
        _handle_event(db, event)

    return Response(status_code=204)


def _signature_ok(body: bytes, header: str) -> bool:
    algo, _, sent = header.partition("=")
    if algo != "sha1" or not sent:
        return False
    digest = hmac.new(get_settings().youtube_websub_secret.encode(), body, hashlib.sha1).hexdigest()
    return hmac.compare_digest(digest, sent)


def _iter_events(feed: ElementTree.Element):
    """Yields ('published'|'deleted', video_id, title, published_dt|None) for each entry in the
    feed. YouTube sends one entry per push, but the format allows several."""
    for entry in feed.findall(f"{_ATOM_NS}entry"):
        video_id = entry.findtext(f"{_YT_NS}videoId")
        if not video_id:
            continue
        yield ("published", video_id, entry.findtext(f"{_ATOM_NS}title") or "",
               _parse_dt(entry.findtext(f"{_ATOM_NS}published")))

    for deleted in feed.findall(f"{_TOMBSTONE_NS}deleted-entry"):
        ref = deleted.get("ref", "")  # "yt:video:VIDEO_ID"
        video_id = ref.rsplit(":", 1)[-1] if ref.startswith("yt:video:") else None
        if video_id:
            yield ("deleted", video_id, "", None)


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)  # 3.11+ parses a trailing 'Z' directly
    except ValueError:
        return None


def _handle_event(db: Session, event: tuple) -> None:
    kind, video_id, title, published_dt = event
    video = get_video_by_youtube_id(db, video_id)
    if video is None:
        logger.info("webhook.youtube.event_for_unknown_video", kind=kind, youtube_video_id=video_id)
        return

    if kind == "deleted":
        logger.error("webhook.youtube.published_video_removed", youtube_video_id=video_id, video_row=str(video.id))
        send_notification(
            NotificationEvent(
                run_id=str(video.run_id),
                severity=NotificationSeverity.CRITICAL,
                message=f"Published video {video_id} (\"{video.title or 'untitled'}\") was removed from the channel.",
            )
        )
        return

    if published_dt is not None and video.published_at is None:
        video.published_at = published_dt
        db.add(video)
        db.commit()
        logger.info("webhook.youtube.backfilled_published_at", youtube_video_id=video_id, published_at=published_dt.isoformat())
    else:
        logger.info("webhook.youtube.entry_noted", youtube_video_id=video_id, title=title[:80])


@router.post("/generic")
async def generic_webhook(request: Request) -> dict:
    payload = await request.json()
    logger.info("webhook.received", payload=payload)
    return {"received": True}
