"""Manage the YouTube PubSubHubbub (WebSub) subscription that feeds api/routes/webhooks.py.

YouTube publishes a per-channel Atom feed at
https://www.youtube.com/xml/feeds/videos.xml?channel_id=<CHANNEL_ID> and delegates push
notifications for it to Google's public hub (https://pubsubhubbub.appspot.com/). Subscribing
tells the hub to POST that feed to our callback whenever a video on the channel changes.

Subscriptions expire (lease_seconds, typically <=10 days), so re-run `subscribe` on a schedule
-- scheduler/beat.py is the natural place, or a cron/systemd timer.

    python -m tools.websub subscribe
    python -m tools.websub unsubscribe

Requires settings.youtube_channel_id and settings.youtube_websub_callback_url; verify_token and
secret are sent when configured.
"""
import sys

import httpx

from core.logging import get_logger
from core.settings import get_settings

logger = get_logger("tools.websub")

HUB_URL = "https://pubsubhubbub.appspot.com/subscribe"


def _topic_url(channel_id: str) -> str:
    return f"https://www.youtube.com/xml/feeds/videos.xml?channel_id={channel_id}"


def _request(mode: str) -> None:
    settings = get_settings()
    if not settings.youtube_channel_id:
        raise RuntimeError("YOUTUBE_CHANNEL_ID is not set")
    if not settings.youtube_websub_callback_url:
        raise RuntimeError("YOUTUBE_WEBSUB_CALLBACK_URL is not set")

    form = {
        "hub.mode": mode,
        "hub.topic": _topic_url(settings.youtube_channel_id),
        "hub.callback": settings.youtube_websub_callback_url,
        "hub.verify": "async",
    }
    if settings.youtube_websub_verify_token:
        form["hub.verify_token"] = settings.youtube_websub_verify_token
    if settings.youtube_websub_secret:
        form["hub.secret"] = settings.youtube_websub_secret

    resp = httpx.post(HUB_URL, data=form, timeout=30)
    resp.raise_for_status()  # hub returns 202 Accepted; the GET verification hits our callback next
    logger.info("websub.request_sent", mode=mode, status=resp.status_code, topic=form["hub.topic"])


def subscribe() -> None:
    _request("subscribe")


def unsubscribe() -> None:
    _request("unsubscribe")


def main(argv: list[str]) -> int:
    action = argv[1] if len(argv) > 1 else ""
    if action == "subscribe":
        subscribe()
    elif action == "unsubscribe":
        unsubscribe()
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
