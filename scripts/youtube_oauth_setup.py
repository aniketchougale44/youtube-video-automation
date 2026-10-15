"""One-time interactive setup: mints a YouTube OAuth refresh token for this app's upload/read
scopes and writes it straight into .env. Run this yourself, locally:

    python scripts/youtube_oauth_setup.py

It opens your browser for you to sign in with your own Google account and approve access to your
own channel — the resulting refresh token never leaves your machine, it's written directly to
.env by this script.

Needs YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET already set in .env first: Google Cloud Console ->
APIs & Services -> Credentials -> Create Credentials -> OAuth client ID -> Desktop app.

If you already ran this once before the yt-analytics.readonly scope below existed, re-run it —
the old refresh token doesn't cover Analytics API calls and analytics_report() will fail with a
403 until you mint a new one.
"""
import sys
from pathlib import Path

from dotenv import set_key
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from core.settings import get_settings

SCOPES = [
    "https://www.googleapis.com/auth/youtube",  # full manage scope: upload + delete + thumbnails
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def main() -> None:
    settings = get_settings()
    if not settings.youtube_client_id or not settings.youtube_client_secret:
        sys.exit(
            "Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in .env first (Google Cloud Console -> "
            "APIs & Services -> Credentials -> Create Credentials -> OAuth client ID -> Desktop app)."
        )

    client_config = {
        "installed": {
            "client_id": settings.youtube_client_id,
            "client_secret": settings.youtube_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    print("Opening your browser — sign in and approve access to your YouTube channel...")
    credentials = flow.run_local_server(port=0)

    # Save the token the moment it exists, before anything that can fail.
    #
    # This used to run after the channels.list() lookup below, and a timeout reaching
    # googleapis.com threw away a sign-in that had already succeeded. The refresh token is handed
    # over exactly once, so losing it here costs the whole browser approval again. Nothing
    # fallible belongs between the approval and this line.
    set_key(str(ENV_PATH), "YOUTUBE_REFRESH_TOKEN", credentials.refresh_token)
    print("YOUTUBE_REFRESH_TOKEN written to .env.")

    # The channel id is a convenience, not part of authorizing: it is already in .env from the
    # last run and does not change. A failure here must not look like a failed authorization.
    try:
        youtube = build("youtube", "v3", credentials=credentials, cache_discovery=False)
        channel = youtube.channels().list(part="id,snippet", mine=True).execute()
        items = channel.get("items", [])
        channel_id = items[0]["id"] if items else ""
        channel_title = items[0]["snippet"]["title"] if items else "(unknown)"
        if channel_id:
            set_key(str(ENV_PATH), "YOUTUBE_CHANNEL_ID", channel_id)
        print(f"\nAuthorized for channel: {channel_title} ({channel_id})")
    except Exception as exc:
        print(f"\nCouldn't look up the channel name ({type(exc).__name__}: {exc}).")
        print("That does NOT affect the token — it is saved. YOUTUBE_CHANNEL_ID is unchanged.")

    print("Restart the worker to pick it up.")


if __name__ == "__main__":
    main()
