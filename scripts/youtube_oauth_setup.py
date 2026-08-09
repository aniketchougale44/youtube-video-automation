"""One-time interactive setup: mints a YouTube OAuth refresh token for this app's upload/read
scopes and writes it straight into .env. Run this yourself, locally:

    python scripts/youtube_oauth_setup.py

It opens your browser for you to sign in with your own Google account and approve access to your
own channel — the resulting refresh token never leaves your machine, it's written directly to
.env by this script.

Needs YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET already set in .env first: Google Cloud Console ->
APIs & Services -> Credentials -> Create Credentials -> OAuth client ID -> Desktop app.
"""
import sys
from pathlib import Path

from dotenv import set_key
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from core.settings import get_settings

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
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

    youtube = build("youtube", "v3", credentials=credentials, cache_discovery=False)
    channel = youtube.channels().list(part="id,snippet", mine=True).execute()
    items = channel.get("items", [])
    channel_id = items[0]["id"] if items else ""
    channel_title = items[0]["snippet"]["title"] if items else "(unknown)"

    set_key(str(ENV_PATH), "YOUTUBE_REFRESH_TOKEN", credentials.refresh_token)
    set_key(str(ENV_PATH), "YOUTUBE_CHANNEL_ID", channel_id)

    print(f"\nAuthorized for channel: {channel_title} ({channel_id})")
    print("YOUTUBE_REFRESH_TOKEN and YOUTUBE_CHANNEL_ID written to .env. Restart the worker to pick them up.")


if __name__ == "__main__":
    main()
