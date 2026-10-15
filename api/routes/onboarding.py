"""Connecting a user's own YouTube channel.

Two steps, because the user supplies the OAuth client as well as the authorisation:

1. They paste the client ID and secret from a Google Cloud project they own.
2. We run the normal OAuth code flow against THAT client, and store the refresh token it returns.

Users bring their own credentials rather than authorising one central app, and that is a product
decision forced by two of Google's rules. YouTube's 10,000 units/day quota is granted per Google
Cloud project, and a publish costs ~2,050 units, so one shared app would give the entire userbase
about four uploads a day between them. Uploading to someone else's channel is also a restricted
scope, which caps an unverified app at 100 users for the lifetime of the project, permanently.
Per-user clients give each account its own full quota and need no verification, at the cost of an
onboarding that asks for more than a single button press.

Every secret is encrypted before it is written (core.crypto) and is never rendered back into a
page: once stored, the UI shows only whether a credential exists, never its value.
"""
import uuid
from datetime import UTC, datetime
from urllib.parse import urlencode

import requests
from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.routes.auth import current_user
from core.crypto import decrypt_secret, encrypt_secret
from core.logging import get_logger
from core.settings import get_settings
from db.base import get_db
from db.models import User, UserCredential

router = APIRouter()
templates = Jinja2Templates(directory="api/templates")
logger = get_logger("api.onboarding")

# youtube: upload + set thumbnails. readonly + yt-analytics: the performance feedback loop that
# strategy_node learns from. Matches scripts/youtube_oauth_setup.py, so a channel connected here
# and one connected by the single-operator script are interchangeable.
SCOPES = [
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]
_AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
_TOKEN_URI = "https://oauth2.googleapis.com/token"


def _redirect_uri(request: Request) -> str:
    """The callback Google sends the user back to, which must match a URI registered in their
    own Google Cloud project byte for byte. Derived from the configured base URL when one is
    set -- behind a proxy the app's own view of its host is the internal one, which would send
    users to an address that does not resolve for them."""
    base = (get_settings().app_base_url or "").rstrip("/")
    if base:
        return f"{base}/onboarding/callback"
    return str(request.url_for("onboarding_callback"))


def _credential_for(db: Session, user: User) -> UserCredential | None:
    return db.execute(
        select(UserCredential).where(UserCredential.user_id == user.id)
    ).scalars().first()


@router.get("", name="onboarding")
@router.get("/")
def onboarding(
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
    error: str = "",
    ok: str = "",
):
    credential = _credential_for(db, user)
    return templates.TemplateResponse(
        request,
        "onboarding.html",
        {
            "email": user.email,
            # Only ever booleans and the channel name. The secrets themselves do not go back out.
            "has_client": bool(credential and decrypt_secret(credential.google_client_id)),
            "connected": bool(credential and decrypt_secret(credential.youtube_refresh_token)),
            "channel_title": credential.youtube_channel_title if credential else None,
            "channel_id": credential.youtube_channel_id if credential else None,
            "last_error": credential.last_error if credential else None,
            "redirect_uri": _redirect_uri(request),
            "error": error,
            "ok": ok,
        },
    )


@router.post("/credentials")
def save_credentials(
    request: Request,
    client_id: str = Form(...),
    client_secret: str = Form(...),
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    client_id, client_secret = client_id.strip(), client_secret.strip()
    if not client_id.endswith(".apps.googleusercontent.com"):
        return _back("That does not look like a Google client ID — it should end in .apps.googleusercontent.com")
    if not client_secret:
        return _back("The client secret is required.")

    credential = _credential_for(db, user)
    if credential is None:
        credential = UserCredential(user_id=user.id)
        db.add(credential)

    credential.google_client_id = encrypt_secret(client_id)
    credential.google_client_secret = encrypt_secret(client_secret)
    # Changing the client invalidates any token minted by the previous one, so the stored token
    # is cleared rather than left to fail confusingly at upload time.
    credential.youtube_refresh_token = ""
    credential.youtube_channel_id = None
    credential.youtube_channel_title = None
    credential.last_error = None
    db.commit()
    logger.info("onboarding.credentials_saved", user_id=str(user.id))
    return RedirectResponse(url="/onboarding?ok=Credentials+saved.+Now+connect+your+channel.", status_code=303)


@router.get("/connect")
def connect(
    request: Request,
    user: User = Depends(current_user),
    db: Session = Depends(get_db),
):
    credential = _credential_for(db, user)
    client_id = decrypt_secret(credential.google_client_id) if credential else ""
    if not client_id:
        return _back("Add your Google client ID and secret first.")

    params = {
        "client_id": client_id,
        "redirect_uri": _redirect_uri(request),
        "response_type": "code",
        "scope": " ".join(SCOPES),
        # offline + consent together are what actually return a refresh token. Without
        # prompt=consent Google omits it on every authorisation after the first, and the account
        # silently ends up connected but unable to upload.
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": str(user.id),
    }
    return RedirectResponse(url=f"{_AUTH_URI}?{urlencode(params)}", status_code=303)


@router.get("/callback", name="onboarding_callback")
def callback(
    request: Request,
    code: str = "",
    state: str = "",
    error: str = "",
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
):
    if error:
        return _back(f"Google returned: {error}")
    if not code:
        return _back("Google did not return an authorisation code.")
    # The session decides whose credentials are written, never `state` from the query string.
    # state is checked only to catch a callback landing in the wrong browser session.
    try:
        if state and uuid.UUID(state) != user.id:
            return _back("That authorisation was started by a different account. Try again.")
    except ValueError:
        return _back("That authorisation looks malformed. Try again.")

    credential = _credential_for(db, user)
    client_id = decrypt_secret(credential.google_client_id) if credential else ""
    client_secret = decrypt_secret(credential.google_client_secret) if credential else ""
    if not (client_id and client_secret):
        return _back("Add your Google client ID and secret first.")

    try:
        response = requests.post(
            _TOKEN_URI,
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": _redirect_uri(request),
                "grant_type": "authorization_code",
            },
            timeout=30,
        )
        payload = response.json()
    except Exception as exc:
        logger.warning("onboarding.token_exchange_failed", user_id=str(user.id), error=str(exc))
        return _back(f"Could not reach Google to finish connecting ({type(exc).__name__}).")

    if response.status_code != 200 or "refresh_token" not in payload:
        detail = payload.get("error_description") or payload.get("error") or "no refresh token returned"
        logger.warning("onboarding.no_refresh_token", user_id=str(user.id), detail=str(detail)[:200])
        return _back(f"Google did not return a refresh token: {detail}")

    credential.youtube_refresh_token = encrypt_secret(payload["refresh_token"])
    credential.connected_at = datetime.now(UTC)
    credential.last_error = None

    # Naming the channel back to the user is the only check that they authorised the account they
    # meant to -- people are frequently signed into several Google accounts at once.
    try:
        from tools import youtube as youtube_tool

        with youtube_tool.acting_for(client_id, client_secret, payload["refresh_token"]):
            channel = youtube_tool._oauth_client().channels().list(part="id,snippet", mine=True).execute()
        items = channel.get("items") or []
        if items:
            credential.youtube_channel_id = items[0]["id"]
            credential.youtube_channel_title = items[0]["snippet"]["title"]
    except Exception as exc:
        # The token is already saved and is the thing that matters; a failed lookup must not
        # discard a successful authorisation (the same mistake scripts/youtube_oauth_setup.py
        # made, which threw away a completed sign-in on a network timeout).
        logger.warning("onboarding.channel_lookup_failed", user_id=str(user.id), error=str(exc))

    db.commit()
    logger.info("onboarding.connected", user_id=str(user.id), channel_id=credential.youtube_channel_id)
    return RedirectResponse(url="/onboarding?ok=Channel+connected.", status_code=303)


@router.post("/disconnect")
def disconnect(user: User = Depends(current_user), db: Session = Depends(get_db)):
    credential = _credential_for(db, user)
    if credential is not None:
        credential.youtube_refresh_token = ""
        credential.youtube_channel_id = None
        credential.youtube_channel_title = None
        credential.connected_at = None
        credential.last_error = None
        db.commit()
    logger.info("onboarding.disconnected", user_id=str(user.id))
    return RedirectResponse(url="/onboarding?ok=Channel+disconnected.", status_code=303)


def _back(message: str) -> RedirectResponse:
    from urllib.parse import quote

    return RedirectResponse(url=f"/onboarding?error={quote(message)}", status_code=303)
