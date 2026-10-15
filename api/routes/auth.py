"""Accounts: sign up, sign in, sign out, and the dependency every protected route hangs off.

Sessions are a single encrypted cookie (core.crypto), not server-side state, so the app stays
horizontally scalable without a session store. The cookie is HttpOnly so page scripts cannot read
it, SameSite=Lax so it does not ride along on cross-site POSTs, and Secure whenever the request
arrived over HTTPS -- derived from the request rather than configured, because a flag that has to
be remembered at deploy time is a flag that gets forgotten, and the failure is silent.

Two dependencies, deliberately distinct:

- `current_user` is for pages. A signed-out visitor is redirected to the login screen.
- `require_user` is for the JSON API. A signed-out caller gets a 401, because a fetch() that
  receives a login page with status 200 reports success and renders HTML into the chat.
"""
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.crypto import (
    SESSION_MAX_AGE_SECONDS,
    SessionInvalid,
    hash_password,
    mint_session,
    read_session,
    verify_password,
)
from core.logging import get_logger
from db.base import get_db
from db.models import User

router = APIRouter()
templates = Jinja2Templates(directory="api/templates")
logger = get_logger("api.auth")

SESSION_COOKIE = "autotube_session"
MIN_PASSWORD_CHARS = 10


class _AuthRedirect(HTTPException):
    """A 303 to the login page, raised as an exception so a dependency can stop a request."""

    def __init__(self, next_path: str = "/"):
        super().__init__(status_code=303, headers={"Location": f"/auth/login?next={next_path}"})


def _set_session_cookie(response, user_id: uuid.UUID, secure: bool) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        mint_session(str(user_id)),
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


def _is_secure(request: Request) -> bool:
    """Honours X-Forwarded-Proto, since in any real deployment TLS terminates at a proxy and the
    app itself sees plain HTTP."""
    forwarded = request.headers.get("x-forwarded-proto", "")
    if forwarded:
        return forwarded.split(",")[0].strip() == "https"
    return request.url.scheme == "https"


def _lookup(db: Session, request: Request) -> User | None:
    try:
        user_id = read_session(request.cookies.get(SESSION_COOKIE, ""))
    except SessionInvalid:
        return None
    try:
        user = db.get(User, uuid.UUID(user_id))
    except (ValueError, AttributeError):
        return None
    # A deactivated account keeps its unexpired cookies; checking is_active on every request is
    # what makes deactivation take effect immediately rather than in up to fourteen days.
    return user if user and user.is_active else None


def current_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = _lookup(db, request)
    if user is None:
        raise _AuthRedirect(request.url.path)
    return user


def require_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = _lookup(db, request)
    if user is None:
        raise HTTPException(401, "sign in to continue")
    return user


def optional_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    return _lookup(db, request)


@router.get("/login")
def login_page(request: Request, next: str = "/", error: str = ""):
    return templates.TemplateResponse(
        request, "login.html", {"mode": "login", "next": next, "error": error}
    )


@router.get("/signup")
def signup_page(request: Request, next: str = "/", error: str = ""):
    return templates.TemplateResponse(
        request, "login.html", {"mode": "signup", "next": next, "error": error}
    )


@router.post("/signup")
def signup(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
    db: Session = Depends(get_db),
):
    email = email.strip().lower()
    if len(password) < MIN_PASSWORD_CHARS:
        return _back("signup", next, f"Password must be at least {MIN_PASSWORD_CHARS} characters.")
    if "@" not in email or "." not in email.split("@")[-1]:
        return _back("signup", next, "That does not look like an email address.")

    existing = db.execute(select(User).where(User.email == email)).scalars().first()
    if existing:
        # Deliberately specific. Address enumeration is a real concern on apps where membership
        # is sensitive; this is a tool you connect your own YouTube channel to, and sending
        # someone round the "wrong password" loop for an account they already have is a worse
        # trade than confirming the address is taken.
        return _back("signup", next, "An account with that email already exists — sign in instead.")

    user = User(email=email, password_hash=hash_password(password), last_login_at=datetime.now(UTC))
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("auth.signup", user_id=str(user.id))

    # Straight to onboarding: an account with no connected channel cannot do the one thing the
    # product exists for.
    response = RedirectResponse(url="/onboarding", status_code=303)
    _set_session_cookie(response, user.id, _is_secure(request))
    return response


@router.post("/login")
def login(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    next: str = Form("/"),
    db: Session = Depends(get_db),
):
    email = email.strip().lower()
    user = db.execute(select(User).where(User.email == email)).scalars().first()

    # One message for "no such account" and "wrong password", and the hash is verified even when
    # the account is missing, so the response does not differ in wording or in timing.
    if user is None:
        verify_password(password, "scrypt$AAAA$AAAA")
        return _back("login", next, "Email or password is incorrect.")
    if not verify_password(password, user.password_hash):
        logger.info("auth.login_failed", user_id=str(user.id))
        return _back("login", next, "Email or password is incorrect.")
    if not user.is_active:
        return _back("login", next, "That account is deactivated.")

    user.last_login_at = datetime.now(UTC)
    db.commit()
    logger.info("auth.login", user_id=str(user.id))

    destination = next if next.startswith("/") and not next.startswith("//") else "/"
    response = RedirectResponse(url=destination, status_code=303)
    _set_session_cookie(response, user.id, _is_secure(request))
    return response


@router.post("/logout")
@router.get("/logout")
def logout():
    response = RedirectResponse(url="/auth/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


def _back(mode: str, next_path: str, error: str) -> RedirectResponse:
    from urllib.parse import quote

    return RedirectResponse(
        url=f"/auth/{mode}?next={quote(next_path, safe='/')}&error={quote(error)}",
        status_code=303,
    )
