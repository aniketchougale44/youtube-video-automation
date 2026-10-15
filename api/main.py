"""FastAPI app: run-trigger API + the human-approval review dashboard."""
from fastapi import Depends, FastAPI, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routes import auth, chat, dashboard, media, onboarding, runs, webhooks
from api.routes.auth import optional_user
from core.logging import configure_logging
from core.settings import get_settings
from db.base import get_db

configure_logging()
settings = get_settings()

app = FastAPI(
    title="AutoTube AI",
    description="Autonomous multi-agent YouTube content pipeline — control API + review dashboard.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory="api/static"), name="static")

# /media is NOT a static mount. It was, and that served every render, voiceover and thumbnail to
# anyone who asked -- no session, no ownership check. Before a video is uploaded, the file under
# media_dir is the only copy that exists, and a run id in a URL is not a secret. It is now a
# route that checks the signed-in account owns the run: see api/routes/media.py.

app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
app.include_router(webhooks.router, prefix="/api/webhooks", tags=["webhooks"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(onboarding.router, prefix="/onboarding", tags=["onboarding"])
app.include_router(media.router, prefix="/media", tags=["media"])


@app.exception_handler(StarletteHTTPException)
async def _auth_redirect_handler(request, exc: StarletteHTTPException):
    """Turns the 303 that `current_user` raises into an actual browser redirect.

    FastAPI serialises HTTPException as a JSON body, so without this a signed-out visitor would
    be shown {"detail": ...} rather than being sent to the login page. Every other status is
    handled exactly as before.
    """
    if exc.status_code == 303 and exc.headers and "Location" in exc.headers:
        return RedirectResponse(url=exc.headers["Location"], status_code=303)
    return await http_exception_handler(request, exc)


@app.get("/", tags=["meta"], include_in_schema=False)
def root(request: Request, db: Session = Depends(get_db)):
    """Front door. Signed out -> login; signed in but no channel -> onboarding; otherwise the
    studio. Keeping this decision in one place stops each page re-deriving it differently."""
    user = optional_user(request, db)
    if user is None:
        return RedirectResponse(url="/auth/login", status_code=303)
    return RedirectResponse(url="/dashboard/studio", status_code=303)


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "env": settings.app_env}
