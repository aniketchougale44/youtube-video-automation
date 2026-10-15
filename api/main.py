"""FastAPI app: run-trigger API + the human-approval review dashboard."""
from fastapi import FastAPI
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routes import auth, chat, dashboard, runs, webhooks
from core.logging import configure_logging
from core.settings import get_settings

configure_logging()
settings = get_settings()

app = FastAPI(
    title="AutoTube AI",
    description="Autonomous multi-agent YouTube content pipeline — control API + review dashboard.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory="api/static"), name="static")
# Renders/thumbnails a run produced under settings.media_dir, so the review dashboard can show the
# real thumbnail images and link the final video instead of grey placeholders. Read-only; the
# dashboard only ever builds URLs under here via dashboard._media_url (which refuses paths outside
# media_dir), and StaticFiles itself blocks path traversal on the serving side.
app.mount("/media", StaticFiles(directory=settings.media_dir), name="media")

app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
app.include_router(webhooks.router, prefix="/api/webhooks", tags=["webhooks"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(auth.router, prefix="/auth", tags=["auth"])


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


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "env": settings.app_env}
