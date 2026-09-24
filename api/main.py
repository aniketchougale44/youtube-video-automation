"""FastAPI app: run-trigger API + the human-approval review dashboard."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routes import dashboard, runs, webhooks
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


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "env": settings.app_env}
