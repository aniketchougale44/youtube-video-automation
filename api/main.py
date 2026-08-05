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

app.include_router(runs.router, prefix="/api/runs", tags=["runs"])
app.include_router(webhooks.router, prefix="/api/webhooks", tags=["webhooks"])
app.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])


@app.get("/health", tags=["meta"])
def health() -> dict:
    return {"status": "ok", "env": settings.app_env}
