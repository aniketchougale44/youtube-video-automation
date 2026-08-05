"""Inbound webhooks (stub). Placeholder for future integrations — e.g. a YouTube PubSubHubbub
push-notification subscription, or a third-party TTS/image-gen provider's async completion callback."""
from fastapi import APIRouter, Request

from core.logging import get_logger

router = APIRouter()
logger = get_logger("api.webhooks")


@router.post("/generic")
async def generic_webhook(request: Request) -> dict:
    payload = await request.json()
    logger.info("webhook.received", payload=payload)
    return {"received": True}
