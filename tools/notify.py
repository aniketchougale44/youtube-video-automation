"""Notification Agent's transport layer. Dispatches NotificationEvent to whichever channels are
configured — Slack is fully wired (it's just a webhook POST); Telegram/email are stub hooks
with the same call shape, ready to fill in when those providers are prioritized."""
import httpx

from agents.schemas.feedback import NotificationEvent
from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import with_resilience

logger = get_logger("tools.notify")


@with_resilience(provider="slack_webhook", max_attempts=2)
def _post_slack(webhook_url: str, text: str) -> None:
    resp = httpx.post(webhook_url, json={"text": text}, timeout=10)
    resp.raise_for_status()


def _format(event: NotificationEvent) -> str:
    text = f"[{event.severity.value.upper()}] {event.message}"
    if event.run_id:
        text += f" (run_id={event.run_id})"
    if event.stage:
        text += f" (stage={event.stage.value})"
    return text


def send_notification(event: NotificationEvent) -> None:
    settings = get_settings()
    text = _format(event)
    logger.info("notify.dispatch", severity=event.severity.value, message=event.message)

    if settings.slack_webhook_url:
        try:
            _post_slack(settings.slack_webhook_url, text)
        except Exception as exc:
            logger.error("notify.slack_failed", error=str(exc))
    else:
        logger.warning("notify.slack_not_configured", text=text)

    if settings.telegram_bot_token and settings.telegram_chat_id:
        logger.info("notify.telegram.stub", text=text)  # STUB: POST to Telegram Bot API sendMessage

    if settings.alert_email_to:
        logger.info("notify.email.stub", text=text)  # STUB: SMTP send via settings.smtp_*
