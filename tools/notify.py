"""Notification Agent's transport layer. Dispatches NotificationEvent to every channel that has
credentials configured — Slack, Telegram, and email are all fully wired; each is independently
optional, and a failure in one channel never blocks the others."""
import smtplib
from email.message import EmailMessage

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


@with_resilience(provider="telegram_bot_api", max_attempts=2)
def _post_telegram(bot_token: str, chat_id: str, text: str) -> None:
    resp = httpx.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": text},
        timeout=10,
    )
    resp.raise_for_status()


@with_resilience(provider="smtp", max_attempts=2)
def _send_email(subject: str, text: str) -> None:
    settings = get_settings()
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.alert_email_from or settings.smtp_user
    message["To"] = settings.alert_email_to
    message.set_content(text)

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
        smtp.starttls()
        if settings.smtp_user:
            smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(message)


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
        try:
            _post_telegram(settings.telegram_bot_token, settings.telegram_chat_id, text)
        except Exception as exc:
            logger.error("notify.telegram_failed", error=str(exc))

    if settings.alert_email_to and settings.smtp_host:
        try:
            _send_email(f"AutoTube AI: {event.severity.value.upper()}", text)
        except Exception as exc:
            logger.error("notify.email_failed", error=str(exc))
