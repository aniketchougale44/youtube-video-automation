"""Chat control surface for the studio dashboard.

One endpoint, POST /api/chat, that turns a sentence into either an answer about the pipeline or a
queued run. The LLM is used only to classify intent and pull out a topic or story -- every answer
about system state is assembled from Postgres and Redis, never generated. That split is deliberate:
an operator asking "what's happening right now" needs a fact, and a model that paraphrases run
status will eventually paraphrase it wrong.

Triggering a run here creates the same Run row the scheduler creates, so a chat-requested video
goes through the identical graph, critics and approval gate as a cron-triggered one. There is no
second, weaker path to publishing.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from agents.schemas.common import RunStatus
from api.routes.dashboard import _approval_rows, _build_snapshot
from core.llm import call_structured
from core.logging import get_logger
from core.settings import get_settings
from db.base import get_db
from db.crud import (
    create_run,
    latest_strategy_weight_adjustments,
    list_runs,
    summarize_recent_performance,
)

logger = get_logger("api.chat")
router = APIRouter()

MAX_STORY_CHARS = 20_000


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_STORY_CHARS + 2_000)


class ChatReply(BaseModel):
    reply: str
    intent: str
    run_id: str | None = None
    suggestions: list[str] = Field(default_factory=list)
    # Populated whenever a finished video is waiting on the operator. The dashboard renders this
    # as a player + approve/change controls inside the conversation, so reviewing never means
    # leaving the chat for a separate screen.
    review: dict | None = None


class _Intent(BaseModel):
    """What the operator wants. Kept deliberately small -- this classifies, it does not answer."""

    intent: str = Field(
        description=(
            "One of: STATUS (asking what is happening / progress / results / cost), "
            "MAKE_TOPIC (asking for a video on a subject they name), "
            "FROM_STORY (they pasted a story/script to turn into a video), "
            "IDEAS (asking for content suggestions), "
            "REVIEW (asking to see / preview / check the video that is waiting), "
            "APPROVE (telling you to publish or upload the video that is waiting), "
            "REJECT (telling you to scrap, bin or cancel the video that is waiting), "
            "REVISE (asking to change the title or description of the waiting video), "
            "HELP (anything else, including greetings)"
        )
    )
    topic: str = Field(default="", description="For MAKE_TOPIC: the subject, as a short title")
    story: str = Field(default="", description="For FROM_STORY: the story text, verbatim")
    new_title: str = Field(
        default="",
        description="For REVISE: the replacement title, if they gave one. Empty if they only said what to change in general terms.",
    )
    new_description: str = Field(
        default="", description="For REVISE: the replacement description, if they gave one."
    )


class _Ideas(BaseModel):
    ideas: list[str] = Field(description="4 short, concrete video ideas for this channel")


_INTENT_SYSTEM = (
    "Classify what the operator wants from a YouTube automation pipeline. A long block of narrative "
    "prose is FROM_STORY even without an explicit instruction -- pasting a story IS the request. A "
    "short phrase naming a subject is MAKE_TOPIC. Questions about progress, runs, videos, costs or "
    "quota are STATUS. For FROM_STORY copy the story text verbatim into `story`; never summarise it. "
    "A finished video may be waiting for the operator's decision: 'show it to me' / 'let me see' is "
    "REVIEW, 'upload it' / 'publish' / 'yes, send it' / 'looks good' is APPROVE, 'scrap it' / 'no, "
    "cancel' is REJECT, and anything asking for a different title or description is REVISE. For "
    "REVISE, put the replacement text in new_title/new_description only when the operator actually "
    "dictated it -- leave them empty when they just said something was wrong with it."
)


def _status_answer(db: Session) -> str:
    """Assembled from real rows, not generated. See the module docstring."""
    settings = get_settings()
    snapshot = _build_snapshot(db)

    runs = snapshot.get("runs", [])
    stats = snapshot.get("stats", {})

    lines: list[str] = []
    active = [r for r in runs if r.get("status") in ("running", "pending", "awaiting_approval")]
    if active:
        for run in active[:3]:
            stage = run.get("stage_label") or "starting"
            done, total = run.get("progress") or 0, run.get("progress_total") or 0
            step = f" ({done}/{total} stages)" if total else ""
            lines.append(f"**{run.get('status')}** — `{run.get('topic')}` at **{stage}**{step}")
    else:
        lines.append("No run is in flight right now.")

    published = [r for r in runs if r.get("youtube_url")]
    if published:
        newest = published[0]
        lines.append(f"Last published: **{newest.get('topic')}** → {newest.get('youtube_url')}")

    failed = [r for r in runs if r.get("error")]
    if failed:
        lines.append(f"⚠ Most recent failure on `{failed[0].get('topic')}`: {str(failed[0]['error'])[:160]}")

    lines.append(
        f"Totals: **{stats.get('published', 0)}** published, "
        f"**{stats.get('needs_attention', 0)}** needing attention, "
        f"${stats.get('cost_total_usd', 0)} measured spend all-time."
    )

    try:
        from tools import quota

        remaining = quota.remaining_units()
        lines.append(f"YouTube quota: **{remaining:,}** units left today (a full publish costs ~2,050).")
    except Exception as exc:  # Redis down is worth saying plainly rather than hiding
        lines.append(f"Quota unavailable ({type(exc).__name__}) — is Redis running?")

    adjustments = latest_strategy_weight_adjustments(db)
    if adjustments:
        lines.append(f"Learned from performance so far: `{adjustments}`.")

    perf = summarize_recent_performance(db)
    if perf and "no" not in perf.lower()[:12]:
        lines.append(perf.strip().split("\n")[1].strip() if "\n" in perf else perf.strip())

    if not settings.require_human_approval:
        lines.append("_Approval gate is off — a finished run publishes straight to YouTube (unlisted)._")

    return "\n\n".join(lines)


def _ideas_answer(db: Session) -> list[str]:
    perf = summarize_recent_performance(db) or "No performance data yet."
    settings = get_settings()
    try:
        out = call_structured(
            f"Channel goals: {settings.channel_goals}\n\nRecent performance:\n{perf}\n\n"
            "Suggest 4 video ideas that fit this channel and what has worked.",
            _Ideas,
            system=(
                "You suggest YouTube video ideas. Each idea is one short line an operator could "
                "paste straight back as a topic request. Be concrete about the subject and format; "
                "no preamble, no numbering, no explanation."
            ),
        )
        return out.ideas[:4]
    except Exception as exc:
        logger.warning("chat.ideas_failed", error=str(exc))
        return []


def _pending_review(db: Session) -> dict | None:
    """The oldest run parked at the approval gate, as a dashboard-style card, or None.

    Oldest rather than newest on purpose: if two videos are waiting, "upload it" should clear the
    queue in the order the operator saw them, not jump to whichever finished last.
    """
    waiting = list_runs(db, status=RunStatus.AWAITING_APPROVAL)
    if not waiting:
        return None
    waiting.sort(key=lambda r: r.updated_at or r.created_at)
    rows = _approval_rows(db, waiting[:1])
    return rows[0] if rows else None


def _review_text(card: dict) -> str:
    """The spoken half of a review. The card itself carries the player and the buttons, so this
    only says what a reviewer cannot see at a glance."""
    parts = [f"**{card['title']}** is ready — have a look before it goes anywhere."]

    compliance = card.get("compliance")
    if compliance:
        if not compliance.get("passed"):
            reasons = ", ".join(compliance.get("blocking_reasons") or []) or "unspecified"
            parts.append(f"⚠️ Compliance did **not** pass: {reasons}")
        else:
            bits = ["compliance passed"]
            risk = compliance.get("copyright_risk_score")
            if risk is not None:
                bits.append(f"copyright risk {risk}")
            if not compliance.get("av_sync_ok"):
                bits.append("**audio/video out of sync**")
            if not compliance.get("spec_check_ok"):
                bits.append("**spec check failed**")
            parts.append(" · ".join(bits))
    else:
        parts.append("I could not read the compliance result from the checkpoint for this one.")

    if not card.get("render_url"):
        parts.append(
            "No rendered file is attached to this run, so there is nothing to play — worth "
            "opening the full review before approving."
        )

    parts.append(
        "Tell me **upload it** to publish, **change the title to …** to edit it first, or "
        "**scrap it** to bin the run."
    )
    return "\n\n".join(parts)


def _no_pending_reply(intent: str) -> ChatReply:
    return ChatReply(
        reply=(
            "Nothing is waiting for a decision right now. When a video finishes I'll show it to "
            "you here first, and it won't upload until you say so."
        ),
        intent=intent,
    )


@router.post("", response_model=ChatReply)
@router.post("/", response_model=ChatReply)
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatReply:
    message = request.message.strip()

    try:
        parsed = call_structured(message, _Intent, system=_INTENT_SYSTEM)
        intent = (parsed.intent or "HELP").strip().upper()
    except Exception as exc:
        logger.warning("chat.intent_failed", error=str(exc))
        # Without an LLM the dashboard should still answer "what's happening", which is the
        # question an operator asks when something looks wrong -- exactly when the LLM may be down.
        return ChatReply(reply=_status_answer(db), intent="STATUS")

    logger.info("chat.intent", intent=intent, chars=len(message))

    if intent == "STATUS":
        return ChatReply(reply=_status_answer(db), intent=intent)

    if intent == "IDEAS":
        ideas = _ideas_answer(db)
        reply = "Here are four that fit your channel and what's performed:" if ideas else (
            "I couldn't reach the LLM for ideas just now."
        )
        return ChatReply(reply=reply, intent=intent, suggestions=ideas)

    if intent == "REVIEW":
        card = _pending_review(db)
        if card is None:
            return _no_pending_reply(intent)
        return ChatReply(reply=_review_text(card), intent=intent, run_id=card["id"], review=card)

    if intent == "APPROVE":
        card = _pending_review(db)
        if card is None:
            return _no_pending_reply(intent)
        _resume(card["id"], approved=True, notes=message)
        return ChatReply(
            reply=(
                f"Publishing **{card['title']}** now — uploading to YouTube as "
                f"**{get_settings().youtube_upload_visibility}**, then the thumbnail. "
                "Ask me what's happening to watch it land."
            ),
            intent=intent,
            run_id=card["id"],
        )

    if intent == "REJECT":
        card = _pending_review(db)
        if card is None:
            return _no_pending_reply(intent)
        _resume(card["id"], approved=False, notes=message)
        return ChatReply(
            reply=(
                f"Scrapped **{card['title']}** — it will not be uploaded. The render stays on disk "
                "in case you want it later."
            ),
            intent=intent,
            run_id=card["id"],
        )

    if intent == "REVISE":
        card = _pending_review(db)
        if card is None:
            return _no_pending_reply(intent)
        new_title = (parsed.new_title or "").strip()
        new_description = (parsed.new_description or "").strip()
        if not new_title and not new_description:
            # The model understood "the title is wrong" but has nothing to write. Asking is better
            # than inventing a title the operator never chose for a video about to go public.
            return ChatReply(
                reply=(
                    "Happy to change it — what should it say instead? The title now is "
                    f"**{card['title']}**.\n\nGive me the exact words, like _change the title to "
                    "Counting to Ten with Friendly Animals_."
                ),
                intent="HELP",
                run_id=card["id"],
                review=card,
            )
        # Editing a parked run in place is not wired up yet, and the half-built version was worse
        # than nothing: writing the checkpoint destroyed the pending interrupt, which emptied the
        # review packet, and the only way found to rebuild it re-entered the gate -- where a queued
        # decision could carry the run straight to upload. Until the edit can be carried in the
        # approval decision itself, say so plainly rather than risk an unasked-for publish.
        wanted = new_title or new_description
        return ChatReply(
            reply=(
                f"I can't edit **{card['title']}** in place yet — that path isn't finished, and the "
                "unfinished version risked publishing the video instead of editing it.\n\n"
                f"What you asked for: _{wanted}_\n\n"
                "For now: **scrap it** and name the topic again with the wording you want, or "
                "**upload it** as it stands and rename it on YouTube, which takes a few seconds "
                "there and costs nothing here."
            ),
            intent="HELP",
            run_id=card["id"],
            review=card,
        )

    if intent == "MAKE_TOPIC":
        topic = (parsed.topic or message)[:300]
        run = create_run(db, user_topic=topic)
        _enqueue(str(run.id))
        return ChatReply(
            reply=(
                f"Queued a video on **{topic}**.\n\n"
                "It skips trend research (saving ~400 quota units, since you named the topic) and "
                "still goes through both critics and the compliance gate. Ask me what's happening "
                "to follow it."
            ),
            intent=intent,
            run_id=str(run.id),
        )

    if intent == "FROM_STORY":
        story = (parsed.story or message)[:MAX_STORY_CHARS]
        if len(story.split()) < 20:
            return ChatReply(
                reply="That looks too short to build a video from — paste the full story and I'll use your words.",
                intent="HELP",
            )
        run = create_run(db, user_script=story)
        _enqueue(str(run.id))
        return ChatReply(
            reply=(
                f"Queued a video from your story ({len(story.split())} words).\n\n"
                "Your wording is segmented into beats rather than rewritten. It still passes through "
                "the originality and compliance critics — worth knowing if the story isn't yours to "
                "publish."
            ),
            intent=intent,
            run_id=str(run.id),
        )

    return ChatReply(
        reply=(
            "I can do four things:\n\n"
            "- **Tell you what's happening** — \"what's going on right now?\"\n"
            "- **Make a video on a topic** — \"make a video about counting to ten\"\n"
            "- **Use your story** — paste it and I'll build the video from your words\n"
            "- **Suggest ideas** — \"give me some ideas\""
        ),
        intent="HELP",
    )


def _enqueue(run_id: str) -> None:
    """Hands the run to Celery. Imported late to avoid a circular import at module load, the same
    way scheduler/jobs.py does."""
    from worker.tasks import run_pipeline_task

    run_pipeline_task.delay(run_id)
    logger.info("chat.run_enqueued", run_id=run_id)


def _resume(run_id: str, approved: bool, notes: str = "") -> None:
    """Hands an approve/reject decision to the same Celery task the review dashboard uses, so a
    chat decision and a dashboard click take an identical path through the graph."""
    from worker.tasks import resume_pipeline_task

    decision = {
        "run_id": run_id,
        "approved": approved,
        "selected_thumbnail_index": 0,
        "reviewer": "chat",
        "notes": notes[:2_000],
    }
    resume_pipeline_task.delay(run_id, decision)
    logger.info("chat.run_resumed", run_id=run_id, approved=approved)


def _revise(run_id: str, title: str = "", description: str = "") -> None:
    from worker.tasks import revise_metadata_task

    revise_metadata_task.delay(run_id, {"title": title, "description": description})
    logger.info("chat.run_revised", run_id=run_id, title=bool(title), description=bool(description))
