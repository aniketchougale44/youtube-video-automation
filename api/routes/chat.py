"""Chat control surface for the studio dashboard.

One endpoint, POST /api/chat. The model is given a set of tools and chooses WHICH one runs and
with what arguments; it never composes the answer. Every tool assembles its reply from Postgres
and Redis, and that text is what the operator sees. The split is deliberate: an operator asking
"what's happening right now" needs a fact, and a model that paraphrases run status will
eventually paraphrase it wrong -- convincingly. Routing is the model's job; the facts are not.

No tool publishes. Approving is matched here by exact phrase, before the model is consulted at
all, so a misread sentence cannot put a video on someone's channel -- see _publish_decision.

Triggering a run here creates the same Run row the scheduler creates, so a chat-requested video
goes through the identical graph, critics and approval gate as a cron-triggered one. There is no
second, weaker path to publishing.
"""

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from agents.schemas.common import RunStatus
from api.routes.auth import require_user
from api.routes.dashboard import _approval_rows, _build_snapshot
from core.limits import RateLimited, check_can_start_run
from core.llm import call_structured, choose_tool
from core.logging import get_logger
from core.settings import get_settings
from db.base import get_db
from db.crud import (
    create_run,
    latest_strategy_weight_adjustments,
    list_runs,
    summarize_recent_performance,
)
from db.models import ChatMessage, Video

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


class _Ideas(BaseModel):
    ideas: list[str] = Field(description="4 short, concrete video ideas for this channel")


def _status_answer(db: Session, user_id=None) -> str:
    """Assembled from real rows, not generated. See the module docstring.

    Scoped to the caller: "what's happening" must mean their runs and their spend, not the
    deployment's.
    """
    settings = get_settings()
    snapshot = _build_snapshot(db, user_id=user_id)

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


def _pending_review(db: Session, user_id=None) -> dict | None:
    """The oldest run parked at the approval gate, as a dashboard-style card, or None.

    Oldest rather than newest on purpose: if two videos are waiting, "upload it" should clear the
    queue in the order the operator saw them, not jump to whichever finished last.
    """
    waiting = list_runs(db, status=RunStatus.AWAITING_APPROVAL, user_id=user_id)
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


# An approval publishes to YouTube, so it may not rest on the classifier reading a sentence the
# way a person would. "upload a video of story for children" is a request to MAKE something, but
# it opens with the same verb as "upload it" and was observed classifying as APPROVE -- which
# would have published whatever was parked instead of building the story that was asked for.
#
# The rule below is deliberately blunt: a real approval is a short reference to something that
# already exists. Anything describing subject matter is not a confirmation, whatever the
# classifier decided. False positives here cost one clarifying question; a false negative costs
# an unwanted publish.
_CONFIRMATION_MAX_WORDS = 8
_NEW_CONTENT_HINTS = (
    "story", "video of", "video about", "make ", "create", "another", "new ", "about ",
)


def _describes_new_content(message: str) -> bool:
    low = message.lower()
    if len(low.split()) > _CONFIRMATION_MAX_WORDS:
        return True
    return any(hint in low for hint in _NEW_CONTENT_HINTS)


def _ambiguous_reply(card: dict, message: str) -> ChatReply:
    return ChatReply(
        reply=(
            f"I want to be sure which you mean, because one of these publishes to YouTube.\n\n"
            f"**{card['title']}** is finished and waiting. Did you mean:\n\n"
            f"- **publish that one** — say _upload it_ or _publish it_\n"
            f"- **make a new video** — say _make a video about …_ with the subject\n\n"
            f"I haven't done either yet."
        ),
        intent="HELP",
        run_id=card["id"],
        review=card,
    )


def _no_pending_reply(intent: str) -> ChatReply:
    return ChatReply(
        reply=(
            "Nothing is waiting for a decision right now. When a video finishes I'll show it to "
            "you here first, and it won't upload until you say so."
        ),
        intent=intent,
    )



MAX_HISTORY = 60


def _remember(db: Session, user_id, role: str, content: str,
              intent: str | None = None, run_id: str | None = None, review: dict | None = None) -> None:
    """Appends one turn. Best-effort: a conversation that cannot be written down is not a reason
    to fail the reply the operator is waiting for."""
    try:
        db.add(ChatMessage(
            user_id=user_id, role=role, content=content, intent=intent,
            run_id=uuid.UUID(run_id) if run_id else None, review=review,
        ))
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("chat.history_write_failed", error=str(exc))


@router.get("/history")
def history(db: Session = Depends(get_db), user=Depends(require_user)) -> dict:
    """The conversation so far, oldest first, so the studio can restore itself on load.

    Capped at the most recent MAX_HISTORY turns: the page replays every one of these into the
    DOM on open, and an operator who has been using this for months should not wait on a
    thousand of them to find the video waiting for a decision.
    """
    rows = db.execute(
        select(ChatMessage)
        .where(ChatMessage.user_id == user.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(MAX_HISTORY)
    ).scalars().all()
    return {
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "intent": m.intent,
                "run_id": str(m.run_id) if m.run_id else None,
                "review": m.review,
                "at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in reversed(rows)  # query is newest-first to apply the limit; replay oldest-first
        ]
    }


@router.delete("/history")
def clear_history(db: Session = Depends(get_db), user=Depends(require_user)) -> dict:
    deleted = db.query(ChatMessage).filter(ChatMessage.user_id == user.id).delete()
    db.commit()
    logger.info("chat.history_cleared", user_id=str(user.id), deleted=deleted)
    return {"deleted": deleted}


@router.post("", response_model=ChatReply)
@router.post("/", response_model=ChatReply)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    user=Depends(require_user),
) -> ChatReply:
    """Records both sides of the turn around the reply.

    A thin wrapper rather than a _remember() call before each return: the handler below has a
    dozen exit points, one per intent, and the one that gets forgotten would be the one whose
    message quietly vanished from the history.
    """
    message = request.message.strip()
    _remember(db, user.id, "user", message)
    reply = _answer(message, db, user)
    _remember(
        db, user.id, "assistant", reply.reply,
        intent=reply.intent, run_id=reply.run_id, review=reply.review,
    )
    return reply


# --------------------------------------------------------------------------- the agent
#
# The model chooses WHICH action to take and with what arguments; it does not compose the answer.
# Every tool below returns text assembled from real rows, and that text is what the operator
# sees. The split is the same one the module docstring describes, and it is the reason a model
# that has a bad day here gets the routing wrong -- answerable with "no, I meant X" -- rather
# than inventing a run status that reads perfectly and is false.
#
# There is deliberately NO tool that publishes. See _publish_decision below.


class GetStatus(BaseModel):
    """What the pipeline is doing right now: runs in flight, the last published video, failures,
    spend and remaining YouTube quota. Use for any question about progress, cost or quota."""


class StartVideo(BaseModel):
    """Start making a video about a subject the operator named."""

    topic: str = Field(description="the subject, as a short title, in the operator's own words")


class StartFromStory(BaseModel):
    """Build a video from a story or script the operator pasted, using their own wording."""

    story: str = Field(description="the story text, copied verbatim, never summarised")


class ShowPendingReview(BaseModel):
    """Show the finished video that is waiting for a decision, so the operator can watch it."""


class SuggestIdeas(BaseModel):
    """Suggest video ideas that fit this channel and what has performed well."""


class ExplainLastFailure(BaseModel):
    """Explain why the most recent run failed, in terms of what to do about it."""


class ListPublished(BaseModel):
    """List the videos published most recently, with their links."""


class GetHelp(BaseModel):
    """Explain what this assistant can do. Use when nothing else fits, or for a greeting."""


_TOOLS = [
    GetStatus, StartVideo, StartFromStory, ShowPendingReview,
    SuggestIdeas, ExplainLastFailure, ListPublished, GetHelp,
]

_AGENT_SYSTEM = (
    "You route an operator's message to exactly one action in a YouTube automation pipeline. "
    "Call the single tool that does what they asked; do not answer in prose. "
    "A long block of narrative prose is StartFromStory even with no instruction attached -- "
    "pasting a story IS the request, and the story must be copied verbatim, never summarised. "
    "A short phrase naming a subject is StartVideo. Questions about progress, runs, cost or "
    "quota are GetStatus. 'why did it fail' is ExplainLastFailure. "
    "A publish verb does not by itself mean publish: 'upload a story for kids' describes "
    "something that does not exist yet, so it is StartVideo. If the operator seems to be "
    "approving or scrapping the video that is waiting, call ShowPendingReview -- you cannot "
    "publish, and showing them the video with its buttons is the correct response."
)


# Approving publishes to YouTube, so it is decided here, by exact phrase, and never by the model.
#
# The guard that already existed asks "does this describe new content", which stops
# "upload a story for kids" publishing. It does not stop a misrouted "what's happening" being
# read as approval -- short, no content words, passes the guard. Matching an explicit
# confirmation closes that: the model cannot cause a publish by misunderstanding, because the
# model is not consulted. Anything it does not match falls through to the agent, which can only
# offer the card and its buttons.
_APPROVE_PHRASES = frozenset({
    "upload it", "upload", "publish it", "publish", "approve", "approve it", "ship it",
    "send it", "post it", "go ahead", "do it", "yes", "yep", "yes please", "looks good",
    "lgtm", "upload it now", "publish it now", "approved",
})
_REJECT_PHRASES = frozenset({
    "scrap it", "scrap", "reject", "reject it", "bin it", "delete it", "cancel it", "cancel",
    "no", "nope", "don't upload", "do not upload", "discard", "discard it", "throw it away",
})


def _normalise(message: str) -> str:
    return " ".join(message.lower().strip().strip(".!").split())


def _publish_decision(message: str, db: Session, user) -> ChatReply | None:
    """Handles approve/scrap without consulting the model, or returns None to let the agent run."""
    phrase = _normalise(message)
    approving = phrase in _APPROVE_PHRASES
    rejecting = phrase in _REJECT_PHRASES
    if not (approving or rejecting):
        return None

    card = _pending_review(db, user.id)
    if card is None:
        return _no_pending_reply("APPROVE" if approving else "REJECT")

    # Belt and braces: the phrase list cannot match anything describing new content, but the
    # guard stays because it is the rule, not an implementation detail of the list.
    if _describes_new_content(message):
        return _ambiguous_reply(card, message)

    if approving:
        _resume(card["id"], approved=True, notes=message)
        return ChatReply(
            reply=(
                f"Publishing **{card['title']}** now — uploading to YouTube as "
                f"**{get_settings().youtube_upload_visibility}**, then the thumbnail. "
                "Ask me what's happening to watch it land."
            ),
            intent="APPROVE",
            run_id=card["id"],
        )

    _resume(card["id"], approved=False, notes=message)
    return ChatReply(
        reply=(
            f"Scrapped **{card['title']}** — it will not be uploaded. The render stays on disk "
            "in case you want it later."
        ),
        intent="REJECT",
        run_id=card["id"],
    )


# --------------------------------------------------------------------------- tool handlers


def _tool_status(args: dict, db: Session, user) -> ChatReply:
    return ChatReply(reply=_status_answer(db, user.id), intent="STATUS")


def _tool_start_video(args: dict, db: Session, user) -> ChatReply:
    topic = (args.get("topic") or "").strip()[:300]
    if not topic:
        return ChatReply(
            reply="What should the video be about? Name the subject and I'll start it.",
            intent="HELP",
        )
    try:
        check_can_start_run(db, user.id)
    except RateLimited as limit:
        return ChatReply(reply=str(limit), intent="HELP")
    run = create_run(db, user_topic=topic, user_id=user.id)
    _enqueue(str(run.id))
    return ChatReply(
        reply=(
            f"Started a video on **{topic}**.\n\n"
            "It skips trend research (saving ~400 quota units, since you named the topic) and "
            "still goes through both critics and the compliance gate. When it's done I'll show "
            "it to you here — nothing publishes until you say so."
        ),
        intent="MAKE_TOPIC",
        run_id=str(run.id),
    )


def _tool_start_from_story(args: dict, db: Session, user) -> ChatReply:
    story = (args.get("story") or "").strip()[:MAX_STORY_CHARS]
    if len(story.split()) < 20:
        return ChatReply(
            reply="That looks too short to build a video from — paste the full story and I'll use your words.",
            intent="HELP",
        )
    try:
        check_can_start_run(db, user.id)
    except RateLimited as limit:
        return ChatReply(reply=str(limit), intent="HELP")
    run = create_run(db, user_script=story, user_id=user.id)
    _enqueue(str(run.id))
    return ChatReply(
        reply=(
            f"Started a video from your story ({len(story.split())} words).\n\n"
            "Your wording is segmented into beats rather than rewritten. It still passes through "
            "the originality and compliance critics — worth knowing if the story isn't yours to "
            "publish. I'll show you the result before anything is uploaded."
        ),
        intent="FROM_STORY",
        run_id=str(run.id),
    )


def _tool_show_review(args: dict, db: Session, user) -> ChatReply:
    card = _pending_review(db, user.id)
    if card is None:
        return _no_pending_reply("REVIEW")
    return ChatReply(reply=_review_text(card), intent="REVIEW", run_id=card["id"], review=card)


def _tool_ideas(args: dict, db: Session, user) -> ChatReply:
    ideas = _ideas_answer(db)
    return ChatReply(
        reply=(
            "Here are four that fit your channel and what's performed:" if ideas
            else "I couldn't reach the LLM for ideas just now."
        ),
        intent="IDEAS",
        suggestions=ideas,
    )


def _tool_explain_failure(args: dict, db: Session, user) -> ChatReply:
    """Reads the failure off the run, and translates the ones with a known cause.

    The raw error is the truth and is always shown, but "invalid_grant" and a 403 from a speech
    endpoint do not tell an operator what to do, and both cost a full render before they appear.
    """
    failed = [
        r for r in list_runs(db, limit=20, user_id=user.id)
        if r.status in (RunStatus.FAILED, RunStatus.ESCALATED)
    ]
    if not failed:
        return ChatReply(reply="Nothing has failed recently — the last runs all completed.", intent="STATUS")

    run = failed[0]
    error = (run.error or "").strip() or "no error was recorded"
    topic = (run.selected_topic or {}).get("title") or (
        (run.stage_status or {}).get("user_inputs") or {}
    ).get("topic") or "that run"

    hint = ""
    low = error.lower()
    if "invalid_grant" in low:
        hint = (
            "\n\nThat is your YouTube authorisation, not the video. Reconnect your channel in "
            "onboarding — and if your Google consent screen is still in Testing, publish it, "
            "because Testing expires the connection every 7 days."
        )
    elif "speech.platform.bing.com" in low or "edge" in low and "403" in low:
        hint = "\n\nThat is the free text-to-speech provider refusing the request, not your script."
    elif "no usable llm provider" in low:
        hint = (
            "\n\nEvery language-model provider failed at once, which is usually a passing outage "
            "rather than a key problem. Starting it again is the right move."
        )
    elif "quota" in low:
        hint = "\n\nThat is the daily YouTube API budget for your Google project, which resets at midnight Pacific."

    return ChatReply(
        reply=f"**{topic}** failed at **{run.current_stage or 'an early stage'}**.\n\n`{error[:400]}`{hint}",
        intent="STATUS",
        run_id=str(run.id),
    )


def _tool_list_published(args: dict, db: Session, user) -> ChatReply:
    runs = [r for r in list_runs(db, limit=30, user_id=user.id) if r.status == RunStatus.PUBLISHED]
    rows = []
    for run in runs[:6]:
        video = db.execute(
            select(Video).where(Video.run_id == run.id).order_by(Video.created_at.desc())
        ).scalars().first()
        if video and video.youtube_video_id:
            rows.append(f"- **{video.title or 'Untitled'}** — https://youtube.com/watch?v={video.youtube_video_id}")
    if not rows:
        return ChatReply(
            reply="Nothing published from this account yet. Name a topic and I'll make the first one.",
            intent="STATUS",
        )
    return ChatReply(reply="Published so far:\n\n" + "\n".join(rows), intent="STATUS")


def _tool_help(args: dict, db: Session, user) -> ChatReply:
    return ChatReply(reply=_HELP_TEXT, intent="HELP")


_HELP_TEXT = (
    "Here's what I can do:\n\n"
    "- **Make a video** — name a subject, like _a story about counting to ten_\n"
    "- **Use your own story** — paste it and I'll build the video from your words\n"
    "- **Show you what's happening** — runs in flight, spend, quota\n"
    "- **Explain a failure** — _why did the last run fail?_\n"
    "- **Show the finished video** — _show me the one waiting_\n"
    "- **Suggest ideas** — _give me some ideas_\n"
    "- **List what's published**\n\n"
    "When a video is finished I'll show it here first. Say **upload it** to publish or "
    "**scrap it** to bin it — nothing goes to YouTube until you do."
)

_HANDLERS = {
    "GetStatus": _tool_status,
    "StartVideo": _tool_start_video,
    "StartFromStory": _tool_start_from_story,
    "ShowPendingReview": _tool_show_review,
    "SuggestIdeas": _tool_ideas,
    "ExplainLastFailure": _tool_explain_failure,
    "ListPublished": _tool_list_published,
    "GetHelp": _tool_help,
}


def _answer(message: str, db: Session, user) -> ChatReply:
    # Publishing is decided before the model is consulted, and the model has no tool for it.
    decision = _publish_decision(message, db, user)
    if decision is not None:
        return decision

    try:
        chosen = choose_tool(message, _TOOLS, system=_AGENT_SYSTEM)
    except Exception as exc:
        logger.warning("chat.tool_choice_failed", error=str(exc))
        # The question an operator asks when something looks wrong is "what's happening" -- which
        # is exactly when the model is most likely to be the thing that is wrong. Answer it from
        # the database instead of apologising.
        return ChatReply(reply=_status_answer(db, user.id), intent="STATUS")

    if chosen is None:
        return _tool_help({}, db, user)

    name, args = chosen
    handler = _HANDLERS.get(name)
    if handler is None:
        logger.warning("chat.unknown_tool", tool=name)
        return _tool_help({}, db, user)

    logger.info("chat.tool", tool=name, chars=len(message))
    return handler(args, db, user)


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
