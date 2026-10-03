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

from api.routes.dashboard import _build_snapshot
from core.llm import call_structured
from core.logging import get_logger
from core.settings import get_settings
from db.base import get_db
from db.crud import create_run, latest_strategy_weight_adjustments, summarize_recent_performance

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


class _Intent(BaseModel):
    """What the operator wants. Kept deliberately small -- this classifies, it does not answer."""

    intent: str = Field(
        description=(
            "One of: STATUS (asking what is happening / progress / results / cost), "
            "MAKE_TOPIC (asking for a video on a subject they name), "
            "FROM_STORY (they pasted a story/script to turn into a video), "
            "IDEAS (asking for content suggestions), "
            "HELP (anything else, including greetings)"
        )
    )
    topic: str = Field(default="", description="For MAKE_TOPIC: the subject, as a short title")
    story: str = Field(default="", description="For FROM_STORY: the story text, verbatim")


class _Ideas(BaseModel):
    ideas: list[str] = Field(description="4 short, concrete video ideas for this channel")


_INTENT_SYSTEM = (
    "Classify what the operator wants from a YouTube automation pipeline. A long block of narrative "
    "prose is FROM_STORY even without an explicit instruction -- pasting a story IS the request. A "
    "short phrase naming a subject is MAKE_TOPIC. Questions about progress, runs, videos, costs or "
    "quota are STATUS. For FROM_STORY copy the story text verbatim into `story`; never summarise it."
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
