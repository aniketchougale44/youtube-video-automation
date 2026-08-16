"""Trend Research Agent + Strategy Agent — real logic.

Trend Research pulls YouTube "most popular" videos (a public, API-key-only call) as the primary
trend signal, has an LLM cluster them into distinct *original* content angles (never verbatim
topics/titles to copy), then scores each candidate on objective, code-computed signals — not LLM
guesses — so scoring stays consistent and auditable:
  - competition_score  <- avg view count of the representative trending videos (log-scaled)
  - freshness_score    <- avg recency of those videos
  - search_volume_score<- Google Trends rising/top query hits for the candidate topic (best-effort;
                           Trends is notoriously rate-limited from datacenter IPs, so a failure here
                           degrades to a view-count-derived proxy instead of failing the whole stage)

Strategy then has an LLM pick one candidate and decide format/length, given channel goals and
whatever past-performance summary the caller injected into state (worker/tasks.py, from Postgres).
"""
import math
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from agents.schemas.common import ContentFormat, ContentType
from agents.schemas.research import StrategyDecision, TopicCandidate, TrendResearchOutput
from core.llm import call_structured
from core.logging import get_logger
from core.settings import get_settings
from graph.nodes._helpers import log_and_trace
from graph.state import PipelineState
from tools import trends as trends_tool
from tools import youtube as youtube_tool

logger = get_logger("graph.nodes.research")

STAGE_TREND = "trend_research"
STAGE_STRATEGY = "strategy"


# --- Trend Research: objective, code-computed scoring (kept as pure functions -> unit-testable) ---

def freshness_score(published_ats: list[str]) -> float:
    """10 = published today, ~0 by 30 days old. Missing/unparseable timestamps default to a
    neutral 5.0 rather than skewing the score in either direction."""
    now = datetime.now(UTC)
    ages_days = []
    for raw in published_ats:
        try:
            ages_days.append((now - datetime.fromisoformat(raw)).total_seconds() / 86400)
        except (ValueError, TypeError):
            continue
    if not ages_days:
        return 5.0
    avg_age = sum(ages_days) / len(ages_days)
    return max(0.0, min(10.0, 10.0 - (avg_age / 3.0)))


def competition_score(view_counts: list[int]) -> float:
    """Log-scaled avg view count as a saturation proxy: ~1k views -> 0, ~1M -> ~7, 10M+ -> 10.
    Higher = more competitive/saturated (matches TopicCandidate.competition_score semantics)."""
    if not view_counts:
        return 5.0
    avg_views = sum(view_counts) / len(view_counts)
    return max(0.0, min(10.0, (math.log10(max(avg_views, 1)) - 3) * 1.4))


def search_volume_score(trend_data: dict | None, fallback: float) -> float:
    """Google Trends rising/top query counts as a demand signal; falls back to the
    view-count-derived competition proxy when Trends is unavailable (rate-limited, etc.)."""
    if not trend_data:
        return fallback
    rising, top = len(trend_data.get("rising", [])), len(trend_data.get("top", []))
    if rising == 0 and top == 0:
        return fallback
    return max(0.0, min(10.0, rising * 1.5 + top * 0.5))


def composite_score(search_volume: float, freshness: float, competition: float) -> float:
    """Rewards demand + freshness, penalizes saturation (low competition_score is good)."""
    return round(0.4 * search_volume + 0.3 * freshness + 0.3 * (10 - competition), 2)


class _TopicIdea(BaseModel):
    title: str = Field(description="An original video concept/angle — not a copied video title")
    description: str
    category: str
    representative_video_ids: list[str] = Field(
        default_factory=list, description="IDs from the provided trending list that inspired this idea"
    )


class _TrendSynthesis(BaseModel):
    ideas: list[_TopicIdea]


_TREND_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a YouTube trend analyst. You are given a list of currently-trending videos (title, "
    "description snippet, view/like counts, publish date). Identify distinct, ORIGINAL content "
    "angles a different channel could make videos about, inspired by the same underlying interest "
    "these trending videos reveal. Never propose a title, script, or description that copies or "
    "closely imitates any specific trending video — extract the underlying topic/format/gap, not "
    "the video itself. Also never propose a remix, cover, parody, or retelling of an existing song, "
    "rhyme, or story (e.g. 'a modern remix of classic nursery rhymes') — the lyrics/plot of the "
    "source material are what make it recognizable, so any script written for that idea ends up "
    "near-identical to the original and fails downstream originality review. Propose wholly new "
    "songs/stories/formats instead, merely inspired by the same theme. Each idea must cite which "
    "trending video IDs informed it."
)


def _dedupe_videos(videos: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped = []
    for v in videos:
        if v["video_id"] not in seen:
            seen.add(v["video_id"])
            deduped.append(v)
    return deduped


def _synthesize_topic_candidates(raw_videos: list[dict], max_candidates: int) -> list[TopicCandidate]:
    videos_by_id = {v["video_id"]: v for v in raw_videos}
    listing = "\n".join(
        f"- id={v['video_id']} | \"{v['title']}\" | views={v['view_count']} | published={v['published_at']}"
        for v in raw_videos
    )
    prompt = (
        f"Propose up to {max_candidates} original content angles based on these trending videos:\n\n{listing}"
    )
    synthesis = call_structured(prompt, _TrendSynthesis, system=_TREND_SYNTHESIS_SYSTEM_PROMPT)

    candidates: list[TopicCandidate] = []
    for idea in synthesis.ideas[:max_candidates]:
        reps = [videos_by_id[vid] for vid in idea.representative_video_ids if vid in videos_by_id] or raw_videos[:3]

        fresh = freshness_score([v["published_at"] for v in reps])
        comp = competition_score([v["view_count"] for v in reps])

        trend_data = None
        try:
            trend_data = trends_tool.related_queries(idea.title)
        except Exception as exc:
            logger.warning("trend_research.trends_lookup_failed", title=idea.title, error=str(exc))

        search_vol = search_volume_score(trend_data, fallback=comp)

        candidates.append(
            TopicCandidate(
                title=idea.title,
                description=idea.description,
                category=idea.category,
                search_volume_score=round(search_vol, 2),
                competition_score=round(comp, 2),
                freshness_score=round(fresh, 2),
                composite_score=composite_score(search_vol, fresh, comp),
                source_video_ids=[v["video_id"] for v in reps],
            )
        )

    candidates.sort(key=lambda c: c.composite_score, reverse=True)
    return candidates


def trend_research_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_TREND, "start")
    settings = get_settings()

    category_ids = [c.strip() for c in settings.trend_category_ids.split(",") if c.strip()] or [None]

    raw_videos: list[dict] = []
    for category_id in category_ids:
        try:
            raw_videos.extend(
                youtube_tool.most_popular(region_code=settings.trend_region_code, category_id=category_id, max_results=15)
            )
        except Exception as exc:
            logger.error("trend_research.most_popular_failed", category_id=category_id, error=str(exc))

    search_queries = [q.strip() for q in settings.trend_search_queries.split(",") if q.strip()]
    for query in search_queries:
        try:
            raw_videos.extend(youtube_tool.search_with_stats(query, max_results=10))
        except Exception as exc:
            logger.error("trend_research.search_failed", query=query, error=str(exc))

    deduped = _dedupe_videos(raw_videos)
    if not deduped:
        raise RuntimeError("trend_research: no trending videos retrieved from any configured category")

    candidates = _synthesize_topic_candidates(deduped, max_candidates=settings.max_trend_candidates)
    output = TrendResearchOutput(candidates=candidates)

    return {
        "trend_output": output.model_dump(mode="json"),
        "trace": [trace, log_and_trace(STAGE_TREND, "complete", raw_videos=len(deduped), candidates=len(candidates))],
    }


# --- Strategy: LLM picks one candidate + decides format, code just validates the pick ---


class _StrategyChoice(BaseModel):
    selected_topic_title: str = Field(description="Must exactly match one candidate title provided")
    content_type: ContentType
    content_format: ContentFormat
    target_length_seconds: int = Field(ge=15, le=3600)
    rationale: str


_STRATEGY_SYSTEM_PROMPT = (
    "You are a YouTube channel strategist. Given ranked topic candidates, the channel's goals, "
    "and a summary of past video performance, choose exactly one topic to produce next and decide "
    "its format. Weigh the candidates' scores, but you may override the top-ranked one if channel "
    "goals or past performance clearly favor another."
)


def strategy_node(state: PipelineState) -> dict:
    trace = log_and_trace(STAGE_STRATEGY, "start")
    settings = get_settings()

    trend_output = TrendResearchOutput.model_validate(state["trend_output"])
    past_performance_summary = state.get("past_performance_summary") or "No prior performance data available yet."

    candidates_listing = "\n".join(
        f"- \"{c.title}\" (search_volume={c.search_volume_score}, competition={c.competition_score}, "
        f"freshness={c.freshness_score}, composite={c.composite_score}): {c.description}"
        for c in trend_output.candidates
    )
    prompt = (
        f"Channel goals: {settings.channel_goals}\n\n"
        f"Past performance summary: {past_performance_summary}\n\n"
        f"Candidate topics (ranked by composite score, highest first):\n{candidates_listing}"
    )
    choice = call_structured(prompt, _StrategyChoice, system=_STRATEGY_SYSTEM_PROMPT)

    selected = next((c for c in trend_output.candidates if c.title == choice.selected_topic_title), None)
    if selected is None:
        logger.warning("strategy.llm_title_mismatch", chosen=choice.selected_topic_title)
        selected = trend_output.candidates[0]

    decision = StrategyDecision(
        selected_topic=selected,
        content_type=choice.content_type,
        content_format=choice.content_format,
        target_length_seconds=choice.target_length_seconds,
        rationale=choice.rationale,
    )

    return {
        "strategy_decision": decision.model_dump(mode="json"),
        "trace": [
            trace,
            log_and_trace(STAGE_STRATEGY, "complete", topic=decision.selected_topic.title, content_type=decision.content_type.value),
        ],
    }
