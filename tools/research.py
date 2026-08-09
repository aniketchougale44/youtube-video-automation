"""Web research client wrapper (Tavily) — used by the Fact-Check Agent to verify flagged claims.
Raises ResearchNotConfiguredError when TAVILY_API_KEY is unset; callers (fact_check_node) catch
this and mark claims UNVERIFIED rather than failing the run — fact-checking is best-effort, not a
hard gate."""
from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import with_resilience

logger = get_logger("tools.research")


class ResearchNotConfiguredError(RuntimeError):
    """Raised when TAVILY_API_KEY isn't set."""


def search(query: str, max_results: int = 5) -> list[dict]:
    settings = get_settings()
    if not settings.tavily_api_key:
        raise ResearchNotConfiguredError("TAVILY_API_KEY is not set")
    return _search_call(query, max_results)


@with_resilience(provider="tavily")
def _search_call(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=get_settings().tavily_api_key)
    response = client.search(query=query, max_results=max_results)

    results = [
        {"url": r["url"], "title": r.get("title", ""), "snippet": r.get("content", "")}
        for r in response.get("results", [])
    ]
    logger.info("research.search", query=query, count=len(results))
    return results
