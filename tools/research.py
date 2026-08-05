"""Web research client wrapper (Tavily) — used by the Fact-Check Agent to verify flagged claims.
Real logic: TavilyClient(api_key=settings.tavily_api_key).search(query, max_results=...)."""
from core.logging import get_logger
from tools.resilience import with_resilience

logger = get_logger("tools.research")


@with_resilience(provider="tavily")
def search(query: str, max_results: int = 5) -> list[dict]:
    logger.info("research.search.stub", query=query)
    return [{"url": "https://example.com/stub-source", "title": f"[STUB] source for '{query}'", "snippet": "[STUB] snippet"}]
