"""Google Trends client wrapper (pytrends). No API key required — pytrends drives the public
Google Trends web UI's endpoints directly."""
from core.logging import get_logger
from tools.cache import cache_get_json, cache_set_json
from tools.resilience import with_resilience

logger = get_logger("tools.trends")


@with_resilience(provider="google_trends")
def related_queries(keyword: str, region: str = "US") -> dict:
    """Rising/top related search queries for `keyword` over the last 7 days. Cached 24h since
    Trends data doesn't meaningfully change within a day and pytrends is rate-limit-sensitive."""
    cache_key = f"trends:related:{region}:{keyword.lower()}"
    cached = cache_get_json(cache_key)
    if cached is not None:
        return cached

    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-US", tz=360)
    pytrends.build_payload([keyword], geo=region, timeframe="now 7-d")
    raw = pytrends.related_queries().get(keyword) or {}

    rising_df = raw.get("rising")
    top_df = raw.get("top")
    result = {
        "keyword": keyword,
        "rising": rising_df["query"].tolist() if rising_df is not None and not rising_df.empty else [],
        "top": top_df["query"].tolist() if top_df is not None and not top_df.empty else [],
    }

    logger.info("trends.related_queries", keyword=keyword, region=region, rising=len(result["rising"]), top=len(result["top"]))
    cache_set_json(cache_key, result)
    return result


@with_resilience(provider="google_trends")
def interest_over_time(keyword: str, region: str = "US") -> list[dict]:
    """7-day interest-over-time series for `keyword`, used as a freshness/momentum signal."""
    cache_key = f"trends:interest:{region}:{keyword.lower()}"
    cached = cache_get_json(cache_key)
    if cached is not None:
        return cached

    from pytrends.request import TrendReq

    pytrends = TrendReq(hl="en-US", tz=360)
    pytrends.build_payload([keyword], geo=region, timeframe="now 7-d")
    df = pytrends.interest_over_time()

    result = []
    if not df.empty:
        result = [{"time": str(idx), "value": int(row[keyword])} for idx, row in df.iterrows()]

    cache_set_json(cache_key, result)
    return result
