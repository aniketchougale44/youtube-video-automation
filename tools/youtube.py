"""YouTube Data API v3 + Analytics API client wrapper.

`most_popular` / `search_competitor_videos` use a simple API key (YOUTUBE_API_KEY) since they only
read public data. `resumable_upload` / `analytics_report` need OAuth2 (YOUTUBE_CLIENT_ID/SECRET +
YOUTUBE_REFRESH_TOKEN) since they act on the channel's own account — those remain stubbed until
the upload/analytics stages are implemented.

Every call reserves quota via tools.quota first; QuotaExceededError propagates so the calling
agent node can defer rather than silently exceed the daily budget. Client construction (and its
config validation) deliberately happens *outside* the retried section below — a missing API key
is a permanent misconfiguration, not a transient failure, and shouldn't burn 3 retry attempts or
help trip the circuit breaker.
"""
from core.logging import get_logger
from core.settings import get_settings
from tools import quota
from tools.resilience import with_resilience

logger = get_logger("tools.youtube")


class YouTubeNotConfiguredError(RuntimeError):
    """Raised when a call needs credentials that aren't set (YOUTUBE_API_KEY, or OAuth for
    account-scoped calls) — callers should surface this clearly rather than silently no-op."""


def _public_client():
    """Read-only client authenticated with just an API key (no OAuth needed)."""
    settings = get_settings()
    if not settings.youtube_api_key:
        raise YouTubeNotConfiguredError("YOUTUBE_API_KEY is not set")
    from googleapiclient.discovery import build

    return build("youtube", "v3", developerKey=settings.youtube_api_key, cache_discovery=False)


def most_popular(region_code: str = "US", category_id: str | None = None, max_results: int = 25) -> list[dict]:
    return _most_popular_call(_public_client(), region_code, category_id, max_results)


@with_resilience(provider="youtube_data_api")
def _most_popular_call(youtube, region_code: str, category_id: str | None, max_results: int) -> list[dict]:
    quota.consume("videos.list")

    request_kwargs = {
        "part": "snippet,statistics",
        "chart": "mostPopular",
        "regionCode": region_code,
        "maxResults": max_results,
    }
    if category_id:
        request_kwargs["videoCategoryId"] = category_id

    response = youtube.videos().list(**request_kwargs).execute()

    videos = [
        {
            "video_id": item["id"],
            "title": item["snippet"]["title"],
            "description": item["snippet"].get("description", ""),
            "category_id": item["snippet"].get("categoryId"),
            "published_at": item["snippet"].get("publishedAt"),
            "channel_title": item["snippet"].get("channelTitle"),
            "view_count": int(item.get("statistics", {}).get("viewCount", 0)),
            "like_count": int(item.get("statistics", {}).get("likeCount", 0)),
            "comment_count": int(item.get("statistics", {}).get("commentCount", 0)),
        }
        for item in response.get("items", [])
    ]
    logger.info("youtube.most_popular", region_code=region_code, category_id=category_id, count=len(videos))
    return videos


def search_competitor_videos(query: str, max_results: int = 25) -> list[dict]:
    return _search_call(_public_client(), query, max_results)


@with_resilience(provider="youtube_data_api")
def _search_call(youtube, query: str, max_results: int) -> list[dict]:
    quota.consume("search.list")

    response = (
        youtube.search()
        .list(part="snippet", q=query, type="video", order="viewCount", maxResults=max_results)
        .execute()
    )

    videos = [
        {
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "channel_title": item["snippet"].get("channelTitle"),
            "published_at": item["snippet"].get("publishedAt"),
        }
        for item in response.get("items", [])
    ]
    logger.info("youtube.search", query=query, count=len(videos))
    return videos


@with_resilience(provider="youtube_analytics_api")
def analytics_report(video_id: str, window: str) -> dict:
    logger.info("youtube.analytics.stub", video_id=video_id, window=window)
    # STUB: needs OAuth (YOUTUBE_CLIENT_ID/SECRET/REFRESH_TOKEN) — wired in with the Performance
    # Monitor Agent's real implementation. Real call: youtubeAnalytics.reports().query(
    #   ids=f"channel=={settings.youtube_channel_id}", metrics="views,likes,averageViewDuration,...")
    return {"video_id": video_id, "window": window, "views": 0, "impressions": 0, "ctr": 0.0, "avg_view_duration_seconds": 0.0}


@with_resilience(provider="youtube_upload_api", max_attempts=5, cooldown_seconds=300)
def resumable_upload(file_path: str, metadata: dict, thumbnail_path: str | None = None, visibility: str = "unlisted") -> str:
    quota.consume("videos.insert")
    logger.info("youtube.upload.stub", file_path=file_path, visibility=visibility)
    # STUB: needs OAuth — wired in with the Upload Agent's real implementation. Real call:
    # MediaFileUpload(file_path, chunksize=-1, resumable=True) + youtube.videos().insert(...).next_chunk()
    if thumbnail_path:
        quota.consume("thumbnails.set")
    return "stub_yt_id_000"
