"""YouTube Data API v3 + Analytics API client wrapper.

`most_popular` / `search_competitor_videos` use a simple API key (YOUTUBE_API_KEY) since they only
read public data. `resumable_upload` / `analytics_report` need OAuth2 (YOUTUBE_CLIENT_ID/SECRET +
YOUTUBE_REFRESH_TOKEN, minted by scripts/youtube_oauth_setup.py) since they act on the channel's
own account.

Every Data API v3 call reserves quota via tools.quota first; QuotaExceededError propagates so the
calling agent node can defer rather than silently exceed the daily budget. The Analytics API
(`analytics_report`) is a separate quota pool from Google, not the Data API's 10,000 units/day
budget, so it doesn't go through tools.quota. Client construction (and its config validation)
deliberately happens *outside* the retried section below — a missing API key is a permanent
misconfiguration, not a transient failure, and shouldn't burn retry attempts or help trip the
circuit breaker.
"""
from datetime import UTC, datetime, timedelta

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


def _oauth_client():
    """Account-scoped client (upload, analytics) authenticated via the refresh token minted by
    scripts/youtube_oauth_setup.py. google-auth transparently exchanges it for a short-lived
    access token on first use and refreshes again whenever that token expires."""
    settings = get_settings()
    missing = [
        name
        for name, value in (
            ("YOUTUBE_CLIENT_ID", settings.youtube_client_id),
            ("YOUTUBE_CLIENT_SECRET", settings.youtube_client_secret),
            ("YOUTUBE_REFRESH_TOKEN", settings.youtube_refresh_token),
        )
        if not value
    ]
    if missing:
        raise YouTubeNotConfiguredError(f"OAuth not configured, missing: {', '.join(missing)}")

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    credentials = Credentials(
        token=None,
        refresh_token=settings.youtube_refresh_token,
        client_id=settings.youtube_client_id,
        client_secret=settings.youtube_client_secret,
        token_uri="https://oauth2.googleapis.com/token",
    )
    return build("youtube", "v3", credentials=credentials, cache_discovery=False)


def _analytics_client():
    """Same OAuth credentials as _oauth_client(), but built against the separate youtubeAnalytics
    v2 API (its own quota pool, distinct product from the Data API v3 client above)."""
    settings = get_settings()
    missing = [
        name
        for name, value in (
            ("YOUTUBE_CLIENT_ID", settings.youtube_client_id),
            ("YOUTUBE_CLIENT_SECRET", settings.youtube_client_secret),
            ("YOUTUBE_REFRESH_TOKEN", settings.youtube_refresh_token),
            ("YOUTUBE_CHANNEL_ID", settings.youtube_channel_id),
        )
        if not value
    ]
    if missing:
        raise YouTubeNotConfiguredError(f"OAuth not configured, missing: {', '.join(missing)}")

    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    credentials = Credentials(
        token=None,
        refresh_token=settings.youtube_refresh_token,
        client_id=settings.youtube_client_id,
        client_secret=settings.youtube_client_secret,
        token_uri="https://oauth2.googleapis.com/token",
    )
    return build("youtubeAnalytics", "v2", credentials=credentials, cache_discovery=False)


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


def search_with_stats(query: str, max_results: int = 15) -> list[dict]:
    """Like search_competitor_videos, but hydrated with view/like/comment counts (a second
    videos().list call) so the results are shape-compatible with most_popular() -- trend_research
    scores candidates on those stats, and search().list alone doesn't return them. Used when
    TREND_SEARCH_QUERIES is set, for niches (e.g. "nursery rhymes for kids") that the generic
    regional mostPopular chart won't reliably surface."""
    return _search_with_stats_call(_public_client(), query, max_results)


@with_resilience(provider="youtube_data_api")
def _search_with_stats_call(youtube, query: str, max_results: int) -> list[dict]:
    quota.consume("search.list")
    search_response = (
        youtube.search().list(part="snippet", q=query, type="video", maxResults=max_results).execute()
    )
    video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
    if not video_ids:
        return []

    quota.consume("videos.list")
    stats_response = youtube.videos().list(part="snippet,statistics", id=",".join(video_ids)).execute()

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
        for item in stats_response.get("items", [])
    ]
    logger.info("youtube.search_with_stats", query=query, count=len(videos))
    return videos


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


def analytics_report(video_id: str, window: str) -> dict:
    """Pulls views/likes/comments/avg-view-duration/retention for one video over the trailing
    `window` ("24h" -> last 1 day, "7d" -> last 7 days) ending today.

    impressions/ctr are NOT included: thumbnail-impression and click-through-rate numbers are
    exclusive to the YouTube Studio UI and are not exposed by the public Analytics API, so those
    two fields always come back 0 — that's a real API limitation, not something left unfinished
    here.
    """
    return _analytics_call(_analytics_client(), video_id, window)


@with_resilience(provider="youtube_analytics_api")
def _analytics_call(analytics, video_id: str, window: str) -> dict:
    settings = get_settings()
    days = 1 if window == "24h" else 7
    end_date = datetime.now(UTC).date()
    start_date = end_date - timedelta(days=days)

    response = (
        analytics.reports()
        .query(
            ids=f"channel=={settings.youtube_channel_id}",
            startDate=start_date.isoformat(),
            endDate=end_date.isoformat(),
            metrics="views,likes,comments,averageViewDuration,averageViewPercentage",
            filters=f"video=={video_id}",
        )
        .execute()
    )

    rows = response.get("rows") or [[0, 0, 0, 0, 0]]
    views, likes, comments, avg_view_duration, avg_view_pct = rows[0]

    logger.info("youtube.analytics.report", video_id=video_id, window=window, views=int(views))
    return {
        "video_id": video_id,
        "window": window,
        "views": int(views),
        "impressions": 0,
        "ctr": 0.0,
        "avg_view_duration_seconds": float(avg_view_duration),
        "retention_pct": float(avg_view_pct),
        "likes": int(likes),
        "comments": int(comments),
    }


def resumable_upload(file_path: str, metadata: dict, thumbnail_path: str | None = None, visibility: str = "unlisted") -> str:
    """Uploads `file_path` to the OAuth-authorized channel. `metadata` is {"title", "description",
    "tags"}; `visibility` is a Visibility value ("private"/"unlisted"/"public").

    Video upload and thumbnail-set are deliberately two separate retried calls, not one: retrying
    them as a single unit means any thumbnail failure (e.g. YouTube's 403 on channels without a
    verified phone number, which is a permanent per-channel restriction, not a transient error)
    re-runs videos.insert too -- silently re-uploading, and re-publishing, a full duplicate video
    for every retry. A thumbnail-set failure is logged and swallowed here instead: the video
    itself already published successfully, which is the part that actually matters.
    """
    client = _oauth_client()
    video_id = _upload_video_call(client, file_path, metadata, visibility)

    if thumbnail_path:
        try:
            _set_thumbnail_call(client, video_id, thumbnail_path)
        except Exception as exc:
            logger.warning("youtube.thumbnail_set_failed", video_id=video_id, error=str(exc))

    logger.info("youtube.upload.complete", video_id=video_id, visibility=visibility)
    return video_id


@with_resilience(provider="youtube_upload_api", max_attempts=3, cooldown_seconds=300)
def _upload_video_call(youtube, file_path: str, metadata: dict, visibility: str) -> str:
    from googleapiclient.http import MediaFileUpload

    quota.consume("videos.insert")
    body = {
        "snippet": {
            "title": metadata.get("title", "Untitled"),
            "description": metadata.get("description", ""),
            "tags": metadata.get("tags", []),
        },
        "status": {"privacyStatus": visibility},
    }
    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        _progress, response = request.next_chunk()
    return response["id"]


@with_resilience(provider="youtube_thumbnail_api", max_attempts=2, cooldown_seconds=60)
def _set_thumbnail_call(youtube, video_id: str, thumbnail_path: str) -> None:
    from googleapiclient.http import MediaFileUpload

    quota.consume("thumbnails.set")
    youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumbnail_path)).execute()
