"""Stock footage/image client — Pexels primary, Pixabay fallback. Returns [] (not an exception)
when neither key is configured, or both searches fail/return nothing — callers (asset_visual_node)
fall back to AI image generation in that case, so a missing/exhausted stock provider never blocks
a run."""
import httpx

from core.logging import get_logger
from core.settings import get_settings
from tools.resilience import CircuitOpenError, with_resilience

logger = get_logger("tools.stock_media")


def search_videos(keywords: list[str], per_page: int = 5) -> list[dict]:
    return _search(keywords, per_page, _pexels_video_search, _pixabay_video_search)


def search_images(keywords: list[str], per_page: int = 5) -> list[dict]:
    return _search(keywords, per_page, _pexels_image_search, _pixabay_image_search)


def _search(keywords: list[str], per_page: int, pexels_fn, pixabay_fn) -> list[dict]:
    settings = get_settings()
    query = " ".join(keywords)

    if settings.pexels_api_key:
        try:
            results = pexels_fn(query, per_page)
            if results:
                return results
        except CircuitOpenError:
            logger.warning("stock_media.pexels_circuit_open")
        except Exception as exc:
            logger.warning("stock_media.pexels_failed", error=str(exc))

    if settings.pixabay_api_key:
        try:
            return pixabay_fn(query, per_page)
        except Exception as exc:
            logger.warning("stock_media.pixabay_failed", error=str(exc))

    return []


@with_resilience(provider="pexels")
def _pexels_video_search(query: str, per_page: int) -> list[dict]:
    settings = get_settings()
    response = httpx.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": settings.pexels_api_key},
        params={"query": query, "per_page": per_page},
        timeout=30,
    )
    response.raise_for_status()
    videos = response.json().get("videos", [])
    results = []
    for video in videos:
        files = video.get("video_files", [])
        best = next((f for f in files if f.get("quality") == "hd"), files[0] if files else None)
        if best:
            results.append({"id": str(video["id"]), "url": best["link"], "source": "pexels", "license": "pexels-license"})
    return results


@with_resilience(provider="pexels")
def _pexels_image_search(query: str, per_page: int) -> list[dict]:
    settings = get_settings()
    response = httpx.get(
        "https://api.pexels.com/v1/search",
        headers={"Authorization": settings.pexels_api_key},
        params={"query": query, "per_page": per_page},
        timeout=30,
    )
    response.raise_for_status()
    photos = response.json().get("photos", [])
    return [
        {"id": str(photo["id"]), "url": photo["src"]["large"], "source": "pexels", "license": "pexels-license"}
        for photo in photos
    ]


@with_resilience(provider="pixabay")
def _pixabay_video_search(query: str, per_page: int) -> list[dict]:
    settings = get_settings()
    response = httpx.get(
        "https://pixabay.com/api/videos/",
        params={"key": settings.pixabay_api_key, "q": query, "per_page": max(per_page, 3)},
        timeout=30,
    )
    response.raise_for_status()
    hits = response.json().get("hits", [])
    return [
        {"id": str(hit["id"]), "url": hit["videos"]["medium"]["url"], "source": "pixabay", "license": "pixabay-license"}
        for hit in hits
    ]


@with_resilience(provider="pixabay")
def _pixabay_image_search(query: str, per_page: int) -> list[dict]:
    settings = get_settings()
    response = httpx.get(
        "https://pixabay.com/api/",
        params={"key": settings.pixabay_api_key, "q": query, "per_page": max(per_page, 3)},
        timeout=30,
    )
    response.raise_for_status()
    hits = response.json().get("hits", [])
    return [
        {"id": str(hit["id"]), "url": hit["largeImageURL"], "source": "pixabay", "license": "pixabay-license"}
        for hit in hits
    ]
