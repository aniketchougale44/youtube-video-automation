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
        best = _pick_rendition(video.get("video_files", []))
        if best:
            results.append({"id": str(video["id"]), "url": best["link"], "source": "pexels", "license": "pexels-license"})
    return results


def _pick_rendition(files: list[dict]) -> dict | None:
    """The smallest landscape rendition that fills a 1920x1080 frame without upscaling.

    Selecting on `quality == "hd"` was wrong twice over. Pexels labels 1280x720 "hd", so beats were
    upscaled to 1080p and came out soft. Worse, it ignores orientation: a 1080x1920 portrait clip
    scored "hd" too, and _resize_to_cover then scales it to 1920x3413 and crops a narrow horizontal
    band out of the middle -- the subject is usually not in that band, so the shot is destroyed. Two
    of seven beats in a real render were portrait clips mangled this way.

    So: landscape only, at least 1920x1080, and of those the smallest, since anything larger is
    bytes downloaded and pixels decoded that the 1080p output throws away. Falling back to the
    widest landscape rendition, then to anything at all, keeps a beat sourced rather than dropping
    it to the mascot when a clip simply has no large landscape version.
    """
    sized = [f for f in files if f.get("link") and f.get("width") and f.get("height")]
    if not sized:
        return files[0] if files else None

    landscape = [f for f in sized if f["width"] > f["height"]]
    if not landscape:
        logger.info("stock_media.no_landscape_rendition", renditions=len(sized))
        return max(sized, key=lambda f: f["width"])

    covering = [f for f in landscape if f["width"] >= 1920 and f["height"] >= 1080]
    if covering:
        return min(covering, key=lambda f: f["width"] * f["height"])
    return max(landscape, key=lambda f: f["width"] * f["height"])


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
