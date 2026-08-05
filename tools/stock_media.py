"""Stock footage/image client wrapper (Pexels primary, Pixabay fallback) — stub implementation.
Real logic: httpx.get("https://api.pexels.com/videos/search", headers={"Authorization": settings.pexels_api_key}, ...)."""
from core.logging import get_logger
from tools.resilience import with_resilience

logger = get_logger("tools.stock_media")


@with_resilience(provider="pexels")
def search_videos(keywords: list[str], per_page: int = 5) -> list[dict]:
    logger.info("stock_media.search_videos.stub", keywords=keywords)
    return [{"id": "stub_video", "url": "https://example.com/stub.mp4", "source": "pexels", "license": "pexels-license"}]


@with_resilience(provider="pexels")
def search_images(keywords: list[str], per_page: int = 5) -> list[dict]:
    logger.info("stock_media.search_images.stub", keywords=keywords)
    return [{"id": "stub_image", "url": "https://example.com/stub.jpg", "source": "pexels", "license": "pexels-license"}]
