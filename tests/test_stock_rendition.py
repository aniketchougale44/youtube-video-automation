"""Stock renditions must be landscape and large enough, or the render is quietly degraded.

Selecting on Pexels' `quality == "hd"` label picked two kinds of bad source. Pexels calls 1280x720
"hd", so beats were upscaled to 1080p and came out soft. And the label says nothing about
orientation: a 1080x1920 portrait clip is "hd" as well, and _resize_to_cover scales it to 1920x3413
and crops a narrow horizontal band from the middle -- whatever the clip was of is usually not in
that band. A measured render had two of seven beats mangled that way, and nothing failed or warned.
"""
from tools.stock_media import _pick_rendition


def _f(w: int, h: int, quality: str = "hd") -> dict:
    return {"width": w, "height": h, "quality": quality, "link": f"https://v/{w}x{h}.mp4"}


def test_prefers_the_smallest_rendition_that_fills_1080p():
    """Anything larger is bytes downloaded and pixels decoded that a 1080p output discards."""
    got = _pick_rendition([_f(3840, 2160), _f(1920, 1080), _f(2560, 1440), _f(1280, 720)])
    assert (got["width"], got["height"]) == (1920, 1080)


def test_never_picks_portrait_over_landscape():
    """The bug that actually shipped: a portrait clip centre-cropped to 16:9."""
    got = _pick_rendition([_f(1080, 1920), _f(1280, 720)])
    assert got["width"] > got["height"], f"picked a portrait rendition: {got}"


def test_rejects_720p_when_a_1080p_landscape_exists():
    got = _pick_rendition([_f(1280, 720), _f(1920, 1080)])
    assert got["height"] == 1080


def test_falls_back_to_the_largest_landscape_when_nothing_reaches_1080p():
    """Upscaled footage still beats dropping the beat to a generic mascot shot."""
    got = _pick_rendition([_f(640, 360), _f(1280, 720), _f(960, 540)])
    assert (got["width"], got["height"]) == (1280, 720)


def test_falls_back_to_portrait_only_when_there_is_no_landscape_at_all():
    got = _pick_rendition([_f(1080, 1920), _f(720, 1280)])
    assert got is not None
    assert (got["width"], got["height"]) == (1080, 1920), "should take the widest available"


def test_renditions_without_dimensions_are_still_usable():
    """Some providers omit width/height; a beat with footage beats a beat without."""
    files = [{"link": "https://v/unknown.mp4", "quality": "hd"}]
    assert _pick_rendition(files) is files[0]


def test_no_files_returns_none():
    assert _pick_rendition([]) is None


def test_square_counts_as_portrait_not_landscape():
    """1:1 is not landscape, and cropping it to 16:9 loses a third of the frame."""
    got = _pick_rendition([_f(1080, 1080), _f(1920, 1080)])
    assert (got["width"], got["height"]) == (1920, 1080)
