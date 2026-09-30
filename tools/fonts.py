"""Font resolution for anything that draws text on rendered media (caption burn-in, thumbnail
title overlay). Tries a short list of common install paths — Windows dev machines and the Linux
Docker image differ — rather than hard-coding one.

`language` picks a script-appropriate family: Latin fonts (Arial/DejaVu/Liberation) have no
Devanagari glyphs at all, so Hindi captions need a separate candidate list (Mangal/Nirmala UI ship
with Windows; the Docker image installs fonts-noto-core, see docker/Dockerfile) -- rendering Hindi
text through a Latin-only font produces tofu boxes instead of the actual characters.
"""
import os

_LATIN_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    # The Docker image installs fonts-noto-core, which ships NO DejaVu and NO Liberation -- so
    # with only those two listed, Latin caption burn-in failed in the container while Hindi
    # worked, and video_assembly_node swallowed it as `captions_failed`. Videos published with no
    # subtitles at all and nothing failed loudly. Noto first, since that is what is actually
    # installed; DejaVu/Liberation stay for images that happen to carry them.
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

_DEVANAGARI_CANDIDATES = [
    r"C:\Windows\Fonts\mangal.ttf",
    r"C:\Windows\Fonts\Nirmala.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
]

# Searched when no hard-coded candidate exists, so a font that is installed under an unexpected
# name still gets used. A hard-coded list is a guess about someone else's filesystem; losing
# captions entirely because a distro renamed a package is a bad trade.
_FALLBACK_DIRS = ["/usr/share/fonts", "/usr/local/share/fonts", r"C:\Windows\Fonts"]
_FALLBACK_PATTERNS = {
    "hi": ("notosansdevanagari", "mangal", "nirmala"),
    "en": ("notosans-", "dejavusans", "liberationsans", "arial"),
}


def _scan_for_font(language: str) -> str | None:
    wanted = _FALLBACK_PATTERNS.get(language, _FALLBACK_PATTERNS["en"])
    for root in _FALLBACK_DIRS:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in sorted(filenames):
                if not name.lower().endswith((".ttf", ".otf")):
                    continue
                if any(w in name.lower() for w in wanted):
                    return os.path.join(dirpath, name)
    return None


def resolve_font_path(language: str = "en") -> str:
    candidates = _DEVANAGARI_CANDIDATES if language == "hi" else _LATIN_CANDIDATES
    for path in candidates:
        if os.path.exists(path):
            return path

    found = _scan_for_font(language)
    if found:
        return found
    raise FileNotFoundError(f"no usable font found for language={language!r}")
