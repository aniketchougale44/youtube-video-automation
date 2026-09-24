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
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]

_DEVANAGARI_CANDIDATES = [
    r"C:\Windows\Fonts\mangal.ttf",
    r"C:\Windows\Fonts\Nirmala.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
]


def resolve_font_path(language: str = "en") -> str:
    candidates = _DEVANAGARI_CANDIDATES if language == "hi" else _LATIN_CANDIDATES
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no usable font found for language={language!r}")
