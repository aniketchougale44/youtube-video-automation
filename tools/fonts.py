"""Font resolution for anything that draws text on rendered media (caption burn-in, thumbnail
title overlay). Tries a short list of common install paths — Windows dev machines and the Linux
Docker image differ — rather than hard-coding one."""
import os

_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
]


def resolve_font_path() -> str:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("no usable font found")
