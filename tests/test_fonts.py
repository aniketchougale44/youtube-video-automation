"""Caption burn-in must find a font, or videos publish silently without subtitles.

Regression test for a live run inside the Docker image: _LATIN_CANDIDATES listed only DejaVu and
Liberation paths, but docker/Dockerfile installs fonts-noto-core, which ships neither. So
resolve_font_path('en') raised, video_assembly_node caught it as `captions_failed`, and the video
was rendered and published with no subtitles at all -- a warning in the log being the only sign.

Hindi worked the whole time, because NotoSansDevanagari *is* in that package. The font dependency
was added for Devanagari and English was never covered by it.
"""
import os

import pytest

from tools import fonts


def test_english_resolves_on_this_machine():
    path = fonts.resolve_font_path("en")
    assert os.path.exists(path), path


def test_hindi_resolves_on_this_machine():
    path = fonts.resolve_font_path("hi")
    assert os.path.exists(path), path


def test_noto_core_alone_is_enough_for_english(monkeypatch, tmp_path):
    """Simulate the Docker image exactly: fonts-noto-core present, no DejaVu, no Liberation."""
    noto = tmp_path / "usr/share/fonts/truetype/noto"
    noto.mkdir(parents=True)
    for name in ("NotoSans-Bold.ttf", "NotoSansDevanagari-Bold.ttf", "NotoSansThaana-Regular.ttf"):
        (noto / name).write_bytes(b"\x00")

    monkeypatch.setattr(fonts, "_LATIN_CANDIDATES", [str(noto / "NotoSans-Bold.ttf")])
    monkeypatch.setattr(fonts, "_DEVANAGARI_CANDIDATES", [str(noto / "NotoSansDevanagari-Bold.ttf")])

    assert fonts.resolve_font_path("en").endswith("NotoSans-Bold.ttf")
    assert fonts.resolve_font_path("hi").endswith("NotoSansDevanagari-Bold.ttf")


def test_scan_finds_a_font_the_hardcoded_list_missed(monkeypatch, tmp_path):
    """The point of the fallback: an installed font under an unexpected path still gets used,
    rather than captions being dropped because a distro renamed a package."""
    odd = tmp_path / "opt/weird/fonts"
    odd.mkdir(parents=True)
    (odd / "DejaVuSans-Bold.ttf").write_bytes(b"\x00")

    monkeypatch.setattr(fonts, "_LATIN_CANDIDATES", ["/nonexistent/arial.ttf"])
    monkeypatch.setattr(fonts, "_FALLBACK_DIRS", [str(tmp_path)])

    assert fonts.resolve_font_path("en").endswith("DejaVuSans-Bold.ttf")


def test_raises_only_when_there_is_genuinely_no_font(monkeypatch, tmp_path):
    monkeypatch.setattr(fonts, "_LATIN_CANDIDATES", ["/nonexistent/arial.ttf"])
    monkeypatch.setattr(fonts, "_FALLBACK_DIRS", [str(tmp_path / "empty")])

    with pytest.raises(FileNotFoundError, match="no usable font"):
        fonts.resolve_font_path("en")


def test_devanagari_is_not_satisfied_by_a_latin_font(monkeypatch, tmp_path):
    """A Latin font has no Devanagari glyphs -- matching one for 'hi' would render tofu boxes
    instead of failing, which is worse than failing."""
    d = tmp_path / "fonts"
    d.mkdir()
    (d / "DejaVuSans-Bold.ttf").write_bytes(b"\x00")

    monkeypatch.setattr(fonts, "_DEVANAGARI_CANDIDATES", ["/nonexistent/mangal.ttf"])
    monkeypatch.setattr(fonts, "_FALLBACK_DIRS", [str(d)])

    with pytest.raises(FileNotFoundError):
        fonts.resolve_font_path("hi")
