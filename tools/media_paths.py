"""Filesystem layout for everything a run renders (voiceover audio, sourced visual assets, the
final video, thumbnails). Every real tool/node writes under here instead of scattering ad-hoc
temp paths, so a run's output is easy to find and clean up as a unit."""
import os

from core.settings import get_settings


def media_path(*parts: str) -> str:
    """Resolves settings.media_dir / *parts, creating parent dirs as needed. Returns a str path
    (not Path) since every schema field / moviepy API expects plain strings.

    Convention: media_path(run_id, "voiceover", "beat_0.mp3"), media_path(run_id, "assets", ...),
    media_path(run_id, "render", "final.mp4"), media_path(run_id, "thumbnails", "candidate_0.png").
    """
    settings = get_settings()
    full_path = os.path.join(settings.media_dir, *parts)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path
