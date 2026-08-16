"""Reusable cartoon-mascot pose pack -- generated once and cached on disk, then reused by every
render forever (see graph/nodes/audio_render.py `_character_clip_for_beat`). This is what replaces
the Ken-Burns pan/zoom on a single AI still for AI_IMAGE beats: a small set of consistent poses is
generated once, then animated frame-by-frame (bounce, mouth-flap, blink, wave) at render time --
motion comes from an actual animation state machine, not a moving crop of a photo.

Consistency across poses:
 - if OPENAI_API_KEY is configured, the base (idle) pose is generated once and every other pose is
   produced via OpenAI's image *edit* endpoint against that same base image, which keeps the
   character's design close to identical across poses (an edit, not an independent generation).
 - otherwise every pose falls back to independent Pollinations text-to-image calls sharing one
   long, detailed character-description prompt -- consistency there is best-effort only, since
   Pollinations has no image-edit/reference mode. This is a documented limitation, not a bug.

Every pose is generated on a solid, distinctive backdrop color (_KEY_COLOR) and immediately
chroma-keyed to a transparent alpha channel, so the render step can freely composite the sprite
over any beat backdrop.
"""
import os

from PIL import Image, ImageDraw

from core.logging import get_logger
from core.settings import get_settings
from tools import image_gen as image_gen_tool
from tools.media_paths import media_path
from tools.resilience import with_resilience

logger = get_logger("tools.character_assets")

_KEY_COLOR = (184, 242, 208)  # mint-green -- requested in the prompt as a hint, not relied on
_KEY_TOLERANCE = 40

_POSE_PROMPTS = {
    "base": "standing straight, arms relaxed at sides, neutral friendly smile, looking at camera",
    "talk": "same character in the same pose, mouth open wide as if mid-sentence talking",
    "blink": "same character in the same pose, eyes fully closed as if blinking, mouth closed",
    "wave": "same character, one arm raised up and waving hello, big happy open-mouth smile",
}

_STYLE_SUFFIX = (
    ", simple flat 2D cartoon illustration, thick clean black outlines, bright flat colors, "
    f"centered full-body character, plain solid background color rgb{_KEY_COLOR}, no text, no watermark"
)


def _pack_dir() -> str:
    settings = get_settings()
    slug = settings.mascot_character_name.lower().replace(" ", "_")
    return os.path.dirname(media_path("characters", slug, "base.png"))


def _chroma_key_to_transparent(path: str) -> None:
    """Removes the flat background the generator actually drew. Text-to-image providers don't
    reliably honor an exact requested background color (Pollinations in particular routinely
    ignores it), so this samples the image's own border pixels as the key color instead of
    assuming the one asked for in the prompt, and flood-fills from several border seed points so
    only the region connected to the edge is cleared -- a similarly-toned patch of fur deep inside
    the character (not touching the border) is left alone."""
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    seed_points = {(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1), (w // 2, 0), (0, h // 2), (w - 1, h // 2), (w // 2, h - 1)}
    for seed in seed_points:
        r, g, b, a = img.getpixel(seed)
        if a == 0:
            continue  # already cleared by an earlier seed
        ImageDraw.floodfill(img, seed, (r, g, b, 0), thresh=_KEY_TOLERANCE)
    img.save(path)


def get_character_pack(force: bool = False) -> dict[str, str]:
    """Returns {pose_name: png_path}, generating + chroma-keying the pack on first use (or when
    force=True) and simply returning cached paths on every call after that."""
    settings = get_settings()
    base_dir = _pack_dir()
    paths = {pose: os.path.join(base_dir, f"{pose}.png") for pose in _POSE_PROMPTS}

    if not force and all(os.path.exists(p) for p in paths.values()):
        return paths

    base_prompt = f"{settings.mascot_character_prompt}, {_POSE_PROMPTS['base']}{_STYLE_SUFFIX}"
    image_gen_tool.generate_image(base_prompt, output_path=paths["base"], size="1024x1024")
    _chroma_key_to_transparent(paths["base"])
    logger.info("character_assets.generated", pose="base", path=paths["base"])

    for pose, pose_prompt in _POSE_PROMPTS.items():
        if pose == "base":
            continue
        full_prompt = f"{settings.mascot_character_prompt}, {pose_prompt}{_STYLE_SUFFIX}"
        method = "independent"
        if settings.openai_api_key:
            try:
                _openai_edit(paths["base"], full_prompt, paths[pose])
                method = "openai_edit"
            except Exception as exc:
                logger.warning("character_assets.openai_edit_failed", pose=pose, error=str(exc))
                image_gen_tool.generate_image(full_prompt, output_path=paths[pose], size="1024x1024")
        else:
            image_gen_tool.generate_image(full_prompt, output_path=paths[pose], size="1024x1024")
        _chroma_key_to_transparent(paths[pose])
        logger.info("character_assets.generated", pose=pose, path=paths[pose], method=method)

    return paths


@with_resilience(provider="character_assets_openai_edit")
def _openai_edit(base_image_path: str, prompt: str, output_path: str) -> None:
    import base64

    import httpx
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    with open(base_image_path, "rb") as f:
        response = client.images.edit(model=settings.openai_image_model, image=f, prompt=prompt, size="1024x1024")
    image = response.data[0]

    if getattr(image, "b64_json", None):
        raw = base64.b64decode(image.b64_json)
    elif getattr(image, "url", None):
        raw = httpx.get(image.url, timeout=60).content
    else:
        raise RuntimeError("character_assets: edit response had neither b64_json nor url")

    with open(output_path, "wb") as f:
        f.write(raw)
