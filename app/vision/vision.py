"""SightLine — vision utilities, prompts, and filtering helpers."""

import base64
import io

from PIL import Image

PROMPTS = {
    "general": (
        "You assist a blind person. Describe ONLY physical objects and distances in 1-2 short sentences, max 28 words. "
        "Prioritize immediate obstacles and spatial direction (ahead/left/right). "
        "NEVER mention screens, webpages, images, or photos."
    ),
    "safety": (
        "Blind person safety check. Use up to 22 words. "
        "If safe, say path is clear. If hazard, name hazard type and where it is."
    ),
    "ocr": (
        "Read only visible text exactly as written. Use up to 30 words."
    ),
    "navigation": (
        "Blind person directions in 1-2 short sentences, up to 26 words. "
        "Include direction (ahead/left/right), nearest obstacle, and landmark if visible."
    ),
}

def trim_to_sentence(text: str, max_words: int = 32) -> str:
    """Return text trimmed to max_words, cutting at last sentence boundary (.!?)."""
    words = text.split()
    if len(words) <= max_words:
        return text
    trimmed = " ".join(words[:max_words])
    for i in range(len(trimmed) - 1, -1, -1):
        if trimmed[i] in ".!?":
            return trimmed[:i + 1]
    return trimmed.rstrip(".,!?;") + "."


def is_black_frame(b64: str) -> bool:
    """Return True if the frame is too dark to be useful (mean brightness < 15)."""
    try:
        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
        pixels = list(img.getdata())
        if not pixels:
            return True
        return (sum(pixels) / len(pixels)) < 15
    except Exception:
        return True  # treat corrupt frames as skip


def b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def is_similar(prev: str, curr: str, threshold: float = 0.70) -> bool:
    """True when descriptions share >70% word overlap."""
    if not prev:
        return False
    prev_words = set(prev.lower().split())
    curr_words  = set(curr.lower().split())
    if not curr_words:
        return False
    return len(prev_words & curr_words) / len(curr_words) > threshold
