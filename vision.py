"""SightLine — vision logic: AMD/Gemini inference, prompts, smart filtering."""

import base64
import io

import httpx
from PIL import Image

AMD_ENDPOINT = "http://165.245.140.111:8000/v1/chat/completions"
AMD_HEALTH   = "http://165.245.140.111:8000/health"
AMD_MODEL    = "llava-hf/llava-v1.6-mistral-7b-hf"
AMD_TIMEOUT  = 15.0

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

_SYSTEM_PROMPT = (
    "You are a concise assistant for blind people. "
    "Respond in 1-2 short sentences, ideally under 30 words. "
    "Never start with 'The image shows' or 'I can see' or 'I see'."
)


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


def describe_amd(b64: str, prompt: str) -> str:
    payload = {
        "model": AMD_MODEL,
        "messages": [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens":  120,
        "temperature": 0.1,
    }
    resp = httpx.post(AMD_ENDPOINT, json=payload, timeout=AMD_TIMEOUT)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    return trim_to_sentence(raw)


def describe_gemini(image: Image.Image, prompt: str, api_key: str) -> str:
    import google.genai as genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[_SYSTEM_PROMPT + "\n\n" + prompt, image],
        config=types.GenerateContentConfig(
            max_output_tokens=120,
            temperature=0.2,
        ),
    )
    text = response.text
    if not text:
        return "SKIP"
    return trim_to_sentence(text.strip())


def b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def amd_available() -> bool:
    try:
        return httpx.get(AMD_HEALTH, timeout=5.0).status_code == 200
    except Exception:
        return False


def is_similar(prev: str, curr: str, threshold: float = 0.70) -> bool:
    """True when descriptions share >70% word overlap."""
    if not prev:
        return False
    prev_words = set(prev.lower().split())
    curr_words  = set(curr.lower().split())
    if not curr_words:
        return False
    return len(prev_words & curr_words) / len(curr_words) > threshold
