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
        "Blind person's assistant. ONE sentence, maximum 15 words. Safety first. "
        "Use: left, right, ahead, close, far. Never say 'image shows' or 'I can see'. "
        "Just describe what's there."
    ),
    "ocr": "Read visible text. One sentence. Exact words only.",
    "navigation": "Blind person navigation. One sentence. Path, obstacles, doors, stairs. Clock directions.",
    "safety": "Blind person safety check. One sentence max. Hazards only. If safe say 'Clear path'.",
}


def describe_amd(b64: str, prompt: str) -> str:
    payload = {
        "model": AMD_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a concise assistant for blind people. Always respond in ONE short sentence, 15 words maximum. Never start with 'The image shows' or 'I can see'.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens":  50,
        "temperature": 0.2,
    }
    resp = httpx.post(AMD_ENDPOINT, json=payload, timeout=AMD_TIMEOUT)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    # Hard backstop: take only the first sentence regardless of model output
    first = raw.split(".")[0].strip()
    return (first + ".") if first else raw


def describe_gemini(image: Image.Image, prompt: str, api_key: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model.generate_content([prompt, image]).text.strip()


def b64_to_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def amd_available() -> bool:
    try:
        return httpx.get(AMD_HEALTH, timeout=5.0).status_code == 200
    except Exception:
        return False


def is_similar(prev: str, curr: str, threshold: float = 0.90) -> bool:
    """True when descriptions share >90% word overlap — catches only near-identical frames."""
    if not prev:
        return False
    prev_words = set(prev.lower().split())
    curr_words  = set(curr.lower().split())
    if not curr_words:
        return False
    return len(prev_words & curr_words) / len(curr_words) > threshold
