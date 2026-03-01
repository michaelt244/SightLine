"""
run_sightline.py — SightLine main loop

Continuously screenshots a screen region, describes it via LLaVA on AMD Cloud
(with automatic Gemini Flash fallback), converts to speech via ElevenLabs,
and plays audio through system output.

Usage:
    python capture/run_sightline.py
    python capture/run_sightline.py --mode ask
    python capture/run_sightline.py --interval 5 --focus navigation
    python capture/run_sightline.py --engine gemini

Flags:
    --mode auto|ask         auto = continuous loop, ask = press Enter each time (default: auto)
    --interval N            seconds between captures in auto mode (default: 3)
    --focus general|ocr|navigation|safety  (default: general)
    --engine amd|gemini     vision backend; amd auto-falls back to gemini (default: amd)

Dependencies:
    pip install mss Pillow google-genai elevenlabs python-dotenv httpx
"""

import argparse
import base64
import io
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import httpx
import mss
from PIL import Image
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# ── AMD Cloud config ───────────────────────────────────────────────────────────
AMD_ENDPOINT = "http://165.245.140.111:8000/v1/chat/completions"
AMD_HEALTH_URL = "http://165.245.140.111:8000/health"
AMD_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"
AMD_TIMEOUT = 15.0

# ── Screen region — update these to match your setup ──────────────────────────
# Run `python capture/find_region.py` then `python capture/test_region.py` to verify.
REGION = {
    "top":    190,
    "left":   404,
    "width":  538,
    "height": 956,
}

# ── ElevenLabs config ─────────────────────────────────────────────────────────
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
MODEL_ID = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"

# ── Vision prompts per focus mode ─────────────────────────────────────────────
PROMPTS = {
    "general": (
        "You are a vision assistant for a blind person. "
        "STRICT RULES: "
        "- Maximum 1 sentence, 15 words or less. "
        "- Safety hazards FIRST, always. "
        "- Use spatial words: left, right, ahead, behind, close, far. "
        "- Never say 'I can see' or 'The image shows'. "
        "- Examples of good responses: "
        "'Person ahead, stairs to your left.' "
        "'Clear path, door on your right.' "
        "'Car approaching from the left, stop.' "
        "'Two people sitting ahead, table between them.' "
        "- If nothing changed or nothing notable: say nothing, return 'SKIP'."
    ),
    "ocr": (
        "You are a vision assistant for a blind person. "
        "Read ALL text visible in the image. Signs, labels, menus, screens. "
        "Read exactly as written."
    ),
    "navigation": (
        "You are a vision assistant for a blind person. "
        "Focus on navigation: clear path ahead? Obstacles? Stairs? Doors? "
        "Use clock directions like 'obstacle at 2 o'clock'."
    ),
    "safety": (
        "You are a vision assistant for a blind person. "
        "Focus ONLY on safety: moving vehicles, stairs, ledges, wet floors, approaching people. "
        "If nothing dangerous say 'Path looks clear.'"
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def check_env():
    missing = [k for k, v in [("GEMINI_API_KEY", GEMINI_API_KEY), ("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY)] if not v]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        print("Fill them in your .env file at the project root.")
        sys.exit(1)


def capture_frame() -> Image.Image:
    with mss.mss() as sct:
        shot = sct.grab(REGION)
        img = Image.frombytes("RGB", shot.size, shot.rgb)

    # Resize so longest side ≤ 1024px
    max_side = 1024
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    return img


def image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def describe_image_amd(image: Image.Image, prompt: str) -> str:
    b64 = image_to_b64(image)
    payload = {
        "model": AMD_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": 100,
        "temperature": 0.3,
    }
    resp = httpx.post(AMD_ENDPOINT, json=payload, timeout=AMD_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def describe_image_gemini(image: Image.Image, prompt: str) -> str:
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=[prompt, image],
    )
    return response.text.strip()


def amd_health_check() -> bool:
    try:
        resp = httpx.get(AMD_HEALTH_URL, timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def is_similar(prev: str, curr: str, threshold: float = 0.70) -> bool:
    """Return True if >threshold fraction of words overlap (scene hasn't changed much)."""
    if not prev:
        return False
    prev_words = set(prev.lower().split())
    curr_words = set(curr.lower().split())
    if not curr_words:
        return False
    overlap = len(prev_words & curr_words) / len(curr_words)
    return overlap > threshold


def speak(text: str):
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format=OUTPUT_FORMAT,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        for chunk in audio:
            f.write(chunk)
        tmp_path = f.name

    subprocess.run(["afplay", tmp_path], check=True)
    os.unlink(tmp_path)


def process_frame(prompt: str, last_description: str, engine: str, state: dict) -> str:
    """Capture → describe → (maybe) speak. Returns the new description.

    state: {"fallback_remaining": int} — mutable, updated in place.
    """
    t0 = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S")

    print(f"[{timestamp}] Capturing frame...")
    image = capture_frame()

    description = None
    used_engine = None

    # Decide whether to hit AMD or go straight to Gemini this round.
    use_gemini = engine == "gemini" or state["fallback_remaining"] > 0

    if not use_gemini:
        try:
            print(f"[{timestamp}] Sending to AMD...")
            description = describe_image_amd(image, prompt)
            used_engine = "AMD"
        except Exception:
            print("⚠️ AMD failed, falling back to Gemini...")
            state["fallback_remaining"] = 3
            use_gemini = True

    if use_gemini:
        if state["fallback_remaining"] > 0:
            state["fallback_remaining"] -= 1
        print(f"[{timestamp}] Sending to Gemini...")
        description = describe_image_gemini(image, prompt)
        used_engine = "GEMINI"

    latency_ms = int((time.time() - t0) * 1000)

    if description == "SKIP":
        print(f"[{timestamp}] [{used_engine}] SKIP — nothing notable ({latency_ms}ms)\n")
        return last_description

    if is_similar(last_description, description):
        print(f"[{timestamp}] Scene unchanged — skipping ({latency_ms}ms)\n")
        return last_description

    print(f"[{timestamp}] [{used_engine}] ({latency_ms}ms) {description}\n")

    speak(description)
    return description


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SightLine — real-time vision assistant")
    parser.add_argument("--mode", choices=["auto", "ask"], default="auto")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--focus", choices=["general", "ocr", "navigation", "safety"], default="general")
    parser.add_argument("--engine", choices=["amd", "gemini"], default="amd",
                        help="Vision backend: amd (default, auto-falls back to gemini) or gemini")
    args = parser.parse_args()

    check_env()

    # Fallback state: how many remaining requests to route to Gemini before retrying AMD.
    state = {"fallback_remaining": 0}

    # Startup health check (only relevant when AMD is the preferred engine).
    if args.engine == "amd":
        if amd_health_check():
            print("🟢 AMD Cloud connected — using LLaVA on AMD Instinct GPU")
        else:
            print("⚠️ AMD unreachable — starting with Gemini fallback")
            state["fallback_remaining"] = 3

    prompt = PROMPTS[args.focus]
    last_description = ""

    print(f"[SightLine] Mode: {args.mode} | Focus: {args.focus} | Engine: {args.engine}", end="")
    if args.mode == "auto":
        print(f" | Interval: {args.interval}s")
    else:
        print()
    print("[SightLine] Press Ctrl+C to stop.\n")

    try:
        if args.mode == "auto":
            while True:
                last_description = process_frame(prompt, last_description, args.engine, state)
                time.sleep(args.interval)

        else:  # ask mode
            while True:
                input("Press Enter to capture a frame (Ctrl+C to quit)...")
                last_description = process_frame(prompt, last_description, args.engine, state)

    except KeyboardInterrupt:
        print("\n[SightLine] Stopped.")


if __name__ == "__main__":
    main()
