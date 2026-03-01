"""
run_sightline.py — SightLine main loop

Continuously screenshots a screen region, describes it via Gemini Flash,
converts to speech via ElevenLabs, and plays audio through system output.

Usage:
    python capture/run_sightline.py
    python capture/run_sightline.py --mode ask
    python capture/run_sightline.py --interval 5 --focus navigation
    python capture/run_sightline.py --focus safety

Flags:
    --mode auto|ask     auto = continuous loop, ask = press Enter each time (default: auto)
    --interval N        seconds between captures in auto mode (default: 3)
    --focus general|ocr|navigation|safety  (default: general)

Dependencies:
    pip install mss Pillow google-genai elevenlabs python-dotenv
"""

import argparse
import io
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import mss
from PIL import Image
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

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


def describe_image(image: Image.Image, prompt: str) -> str:
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=[prompt, image],
    )
    return response.text.strip()


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


def process_frame(prompt: str, last_description: str) -> str:
    """Capture → describe → (maybe) speak. Returns the new description."""
    t0 = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S")

    print(f"[{timestamp}] Capturing frame...")
    image = capture_frame()

    print(f"[{timestamp}] Sending to Gemini...")
    description = describe_image(image, prompt)

    latency_ms = int((time.time() - t0) * 1000)

    if description == "SKIP":
        print(f"[{timestamp}] Model returned SKIP — nothing notable ({latency_ms}ms)\n")
        return last_description

    if is_similar(last_description, description):
        print(f"[{timestamp}] Scene unchanged — skipping ({latency_ms}ms)\n")
        return last_description

    print(f"[{timestamp}] ({latency_ms}ms) {description}\n")

    speak(description)
    return description


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SightLine — real-time vision assistant")
    parser.add_argument("--mode", choices=["auto", "ask"], default="auto")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--focus", choices=["general", "ocr", "navigation", "safety"], default="general")
    args = parser.parse_args()

    check_env()

    prompt = PROMPTS[args.focus]
    last_description = ""

    print(f"[SightLine] Mode: {args.mode} | Focus: {args.focus}", end="")
    if args.mode == "auto":
        print(f" | Interval: {args.interval}s")
    else:
        print()
    print("[SightLine] Press Ctrl+C to stop.\n")

    try:
        if args.mode == "auto":
            while True:
                last_description = process_frame(prompt, last_description)
                time.sleep(args.interval)

        else:  # ask mode
            while True:
                input("Press Enter to capture a frame (Ctrl+C to quit)...")
                last_description = process_frame(prompt, last_description)

    except KeyboardInterrupt:
        print("\n[SightLine] Stopped.")


if __name__ == "__main__":
    main()
