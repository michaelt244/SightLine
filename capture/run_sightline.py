"""SightLine — captures a WhatsApp video call, describes the scene, speaks the result."""

import argparse
import base64
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import httpx
import mss
from PIL import Image
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

AMD_ENDPOINT       = "http://165.245.140.111:8000/v1/chat/completions"
AMD_HEALTH         = "http://165.245.140.111:8000/health"
AMD_MODEL          = "llava-hf/llava-v1.6-mistral-7b-hf"
AMD_TIMEOUT        = 15.0
AMD_UPLOAD_URL     = "http://165.245.140.111:8081/upload-frame"
AMD_UPLOAD_TIMEOUT = 3.0  # don't block the capture loop on a slow upload

# Update these with coordinates from find_region.py + test_region.py
REGION = {
    "top":    190,
    "left":   404,
    "width":  538,
    "height": 956,
}

VOICE_ID      = "21m00Tcm4TlvDq8ikWAM"  # Rachel
TTS_MODEL     = "eleven_turbo_v2_5"
OUTPUT_FORMAT = "mp3_44100_128"

PROMPTS = {
    "general": (
        "You are a vision assistant for a blind person. "
        "Maximum 1 sentence, 15 words or less. "
        "Safety hazards FIRST. "
        "Use spatial words: left, right, ahead, behind, close, far. "
        "IMPORTANT: Always mention approximate distance to the nearest object or obstacle "
        "(close, very close, arms reach, far). "
        "If an obstacle is getting closer compared to what you might expect from a walking person, emphasize urgency. "
        "Never say 'I can see'. "
        "If the scene is truly identical and static, return SKIP."
    ),
    "ocr": (
        "You are a vision assistant for a blind person. "
        "Read ALL text visible in the image. Signs, labels, menus, screens. "
        "Read exactly as written. One sentence."
    ),
    "navigation": (
        "Focus on navigation for a blind person: clear path? Obstacles? Stairs? Doors? "
        "Use clock directions. One sentence."
    ),
    "safety": (
        "Focus ONLY on safety for a blind person: moving vehicles, stairs, ledges, "
        "wet floors, approaching people. "
        "If nothing dangerous say 'Path looks clear.'"
    ),
}

latest_frame_b64: str = ""
events: deque = deque(maxlen=500)


def log(type_: str, text: str, engine: str = "", latency_ms: int = 0):
    events.append({
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "type":       type_,
        "engine":     engine,
        "text":       text,
        "latency_ms": latency_ms,
    })


def _upload_frame(b64: str):
    try:
        httpx.post(AMD_UPLOAD_URL, json={"frame": b64}, timeout=AMD_UPLOAD_TIMEOUT)
    except Exception:
        pass


def check_env():
    missing = [k for k, v in [
        ("GEMINI_API_KEY",     GEMINI_API_KEY),
        ("ELEVENLABS_API_KEY", ELEVENLABS_API_KEY),
    ] if not v]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        sys.exit(1)


def capture_frame(save_frames: bool, frames_dir: Path) -> Image.Image:
    global latest_frame_b64

    with mss.mss() as sct:
        shot = sct.grab(REGION)
        img = Image.frombytes("RGB", shot.size, shot.rgb)

    w, h = img.size
    if max(w, h) > 1024:
        scale = 1024 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()
    latest_frame_b64 = base64.b64encode(raw).decode()

    threading.Thread(target=_upload_frame, args=(latest_frame_b64,), daemon=True).start()

    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
        fname = frames_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        img.save(fname, format="JPEG", quality=85)

    return img


def describe_amd(b64: str, prompt: str) -> str:
    payload = {
        "model": AMD_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens":  100,
        "temperature": 0.3,
    }
    resp = httpx.post(AMD_ENDPOINT, json=payload, timeout=AMD_TIMEOUT)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def describe_gemini(image: Image.Image, prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model.generate_content([prompt, image]).text.strip()


def amd_health_check() -> bool:
    try:
        resp = httpx.get(AMD_HEALTH, timeout=5.0)
        return resp.status_code == 200
    except Exception:
        return False


def is_similar(prev: str, curr: str, threshold: float = 0.90) -> bool:
    """True when descriptions are nearly identical — 90% word overlap catches only exact repeats."""
    if not prev:
        return False
    prev_words = set(prev.lower().split())
    curr_words = set(curr.lower().split())
    if not curr_words:
        return False
    return len(prev_words & curr_words) / len(curr_words) > threshold


def speak(text: str):
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio  = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=TTS_MODEL,
        output_format=OUTPUT_FORMAT,
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        for chunk in audio:
            f.write(chunk)
        tmp_path = f.name

    subprocess.run(["afplay", tmp_path], check=True)
    os.unlink(tmp_path)


def process_frame(
    prompt: str,
    last_description: str,
    engine: str,
    state: dict,
    save_frames: bool,
    frames_dir: Path,
) -> str:
    t0 = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S")

    image = capture_frame(save_frames, frames_dir)

    description  = ""
    used_engine  = ""
    use_gemini   = engine == "gemini" or state["fallback_remaining"] > 0

    if not use_gemini:
        try:
            description = describe_amd(latest_frame_b64, prompt)
            used_engine = "AMD"
        except Exception:
            print("⚠️ AMD failed, falling back to Gemini...")
            log("system", "AMD failed, switching to Gemini fallback")
            state["fallback_remaining"] = 3
            use_gemini = True

    if use_gemini:
        if state["fallback_remaining"] > 0:
            state["fallback_remaining"] -= 1
        description = describe_gemini(image, prompt)
        used_engine = "GEMINI"

    latency_ms = int((time.time() - t0) * 1000)

    if description == "SKIP":
        state["consecutive_skips"] += 1
        print(f"[{timestamp}] [{used_engine}] SKIP ({latency_ms}ms) [silent×{state['consecutive_skips']}]")
        log("description", "SKIP", used_engine, latency_ms)
        return last_description

    force_speak = state["consecutive_skips"] >= 3
    if is_similar(last_description, description) and not force_speak:
        state["consecutive_skips"] += 1
        print(f"[{timestamp}] Nearly identical — skipping ({latency_ms}ms) [silent×{state['consecutive_skips']}]")
        return last_description

    if force_speak:
        print(f"[{timestamp}] [{used_engine}] (forced after {state['consecutive_skips']} skips) {description} ({latency_ms}ms)")
    else:
        print(f"[{timestamp}] [{used_engine}] {description} ({latency_ms}ms)")

    state["consecutive_skips"] = 0
    log("description", description, used_engine, latency_ms)

    try:
        speak(description)
    except Exception as e:
        print(f"[{timestamp}] [TTS ERROR] {e}")
        log("error", str(e))

    return description


def main():
    parser = argparse.ArgumentParser(description="SightLine — real-time vision assistant")
    parser.add_argument("--engine",      choices=["amd", "gemini"], default="amd")
    parser.add_argument("--interval",    type=float, default=3.0)
    parser.add_argument("--focus",       choices=["general", "ocr", "navigation", "safety"], default="general")
    parser.add_argument("--save-frames", action="store_true")
    args = parser.parse_args()

    check_env()

    frames_dir = Path(__file__).parent.parent / "frames"

    print()
    print("  ███████╗██╗ ██████╗ ██╗  ██╗████████╗██╗     ██╗███╗   ██╗███████╗")
    print("  ██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝██║     ██║████╗  ██║██╔════╝")
    print("  ███████╗██║██║  ███╗███████║   ██║   ██║     ██║██╔██╗ ██║█████╗  ")
    print("  ╚════██║██║██║   ██║██╔══██║   ██║   ██║     ██║██║╚██╗██║██╔══╝  ")
    print("  ███████║██║╚██████╔╝██║  ██║   ██║   ███████╗██║██║ ╚████║███████╗")
    print("  ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝")
    print()
    print(f"  Engine: {args.engine.upper()}  |  Focus: {args.focus}  |  Interval: {args.interval}s")
    print()

    state = {
        "fallback_remaining": 0,
        "consecutive_skips":  0,
    }

    if args.engine == "amd":
        if amd_health_check():
            print("🟢 AMD Cloud connected — using LLaVA on AMD Instinct GPU")
        else:
            print("⚠️ AMD unreachable — falling back to Gemini")
            state["fallback_remaining"] = 3

    log("system", f"Started — engine={args.engine} focus={args.focus} interval={args.interval}")

    print()
    print("📸 Starting capture... (Ctrl+C to stop)")
    print()

    prompt           = PROMPTS[args.focus]
    last_description = ""

    try:
        while True:
            last_description = process_frame(
                prompt, last_description, args.engine, state, args.save_frames, frames_dir
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n[SightLine] Stopped.")
        log("system", "Stopped by user")


if __name__ == "__main__":
    main()
