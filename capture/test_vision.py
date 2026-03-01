"""
test_vision.py — End-to-end test: image → Gemini Flash → ElevenLabs → audio

Usage:
    python capture/test_vision.py

Requires:
    pip install google-genai elevenlabs python-dotenv Pillow

Env vars (in .env at project root):
    GEMINI_API_KEY
    ELEVENLABS_API_KEY
"""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (one level up from capture/)
load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
MODEL_ID = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"
IMAGE_PATH = "test_frame.jpg"
AUDIO_PATH = "description.mp3"

VISION_PROMPT = (
    "You are a vision assistant for a blind person. "
    "Describe what you see in 2-3 concise sentences. "
    "Prioritize safety hazards first, then people, then spatial layout, then text/signs. "
    "Use spatial language like 'to your left', 'directly ahead'. "
    "Be warm but brief."
)


def check_env():
    missing = []
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not ELEVENLABS_API_KEY:
        missing.append("ELEVENLABS_API_KEY")
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        print("Add them to your .env file at the project root.")
        sys.exit(1)


def describe_image(image_path: str) -> str:
    from google import genai
    from PIL import Image

    print(f"[Gemini] Loading {image_path}...")
    client = genai.Client(api_key=GEMINI_API_KEY)
    image = Image.open(image_path)

    print("[Gemini] Sending to gemini-flash-latest...")
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=[VISION_PROMPT, image],
    )
    description = response.text.strip()
    print(f"\n[Gemini] Description:\n{description}\n")
    return description


def speak(text: str, output_path: str):
    from elevenlabs.client import ElevenLabs

    print("[ElevenLabs] Generating audio...")
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    audio = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format=OUTPUT_FORMAT,
    )

    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    print(f"[ElevenLabs] Saved audio to {output_path}")


def play(audio_path: str):
    print(f"[Audio] Playing {audio_path}...")
    subprocess.run(["afplay", audio_path])


def main():
    check_env()

    if not Path(IMAGE_PATH).exists():
        print(f"[ERROR] {IMAGE_PATH} not found.")
        print("Run 'python capture/test_region.py' first to generate it.")
        sys.exit(1)

    description = describe_image(IMAGE_PATH)
    speak(description, AUDIO_PATH)
    play(AUDIO_PATH)

    print("\n[Done] Full pipeline test complete.")


if __name__ == "__main__":
    main()
