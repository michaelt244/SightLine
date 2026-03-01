"""test_vision.py — end-to-end test: test_frame.jpg → Gemini → ElevenLabs → audio playback."""

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

VOICE_ID      = "JBFqnCBsd6RMkjVDRZzb"
MODEL_ID      = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"
IMAGE_PATH    = "test_frame.jpg"
AUDIO_PATH    = "description.mp3"

VISION_PROMPT = (
    "You are a vision assistant for a blind person. "
    "Describe what you see in 2-3 concise sentences. "
    "Prioritize safety hazards first, then people, then spatial layout, then text/signs. "
    "Use spatial language like 'to your left', 'directly ahead'. Be warm but brief."
)


def check_env():
    missing = [k for k in ("GEMINI_API_KEY", "ELEVENLABS_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"[ERROR] Missing env vars: {', '.join(missing)}")
        sys.exit(1)


def describe_image(image_path: str) -> str:
    from google import genai
    from PIL import Image

    client = genai.Client(api_key=GEMINI_API_KEY)
    image  = Image.open(image_path)
    print(f"[Gemini] Describing {image_path}...")
    response    = client.models.generate_content(model="gemini-flash-latest", contents=[VISION_PROMPT, image])
    description = response.text.strip()
    print(f"[Gemini] {description}\n")
    return description


def speak(text: str, output_path: str):
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio  = client.text_to_speech.convert(
        text=text, voice_id=VOICE_ID, model_id=MODEL_ID, output_format=OUTPUT_FORMAT,
    )
    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    print(f"[ElevenLabs] Saved to {output_path}")


def main():
    check_env()

    if not Path(IMAGE_PATH).exists():
        print(f"[ERROR] {IMAGE_PATH} not found. Run capture/test_region.py first.")
        sys.exit(1)

    description = describe_image(IMAGE_PATH)
    speak(description, AUDIO_PATH)
    subprocess.run(["afplay", AUDIO_PATH])
    print("[Done]")


if __name__ == "__main__":
    main()
