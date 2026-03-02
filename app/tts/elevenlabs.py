import os
import subprocess
import tempfile

from app.tts.base import TTSEngine

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel
TTS_MODEL = "eleven_turbo_v2_5"
OUTPUT_FORMAT = "mp3_44100_128"


class ElevenLabsEngine(TTSEngine):
    def __init__(self, api_key: str):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "elevenlabs"

    def speak(self, text: str) -> None:
        from elevenlabs.client import ElevenLabs

        client = ElevenLabs(api_key=self._api_key)
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id=TTS_MODEL,
            output_format=OUTPUT_FORMAT,
        )
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            for chunk in audio:
                f.write(chunk)
            tmp_path = f.name
        subprocess.run(["afplay", tmp_path], check=False)
        os.unlink(tmp_path)

