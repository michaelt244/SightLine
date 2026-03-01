"""SightLine voice agent — ElevenLabs TTS and conversational agent sessions."""

import os
from typing import Optional

ELEVENLABS_API_KEY  = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # Bella


class VoiceAgent:
    def __init__(self):
        if not ELEVENLABS_API_KEY:
            print("[Voice] ELEVENLABS_API_KEY not set — voice disabled")
            self._enabled = False
            return

        self._enabled = True
        try:
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        except ImportError:
            print("[Voice] elevenlabs package not installed")
            self._enabled = False

    def speak(self, text: str) -> Optional[bytes]:
        """Convert text to MP3 bytes. Returns None if voice is disabled."""
        if not self._enabled:
            return None
        audio = self._client.generate(
            text=text,
            voice=ELEVENLABS_VOICE_ID,
            model="eleven_turbo_v2",
        )
        return b"".join(audio)

    def get_agent_signed_url(self) -> Optional[str]:
        """Return a signed WebSocket URL for a live ElevenLabs Conversational Agent session."""
        if not self._enabled or not ELEVENLABS_AGENT_ID:
            return None
        try:
            result = self._client.conversational_ai.get_signed_url(agent_id=ELEVENLABS_AGENT_ID)
            return result.signed_url
        except Exception as e:
            print(f"[Voice] Failed to get signed URL: {e}")
            return None
