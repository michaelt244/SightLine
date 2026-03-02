import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ENV_LOADED = False


def _load_env_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    root_env = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(root_env)
    _ENV_LOADED = True


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = ""
    elevenlabs_api_key: str = ""
    webhook_upload_url: str = ""
    elevenlabs_agent_id: str = "agent_7901kjkkm9jpesna15bwehhwtjr6"
    default_engine: str = "amd"
    default_voice: str = "mac"
    default_focus: str = "general"
    default_port: int = 8080

    @classmethod
    def from_env(cls) -> "Settings":
        _load_env_once()
        default_port_raw = os.environ.get("SIGHTLINE_PORT", "8080").strip() or "8080"
        try:
            default_port = int(default_port_raw)
        except ValueError:
            default_port = 8080
        default_engine = os.environ.get("SIGHTLINE_ENGINE", "amd").strip() or "amd"
        if default_engine not in {"amd", "gemini"}:
            default_engine = "amd"
        default_voice = os.environ.get("SIGHTLINE_VOICE", "mac").strip() or "mac"
        if default_voice not in {"mac", "elevenlabs", "none"}:
            default_voice = "mac"
        default_focus = os.environ.get("SIGHTLINE_FOCUS", "general").strip() or "general"
        if default_focus not in {"general", "ocr", "navigation", "safety"}:
            default_focus = "general"
        return cls(
            gemini_api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
            elevenlabs_api_key=os.environ.get("ELEVENLABS_API_KEY", "").strip(),
            webhook_upload_url=os.environ.get("WEBHOOK_UPLOAD_URL", "").strip(),
            elevenlabs_agent_id=os.environ.get(
                "ELEVENLABS_AGENT_ID", "agent_7901kjkkm9jpesna15bwehhwtjr6"
            ).strip(),
            default_engine=default_engine,
            default_voice=default_voice,
            default_focus=default_focus,
            default_port=default_port,
        )
