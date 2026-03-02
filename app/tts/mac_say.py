import subprocess
from typing import Optional

from app.tts.base import TTSEngine


class MacSayEngine(TTSEngine):
    def __init__(self, rate: str = "210"):
        self._rate = rate
        self._process: Optional[subprocess.Popen] = None

    @property
    def name(self) -> str:
        return "mac"

    def speak(self, text: str) -> None:
        self._process = subprocess.Popen(["say", "-r", self._rate, text])
        self._process.wait()
        self._process = None

    def interrupt(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        self._process = None

