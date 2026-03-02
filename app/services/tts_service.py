import queue
import threading
import time

from app.core.logger import get_logger
from app.tts.base import TTSEngine
from app.tts.elevenlabs import ElevenLabsEngine
from app.tts.mac_say import MacSayEngine

logger = get_logger("sightline.tts_service")


class _SilentEngine(TTSEngine):
    @property
    def name(self) -> str:
        return "none"

    def speak(self, text: str) -> None:
        return


class TTSService:
    def __init__(self, voice: str, elevenlabs_api_key: str = ""):
        self._voice = voice
        self._queue: queue.Queue = queue.Queue()
        self._last_spoken_at: float = 0.0
        self._engine = self._build_engine(voice, elevenlabs_api_key)
        self._worker_thread: threading.Thread | None = None

    def _build_engine(self, voice: str, elevenlabs_api_key: str) -> TTSEngine:
        if voice == "mac":
            return MacSayEngine()
        if voice == "elevenlabs":
            return ElevenLabsEngine(api_key=elevenlabs_api_key)
        return _SilentEngine()

    @property
    def last_spoken_at(self) -> float:
        return self._last_spoken_at

    def start(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()

    def _speech_worker(self) -> None:
        while True:
            text = self._queue.get()
            if not text:
                self._queue.task_done()
                continue
            try:
                self._engine.speak(text)
                self._last_spoken_at = time.time()
            except Exception as e:
                logger.error(
                    "Speech synthesis failed",
                    extra={"event": "tts_error", "context": {"error_type": e.__class__.__name__}},
                )
            finally:
                self._queue.task_done()

    def enqueue(self, text: str, priority: bool = False) -> None:
        if priority:
            self._engine.interrupt()
            self._drain_queue()
        self._queue.put(text)

    def pause(self) -> None:
        self._engine.interrupt()
        self._drain_queue()

    def _drain_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
