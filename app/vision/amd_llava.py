import httpx

from app.vision.base import VisionEngine
from app.vision.vision import trim_to_sentence

SYSTEM_PROMPT = (
    "You are a concise assistant for blind people. "
    "Respond in 1-2 short sentences, ideally under 30 words. "
    "Never start with 'The image shows' or 'I can see' or 'I see'."
)


class AmdLlavaEngine(VisionEngine):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        model: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        timeout_seconds: float = 15.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._endpoint = f"{self._base_url}/v1/chat/completions"
        self._health = f"{self._base_url}/health"
        self._model = model
        self._timeout = timeout_seconds

    @property
    def name(self) -> str:
        return "AMD"

    def describe(self, frame_b64: str, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            "max_tokens": 120,
            "temperature": 0.1,
        }
        resp = httpx.post(self._endpoint, json=payload, timeout=self._timeout)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        return trim_to_sentence(raw)

    def available(self) -> bool:
        try:
            return httpx.get(self._health, timeout=5.0).status_code == 200
        except Exception:
            return False
