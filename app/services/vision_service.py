from app.vision.amd_llava import AmdLlavaEngine
from app.vision.gemini_flash import GeminiFlashEngine
from app.vision.vision import PROMPTS, is_black_frame, is_similar


class VisionService:
    def __init__(
        self,
        gemini_api_key: str,
        amd_base_url: str = "http://127.0.0.1:8000",
        amd_model: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        amd_timeout_seconds: float = 15.0,
    ):
        self.amd_engine = AmdLlavaEngine(
            base_url=amd_base_url,
            model=amd_model,
            timeout_seconds=amd_timeout_seconds,
        )
        self.gemini_engine = GeminiFlashEngine(api_key=gemini_api_key) if gemini_api_key else None

    def amd_available(self) -> bool:
        return self.amd_engine.available()

    def prompt_for_mode(self, mode: str) -> str:
        return PROMPTS[mode]

    def is_black_frame(self, frame_b64: str) -> bool:
        return is_black_frame(frame_b64)

    def is_similar(self, previous: str, current: str, threshold: float = 0.70) -> bool:
        return is_similar(previous, current, threshold=threshold)

    def describe_with_fallback(
        self,
        frame_b64: str,
        prompt: str,
        prefer_gemini: bool,
        fallback_remaining: int,
    ) -> tuple[str, str, int]:
        description = ""
        used_engine = ""
        use_gemini = prefer_gemini or (fallback_remaining > 0)

        if not use_gemini:
            try:
                description = self.amd_engine.describe(frame_b64, prompt)
                used_engine = self.amd_engine.name
            except Exception as e:
                print(f"AMD failed ({e}), falling back to Gemini...")
                fallback_remaining = 3
                use_gemini = True

        if use_gemini:
            if not self.gemini_engine:
                raise RuntimeError("Gemini engine unavailable: GEMINI_API_KEY not set")
            if fallback_remaining > 0:
                fallback_remaining -= 1
            description = self.gemini_engine.describe(frame_b64, prompt)
            used_engine = self.gemini_engine.name

        return description, used_engine, fallback_remaining
