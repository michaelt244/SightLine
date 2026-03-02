import google.genai as genai
from google.genai import types

from app.vision.base import VisionEngine
from app.vision.vision import b64_to_image, trim_to_sentence

SYSTEM_PROMPT = (
    "You are a concise assistant for blind people. "
    "Respond in 1-2 short sentences, ideally under 30 words. "
    "Never start with 'The image shows' or 'I can see' or 'I see'."
)


class GeminiFlashEngine(VisionEngine):
    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    @property
    def name(self) -> str:
        return "GEMINI"

    def describe(self, frame_b64: str, prompt: str) -> str:
        image = b64_to_image(frame_b64)
        response = self._client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=[SYSTEM_PROMPT + "\n\n" + prompt, image],
            config=types.GenerateContentConfig(
                max_output_tokens=120,
                temperature=0.2,
            ),
        )
        text = response.text
        if not text:
            return "SKIP"
        return trim_to_sentence(text.strip())

