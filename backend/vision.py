"""
SightLine vision pipeline — wraps LLaVA/Llama Vision via vLLM or Ollama.

VISION_BACKEND=vllm    → OpenAI-compatible API (AMD Cloud)
VISION_BACKEND=ollama  → local Ollama
VISION_BACKEND=mock    → static responses for front-end dev
"""

import base64
import os
from typing import Optional

import httpx

VISION_BACKEND  = os.getenv("VISION_BACKEND", "mock")
VLLM_BASE_URL   = os.getenv("VLLM_BASE_URL",  "http://localhost:8001/v1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.2-vision:11b")

PROMPTS = {
    "general": (
        "You are an AI assistant helping a blind person understand their surroundings. "
        "Describe what you see in 2-3 concise sentences. Focus on: people and their positions, "
        "objects, spatial layout, and anything actionable. Be warm and direct."
    ),
    "ocr": (
        "Read all text visible in this image. Quote it exactly, left-to-right, top-to-bottom. "
        "If no text is visible, say so briefly."
    ),
    "navigation": (
        "You are helping a blind person navigate safely. Describe the spatial layout: "
        "what is directly ahead, to the left, to the right. Estimate distances (e.g. '2 feet ahead'). "
        "Mention any openings, corridors, or paths. Be precise and concise."
    ),
    "safety": (
        "You are a safety assistant for a blind person. Scan this image for hazards FIRST: "
        "stairs, steps, drops, moving vehicles, wet floors, obstacles at head/shin height, crowds. "
        "State hazards immediately, then give a brief scene overview. Use alert language for dangers."
    ),
}

SAFETY_KEYWORDS = {
    "stairs_detected": ["stair", "step", "steps"],
    "obstacle_ahead":  ["obstacle", "blocking", "in your path"],
    "vehicle_nearby":  ["car", "vehicle", "bus", "truck", "moving"],
    "drop_ahead":      ["drop", "edge", "ledge", "cliff"],
    "wet_floor":       ["wet", "slippery", "puddle"],
}


class VisionPipeline:
    def __init__(self):
        self.model_name = self._detect_model()
        print(f"[Vision] Backend: {VISION_BACKEND} | Model: {self.model_name}")

    def _detect_model(self) -> str:
        if VISION_BACKEND == "vllm":
            return "meta-llama/Llama-3.2-11B-Vision-Instruct"
        if VISION_BACKEND == "ollama":
            return OLLAMA_MODEL
        return "mock"

    async def describe(self, image_bytes: bytes, mode: str = "general") -> dict:
        prompt = PROMPTS.get(mode, PROMPTS["general"])

        if VISION_BACKEND == "vllm":
            text = await self._call_vllm(image_bytes, prompt)
        elif VISION_BACKEND == "ollama":
            text = await self._call_ollama(image_bytes, prompt)
        else:
            text = self._mock_describe(mode)

        return {
            "description":   text,
            "safety_alerts": self._extract_safety_alerts(text),
            "confidence":    0.9 if VISION_BACKEND != "mock" else 1.0,
        }

    async def ask(self, image_bytes: Optional[bytes], question: str, context: list[str]) -> dict:
        context_block = (
            "Recent scene descriptions:\n" + "\n".join(f"- {c}" for c in context[-3:]) + "\n\n"
            if context else ""
        )
        prompt = (
            f"{context_block}The user asks: {question}\n"
            "Answer concisely. If the answer is visible in the image, use it. "
            "If relying on context, say so briefly."
        )

        if image_bytes and VISION_BACKEND == "vllm":
            text = await self._call_vllm(image_bytes, prompt)
        elif image_bytes and VISION_BACKEND == "ollama":
            text = await self._call_ollama(image_bytes, prompt)
        else:
            text = self._mock_ask(question, context)

        return {"answer": text, "confidence": 0.88}

    async def _call_vllm(self, image_bytes: bytes, prompt: str) -> str:
        b64     = base64.b64encode(image_bytes).decode()
        payload = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "max_tokens":  256,
            "temperature": 0.3,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{VLLM_BASE_URL}/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()

    async def _call_ollama(self, image_bytes: bytes, prompt: str) -> str:
        b64     = base64.b64encode(image_bytes).decode()
        payload = {
            "model":   OLLAMA_MODEL,
            "prompt":  prompt,
            "images":  [b64],
            "stream":  False,
            "options": {"temperature": 0.3, "num_predict": 256},
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()["response"].strip()

    def _extract_safety_alerts(self, text: str) -> list[str]:
        lower = text.lower()
        return [
            key for key, words in SAFETY_KEYWORDS.items()
            if any(w in lower for w in words)
        ]

    def _mock_describe(self, mode: str) -> str:
        return {
            "general":    "You're in a medium-sized room with several people seated at tables. There's an open path ahead of you.",
            "ocr":        "The sign reads: 'EXIT — Push bar to open'.",
            "navigation": "The corridor extends straight ahead for about 10 feet, then turns left. There's a door on your right.",
            "safety":     "Clear path ahead. No immediate hazards detected. Two people are seated to your left.",
        }.get(mode, "Clear path ahead.")

    def _mock_ask(self, question: str, context: list[str]) -> str:
        return f"[Mock] Based on the scene, here is a placeholder answer to '{question}'."
