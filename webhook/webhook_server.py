"""SightLine webhook server — runs on the AMD machine, serves the ElevenLabs tool endpoint."""

import asyncio
import base64
import io
import re
import time
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import Settings
from app.core.logger import configure_logging, get_logger

settings = Settings.from_env()
configure_logging()
logger = get_logger("sightline.webhook")

AMD_ENDPOINT = f"{settings.amd_base_url}/v1/chat/completions"
AMD_MODEL = settings.amd_model
AMD_TIMEOUT = settings.amd_timeout_seconds

_latest_frame_b64:  Optional[str]   = None
_latest_frame_time: Optional[float] = None
sightline_active:   bool            = True

PROMPTS = {
    "general": (
        "You are a vision assistant for a blind person. "
        "Describe what you see in 2 short sentences, target 20-35 words. "
        "Safety hazards FIRST. Use spatial language: left, right, ahead, and distance if possible. "
        "Never say 'I can see'."
    ),
    "ocr": (
        "Read ALL text visible in the image. Signs, labels, menus. "
        "Read exactly as written. Include complete lines where possible."
    ),
    "safety": (
        "Focus ONLY on safety for a blind person: vehicles, stairs, ledges, obstacles. "
        "If nothing dangerous, say path looks clear and mention the nearest non-hazard object."
    ),
    "navigation": (
        "Navigation info for a blind person: clear path? Obstacles? Stairs? Doors? "
        "Use clock directions and approximate distance. Use 1-2 short sentences."
    ),
}

app = FastAPI(title="SightLine Webhook", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _iter_strings(value):
    """Yield all string leaves from nested dict/list payloads."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_strings(v)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)


def _extract_command(body: dict) -> Optional[str]:
    """Find on/off command from structured params or free-form text."""
    direct = (
        body.get("command")
        or body.get("parameters", {}).get("command")
        or body.get("tool_call", {}).get("parameters", {}).get("command")
    )
    if isinstance(direct, str):
        cmd = direct.strip().lower()
        if cmd in {"on", "off"}:
            return cmd

    all_text = " ".join(s.lower() for s in _iter_strings(body))
    normalized = re.sub(r"\s+", " ", all_text).strip()
    if not normalized:
        return None

    # Prefer explicit phrases to avoid accidental toggles.
    off_patterns = (
        "sightline turn off",
        "turn sightline off",
        "turn off sightline",
        "sightline off",
        "stop sightline",
        "pause sightline",
    )
    on_patterns = (
        "sightline turn on",
        "turn sightline on",
        "turn on sightline",
        "sightline on",
        "resume sightline",
        "start sightline",
    )

    if any(p in normalized for p in off_patterns):
        return "off"
    if any(p in normalized for p in on_patterns):
        return "on"
    return None


def _apply_control(command: str, source: str) -> str:
    global sightline_active
    if command == "off":
        sightline_active = False
        logger.info("SightLine paused", extra={"event": "control_pause", "context": {"source": source}})
        return "SightLine paused"
    if command == "on":
        sightline_active = True
        logger.info("SightLine resumed", extra={"event": "control_resume", "context": {"source": source}})
        return "SightLine resumed"
    return f"Unknown command: {command}"


@app.post("/upload-frame")
async def upload_frame(request: Request):
    global _latest_frame_b64, _latest_frame_time
    body  = await request.json()
    frame = body.get("frame")
    if not frame:
        return JSONResponse({"ok": False, "error": "missing 'frame' field"}, status_code=400)
    _latest_frame_b64  = frame
    _latest_frame_time = time.time()
    return JSONResponse({"ok": True})


async def _call_amd(b64: str, prompt: str) -> str:
    payload = {
        "model": AMD_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens":  100,
        "temperature": 0.3,
    }
    async with httpx.AsyncClient(timeout=AMD_TIMEOUT) as client:
        resp = await client.post(AMD_ENDPOINT, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_gemini(b64: str, prompt: str) -> str:
    import google.genai as genai
    from google.genai import types
    from PIL import Image

    if not settings.gemini_api_key:
        raise RuntimeError("Gemini fallback unavailable: GEMINI_API_KEY is not set")

    client = genai.Client(api_key=settings.gemini_api_key)
    image  = Image.open(io.BytesIO(base64.b64decode(b64)))
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=[prompt, image],
        config=types.GenerateContentConfig(max_output_tokens=60, temperature=0.2),
    )

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    # Some Gemini responses (safety block / empty candidate) do not expose .text.
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        chunks = [getattr(part, "text", "") for part in parts if getattr(part, "text", None)]
        merged = " ".join(chunks).strip()
        if merged:
            return merged

    raise RuntimeError("Gemini returned no text output")


async def _run_vision(b64: str, prompt: str) -> str:
    try:
        return await _call_amd(b64, prompt)
    except Exception:
        logger.warning("AMD failed, falling back to Gemini", extra={"event": "webhook_amd_fallback"})
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _call_gemini, b64, prompt)
        except Exception as ge:
            raise RuntimeError("Both vision backends failed") from ge


def _too_short(text: str, min_words: int = 8) -> bool:
    return len((text or "").split()) < min_words


@app.post("/tools/control")
async def control(request: Request):
    body = await request.json()
    command = _extract_command(body)
    if command in {"on", "off"}:
        return JSONResponse({"result": _apply_control(command, source="tool")})
    return JSONResponse({"result": "Unknown command. Use on or off."}, status_code=400)


@app.get("/tools/status")
async def status():
    return {"active": sightline_active}


@app.post("/tools/describe_scene")
async def handle_describe_scene(request: Request):
    body = await request.json()
    command = _extract_command(body)
    if command in {"on", "off"}:
        return JSONResponse({"result": _apply_control(command, source="describe_scene")})

    if not sightline_active:
        return JSONResponse({"result": "SightLine is currently paused. Say SightLine turn on to resume."})

    # ElevenLabs places parameters at different paths depending on SDK version
    mode = (
        body.get("parameters", {}).get("mode")
        or body.get("tool_call", {}).get("parameters", {}).get("mode")
        or body.get("mode")
        or "general"
    )
    if mode not in PROMPTS:
        mode = "general"

    if _latest_frame_b64 is None:
        return JSONResponse({"result": "No camera feed yet. Start run_sightline.py on the laptop."})

    try:
        description = await _run_vision(_latest_frame_b64, PROMPTS[mode])
        if _too_short(description):
            longer_prompt = (
                f"{PROMPTS[mode]} "
                "Your previous answer was too short. Return a fuller response with at least 14 words."
            )
            description = await _run_vision(_latest_frame_b64, longer_prompt)
    except Exception as e:
        logger.error(
            "describe_scene failed",
            extra={"event": "describe_scene_failed", "context": {"error_type": e.__class__.__name__}},
        )
        return JSONResponse({
            "result": "Vision backend is temporarily unavailable. Please try again in a moment."
        })
    logger.info("Scene described", extra={"event": "scene_described", "context": {"mode": mode}})
    return JSONResponse({"result": description})


@app.get("/health")
async def health():
    frame_age = round(time.time() - _latest_frame_time, 1) if _latest_frame_time else None
    return {
        "status":            "ok",
        "frame_available":   _latest_frame_b64 is not None,
        "frame_age_seconds": frame_age,
    }


if __name__ == "__main__":
    import uvicorn

    logger.info(
        "SightLine webhook starting",
        extra={
            "event": "webhook_start",
            "context": {
                "vision_model_url": settings.amd_base_url,
                "gemini_fallback_enabled": bool(settings.gemini_api_key),
                "port": settings.webhook_port,
            },
        },
    )
    if settings.webhook_public_base_url:
        logger.info(
            "Webhook public endpoints configured",
            extra={
                "event": "webhook_public_endpoints",
                "context": {"public_base_url": settings.webhook_public_base_url},
            },
        )
    else:
        logger.info(
            "Webhook running without public base URL",
            extra={"event": "webhook_public_base_missing"},
        )

    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY not set for webhook fallback", extra={"event": "gemini_key_missing"})

    uvicorn.run(app, host="0.0.0.0", port=settings.webhook_port)
