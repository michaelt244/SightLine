"""SightLine webhook server — runs on the AMD machine, serves the ElevenLabs tool endpoint."""

import asyncio
import base64
import io
import os
import time
from typing import Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# AMD GPU machine running vLLM
AMD_ENDPOINT = "http://165.245.140.111:8000/v1/chat/completions"
AMD_MODEL    = "llava-hf/llava-v1.6-mistral-7b-hf"
AMD_TIMEOUT  = 15.0

_latest_frame_b64:  Optional[str]   = None
_latest_frame_time: Optional[float] = None
sightline_active:   bool            = True

PROMPTS = {
    "general": (
        "You are a vision assistant for a blind person. "
        "Describe what you see in 1-2 sentences, max 20 words. "
        "Safety hazards FIRST. Use spatial language: left, right, ahead. "
        "Never say 'I can see'."
    ),
    "ocr": (
        "Read ALL text visible in the image. Signs, labels, menus. "
        "Read exactly as written. One sentence."
    ),
    "safety": (
        "Focus ONLY on safety for a blind person: vehicles, stairs, ledges, obstacles. "
        "If nothing dangerous say 'Path looks clear.'"
    ),
    "navigation": (
        "Navigation info for a blind person: clear path? Obstacles? Stairs? Doors? "
        "Use clock directions. One sentence."
    ),
}

app = FastAPI(title="SightLine Webhook", version="2.0")


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
    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=GEMINI_API_KEY)
    model  = genai.GenerativeModel("gemini-2.0-flash")
    image  = Image.open(io.BytesIO(base64.b64decode(b64)))
    return model.generate_content([prompt, image]).text.strip()


async def _run_vision(b64: str, prompt: str) -> str:
    try:
        return await _call_amd(b64, prompt)
    except Exception as e:
        print(f"AMD failed ({e}), falling back to Gemini...")
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _call_gemini, b64, prompt)


@app.post("/tools/control")
async def control(request: Request):
    global sightline_active
    body    = await request.json()
    command = body.get("command", "").lower()
    if command == "off":
        sightline_active = False
        print("[CONTROL] SightLine paused")
        return JSONResponse({"result": "SightLine paused"})
    if command == "on":
        sightline_active = True
        print("[CONTROL] SightLine resumed")
        return JSONResponse({"result": "SightLine resumed"})
    return JSONResponse({"result": f"Unknown command: {command}"}, status_code=400)


@app.get("/tools/status")
async def status():
    return {"active": sightline_active}


@app.post("/tools/describe_scene")
async def handle_describe_scene(request: Request):
    if not sightline_active:
        return JSONResponse({"result": "SightLine is currently paused. Say turn on to resume."})

    body = await request.json()

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

    description = await _run_vision(_latest_frame_b64, PROMPTS[mode])
    print(f"[WEBHOOK] mode={mode} → {description}")
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

    print()
    print("  SightLine Webhook Server")
    print(f"  Vision model    : {AMD_ENDPOINT}")
    print(f"  Gemini fallback : {'enabled' if GEMINI_API_KEY else 'DISABLED (GEMINI_API_KEY not set)'}")
    print()
    print("  Laptop uploads frames to:  POST http://165.245.140.111:8081/upload-frame")
    print("  ElevenLabs calls:          POST http://<ngrok-url>/tools/describe_scene")
    print()

    if not GEMINI_API_KEY:
        print("Set GEMINI_API_KEY for fallback:  export GEMINI_API_KEY='...'")
        print()

    uvicorn.run(app, host="0.0.0.0", port=8081)
