"""SightLine — server: receives frames, runs vision, speaks descriptions."""

import argparse
import asyncio
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Set

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import Settings
from app.services.tts_service import TTSService
from app.services.vision_service import VisionService
from app.vision.vision import PROMPTS

settings = Settings.from_env()

latest_frame_b64: str            = ""
log_entries:      deque          = deque(maxlen=100)
ws_clients:       Set[WebSocket] = set()
_processing_lock: asyncio.Lock   = asyncio.Lock()
_auto_active:     bool           = True
start_time = time.time()

SPEECH_DEBOUNCE_SECS = 2.0
HAZARD_WORDS = frozenset({
    'car', 'vehicle', 'stairs', 'step', 'edge', 'ledge', 'hole',
    'stop', 'careful', 'watch out', 'moving', 'danger', 'warning',
    'hazard', 'traffic', 'fast', 'obstacle',
})

state = {
    "fallback_remaining": 0,
    "consecutive_skips":  0,
    "last_description":   "",
    "frame_count":        0,
    "last_latency_ms":    0,
    "focus_mode":         "general",
}

args = None
vision_service = VisionService(
    gemini_api_key=settings.gemini_api_key,
    amd_base_url=settings.amd_base_url,
    amd_model=settings.amd_model,
    amd_timeout_seconds=settings.amd_timeout_seconds,
)
tts_service: TTSService | None = None


async def broadcast(entry: dict):
    global ws_clients
    if not ws_clients:
        return
    message = json.dumps(entry)
    dead = set()
    for client in list(ws_clients):
        try:
            await client.send_text(message)
        except Exception:
            dead.add(client)
    ws_clients -= dead


async def process_frame(b64: str):
    if not _auto_active:
        return
    if _processing_lock.locked():
        return

    async with _processing_lock:
        # Skip black / corrupt frames before spending on inference
        if await asyncio.get_running_loop().run_in_executor(None, vision_service.is_black_frame, b64):
            return

        loop   = asyncio.get_running_loop()
        t0     = time.time()
        prompt = vision_service.prompt_for_mode(state["focus_mode"])
        description, used_engine, fallback_remaining = await loop.run_in_executor(
            None,
            vision_service.describe_with_fallback,
            b64,
            prompt,
            args.engine == "gemini",
            state["fallback_remaining"],
        )
        state["fallback_remaining"] = fallback_remaining

        latency_ms = int((time.time() - t0) * 1000)
        state["last_latency_ms"] = latency_ms
        state["frame_count"] += 1

        if description == "SKIP":
            state["consecutive_skips"] += 1
            print(f"[{used_engine}] SKIP ({latency_ms}ms) [silent×{state['consecutive_skips']}]")
            return

        force_speak = state["consecutive_skips"] >= 3
        if vision_service.is_similar(state["last_description"], description) and not force_speak:
            state["consecutive_skips"] += 1
            print(f"[{used_engine}] Similar — skipping ({latency_ms}ms) [silent×{state['consecutive_skips']}]")
            return

        if force_speak:
            print(f"[{used_engine}] (forced after {state['consecutive_skips']} skips) {description} ({latency_ms}ms)")
        else:
            print(f"[{used_engine}] {description} ({latency_ms}ms)")

        state["consecutive_skips"] = 0
        state["last_description"]  = description

        entry = {
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "type":       "description",
            "engine":     used_engine,
            "text":       description,
            "latency_ms": latency_ms,
        }
        log_entries.append(entry)
        await broadcast(entry)

        if not _auto_active:
            return

        is_hazard = any(w in description.lower() for w in HAZARD_WORDS)
        now = time.time()
        last_spoken_at = tts_service.last_spoken_at if tts_service else 0.0
        if not is_hazard and not force_speak and (now - last_spoken_at < SPEECH_DEBOUNCE_SECS):
            print(f"[VOICE] Debounced ({now - last_spoken_at:.1f}s since last)")
            return

        print(f"[VOICE] Enqueuing: {description}")
        if tts_service:
            tts_service.enqueue(description, priority=is_hazard)


app = FastAPI(title="SightLine", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    if tts_service:
        tts_service.start()
        print("[TTS] Speech queue worker started")


@app.post("/upload-frame")
async def upload_frame(request: Request):
    global latest_frame_b64
    body  = await request.json()
    frame = body.get("frame")
    if not frame:
        return JSONResponse({"ok": False, "error": "missing 'frame' field"}, status_code=400)
    latest_frame_b64 = frame
    asyncio.create_task(process_frame(frame))
    return JSONResponse({"ok": True})


@app.get("/api/latest-frame")
async def get_latest_frame():
    return JSONResponse({"frame": latest_frame_b64 or None})


@app.get("/api/logs")
async def get_logs():
    return JSONResponse(list(log_entries))


@app.get("/api/status")
async def get_status():
    return JSONResponse({
        "engine":          args.engine.upper() if args else "—",
        "running":         True,
        "frames":          state["frame_count"],
        "latency_ms":      state["last_latency_ms"],
        "focus_mode":      state["focus_mode"],
        "uptime_seconds":  round(time.time() - start_time, 1),
    })


@app.get("/api/config")
async def get_config():
    return JSONResponse({
        "webhook_upload_url": settings.webhook_upload_url or None,
    })


@app.post("/api/mode")
async def set_mode(request: Request):
    body = await request.json()
    mode = body.get("mode", "")
    if mode not in PROMPTS:
        return JSONResponse({"ok": False, "error": f"Unknown mode: {mode}"}, status_code=400)
    state["focus_mode"] = mode
    print(f"[MODE] {mode}")
    return JSONResponse({"ok": True, "mode": mode})


@app.post("/api/pause")
async def pause_auto():
    global _auto_active
    _auto_active = False
    if tts_service:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, tts_service.pause)
    print("[AUTO] Paused — voice command mode")
    return JSONResponse({"ok": True, "active": False})


@app.post("/api/resume")
async def resume_auto():
    global _auto_active
    _auto_active = True
    print("[AUTO] Resumed — auto detect mode")
    return JSONResponse({"ok": True, "active": True})


@app.websocket("/ws/live")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    try:
        for entry in list(log_entries)[-20:]:
            await websocket.send_text(json.dumps(entry))
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        ws_clients.discard(websocket)


@app.get("/")
async def dashboard():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


def main():
    global args, tts_service

    parser = argparse.ArgumentParser(description="SightLine — vision server")
    parser.add_argument("--engine", choices=["amd", "gemini"], default=settings.default_engine,
                        help="Vision backend (default: amd, auto-falls back to gemini)")
    parser.add_argument("--voice",  choices=["mac", "elevenlabs", "none"], default=settings.default_voice,
                        help="TTS backend: mac = system say, elevenlabs = ElevenLabs API")
    parser.add_argument("--focus",  choices=["general", "ocr", "navigation", "safety"], default=settings.default_focus,
                        help="Initial focus mode")
    parser.add_argument("--port",   type=int, default=settings.default_port,
                        help="Port to listen on (default: 8080)")
    args = parser.parse_args()

    state["focus_mode"] = args.focus

    if args.voice == "elevenlabs" and not settings.elevenlabs_api_key:
        print("[ERROR] ELEVENLABS_API_KEY not set")
        sys.exit(1)
    tts_service = TTSService(voice=args.voice, elevenlabs_api_key=settings.elevenlabs_api_key)

    print()
    print("  ███████╗██╗ ██████╗ ██╗  ██╗████████╗██╗     ██╗███╗   ██╗███████╗")
    print("  ██╔════╝██║██╔════╝ ██║  ██║╚══██╔══╝██║     ██║████╗  ██║██╔════╝")
    print("  ███████╗██║██║  ███╗███████║   ██║   ██║     ██║██╔██╗ ██║█████╗  ")
    print("  ╚════██║██║██║   ██║██╔══██║   ██║   ██║     ██║██║╚██╗██║██╔══╝  ")
    print("  ███████║██║╚██████╔╝██║  ██║   ██║   ███████╗██║██║ ╚████║███████╗")
    print("  ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝")
    print()
    print(f"  Engine: {args.engine.upper()}  |  Voice: {args.voice}  |  Focus: {args.focus}  |  Port: {args.port}")
    print()

    if args.engine == "amd":
        if vision_service.amd_available():
            print("  AMD Cloud connected — LLaVA on AMD Instinct GPU")
        else:
            print("  AMD unreachable — will fall back to Gemini per request")
            state["fallback_remaining"] = 3

    print()
    print(f"  Dashboard → http://localhost:{args.port}/")
    print(f"  Logs      → GET  http://localhost:{args.port}/api/logs")
    print(f"  WebSocket → ws://localhost:{args.port}/ws/live")
    print()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
