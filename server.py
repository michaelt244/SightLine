"""SightLine — server: receives frames, runs vision, speaks descriptions."""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Set

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

import vision

load_dotenv(Path(__file__).parent / ".env")

GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

VOICE_ID      = "21m00Tcm4TlvDq8ikWAM"  # Rachel
TTS_MODEL     = "eleven_turbo_v2_5"
OUTPUT_FORMAT = "mp3_44100_128"

latest_frame_b64: str            = ""
log_entries:      deque          = deque(maxlen=100)
ws_clients:       Set[WebSocket] = set()
_processing_lock: asyncio.Lock   = asyncio.Lock()
_auto_active:     bool           = True
_speaking:        bool           = False
_resume_after:    float          = 0.0
start_time = time.time()

state = {
    "fallback_remaining": 0,
    "consecutive_skips":  0,
    "last_description":   "",
    "frame_count":        0,
    "last_latency_ms":    0,
    "focus_mode":         "general",
}

args = None


_say_process = None


def kill_say():
    """Kill any running Mac 'say' process immediately."""
    global _say_process
    if _say_process and _say_process.poll() is None:
        _say_process.kill()
        _say_process.wait()
    _say_process = None


def speak_mac(text: str):
    global _say_process
    if not _auto_active:
        return
    kill_say()
    _say_process = subprocess.Popen(["say", "-r", "200", text])
    _say_process.wait()  # Block thread until speech finishes (or is killed)


def speak_elevenlabs(text: str):
    from elevenlabs.client import ElevenLabs
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio  = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=TTS_MODEL,
        output_format=OUTPUT_FORMAT,
    )
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        for chunk in audio:
            f.write(chunk)
        tmp_path = f.name
    subprocess.run(["afplay", tmp_path], check=False)
    os.unlink(tmp_path)


async def play_audio(text: str):
    loop = asyncio.get_running_loop()
    if args.voice == "mac":
        await loop.run_in_executor(None, speak_mac, text)
    elif args.voice == "elevenlabs":
        await loop.run_in_executor(None, speak_elevenlabs, text)


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
    global _speaking
    if not _auto_active:
        return
    if time.time() < _resume_after:
        return
    if _speaking:
        return
    if _processing_lock.locked():
        return

    async with _processing_lock:
        loop   = asyncio.get_running_loop()
        t0     = time.time()
        prompt = vision.PROMPTS[state["focus_mode"]]

        description = ""
        used_engine = ""
        use_gemini  = (args.engine == "gemini") or (state["fallback_remaining"] > 0)

        if not use_gemini:
            try:
                description = await loop.run_in_executor(None, vision.describe_amd, b64, prompt)
                used_engine = "AMD"
            except Exception as e:
                print(f"AMD failed ({e}), falling back to Gemini...")
                state["fallback_remaining"] = 3
                use_gemini = True

        if use_gemini:
            if state["fallback_remaining"] > 0:
                state["fallback_remaining"] -= 1
            image       = await loop.run_in_executor(None, vision.b64_to_image, b64)
            description = await loop.run_in_executor(
                None, vision.describe_gemini, image, prompt, GEMINI_API_KEY
            )
            used_engine = "GEMINI"

        latency_ms = int((time.time() - t0) * 1000)
        state["last_latency_ms"] = latency_ms
        state["frame_count"] += 1

        if description == "SKIP":
            state["consecutive_skips"] += 1
            print(f"[{used_engine}] SKIP ({latency_ms}ms) [silent×{state['consecutive_skips']}]")
            return

        force_speak = state["consecutive_skips"] >= 3
        if vision.is_similar(state["last_description"], description) and not force_speak:
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

        _speaking = True
        try:
            print(f"[VOICE] Speaking: {description}")
            await play_audio(description)
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        finally:
            _speaking = False


app = FastAPI(title="SightLine", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/api/mode")
async def set_mode(request: Request):
    body = await request.json()
    mode = body.get("mode", "")
    if mode not in vision.PROMPTS:
        return JSONResponse({"ok": False, "error": f"Unknown mode: {mode}"}, status_code=400)
    state["focus_mode"] = mode
    print(f"[MODE] {mode}")
    return JSONResponse({"ok": True, "mode": mode})


@app.post("/api/pause")
async def pause_auto():
    global _auto_active, _speaking
    _auto_active = False
    _speaking    = False
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, kill_say)
    print("[AUTO] Paused — voice command mode (say killed)")
    return JSONResponse({"ok": True, "active": False})


@app.post("/api/resume")
async def resume_auto():
    global _auto_active, _resume_after
    _auto_active  = True
    _resume_after = time.time() + 2.0  # 2s cooldown before first frame is processed
    print("[AUTO] Resumed — auto detect mode (2s cooldown)")
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
    global args

    parser = argparse.ArgumentParser(description="SightLine — vision server")
    parser.add_argument("--engine", choices=["amd", "gemini"], default="amd",
                        help="Vision backend (default: amd, auto-falls back to gemini)")
    parser.add_argument("--voice",  choices=["mac", "elevenlabs", "none"], default="mac",
                        help="TTS backend: mac = system say, elevenlabs = ElevenLabs API")
    parser.add_argument("--focus",  choices=["general", "ocr", "navigation", "safety"], default="general",
                        help="Initial focus mode")
    parser.add_argument("--port",   type=int, default=8080,
                        help="Port to listen on (default: 8080)")
    args = parser.parse_args()

    state["focus_mode"] = args.focus

    if args.voice == "elevenlabs" and not ELEVENLABS_API_KEY:
        print("[ERROR] ELEVENLABS_API_KEY not set")
        sys.exit(1)

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
        if vision.amd_available():
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
