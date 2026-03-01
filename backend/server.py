"""SightLine backend — FastAPI server handling vision, context, and WebSocket streaming."""

import base64
import json
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vision import VisionPipeline
from voice import VoiceAgent

app = FastAPI(title="SightLine", version="0.1.0")

context_memory: deque[str] = deque(maxlen=5)
vision = VisionPipeline()
voice  = VoiceAgent()


class DescribeRequest(BaseModel):
    image: str          # base64-encoded JPEG
    mode:  str = "general"


class DescribeResponse(BaseModel):
    description:   str
    safety_alerts: list[str]
    confidence:    float


class AskRequest(BaseModel):
    image:    str        # base64-encoded JPEG
    question: str
    context:  list[str] = []


class AskResponse(BaseModel):
    answer:     str
    confidence: float


@app.get("/health")
async def health():
    return {"status": "ok", "model": vision.model_name}


@app.post("/describe", response_model=DescribeResponse)
async def describe(req: DescribeRequest):
    image_bytes = base64.b64decode(req.image)
    result      = await vision.describe(image_bytes, mode=req.mode)
    context_memory.append(result["description"])
    return DescribeResponse(
        description=result["description"],
        safety_alerts=result.get("safety_alerts", []),
        confidence=result.get("confidence", 0.0),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    image_bytes  = base64.b64decode(req.image)
    full_context = list(context_memory) + req.context
    result       = await vision.ask(image_bytes, question=req.question, context=full_context)
    return AskResponse(answer=result["answer"], confidence=result.get("confidence", 0.0))


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Client → Server:
        {"type": "frame",    "image": "<base64>"}
        {"type": "question", "text":  "<string>"}

    Server → Client:
        {"type": "description", "text": "...", "safety_alerts": [...]}
        {"type": "answer",      "text": "..."}
        {"type": "error",       "message": "..."}
    """
    await ws.accept()
    image_bytes = None

    try:
        while True:
            msg = json.loads(await ws.receive_text())

            if msg["type"] == "frame":
                image_bytes = base64.b64decode(msg["image"])
                result      = await vision.describe(image_bytes, mode="general")
                context_memory.append(result["description"])
                await ws.send_json({
                    "type":          "description",
                    "text":          result["description"],
                    "safety_alerts": result.get("safety_alerts", []),
                })

            elif msg["type"] == "question":
                result = await vision.ask(
                    image_bytes=image_bytes,
                    question=msg["text"],
                    context=list(context_memory),
                )
                await ws.send_json({"type": "answer", "text": result["answer"]})

            else:
                await ws.send_json({"type": "error", "message": f"unknown type: {msg['type']}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})


BRIDGE_DIR = Path(__file__).parent.parent / "bridge"

if BRIDGE_DIR.exists():
    app.mount("/bridge", StaticFiles(directory=str(BRIDGE_DIR), html=True), name="bridge")

    @app.get("/bridge", response_class=HTMLResponse, include_in_schema=False)
    async def bridge_root():
        return (BRIDGE_DIR / "index.html").read_text()
