# SightLine

Real-time vision assistant for blind people using Meta Ray-Ban glasses and WhatsApp video calls.

A friend on a phone calls the blind user via WhatsApp. The caregiver's phone screen is shared to this browser dashboard, which captures frames and sends them to an AI vision model. Descriptions are spoken aloud in real time.

## Architecture

```
Browser (static/index.html)
  └─ getDisplayMedia() → canvas JPEG frames every 3s
       ├─ POST /upload-frame  → server.py (local, port 8080)
       └─ POST /upload-frame  → AMD webhook (port 8081, optional)

server.py  (FastAPI, port 8080)
  ├─ vision.py  → AMD LLaVA or Gemini Flash 2.0
  ├─ TTS        → macOS say or ElevenLabs
  ├─ WS /ws/live  → real-time log to dashboard
  └─ GET /      → serves static/index.html

webhook_server.py  (AMD machine, port 8081)
  └─ ElevenLabs conversational agent calls /tools/describe_scene
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add GEMINI_API_KEY, ELEVENLABS_API_KEY
# Optional for ElevenLabs scene tool context:
# WEBHOOK_UPLOAD_URL=http://165.245.140.111:8081/upload-frame
```

## Run

```bash
# Default: AMD vision, Mac TTS
python server.py

# Gemini only, ElevenLabs voice
python server.py --engine gemini --voice elevenlabs

# Options
python server.py --engine amd|gemini --voice mac|elevenlabs|none --focus general|ocr|navigation|safety --port 8080
```

Open `http://localhost:8080/`, click **Start**, share the WhatsApp window.

## AMD webhook (optional)

Runs on the AMD Cloud machine alongside vLLM. Lets the ElevenLabs conversational agent answer spoken questions about the scene.

```bash
# On AMD machine
python webhook_server.py        # port 8081
ngrok http 8081                 # expose publicly
python setup_agent_tool.py --webhook-url https://<ngrok-id>.ngrok-free.app/tools/describe_scene

# On laptop/server machine (same place you run server.py):
export WEBHOOK_UPLOAD_URL=http://165.245.140.111:8081/upload-frame
python server.py
```

After re-registering tools, voice commands like `SightLine turn off` and `SightLine turn on` will toggle the assistant.

## Files

| File | Purpose |
|------|---------|
| `server.py` | FastAPI server, audio, WebSocket, static serving |
| `vision.py` | AMD/Gemini vision, prompts, smart filtering |
| `static/index.html` | Browser dashboard (frame capture + live log) |
| `webhook_server.py` | AMD machine webhook for ElevenLabs agent |
| `setup_agent_tool.py` | Register `describe_scene` tool on ElevenLabs agent |
