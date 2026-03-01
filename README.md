# SightLine

Real-time vision assistant for blind people using Meta Ray-Ban glasses. A laptop captures a WhatsApp video call screen region showing the glasses camera feed, sends frames to a LLaVA vision model running on AMD Cloud GPU (with Gemini Flash fallback), converts descriptions to speech via ElevenLabs, and routes audio back through the call. An ElevenLabs Conversational AI agent handles voice Q&A, calling a webhook server on the AMD machine to describe the current scene on demand.

## Setup

```bash
cp .env.example .env   # fill in your API keys
pip install -r requirements.txt
```

**Find your screen region** (run once to calibrate):
```bash
python capture/find_region.py    # screenshots your screen, gives instructions
python capture/test_region.py    # verify the coordinates capture only the video
```

**Run** (two machines):
```bash
# AMD server
python webhook_server.py

# Laptop
python capture/run_sightline.py
```

**Register the ElevenLabs tool** (run once after ngrok is up):
```bash
ngrok http 8081
python setup_agent_tool.py --webhook-url https://<ngrok-id>.ngrok-free.app/tools/describe_scene
```

## Flags

```
--engine amd|gemini    vision backend (default: amd, auto-falls back to gemini)
--interval N           seconds between captures (default: 3)
--focus general|ocr|navigation|safety
--save-frames          save every captured frame to frames/
```
