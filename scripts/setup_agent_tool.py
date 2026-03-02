"""setup_agent_tool.py — register the describe_scene webhook tool on the ElevenLabs agent."""

import argparse
import json
import sys

from app.core.config import Settings

settings = Settings.from_env()

DESCRIBE_TOOL_DEFINITION = {
    "type": "webhook",
    "name": "describe_scene",
    "description": (
        "Sees what the blind user's camera is currently looking at and returns a scene description. "
        "Call this whenever the user asks about their surroundings, what's nearby, if it's safe, "
        "or any question about the physical environment."
    ),
    "api": {"method": "POST"},
    "input_schema": {
        "type": "object",
        "properties": {
            "mode": {
                "type":        "string",
                "enum":        ["general", "ocr", "safety", "navigation"],
                "default":     "general",
                "description": (
                    "Use 'general' for everyday navigation, 'safety' when the user seems worried, "
                    "'ocr' to read signs or text, 'navigation' for route guidance."
                ),
            }
        },
        "required": [],
    },
}

CONTROL_TOOL_DEFINITION = {
    "type": "webhook",
    "name": "control_sightline",
    "description": (
        "Turns SightLine on or off. Call this when the user says things like "
        "'SightLine turn off', 'pause SightLine', or 'turn SightLine on'."
    ),
    "api": {"method": "POST"},
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "enum": ["on", "off"],
                "description": "Use 'off' to pause scene descriptions, 'on' to resume.",
            }
        },
        "required": ["command"],
    },
}


def build_tools(webhook_url: str):
    url = webhook_url.rstrip("/")
    if url.endswith("/tools/describe_scene"):
        base = url[: -len("/tools/describe_scene")]
        describe_url = url
    elif url.endswith("/tools/control"):
        base = url[: -len("/tools/control")]
        describe_url = f"{base}/tools/describe_scene"
    else:
        base = url
        describe_url = f"{base}/tools/describe_scene"
    control_url = f"{base}/tools/control"

    describe_tool = {**DESCRIBE_TOOL_DEFINITION, "api": {"url": describe_url, "method": "POST"}}
    control_tool = {**CONTROL_TOOL_DEFINITION, "api": {"url": control_url, "method": "POST"}}
    return [describe_tool, control_tool]


def print_curl(webhook_url: str):
    body = {"tools": build_tools(webhook_url)}
    print()
    print(f"curl -s -X PATCH \\")
    print(f"  https://api.elevenlabs.io/v1/convai/agents/{settings.elevenlabs_agent_id} \\")
    print("  -H 'xi-api-key: $ELEVENLABS_API_KEY' \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{json.dumps(body)}'")
    print()


def register_via_sdk(webhook_url: str) -> bool:
    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        client.conversational_ai.agents.update(agent_id=settings.elevenlabs_agent_id, tools=build_tools(webhook_url))
        print(f"✅ Registered on agent {settings.elevenlabs_agent_id}")
        return True
    except AttributeError:
        return False  # SDK version doesn't expose this method
    except Exception as e:
        print(f"SDK error: {e}")
        return False


def register_via_rest(webhook_url: str) -> bool:
    import httpx

    resp = httpx.patch(
        f"https://api.elevenlabs.io/v1/convai/agents/{settings.elevenlabs_agent_id}",
        headers={"xi-api-key": settings.elevenlabs_api_key, "Content-Type": "application/json"},
        json={"tools": build_tools(webhook_url)},
        timeout=15.0,
    )
    if resp.status_code in (200, 204):
        print(f"Registered on agent {settings.elevenlabs_agent_id}")
        return True
    print(f"API returned {resp.status_code}: {resp.text}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Register describe_scene on the SightLine ElevenLabs agent")
    parser.add_argument(
        "--webhook-url",
        help="Public webhook URL base or describe endpoint (e.g. https://your-public-host/tools/describe_scene)",
    )
    parser.add_argument("--print-curl",  action="store_true", help="Print the equivalent curl command")
    args = parser.parse_args()

    if not settings.elevenlabs_api_key:
        print("[ERROR] ELEVENLABS_API_KEY not set")
        sys.exit(1)

    if args.print_curl:
        print_curl(args.webhook_url or "https://YOUR_NGROK_URL/tools/describe_scene")
        return

    if not args.webhook_url:
        print("Usage: python scripts/setup_agent_tool.py --webhook-url https://<public-host>/tools/describe_scene")
        print()
        print("  Start ngrok first:  ngrok http 8081")
        print("  Or print curl:      python setup_agent_tool.py --print-curl")
        sys.exit(1)

    print(f"Registering on agent {settings.elevenlabs_agent_id}...")
    print(f"Webhook: {args.webhook_url}\n")

    if not register_via_sdk(args.webhook_url):
        print("(Trying REST API...)")
        if not register_via_rest(args.webhook_url):
            print()
            print("Manual fallback — add in ElevenLabs dashboard:")
            tools = build_tools(args.webhook_url)
            print("  Agents → Your Agent → Tools → Add Tool → Webhook (add both)")
            print(f"  Name: describe_scene    |  URL: {tools[0]['api']['url']}  |  Method: POST")
            print(f"  Name: control_sightline |  URL: {tools[1]['api']['url']}  |  Method: POST")
            print()
            print_curl(args.webhook_url)
            sys.exit(1)

    print()
    print("Next steps:")
    print("  python capture/run_sightline.py   (start capturing)")
    print("  python webhook_server.py           (start webhook on 8081)")
    print("  ngrok http 8081                    (expose publicly)")
    print()


if __name__ == "__main__":
    main()
