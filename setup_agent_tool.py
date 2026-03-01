"""setup_agent_tool.py — register the describe_scene webhook tool on the ElevenLabs agent."""

import argparse
import json
import os
import sys

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
AGENT_ID           = "agent_7901kjkkm9jpesna15bwehhwtjr6"

TOOL_DEFINITION = {
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


def print_curl(webhook_url: str):
    tool = {**TOOL_DEFINITION, "api": {"url": webhook_url, "method": "POST"}}
    body = {"tools": [tool]}
    print()
    print(f"curl -s -X PATCH \\")
    print(f"  https://api.elevenlabs.io/v1/convai/agents/{AGENT_ID} \\")
    print(f"  -H 'xi-api-key: {ELEVENLABS_API_KEY or \"YOUR_ELEVENLABS_API_KEY\"}' \\")
    print(f"  -H 'Content-Type: application/json' \\")
    print(f"  -d '{json.dumps(body)}'")
    print()


def register_via_sdk(webhook_url: str) -> bool:
    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        tool   = {**TOOL_DEFINITION, "api": {"url": webhook_url, "method": "POST"}}
        client.conversational_ai.agents.update(agent_id=AGENT_ID, tools=[tool])
        print(f"✅ Registered on agent {AGENT_ID}")
        return True
    except AttributeError:
        return False  # SDK version doesn't expose this method
    except Exception as e:
        print(f"SDK error: {e}")
        return False


def register_via_rest(webhook_url: str) -> bool:
    import httpx

    tool = {**TOOL_DEFINITION, "api": {"url": webhook_url, "method": "POST"}}
    resp = httpx.patch(
        f"https://api.elevenlabs.io/v1/convai/agents/{AGENT_ID}",
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json={"tools": [tool]},
        timeout=15.0,
    )
    if resp.status_code in (200, 204):
        print(f"Registered on agent {AGENT_ID}")
        return True
    print(f"API returned {resp.status_code}: {resp.text}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Register describe_scene on the SightLine ElevenLabs agent")
    parser.add_argument("--webhook-url", help="Public webhook URL (e.g. https://abc.ngrok-free.app/tools/describe_scene)")
    parser.add_argument("--print-curl",  action="store_true", help="Print the equivalent curl command")
    args = parser.parse_args()

    if not ELEVENLABS_API_KEY:
        print("[ERROR] ELEVENLABS_API_KEY not set")
        sys.exit(1)

    if args.print_curl:
        print_curl(args.webhook_url or "https://YOUR_NGROK_URL/tools/describe_scene")
        return

    if not args.webhook_url:
        print("Usage: python setup_agent_tool.py --webhook-url https://<ngrok-id>.ngrok-free.app/tools/describe_scene")
        print()
        print("  Start ngrok first:  ngrok http 8081")
        print("  Or print curl:      python setup_agent_tool.py --print-curl")
        sys.exit(1)

    print(f"Registering on agent {AGENT_ID}...")
    print(f"Webhook: {args.webhook_url}\n")

    if not register_via_sdk(args.webhook_url):
        print("(Trying REST API...)")
        if not register_via_rest(args.webhook_url):
            print()
            print("Manual fallback — add in ElevenLabs dashboard:")
            print("  Agents → Your Agent → Tools → Add Tool → Webhook")
            print(f"  Name: describe_scene  |  URL: {args.webhook_url}  |  Method: POST")
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
