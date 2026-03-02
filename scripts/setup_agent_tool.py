"""setup_agent_tool.py — register the describe_scene webhook tool on the ElevenLabs agent."""

import argparse
import json
import sys

from app.core.config import Settings
from app.core.logger import configure_logging, get_logger

settings = Settings.from_env()
configure_logging()
logger = get_logger("sightline.setup_agent_tool")

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
    curl_cmd = (
        "curl -s -X PATCH "
        f"https://api.elevenlabs.io/v1/convai/agents/{settings.elevenlabs_agent_id} "
        "-H 'xi-api-key: $ELEVENLABS_API_KEY' "
        "-H 'Content-Type: application/json' "
        f"-d '{json.dumps(body)}'"
    )
    logger.info("Equivalent curl command", extra={"event": "agent_tool_curl", "context": {"command": curl_cmd}})


def register_via_sdk(webhook_url: str) -> bool:
    try:
        from elevenlabs.client import ElevenLabs
        client = ElevenLabs(api_key=settings.elevenlabs_api_key)
        client.conversational_ai.agents.update(agent_id=settings.elevenlabs_agent_id, tools=build_tools(webhook_url))
        logger.info(
            "Registered tools via SDK",
            extra={"event": "agent_tools_registered_sdk", "context": {"agent_id": settings.elevenlabs_agent_id}},
        )
        return True
    except AttributeError:
        return False  # SDK version doesn't expose this method
    except Exception as e:
        logger.error(
            "SDK registration failed",
            extra={"event": "agent_tools_sdk_error", "context": {"error_type": e.__class__.__name__}},
        )
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
        logger.info(
            "Registered tools via REST",
            extra={"event": "agent_tools_registered_rest", "context": {"agent_id": settings.elevenlabs_agent_id}},
        )
        return True
    logger.error(
        "REST registration failed",
        extra={"event": "agent_tools_rest_error", "context": {"status_code": resp.status_code}},
    )
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
        logger.error("ELEVENLABS_API_KEY not set", extra={"event": "config_error"})
        sys.exit(1)

    if args.print_curl:
        print_curl(args.webhook_url or "https://YOUR_NGROK_URL/tools/describe_scene")
        return

    if not args.webhook_url:
        logger.error(
            "Missing required argument --webhook-url",
            extra={"event": "usage_error", "context": {"example": "https://<public-host>/tools/describe_scene"}},
        )
        sys.exit(1)

    logger.info(
        "Registering agent tools",
        extra={
            "event": "agent_tool_registration_start",
            "context": {"agent_id": settings.elevenlabs_agent_id, "webhook_url": args.webhook_url},
        },
    )

    if not register_via_sdk(args.webhook_url):
        logger.info("Trying REST API fallback", extra={"event": "agent_tool_rest_fallback"})
        if not register_via_rest(args.webhook_url):
            tools = build_tools(args.webhook_url)
            logger.error(
                "Automatic registration failed; manual dashboard setup required",
                extra={
                    "event": "agent_tool_manual_fallback",
                    "context": {
                        "describe_url": tools[0]["api"]["url"],
                        "control_url": tools[1]["api"]["url"],
                    },
                },
            )
            print_curl(args.webhook_url)
            sys.exit(1)

    logger.info(
        "Registration completed",
        extra={
            "event": "agent_tool_registration_complete",
            "context": {
                "next_steps": [
                    "python capture/run_sightline.py",
                    "python webhook/webhook_server.py",
                    "ngrok http 8081",
                ]
            },
        },
    )


if __name__ == "__main__":
    main()
