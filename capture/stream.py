"""SightLine stream capture — pulls frames from an Instagram Live and sends them to the backend."""

import argparse
import asyncio
import base64
import os
import subprocess
import tempfile
import time
from pathlib import Path

import httpx

BACKEND_URL    = os.getenv("BACKEND_URL",    "http://localhost:8000")
FRAME_INTERVAL = float(os.getenv("FRAME_INTERVAL", "2"))


def get_stream_url(username: str) -> str:
    ig_url = f"https://www.instagram.com/{username}/live/"
    print(f"[Capture] Resolving stream URL for {ig_url} ...")
    result = subprocess.run(
        ["yt-dlp", "--get-url", "--no-warnings", ig_url],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stderr.strip()}")
    url = result.stdout.strip().splitlines()[0]
    print(f"[Capture] Stream URL: {url[:80]}...")
    return url


def capture_frame(stream_url: str, output_path: str) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", stream_url, "-frames:v", "1", "-q:v", "2", "-vf", "scale=640:-1", output_path],
        capture_output=True, timeout=15,
    )
    return result.returncode == 0


async def send_frame(client: httpx.AsyncClient, image_bytes: bytes, mode: str = "general") -> dict:
    b64  = base64.b64encode(image_bytes).decode()
    resp = await client.post(f"{BACKEND_URL}/describe", json={"image": b64, "mode": mode}, timeout=30)
    resp.raise_for_status()
    return resp.json()


async def capture_loop(stream_url: str, mode: str = "general"):
    print(f"[Capture] Starting — interval: {FRAME_INTERVAL}s, mode: {mode}")

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_path = str(Path(tmpdir) / "frame.jpg")

        async with httpx.AsyncClient() as client:
            while True:
                loop_start = time.monotonic()

                if not capture_frame(stream_url, frame_path):
                    print("[Capture] ffmpeg failed — retrying...")
                    await asyncio.sleep(1)
                    continue

                image_bytes = Path(frame_path).read_bytes()

                try:
                    result      = await send_frame(client, image_bytes, mode=mode)
                    description = result.get("description", "")
                    alerts      = result.get("safety_alerts", [])
                    print(f"[Capture] {description[:120]}")
                    if alerts:
                        print(f"[Capture] SAFETY: {alerts}")
                except httpx.HTTPError as e:
                    print(f"[Capture] Backend error: {e}")

                await asyncio.sleep(max(0, FRAME_INTERVAL - (time.monotonic() - loop_start)))


def main():
    parser = argparse.ArgumentParser(description="SightLine stream capture")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--username", help="Instagram username (must be currently live)")
    group.add_argument("--url",      help="Direct HLS/RTMP stream URL")
    parser.add_argument("--mode", default="general", choices=["general", "ocr", "navigation", "safety"])
    args = parser.parse_args()

    stream_url = args.url if args.url else get_stream_url(args.username)

    try:
        asyncio.run(capture_loop(stream_url, mode=args.mode))
    except KeyboardInterrupt:
        print("\n[Capture] Stopped.")


if __name__ == "__main__":
    main()
