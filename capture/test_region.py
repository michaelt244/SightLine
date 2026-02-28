"""
test_region.py — Step 2: Verify your Instagram Live video coordinates.

1. Paste the coordinates you found with find_region.py into REGION below.
2. Run:  python capture/test_region.py
3. Open test_frame.jpg and confirm it shows ONLY the Instagram Live video.

Usage:
    python capture/test_region.py
"""

import mss
from PIL import Image
import io

# ── Paste your coordinates here ──────────────────────────────────────────────
REGION = {
    "top":    190,   # Y of the top-left corner of the video
    "left":   404,   # X of the top-left corner of the video
    "width":  538,   # (X of bottom-right) - (X of top-left)
    "height": 956,   # (Y of bottom-right) - (Y of top-left)
}
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_PATH = "test_frame.jpg"

def main():
    print(f"Capturing region: {REGION}")

    with mss.mss() as sct:
        screenshot = sct.grab(REGION)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        img.save(OUTPUT_PATH, "JPEG", quality=85)

    print(f"Saved: {OUTPUT_PATH}")
    print()
    print("Open test_frame.jpg — does it show ONLY the Instagram Live video?")
    print("  YES → coordinates are correct. Use these values in capture_screen.py.")
    print("  NO  → adjust REGION values and run this script again.")

if __name__ == "__main__":
    main()
