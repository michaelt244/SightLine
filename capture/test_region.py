"""test_region.py — Step 2: verify the video call region coordinates."""

import mss
from PIL import Image

REGION = {
    "top":    190,
    "left":   404,
    "width":  538,
    "height": 956,
}

OUTPUT_PATH = "test_frame.jpg"


def main():
    print(f"Capturing region: {REGION}")
    with mss.mss() as sct:
        shot = sct.grab(REGION)
        img  = Image.frombytes("RGB", shot.size, shot.rgb)
        img.save(OUTPUT_PATH, "JPEG", quality=85)
    print(f"Saved: {OUTPUT_PATH}")
    print()
    print("Open test_frame.jpg — does it show only the video call?")
    print("  YES → copy REGION into run_sightline.py")
    print("  NO  → adjust values and run again")


if __name__ == "__main__":
    main()
