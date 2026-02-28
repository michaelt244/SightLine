"""
find_region.py — Step 1: Capture your full screen to find Instagram Live coordinates.

Usage:
    python capture/find_region.py
"""

import mss
import mss.tools

def main():
    with mss.mss() as sct:
        # Monitor 1 = primary monitor (index 0 is the combined "all monitors" virtual screen)
        monitor = sct.monitors[1]

        width = monitor["width"]
        height = monitor["height"]

        print(f"Screen dimensions: {width} x {height} px")
        print("Capturing full screenshot...")

        screenshot = sct.grab(monitor)
        output_path = "full_screenshot.png"
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=output_path)

        print(f"Saved: {output_path}\n")
        print("=" * 60)
        print("HOW TO FIND YOUR INSTAGRAM LIVE VIDEO COORDINATES")
        print("=" * 60)
        print("""
1. Open full_screenshot.png in Preview
   (double-click the file, or: open full_screenshot.png)

2. In Preview, go to:  Tools → Show Inspector  (or press Cmd+I)
   The inspector shows pixel coordinates as you hover.

3. Hover over the TOP-LEFT corner of the Instagram Live
   video area and note the coordinates shown at the bottom
   of the Preview window (format: X, Y).

4. Hover over the BOTTOM-RIGHT corner of the video area
   and note those coordinates.

5. Calculate your region values:
      top    = Y coordinate of the top-left corner
      left   = X coordinate of the top-left corner
      width  = (X of bottom-right) - (X of top-left)
      height = (Y of bottom-right) - (Y of top-left)

6. Open capture/test_region.py and paste your values into
   the REGION dict at the top of the file.

7. Run:  python capture/test_region.py
   to verify the coordinates capture just the video.
""")
        print("=" * 60)

if __name__ == "__main__":
    main()
