"""find_region.py — Step 1: screenshot your screen to find the WhatsApp video call coordinates."""

import mss
import mss.tools


def main():
    with mss.mss() as sct:
        # monitors[0] is a virtual combined screen; monitors[1] is the primary monitor
        monitor = sct.monitors[1]
        print(f"Screen: {monitor['width']} x {monitor['height']} px")
        print("Capturing full screenshot...")

        screenshot  = sct.grab(monitor)
        output_path = "full_screenshot.png"
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=output_path)
        print(f"Saved: {output_path}\n")

    print("=" * 60)
    print("HOW TO FIND YOUR VIDEO CALL COORDINATES")
    print("=" * 60)
    print("""
1. Open full_screenshot.png in Preview (double-click or: open full_screenshot.png)

2. Go to Tools → Show Inspector (Cmd+I)
   The inspector shows pixel coordinates as you hover.

3. Hover over the TOP-LEFT corner of the video area.
   Note the X, Y coordinates.

4. Hover over the BOTTOM-RIGHT corner of the video area.
   Note the X, Y coordinates.

5. Calculate your REGION values:
      top    = Y of the top-left corner
      left   = X of the top-left corner
      width  = X(bottom-right) - X(top-left)
      height = Y(bottom-right) - Y(top-left)

6. Paste those values into the REGION dict in capture/test_region.py

7. Run: python capture/test_region.py
   Open test_frame.jpg to confirm it captures only the video.
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
