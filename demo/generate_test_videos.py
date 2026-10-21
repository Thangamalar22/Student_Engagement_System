"""
Generate two minimal synthetic videos for testing multi_student_sim.py.

These clips contain a plain-colour background with a white rectangle
(representing a face bounding area) so MediaPipe can attempt detection.
They are NOT real face videos — MediaPipe will mostly return no-face.
They are only for verifying that the simulator:
  - Opens the files correctly
  - Handles no-face-detected gracefully
  - Writes DB rows
  - Loops properly

For a real demo, replace student01.mp4 and student02.mp4 with actual
webcam recordings of a person sitting in front of the camera.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'videos')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS      = 20
DURATION = 15          # seconds
W, H     = 640, 480

STUDENTS = [
    ("student01.mp4", (45, 105, 180), "Student 01"),   # blue-ish background
    ("student02.mp4", (45, 160, 80),  "Student 02"),   # green-ish background
]

for fname, bg_color, label in STUDENTS:
    out_path = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(out_path):
        print(f"[SKIP] {fname} already exists — delete it to regenerate.")
        continue

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    for f in range(FPS * DURATION):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = bg_color           # solid colour background

        # Simulate slight brightness oscillation
        t = f / FPS
        brightness = int(40 * abs(np.sin(t * 0.5)))
        frame = np.clip(frame.astype(np.int32) + brightness, 0, 255).astype(np.uint8)

        # Text overlay
        cv2.putText(frame, f"SYNTHETIC VIDEO — {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Replace with real webcam footage.", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Frame {f:04d} / {FPS*DURATION}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        vw.write(frame)

    vw.release()
    size_kb = os.path.getsize(out_path) // 1024
    print(f"[OK] {fname}  {DURATION}s @ {FPS}fps  -> {size_kb} KB  ({out_path})")

print("\nSynthetic videos written.  Now run:")
print("  python demo/multi_student_sim.py")
