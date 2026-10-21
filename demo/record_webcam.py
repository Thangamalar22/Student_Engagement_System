# demo/record_webcam.py
# -------------------------------------------------------
# Utility: Record webcam footage for multi-student demo clips.
#
# Usage:
#   python demo/record_webcam.py --output videos/student01.mp4 --duration 30
# -------------------------------------------------------

import sys
import os
import time
import argparse
import cv2

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

try:
    from core.config import WEBCAM_INDEX
except ImportError:
    WEBCAM_INDEX = 0


def record_clip(
    output_path: str,
    duration: float = 30.0,
    fps: float = 20.0,
    webcam_index: int = WEBCAM_INDEX,
    width: int = 640,
    height: int = 480,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cap = cv2.VideoCapture(webcam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {webcam_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot initialize VideoWriter for {output_path}")

    print("=" * 60)
    print("  Student Video Recorder")
    print(f"  Target file: {os.path.abspath(output_path)}")
    print(f"  Duration   : {duration:.0f} seconds @ {fps} FPS")
    print("  Get ready in front of the camera...")
    print("=" * 60)

    # 3-second countdown
    for count in range(3, 0, -1):
        start_wait = time.time()
        while time.time() - start_wait < 1.0:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Starting in {count}...", (width // 2 - 160, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow("Webcam Recorder", frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                cap.release()
                writer.release()
                cv2.destroyAllWindows()
                return

    print("  [RECORDING STARTED] Press 'q' to stop early.\n")
    start_time = time.time()
    frames_written = 0
    frame_interval = 1.0 / fps
    last_write = 0.0

    while True:
        now = time.time()
        elapsed = now - start_time
        remaining = max(0.0, duration - elapsed)

        if elapsed >= duration:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # Write frame at requested frame rate
        if (now - last_write) >= (frame_interval * 0.9):
            writer.write(frame)
            frames_written += 1
            last_write = now

        # On-screen HUD for recording
        display = frame.copy()
        # Red recording dot
        cv2.circle(display, (25, 25), 8, (0, 0, 255), -1)
        cv2.putText(display, f"REC: {remaining:4.1f}s remaining", (45, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(display, f"Frames: {frames_written}", (45, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(display, "Tip: Look around, blink, or take notes to record varied signals",
                    (15, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)

        cv2.imshow("Webcam Recorder", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n  [STOPPED] User pressed 'q'")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    size_kb = os.path.getsize(output_path) // 1024 if os.path.exists(output_path) else 0
    print("\n" + "=" * 60)
    print("  Recording Complete!")
    print(f"  Frames captured : {frames_written}")
    print(f"  File size       : {size_kb} KB")
    print(f"  Saved to        : {os.path.abspath(output_path)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record webcam video clip for student demo")
    parser.add_argument("--output", type=str, default="videos/student01.mp4",
                        help="Path to save MP4 video clip")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Recording duration in seconds (default: 30)")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Frame rate (default: 20)")
    parser.add_argument("--webcam", type=int, default=WEBCAM_INDEX,
                        help="Webcam device index (default: 0)")
    args = parser.parse_args()

    record_clip(
        output_path=args.output,
        duration=args.duration,
        fps=args.fps,
        webcam_index=args.webcam,
    )
