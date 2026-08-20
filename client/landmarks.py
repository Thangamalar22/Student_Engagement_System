# client/phase1_landmarks.py
# -------------------------------------------------------
# Phase 1: Webcam capture + MediaPipe Face Mesh
#
# What this does:
#   1. Opens webcam, throttles to TARGET_FPS
#   2. Runs MediaPipe Face Mesh on each frame
#   3. Draws landmarks + selected connections as visual confirmation
#   4. Overlays basic diagnostics (fps, face detected y/n)
#   5. Prints per-frame landmark count to console
#
# What this does NOT do yet:
#   - Any computation on landmarks (that's Phase 2)
#   - Any emotion inference (Phase 3)
# -------------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os

# Allow imports from project root regardless of where script is run from
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.config import (
    TARGET_FPS, WEBCAM_INDEX, MAX_FACES,
    DETECTION_CONFIDENCE, TRACKING_CONFIDENCE,
    REFINE_LANDMARKS, LANDMARK_COLOR, CONNECTION_COLOR,
    LANDMARK_RADIUS, FONT
)


# ── MediaPipe setup ─────────────────────────────────────────────────────────

mp_face_mesh  = mp.solutions.face_mesh
mp_drawing    = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Drawing specs — we define our own so we can control colour/thickness
LANDMARK_SPEC = mp_drawing.DrawingSpec(
    color=LANDMARK_COLOR,
    thickness=1,
    circle_radius=LANDMARK_RADIUS
)
CONNECTION_SPEC = mp_drawing.DrawingSpec(
    color=CONNECTION_COLOR,
    thickness=1
)

# ── Useful landmark index sets ───────────────────────────────────────────────
# MediaPipe Face Mesh uses 468 landmarks (+ 10 iris points if refine=True).
# We label a few groups here for Phase 2 reference; they're not used yet
# but having them named makes the next phase easier to follow.

# Indices for left/right eye corners (used later for EAR + gaze)
LEFT_EYE_INDICES  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
                      388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173,
                      157, 158, 159, 160, 161, 246]

# Six landmarks used for head-pose solvePnP (Phase 2)
# Nose tip, Chin, Left eye corner, Right eye corner, Left mouth, Right mouth
POSE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]


# ── Helper: draw a semi-transparent HUD overlay ─────────────────────────────

def draw_hud(frame: np.ndarray, text_lines: list[str],
             origin: tuple = (10, 30), line_height: int = 28,
             bg_alpha: float = 0.45) -> np.ndarray:
    """
    Draw a dark translucent rectangle behind status text.
    Returns a copy of the frame with the HUD drawn on it.

    Parameters
    ----------
    frame       : BGR image
    text_lines  : list of strings to display, one per line
    origin      : (x, y) of the top-left corner of the first line
    line_height : vertical spacing between lines in pixels
    bg_alpha    : opacity of the background rectangle (0=transparent,1=solid)
    """
    overlay = frame.copy()
    x, y = origin
    font_scale, thickness = 0.55, 1

    # Compute bounding box for the background rect
    max_width = max(
        cv2.getTextSize(line, FONT, font_scale, thickness)[0][0]
        for line in text_lines
    )
    rect_h = line_height * len(text_lines) + 10
    cv2.rectangle(overlay,
                  (x - 5, y - 20),
                  (x + max_width + 10, y + rect_h),
                  (0, 0, 0), cv2.FILLED)

    # Blend overlay with original
    frame = cv2.addWeighted(overlay, bg_alpha, frame, 1 - bg_alpha, 0)

    # Draw text lines on top
    for i, line in enumerate(text_lines):
        color = (255, 255, 255)  # white text
        cv2.putText(frame, line, (x, y + i * line_height),
                    FONT, font_scale, color, thickness, cv2.LINE_AA)
    return frame


# ── Helper: draw selected iris landmarks differently ────────────────────────

def highlight_iris(frame: np.ndarray, landmarks,
                   img_w: int, img_h: int) -> None:
    """
    When REFINE_LANDMARKS=True, indices 468-477 are iris points.
    Draw them in a distinct colour so we can visually confirm
    they're being picked up (used for gaze in Phase 2).
    """
    iris_indices = list(range(468, 478))  # 10 iris landmarks
    for idx in iris_indices:
        if idx < len(landmarks.landmark):
            lm = landmarks.landmark[idx]
            cx = int(lm.x * img_w)
            cy = int(lm.y * img_h)
            cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)  # yellow dots


# ── Main capture loop ────────────────────────────────────────────────────────

def run_phase1():
    """
    Main entry point for Phase 1.
    Opens webcam, runs Face Mesh, shows annotated video, prints diagnostics.
    Press 'q' to quit, 's' to save a single frame screenshot.
    """

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam at index {WEBCAM_INDEX}. "
            "Try changing WEBCAM_INDEX in core/config.py"
        )

    # We'll request native resolution; MediaPipe handles any size fine
    # but FHD is overkill — 640×480 saves bandwidth and compute
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_interval = 1.0 / TARGET_FPS   # seconds between processed frames
    last_frame_time = 0.0

    # Frame counters for diagnostics
    frame_count      = 0
    face_frame_count = 0
    actual_fps_vals  = []

    print("=" * 55)
    print("  Phase 1 — Face Mesh Pipeline")
    print("  Press 'q' to quit | 's' to save screenshot")
    print("=" * 55)

    with mp_face_mesh.FaceMesh(
        max_num_faces      = MAX_FACES,
        refine_landmarks   = REFINE_LANDMARKS,  # enables iris pts
        min_detection_confidence = DETECTION_CONFIDENCE,
        min_tracking_confidence  = TRACKING_CONFIDENCE
    ) as face_mesh:

        while True:
            ret, raw_frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame — retrying...")
                time.sleep(0.1)
                continue

            # ── FPS throttle ─────────────────────────────────────────────
            now = time.time()
            if (now - last_frame_time) < frame_interval:
                # Still show the raw feed so the window doesn't freeze,
                # but don't run inference on every camera frame
                cv2.imshow("Phase 1 — Face Mesh", raw_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            elapsed = now - last_frame_time
            last_frame_time = now
            actual_fps = 1.0 / elapsed if elapsed > 0 else 0
            actual_fps_vals.append(actual_fps)
            frame_count += 1

            # ── Preprocessing ─────────────────────────────────────────────
            # MediaPipe expects RGB; OpenCV gives BGR — flip it
            frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)

            # Mark as non-writeable to pass by reference (small perf gain)
            frame_rgb.flags.writeable = False
            results = face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # Convert back to BGR for display
            display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            img_h, img_w = display_frame.shape[:2]

            face_detected = results.multi_face_landmarks is not None

            # ── Draw landmarks ─────────────────────────────────────────────
            if face_detected:
                face_frame_count += 1

                for face_landmarks in results.multi_face_landmarks:
                    # Draw the full mesh tessellation (468 points + connections)
                    mp_drawing.draw_landmarks(
                        image                  = display_frame,
                        landmark_list          = face_landmarks,
                        connections            = mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec  = None,          # don't dot every point
                        connection_drawing_spec= mp_drawing_styles
                            .get_default_face_mesh_tesselation_style()
                    )
                    # Draw contours (eyes, lips, face oval) more prominently
                    mp_drawing.draw_landmarks(
                        image                  = display_frame,
                        landmark_list          = face_landmarks,
                        connections            = mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec  = LANDMARK_SPEC,
                        connection_drawing_spec= CONNECTION_SPEC
                    )
                    # Draw iris landmarks in yellow if available
                    if REFINE_LANDMARKS:
                        highlight_iris(display_frame, face_landmarks, img_w, img_h)

                    # ── Console output (every frame for now) ────────────────
                    n_landmarks = len(face_landmarks.landmark)
                    print(f"[Frame {frame_count:04d}] "
                          f"Landmarks: {n_landmarks} | "
                          f"FPS: {actual_fps:.1f}")

            else:
                print(f"[Frame {frame_count:04d}] No face detected")

            # ── HUD overlay ────────────────────────────────────────────────
            detection_rate = (face_frame_count / frame_count * 100
                              if frame_count > 0 else 0)
            avg_fps = np.mean(actual_fps_vals[-20:]) if actual_fps_vals else 0

            hud_lines = [
                f"Phase 1 - Face Mesh",
                f"FPS (actual): {avg_fps:.1f} / target {TARGET_FPS}",
                f"Face detected: {'YES' if face_detected else 'NO'}",
                f"Detection rate: {detection_rate:.0f}%",
                f"Landmarks: 468 + 10 iris" if (face_detected and REFINE_LANDMARKS)
                           else "Landmarks: 468 (no iris)",
                f"Frame: {frame_count}",
                f"Press 'q'=quit  's'=screenshot"
            ]
            display_frame = draw_hud(display_frame, hud_lines)

            # ── Show ────────────────────────────────────────────────────────
            cv2.imshow("Phase 1 — Face Mesh", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"data/screenshot_frame{frame_count:04d}.jpg"
                cv2.imwrite(fname, display_frame)
                print(f"[Screenshot saved] {fname}")

    # ── Cleanup ──────────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    print("\n── Session Summary ─────────────────────────")
    print(f"  Total frames processed : {frame_count}")
    print(f"  Frames with face       : {face_frame_count}")
    print(f"  Detection rate         : {detection_rate:.1f}%")
    print(f"  Avg FPS (last 20)      : {avg_fps:.1f}")
    print("────────────────────────────────────────────\n")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_phase1()