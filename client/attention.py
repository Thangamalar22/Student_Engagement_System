# client/attention.py
# -------------------------------------------------------
# Phase 2 + Phase 3: Attention Signals & Emotion Classification
# -------------------------------------------------------

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.config import (
    TARGET_FPS, WEBCAM_INDEX, MAX_FACES,
    DETECTION_CONFIDENCE, TRACKING_CONFIDENCE, REFINE_LANDMARKS,
    EAR_LEFT_EYE_IDX, EAR_RIGHT_EYE_IDX,
    CALIB_POINTS, CALIB_SECONDS_PER_POINT, EAR_DEFAULT_THRESHOLD
)
from client.utils.ear import compute_both_ears
from client.utils.head_pose import estimate_head_pose, draw_pose_axes
from client.utils.calibration import CalibrationProfile, compute_gaze_features
from client.utils.emotion_classifier import EmotionClassifier
from client.landmarks import draw_hud


mp_face_mesh = mp.solutions.face_mesh


def corner_pixel(corner: str, w: int, h: int, pad: int = 40) -> tuple:
    return {
        "center":       (w // 2, h // 2),
        "top_left":     (pad, pad),
        "top_right":    (w - pad, pad),
        "bottom_left":  (pad, h - pad),
        "bottom_right": (w - pad, h - pad),
    }[corner]


def draw_calibration_target(frame, corner, w, h, phase_progress):
    cx, cy = corner_pixel(corner, w, h)
    radius = int(20 + 20 * abs(np.sin(phase_progress * np.pi)))
    cv2.circle(frame, (cx, cy), radius, (0, 255, 255), 3)
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
    cv2.putText(frame, f"LOOK HERE ({corner})", (w // 2 - 200, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def run_calibration(student_id: str):
    print(f"\n[Calibration] Starting for student '{student_id}'")
    print("[Calibration] Look directly at each target as it appears.")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    corner_samples = {c: [] for c in CALIB_POINTS}
    all_ears = []
    frame_interval = 1.0 / TARGET_FPS

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=REFINE_LANDMARKS,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    ) as fm:

        for i in range(3, 0, -1):
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"Starting in {i}...", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow("Calibration", frame)
            cv2.waitKey(1000)

        for corner in CALIB_POINTS:
            print(f"[Calibration] -> {corner}")
            start = time.time()
            last_process = 0.0

            while time.time() - start < CALIB_SECONDS_PER_POINT:
                ret, frame = cap.read()
                if not ret: continue
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                now = time.time()
                if now - last_process >= frame_interval:
                    last_process = now
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = fm.process(rgb)

                    if res.multi_face_landmarks:
                        lm = res.multi_face_landmarks[0]
                        pose = estimate_head_pose(lm, w, h)
                        _, _, ear_avg = compute_both_ears(
                            lm, EAR_LEFT_EYE_IDX, EAR_RIGHT_EYE_IDX, w, h)
                        ix, iy = compute_gaze_features(lm, w, h)

                        if pose["success"]:
                            corner_samples[corner].append((pose["yaw"], pose["pitch"], ix, iy))
                            all_ears.append(ear_avg)

                progress = (time.time() - start) / CALIB_SECONDS_PER_POINT
                draw_calibration_target(frame, corner, w, h, progress)
                cv2.imshow("Calibration", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return None

    cap.release()
    cv2.destroyAllWindows()

    profile = CalibrationProfile(student_id)
    profile.build_from_samples(corner_samples, all_ears, margin_deg=12.0, margin_iris=0.10)
    path = profile.save()
    print(f"\n[Calibration] Saved to {path}")
    return profile


def run_live(student_id: str):
    profile = CalibrationProfile.load(student_id)
    if profile is None:
        print(f"[Live] No calibration found for '{student_id}'. Run with --calibrate.")
        profile = CalibrationProfile(student_id)
    else:
        print(f"[Live] Loaded calibration for '{student_id}'")

    classifier = EmotionClassifier()

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_interval = 1.0 / TARGET_FPS
    last_time = 0.0

    ear_thresh = profile.ear_threshold or EAR_DEFAULT_THRESHOLD
    consec_closed = 0
    blink_count = 0
    drowsy_flag = False

    cached = {
        "ear_avg": 0.0,
        "pose": {"success": False, "yaw": 0, "pitch": 0, "roll": 0},
        "on_screen": False,
        "gaze_xy": (0.5, 0.5),
        "emotion": {"raw_emotion": "neutral", "confidence": 0.0, "engagement_state": "Neutral"},
        "bbox": None
    }

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=REFINE_LANDMARKS,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE
    ) as fm:

        print("\n[Live] Press 'q' to quit, 'r' to recalibrate\n")

        while True:
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            now = time.time()
            if (now - last_time) >= frame_interval:
                last_time = now

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = fm.process(rgb)

                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0]

                    _, _, ear_avg = compute_both_ears(
                        lm, EAR_LEFT_EYE_IDX, EAR_RIGHT_EYE_IDX, w, h)

                    pose_result = estimate_head_pose(lm, w, h)
                    gaze_xy = compute_gaze_features(lm, w, h)

                    on_screen = False
                    if pose_result["success"]:
                        on_screen = profile.is_on_screen(
                            pose_result["yaw"], pose_result["pitch"],
                            gaze_xy[0], gaze_xy[1])

                    # Phase 3: Emotion classification
                    face_crop, bbox = classifier.extract_face_roi(frame, lm, w, h)
                    emo_result = classifier.predict(face_crop)

                    cached["ear_avg"]   = ear_avg
                    cached["pose"]      = pose_result
                    cached["gaze_xy"]   = gaze_xy
                    cached["on_screen"] = on_screen
                    cached["emotion"]   = emo_result
                    cached["bbox"]      = bbox

                    if ear_avg < ear_thresh and ear_avg > 0:
                        consec_closed += 1
                    else:
                        if consec_closed >= 2:
                            blink_count += 1
                        consec_closed = 0
                    drowsy_flag = consec_closed >= 5
                else:
                    cached["on_screen"] = False
                    cached["bbox"] = None

            # Render overlays on every displayed frame
            pose_result = cached["pose"]
            if pose_result.get("success") and pose_result.get("rvec") is not None:
                draw_pose_axes(frame, pose_result)

            if cached["bbox"] is not None:
                bx, by, bw, bh = cached["bbox"]
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 200, 0), 1)

            ear_avg   = cached["ear_avg"]
            on_screen = cached["on_screen"]
            gaze_xy   = cached["gaze_xy"]
            emo       = cached["emotion"]

            hud = [
                f"Phase 3 - Attention & Emotion",
                f"Student: {student_id}",
                f"EAR: {ear_avg:.3f}  (thresh {ear_thresh:.3f})",
                (f"Head: Y:{pose_result['yaw']:+.1f} P:{pose_result['pitch']:+.1f} R:{pose_result['roll']:+.1f}"
                 if pose_result.get("success") else "Head: no pose"),
                f"Gaze (iris): x={gaze_xy[0]:.2f} y={gaze_xy[1]:.2f}",
                f"On-screen: {'YES' if on_screen else 'NO'}",
                f"Emotion: {emo['raw_emotion'].upper()} ({emo['confidence']:.2f})",
                f"State: {emo['engagement_state']}",
                f"Blinks: {blink_count}  Drowsy: {'YES' if drowsy_flag else 'NO'}"
            ]
            frame = draw_hud(frame, hud)

            border_color = (0, 200, 0) if on_screen else (0, 0, 200)
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 4)

            cv2.imshow("Phase 3 — Attention & Emotion", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                cap.release()
                cv2.destroyAllWindows()
                profile = run_calibration(student_id)
                if profile is None: return
                ear_thresh = profile.ear_threshold
                cap = cv2.VideoCapture(WEBCAM_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str, default="student01")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if args.calibrate:
        run_calibration(args.student)
    else:
        run_live(args.student)