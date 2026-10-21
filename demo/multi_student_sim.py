# demo/multi_student_sim.py
# -------------------------------------------------------
# TASK 7 — Multi-Student Pre-Recorded Video Demo
#
# Simulates a classroom with multiple students, each
# represented by a separate video file.
#
# Directory layout expected:
#
#   videos/
#       student01.mp4
#       student02.mp4
#       student03.mp4
#
# Each video is processed independently with its own:
#   - MediaPipe FaceMesh
#   - FaceTracker  (single-face, per-video)
#   - DrowsinessMonitor
#   - EmotionClassifier
#   - EngagementMonitor
#   - AlertManager
#
# All results are written into the SAME SQLite session so
# the Streamlit dashboard shows every student together.
#
# Processing is interleaved frame-by-frame (round-robin)
# so no single student starves the others.
#
# Usage:
#   python demo/multi_student_sim.py
#   python demo/multi_student_sim.py --video-dir path/to/videos
#   python demo/multi_student_sim.py --fps 5 --display
# -------------------------------------------------------

import sys
import os
import time
import argparse
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import cv2
import numpy as np
import mediapipe as mp

from core.config import (
    MAX_FACES, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE, REFINE_LANDMARKS,
    EAR_LEFT_EYE_IDX, EAR_RIGHT_EYE_IDX,
)
from core import db
from core.alert_manager import AlertManager
from client.face_tracker import FaceTracker
from client.drowsiness import DrowsinessMonitor
from client.engagement import EngagementMonitor
from client.utils.ear import compute_both_ears
from client.utils.head_pose import estimate_head_pose
from client.utils.calibration import CalibrationProfile, compute_gaze_features
from client.utils.emotion_classifier import EmotionClassifier


# ── Video directory ───────────────────────────────────────────────────────────
DEFAULT_VIDEO_DIR = os.path.join(ROOT, "videos")

# Expected filenames (tried in order; first match wins)
CANDIDATE_NAMES = [
    ("Student 01", ["student01.mp4", "student_01.mp4", "student1.mp4"]),
    ("Student 02", ["student02.mp4", "student_02.mp4", "student2.mp4"]),
    ("Student 03", ["student03.mp4", "student_03.mp4", "student3.mp4"]),
    ("Student 04", ["student04.mp4", "student_04.mp4", "student4.mp4"]),
    ("Student 05", ["student05.mp4", "student_05.mp4", "student5.mp4"]),
]

# Log one DB row per this many processed frames per student (≈1 s at 5 FPS)
DB_LOG_EVERY = 5


# ═══════════════════════════════════════════════════════════════════════════════
# Per-student state
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class StudentStream:
    """Everything needed to process one video independently."""
    label:       str
    video_path:  str
    cap:         cv2.VideoCapture

    # Per-student analytics objects (completely isolated)
    face_mesh:        object = field(default=None, repr=False)
    face_tracker:     FaceTracker         = field(default_factory=FaceTracker)
    drowsiness_mon:   DrowsinessMonitor   = field(default_factory=DrowsinessMonitor)
    engagement_mon:   EngagementMonitor   = field(default_factory=EngagementMonitor)
    calib_profile:    Optional[object]    = field(default=None, repr=False)

    # Shared (stateless) — created once globally and passed in
    emotion_clf:   Optional[object] = field(default=None, repr=False)
    alert_mgr:     Optional[object] = field(default=None, repr=False)

    # Frame / timing state
    frame_count:   int   = 0
    face_detected: bool  = False
    finished:      bool  = False
    error:         str   = ""

    # Latest feature snapshot (for console printing)
    last_feat:     Dict  = field(default_factory=dict)


def _open_video(path: str) -> Optional[cv2.VideoCapture]:
    """Open a video file; return None if it cannot be read."""
    if not os.path.isfile(path):
        return None
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        return None
    # Sanity-read first frame
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind
    return cap


def _make_face_mesh() -> object:
    mp_fm = mp.solutions.face_mesh
    return mp_fm.FaceMesh(
        max_num_faces=1,                              # one face per video stream
        refine_landmarks=REFINE_LANDMARKS,
        min_detection_confidence=DETECTION_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Per-frame processing for one student
# ═══════════════════════════════════════════════════════════════════════════════

def _process_frame(
    stream:     StudentStream,
    session_id: int,
    timestamp:  float,
    display:    bool,
) -> Optional[np.ndarray]:
    """
    Read one frame from `stream`, run the full analytics pipeline,
    log to DB, check alerts, and return the annotated frame (or None).

    Returns None if the video has ended or a read error occurred.
    """
    ret, frame = stream.cap.read()
    if not ret:
        stream.finished = True
        return None

    stream.frame_count += 1
    h, w = frame.shape[:2]

    # ── MediaPipe inference ──────────────────────────────────────────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = stream.face_mesh.process(rgb)
    rgb.flags.writeable = True

    annotated = frame.copy() if display else frame

    if not results.multi_face_landmarks:
        stream.face_detected = False
        stream.face_tracker.update([], timestamp=timestamp)
        return annotated

    stream.face_detected = True

    # ── Extract the first (and only) face landmarks ──────────────────────────
    mp_landmarks = results.multi_face_landmarks[0]
    lm = mp_landmarks.landmark

    # Build detection dict compatible with FaceTracker
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

    detection = {"bbox": bbox, "landmarks": mp_landmarks}
    tracked = stream.face_tracker.update([detection], timestamp=timestamp)

    if not tracked:
        return annotated

    face = tracked[0]
    face_landmarks = face["landmarks"]

    # ── Signal extraction ────────────────────────────────────────────────────
    _, _, ear_avg = compute_both_ears(
        face_landmarks, EAR_LEFT_EYE_IDX, EAR_RIGHT_EYE_IDX, w, h
    )
    pose_result = estimate_head_pose(face_landmarks, w, h)
    pitch = pose_result["pitch"] if pose_result["success"] else 0.0
    yaw   = pose_result["yaw"]   if pose_result["success"] else 0.0
    roll  = pose_result["roll"]  if pose_result["success"] else 0.0

    calibrated_ear = stream.calib_profile.ear_threshold if stream.calib_profile else None
    drowsy = stream.drowsiness_mon.update(
        stream.label, ear_avg, pitch, timestamp=timestamp, ear_threshold=calibrated_ear
    )
    perclos        = drowsy.get("perclos", 0.0)
    drowsy_state   = drowsy.get("drowsiness_state", "ALERT")

    gaze_xy        = compute_gaze_features(face_landmarks, w, h)
    if stream.calib_profile is not None and pose_result["success"]:
        gaze_on_screen = stream.calib_profile.is_on_screen(
            yaw, pitch, gaze_xy[0], gaze_xy[1]
        )
    else:
        # Fallback without calibration: comfortable yaw range on screen
        gaze_on_screen = abs(yaw) <= 25.0

    face_crop, _ = stream.emotion_clf.extract_face_roi(frame, face_landmarks, w, h)
    emotion_res   = stream.emotion_clf.predict(face_crop)
    emotion_label = emotion_res.get("raw_emotion", "neutral")
    emotion_conf  = emotion_res.get("confidence", 0.0)

    # ── Engagement score ─────────────────────────────────────────────────────
    eng = stream.engagement_mon.update(
        student_id=stream.label,
        timestamp=timestamp,
        gaze_on_screen=gaze_on_screen,
        ear=ear_avg,
        perclos=perclos,
        drowsiness_state=drowsy_state,
        yaw=yaw,
        pitch=pitch,
        emotion_label=emotion_label,
        emotion_confidence=emotion_conf,
    )
    raw_score      = eng["raw_engagement_score"]
    smoothed_score = eng["smoothed_engagement_score"]

    stream.last_feat = {
        "ear": ear_avg, "perclos": perclos, "drowsy_state": drowsy_state,
        "emotion": emotion_label, "emotion_conf": emotion_conf,
        "raw": raw_score, "smoothed": smoothed_score,
        "yaw": yaw, "pitch": pitch,
        "gaze_on": gaze_on_screen,
    }

    # ── DB logging ────────────────────────────────────────────────────────────
    if stream.frame_count % DB_LOG_EVERY == 0:
        db.log_features(
            session_id=session_id,
            label=stream.label,
            timestamp=timestamp,
            gaze_score=eng["components"].get("gaze_component", 0.0),
            gaze_on_screen=gaze_on_screen,
            ear=ear_avg,
            blink=False,
            perclos=perclos,
            drowsiness_score=eng["components"].get("eye_component", 0.0),
            drowsiness_state=drowsy_state,
            head_pitch=pitch,
            head_yaw=yaw,
            head_roll=roll,
            emotion=emotion_label,
            emotion_confidence=emotion_conf,
            engagement_score=raw_score,
            smoothed_engagement_score=smoothed_score,
        )

    # ── Alert check ───────────────────────────────────────────────────────────
    if stream.alert_mgr is not None:
        stream.alert_mgr.update(
            student_id=stream.label,
            label=stream.label,
            score=smoothed_score,
            timestamp=timestamp,
        )

    # ── Optional overlay ──────────────────────────────────────────────────────
    if display:
        _draw_overlay(annotated, stream, face, w, h, face_landmarks)

    return annotated


def _draw_overlay(frame, stream, face, w, h, face_landmarks):
    """Minimal overlay: bounding box + info text."""
    x, y, bw, bh = face["bbox"]
    f = stream.last_feat
    smoothed = f.get("smoothed", 0.0)

    if smoothed >= 70:
        col = (50, 200, 80)
    elif smoothed >= 45:
        col = (30, 200, 200)
    else:
        col = (50, 50, 220)

    cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, 1)
    calib_badge = "[CAL] " if stream.calib_profile else ""
    gaze_badge = "ON" if f.get("gaze_on", True) else "OFF"
    cv2.putText(frame,
                f"{calib_badge}{stream.label} | EAR:{f.get('ear',0):.2f} | Gaze:{gaze_badge} "
                f"| {f.get('drowsy_state','ALERT')}",
                (x, max(0, y - 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"Eng: {smoothed:.0f}/100 | {f.get('emotion','')}",
                (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════════════════════════
# Main simulator loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulator(
    video_dir: str,
    target_fps: float = 5.0,
    display: bool = False,
    loop: bool = True,
) -> None:

    print("=" * 60)
    print("  Multi-Student Video Simulator")
    print(f"  Video directory: {os.path.abspath(video_dir)}")
    print(f"  Target FPS     : {target_fps}")
    print(f"  Display        : {display}")
    print(f"  Loop videos    : {loop}")
    print("=" * 60)

    # ── Discover videos ───────────────────────────────────────────────────────
    streams: List[StudentStream] = []
    for label, candidates in CANDIDATE_NAMES:
        for fname in candidates:
            path = os.path.join(video_dir, fname)
            cap = _open_video(path)
            if cap is not None:
                calib_id = label.lower().replace(" ", "")
                prof = CalibrationProfile.load(calib_id)
                calib_str = f" (calibrated: {calib_id})" if prof else ""
                print(f"  [FOUND] {label:12}  <--  {fname}{calib_str}")
                streams.append(StudentStream(
                    label=label,
                    video_path=path,
                    cap=cap,
                    calib_profile=prof,
                ))
                break
        else:
            # Check if directory exists at all
            tried = [os.path.join(video_dir, c) for c in candidates]
            print(f"  [SKIP ] {label:12}  (not found; tried {candidates[0]})")

    if not streams:
        print(
            "\n  ERROR: No video files found.\n"
            f"  Create the '{os.path.abspath(video_dir)}' directory and\n"
            "  place student01.mp4, student02.mp4, ... inside it.\n"
            "  See README.md for details."
        )
        return

    print(f"\n  {len(streams)} student stream(s) loaded.\n")

    # ── Shared objects ────────────────────────────────────────────────────────
    emotion_clf = EmotionClassifier()

    # ── DB init ───────────────────────────────────────────────────────────────
    db.init_db()
    session_id = db.create_session(
        notes=f"Video demo — {len(streams)} student(s) — {os.path.basename(video_dir)}"
    )
    print(f"  [DB] Session ID: {session_id}")
    print(f"  [DB] Path: {os.path.abspath(db.DB_PATH)}\n")

    def _on_alert(student_id, label, alert_dict):
        db.log_alert(
            session_id=session_id,
            label=label,
            timestamp=alert_dict["timestamp"],
            reason=alert_dict["reason"],
            duration=alert_dict["duration"],
            score=alert_dict["score"],
        )
        print(f"\n  [ALERT] {label}: {alert_dict['reason']}")

    alert_mgr = AlertManager(on_alert=_on_alert)

    # Wire shared objects into each stream and open FaceMesh
    for s in streams:
        s.face_mesh   = _make_face_mesh()
        s.emotion_clf = emotion_clf
        s.alert_mgr   = alert_mgr

    # ── Display window setup ──────────────────────────────────────────────────
    win_name = "Multi-Student Demo"
    if display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # ── Main loop (round-robin across streams) ────────────────────────────────
    frame_interval = 1.0 / target_fps
    total_frames   = 0
    start_wall     = time.time()
    last_print     = start_wall
    print("  Running... (press Ctrl+C to stop)\n")

    try:
        while True:
            loop_start = time.time()
            timestamp  = loop_start - start_wall    # relative seconds
            all_done   = True

            # Collect annotated frames for optional side-by-side display
            display_frames: List[np.ndarray] = []

            for s in streams:
                if s.finished:
                    if loop:
                        # Rewind
                        s.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        s.finished = False
                    else:
                        display_frames.append(
                            np.zeros((240, 320, 3), dtype=np.uint8)
                        )
                        continue

                all_done = False
                annotated = _process_frame(s, session_id, loop_start, display)
                if annotated is not None and display:
                    # Resize to common thumbnail size
                    thumb = cv2.resize(annotated, (320, 240))
                    display_frames.append(thumb)

            total_frames += 1

            # ── Console status every 5 s ──────────────────────────────────────
            now = time.time()
            if now - last_print >= 5.0:
                elapsed = now - start_wall
                print(f"  t={elapsed:.0f}s  frames={total_frames}")
                for s in streams:
                    f = s.last_feat
                    print(
                        f"    {s.label:12} | "
                        f"EAR={f.get('ear',0):.2f}  "
                        f"PERCLOS={f.get('perclos',0)*100:.1f}%  "
                        f"state={f.get('drowsy_state','—'):15}  "
                        f"raw={f.get('raw',0):.0f}  "
                        f"smooth={f.get('smoothed',0):.0f}  "
                        f"emotion={f.get('emotion','—')}"
                    )
                last_print = now

            # ── Display ───────────────────────────────────────────────────────
            if display and display_frames:
                grid = np.hstack(display_frames)
                cv2.imshow(win_name, grid)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n  [q] Quit requested.")
                    break

            if all_done and not loop:
                print("\n  All videos finished.")
                break

            # ── Rate limiting ─────────────────────────────────────────────────
            elapsed_frame = time.time() - loop_start
            sleep_time = frame_interval - elapsed_frame
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n  [Ctrl-C] Stopping simulator.")

    finally:
        # ── Cleanup ───────────────────────────────────────────────────────────
        for s in streams:
            s.cap.release()
            if s.face_mesh:
                s.face_mesh.close()
        if display:
            cv2.destroyAllWindows()

        db.close_session(session_id)

        elapsed_total = time.time() - start_wall
        print(f"\n{'=' * 60}")
        print(f"  Simulation complete.")
        print(f"  Total wall time : {elapsed_total:.1f}s")
        print(f"  Total frames    : {total_frames}")
        print(f"  DB session      : {session_id}")
        print(f"  DB location     : {os.path.abspath(db.DB_PATH)}")
        print(f"\n  View results with:")
        print(f"    .venv311\\Scripts\\streamlit run dashboard/app.py")
        print(f"{'=' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Multi-student engagement simulator using pre-recorded videos."
    )
    p.add_argument(
        "--video-dir", default=DEFAULT_VIDEO_DIR,
        help=f"Directory containing studentNN.mp4 files (default: {DEFAULT_VIDEO_DIR})"
    )
    p.add_argument(
        "--fps", type=float, default=5.0,
        help="Target processing rate in frames-per-second (default: 5)"
    )
    p.add_argument(
        "--display", action="store_true",
        help="Show side-by-side OpenCV window"
    )
    p.add_argument(
        "--no-loop", dest="loop", action="store_false",
        help="Exit when all videos finish (default: loop forever)"
    )
    p.set_defaults(loop=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_simulator(
        video_dir=args.video_dir,
        target_fps=args.fps,
        display=args.display,
        loop=args.loop,
    )
