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
    LANDMARK_RADIUS, FONT,
    EAR_LEFT_EYE_IDX, EAR_RIGHT_EYE_IDX,
    DISPLAY_MODE
)
from client.face_tracker import FaceTracker
from client.drowsiness import DrowsinessMonitor
from client.utils.ear import compute_both_ears
from client.utils.head_pose import estimate_head_pose
from client.utils.calibration import CalibrationProfile, compute_gaze_features
from client.utils.emotion_classifier import EmotionClassifier
from client.feature_extractor import extract_features
from client.engagement import EngagementMonitor
from core import db
from core.alert_manager import AlertManager, StudentNudgeManager
import argparse


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
    
    # Use subtle gray for clean mode, bright yellow for debug mode
    color = (200, 200, 200) if DISPLAY_MODE == "clean" else (0, 255, 255)
    radius = 1 if DISPLAY_MODE == "clean" else 2
    
    for idx in iris_indices:
        if idx < len(landmarks.landmark):
            lm = landmarks.landmark[idx]
            cx = int(lm.x * img_w)
            cy = int(lm.y * img_h)
            cv2.circle(frame, (cx, cy), radius, color, -1)


# ── Main capture loop ────────────────────────────────────────────────────────

def run_phase1(default_student: str = "student01"):
    """
    Main entry point for live capture and engagement estimation.
    Opens webcam, runs Face Mesh, tracks faces, loads per-student calibration,
    computes fused engagement, logs to DB, and handles alerts and private nudges.
    Press 'q' to quit, 's' to save a single frame screenshot.
    """

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open webcam at index {WEBCAM_INDEX}. "
            "Try changing WEBCAM_INDEX in core/config.py"
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_interval = 1.0 / TARGET_FPS  
    last_frame_time = 0.0
    frame_count      = 0
    face_frame_count = 0
    actual_fps_vals  = []

    print("=" * 55)
    print("  Student Engagement System — Live Client")
    if default_student:
        print(f"  Target student profile: {default_student}")
    print("  Press 'q' to quit | 's' to save screenshot")
    print("=" * 55)

    tracker = FaceTracker()
    drowsiness_monitor = DrowsinessMonitor()
    emotion_classifier = EmotionClassifier()
    engagement_monitor = EngagementMonitor()

    # ── Calibration cache ───────────────────────────────────────────────────
    calib_cache = {}

    def _resolve_profile(sid, lbl):
        candidates = []
        if default_student:
            candidates.append(default_student)
        candidates.extend([str(lbl), str(sid)])
        if isinstance(sid, int):
            candidates.append(f"student{sid:02d}")

        for cand in candidates:
            if not cand:
                continue
            cand_clean = str(cand).lower().replace(" ", "").replace("_", "")
            if cand_clean in calib_cache:
                return calib_cache[cand_clean]
            prof = CalibrationProfile.load(cand_clean)
            if prof is not None:
                calib_cache[cand_clean] = prof
                print(f"[Calibration] Loaded profile '{cand_clean}' for {lbl}")
                return prof
        return None

    # ── Database + alert manager + student nudge manager ────────────────────
    db.init_db()
    session_id = db.create_session(notes=f"landmarks.py webcam session ({default_student or 'multi'})")
    print(f"[DB] Session ID: {session_id}")

    def _on_alert(student_id, label, alert_dict):
        """Callback: persist alert to DB and print to console."""
        db.log_alert(
            session_id=session_id,
            label=label,
            timestamp=alert_dict["timestamp"],
            reason=alert_dict["reason"],
            duration=alert_dict["duration"],
            score=alert_dict["score"],
        )
        print(f"\n[ALERT - Teacher] {label} | {alert_dict['reason']}")

    def _on_nudge(student_id, label, nudge_dict):
        """Callback: print private nudge to console."""
        print(f"\n[NUDGE - Student Private] {label} | {nudge_dict['message']} (score={nudge_dict['score']:.0f})")

    alert_manager = AlertManager(on_alert=_on_alert)
    nudge_manager = StudentNudgeManager(on_nudge=_on_nudge)

    # Log every N-th processed frame to avoid flooding the database
    _LOG_EVERY_N_FRAMES = 5   # at 5 FPS this is every ~1 second

    # Cache for smooth drawing on every frame
    cached_tracked_faces = []
    cached_features = {}
    cached_face_detected = False
    cached_detection_rate = 0.0
    cached_avg_fps = 0.0

    with mp_face_mesh.FaceMesh(
        max_num_faces      = MAX_FACES,
        refine_landmarks   = REFINE_LANDMARKS,
        min_detection_confidence = DETECTION_CONFIDENCE,
        min_tracking_confidence  = TRACKING_CONFIDENCE
    ) as face_mesh:

        while True:
            ret, raw_frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame — retrying...")
                time.sleep(0.1)
                continue

            now = time.time()
            # MediaPipe expects RGB; OpenCV gives BGR — flip it
            frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            img_h, img_w = display_frame.shape[:2]

            # ── Inference (throttled) ────────────────────────────────────
            if (now - last_frame_time) >= frame_interval:
                elapsed = now - last_frame_time
                last_frame_time = now
                actual_fps = 1.0 / elapsed if elapsed > 0 else 0
                actual_fps_vals.append(actual_fps)
                frame_count += 1

                # Mark as non-writeable to pass by reference (small perf gain)
                frame_rgb.flags.writeable = False
                results = face_mesh.process(frame_rgb)
                frame_rgb.flags.writeable = True

                cached_face_detected = results.multi_face_landmarks is not None

                if cached_face_detected:
                    face_frame_count += 1
                    detections = []
                    for face_landmarks in results.multi_face_landmarks:
                        xs = [lm.x * img_w for lm in face_landmarks.landmark]
                        ys = [lm.y * img_h for lm in face_landmarks.landmark]
                        x_min, x_max = int(min(xs)), int(max(xs))
                        y_min, y_max = int(min(ys)), int(max(ys))
                        
                        margin = 20
                        x_min = max(0, x_min - margin)
                        y_min = max(0, y_min - margin)
                        x_max = min(img_w, x_max + margin)
                        y_max = min(img_h, y_max + margin)
                        
                        w, h = x_max - x_min, y_max - y_min
                        
                        detections.append({
                            'bbox': (x_min, y_min, w, h),
                            'landmarks': face_landmarks
                        })
                        
                    cached_tracked_faces = tracker.update(detections, timestamp=now)
                    
                    # --- Feature Extraction ---
                    cached_features.clear()
                    for face in cached_tracked_faces:
                        student_id = face['track_id']
                        face_landmarks = face['landmarks']
                        
                        prof = _resolve_profile(student_id, face['label'])
                        calibrated_ear_thresh = prof.ear_threshold if prof else None

                        drowsy_state = drowsiness_monitor.update(
                            student_id, ear_avg, pitch, timestamp=now, ear_threshold=calibrated_ear_thresh
                        )
                        perclos          = drowsy_state.get('perclos', 0.0)
                        drowsiness_label = drowsy_state.get('drowsiness_state', 'ALERT')

                        gaze_xy = compute_gaze_features(face_landmarks, img_w, img_h)
                        if prof is not None and pose_result['success']:
                            gaze_on_screen = prof.is_on_screen(yaw, pitch, gaze_xy[0], gaze_xy[1])
                        else:
                            # Without calibration, assume on-screen if head yaw is within comfortable range
                            gaze_on_screen = abs(yaw) <= 25.0

                        face_crop, _ = emotion_classifier.extract_face_roi(display_frame, face_landmarks, img_w, img_h)
                        emotion_result = emotion_classifier.predict(face_crop)
                        emotion_label  = emotion_result.get('raw_emotion', 'neutral')
                        emotion_conf   = emotion_result.get('confidence', 0.0)

                        # ── Engagement score fusion ──────────────────────
                        eng = engagement_monitor.update(
                            student_id=student_id,
                            timestamp=now,
                            gaze_on_screen=gaze_on_screen,
                            ear=ear_avg,
                            perclos=perclos,
                            drowsiness_state=drowsiness_label,
                            yaw=yaw,
                            pitch=pitch,
                            emotion_label=emotion_label,
                            emotion_confidence=emotion_conf,
                        )
                        raw_score      = eng['raw_engagement_score']
                        smoothed_score = eng['smoothed_engagement_score']

                        feat = extract_features(
                            student_id=student_id,
                            timestamp=now,
                            face_detected=True,
                            ear=ear_avg,
                            blink=False,
                            perclos=perclos,
                            drowsiness_state=drowsiness_label,
                            pose_result=pose_result,
                            gaze_xy=gaze_xy,
                            gaze_on_screen=gaze_on_screen,
                            emotion_result=emotion_result
                        )
                        # Store engagement score and calibration metadata
                        feat['engagement_score'] = smoothed_score
                        feat['is_calibrated'] = (prof is not None)
                        feat['gaze_on_screen'] = gaze_on_screen
                        cached_features[student_id] = feat

                        # ── Console print (compact, privacy-safe) ────────
                        label = face['label']
                        calib_str = " (Calibrated)" if prof else ""
                        print(f"\n{label}{calib_str}  raw={raw_score:.0f}  smoothed={smoothed_score:.0f}")
                        print(f"  EAR={ear_avg:.2f}  PERCLOS={perclos:.3f}  state={drowsiness_label}")
                        print(f"  yaw={yaw:+.1f}  pitch={pitch:+.1f}")
                        print(f"  gaze_on_screen={gaze_on_screen}  emotion={emotion_label}({emotion_conf:.2f})")
                        print(f"  components: {eng['components']}")

                        # ── Database logging (every N-th frame) ──────────
                        if frame_count % _LOG_EVERY_N_FRAMES == 0:
                            db.log_features(
                                session_id=session_id,
                                label=label,
                                timestamp=now,
                                gaze_score=eng['components'].get('gaze_component', 0.0),
                                gaze_on_screen=gaze_on_screen,
                                ear=ear_avg,
                                blink=False,
                                perclos=perclos,
                                drowsiness_score=eng['components'].get('eye_component', 0.0),
                                drowsiness_state=drowsiness_label,
                                head_pitch=pitch,
                                head_yaw=yaw,
                                head_roll=pose_result.get('roll', 0.0) if pose_result else 0.0,
                                emotion=emotion_label,
                                emotion_confidence=emotion_conf,
                                engagement_score=raw_score,
                                smoothed_engagement_score=smoothed_score,
                            )

                        # ── Alert check (Teacher) ─────────────────────────
                        alert_manager.update(
                            student_id=student_id,
                            label=label,
                            score=smoothed_score,
                            timestamp=now,
                        )

                        # ── Nudge check (Student Private) ─────────────────
                        nudge_manager.update(
                            student_id=student_id,
                            label=label,
                            score=smoothed_score,
                            timestamp=now,
                        )

                else:
                    tracker.update([], timestamp=now)
                    cached_tracked_faces = []
                    cached_features.clear()

                cached_detection_rate = (face_frame_count / frame_count * 100 if frame_count > 0 else 0)
                cached_avg_fps = np.mean(actual_fps_vals[-20:]) if actual_fps_vals else 0

            # ── Drawing (every frame) ────────────────────────────────────
            n_landmarks_total = 0

            for face in cached_tracked_faces:
                face_landmarks = face['landmarks']
                x, y, bw, bh = face['bbox']
                label = face['label']
                student_id = face['track_id']

                # Read pre-computed signals from the inference cache
                feat = cached_features.get(student_id, {})
                ear_avg        = feat.get('ear', 0.0)
                perclos_pct    = int(feat.get('perclos', 0.0) * 100)
                state_str      = feat.get('drowsiness_state', 'ALERT')
                smoothed_score = feat.get('engagement_score', 0.0)
                is_calib       = feat.get('is_calibrated', False)
                gaze_on        = feat.get('gaze_on_screen', True)

                # Draw bounding box — color encodes engagement level
                if smoothed_score >= 70:
                    box_color = (0, 200, 80)    # green  — engaged
                elif smoothed_score >= 45:
                    box_color = (0, 200, 220)   # yellow — moderate
                else:
                    box_color = (50, 50, 220)   # red    — low engagement
                box_thickness = 1 if DISPLAY_MODE == "clean" else 2
                cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), box_color, box_thickness)

                # Line 1: Student ID  |  EAR  |  Gaze  |  drowsiness state
                calib_badge = "[CAL]" if is_calib else ""
                gaze_badge  = "ON" if gaze_on else "OFF"
                info_top = f"{label}{calib_badge} | EAR:{ear_avg:.2f} | Gaze:{gaze_badge} | {state_str}"
                # Line 2: engagement estimate
                info_bot = f"Engagement: {smoothed_score:.0f}/100"

                text_color = (0, 255, 0)
                if state_str == "MILD_DROWSINESS":
                    text_color = (0, 255, 255)
                elif state_str == "DROWSY":
                    text_color = (0, 0, 255)
                elif state_str == "NOTE_TAKING":
                    text_color = (255, 200, 50)

                cv2.putText(display_frame, info_top, (x, max(0, y - 22)), FONT, 0.45, text_color, 1, cv2.LINE_AA)
                cv2.putText(display_frame, info_bot, (x, max(0, y - 6)),  FONT, 0.45, box_color,  1, cv2.LINE_AA)

                # Private student nudge banner (if active)
                active_nudge = nudge_manager.get_active_nudge(student_id, now)
                if active_nudge:
                    nudge_y = min(img_h - 10, y + bh + 22)
                    cv2.rectangle(display_frame, (x, nudge_y - 14), (x + max(bw, 280), nudge_y + 6), (0, 140, 255), cv2.FILLED)
                    cv2.putText(display_frame, f"NUDGE: {active_nudge}", (x + 4, nudge_y),
                                FONT, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

                if DISPLAY_MODE == "debug":
                    # Draw the full mesh tessellation in debug mode
                    mp_drawing.draw_landmarks(
                        image                  = display_frame,
                        landmark_list          = face_landmarks,
                        connections            = mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec  = None,
                        connection_drawing_spec= mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                
                # Always draw contours (eyes, lips, face oval), which use the subtle colors from config
                mp_drawing.draw_landmarks(
                    image                  = display_frame,
                    landmark_list          = face_landmarks,
                    connections            = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec  = LANDMARK_SPEC if DISPLAY_MODE == "debug" else None, # Hide dots in clean mode
                    connection_drawing_spec= CONNECTION_SPEC
                )
                
                if REFINE_LANDMARKS:
                    highlight_iris(display_frame, face_landmarks, img_w, img_h)
                    
                n_landmarks_total += len(face_landmarks.landmark)

            if DISPLAY_MODE == "clean":
                hud_lines = [
                    "Student Engagement System",
                    f"Faces: {len(cached_tracked_faces)} | FPS: {cached_avg_fps:.1f}",
                    "Press 'q'=quit  's'=screenshot"
                ]
            else:
                hud_lines = [
                    f"Phase 1 - Face Mesh (Debug)",
                    f"FPS (actual): {cached_avg_fps:.1f} / target {TARGET_FPS}",
                    f"Face detected: {'YES' if cached_face_detected else 'NO'}",
                    f"Detection rate: {cached_detection_rate:.0f}%",
                    f"Landmarks: 468 + 10 iris" if (cached_face_detected and REFINE_LANDMARKS) else "Landmarks: 468 (no iris)",
                    f"Frame: {frame_count}",
                    f"Press 'q'=quit  's'=screenshot"
                ]
            
            display_frame = draw_hud(display_frame, hud_lines)

            cv2.imshow("Phase 1 - Face Mesh", display_frame)

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
    print(f"  Detection rate         : {cached_detection_rate:.1f}%")
    print(f"  Avg FPS (last 20)      : {cached_avg_fps:.1f}")
    db.close_session(session_id)
    print(f"  DB session {session_id} closed")
    import os as _os
    print(f"  Database location      : {_os.path.abspath(db.DB_PATH)}")
    print("────────────────────────────────────────────\n")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live webcam student engagement monitor")
    parser.add_argument("--student", type=str, default="student01",
                        help="Student ID / profile name to bind for calibration (e.g. student01)")
    args = parser.parse_args()
    run_phase1(default_student=args.student)