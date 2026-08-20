# client/utils/calibration.py
# -------------------------------------------------------
# Per-student 4-point calibration.
#
# The student looks at each of 4 screen corners for ~2 seconds.
# We record:
#   - head yaw/pitch at each corner
#   - iris position (relative to eye box) at each corner
#   - baseline EAR (average when eyes are open)
#
# From this we derive:
#   - yaw_range   : [min_yaw,   max_yaw]
#   - pitch_range : [min_pitch, max_pitch]
#   - iris_x_range, iris_y_range
#   - ear_baseline (avg EAR across all calib frames)
#
# A live frame is "on-screen" if BOTH head pose AND iris position
# fall inside the calibrated ranges (with a small tolerance margin).
# -------------------------------------------------------

import json
import os
import numpy as np
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from core.config import (
    CALIB_POINTS, CALIB_DIR, LEFT_IRIS_IDX, RIGHT_IRIS_IDX,
    LEFT_EYE_CORNERS, RIGHT_EYE_CORNERS, EAR_CALIB_MULTIPLIER,
    EAR_DEFAULT_THRESHOLD
)


# ── Iris helpers ────────────────────────────────────────────────────────────

def _iris_center(landmarks, iris_idx, img_w, img_h):
    """Average the 4 iris boundary landmarks to get iris center in pixels."""
    xs, ys = [], []
    for idx in iris_idx:
        lm = landmarks.landmark[idx]
        xs.append(lm.x * img_w)
        ys.append(lm.y * img_h)
    return np.array([np.mean(xs), np.mean(ys)])


def _normalized_iris_position(landmarks, iris_idx, corner_ids,
                                img_w, img_h):
    """
    Return iris position within its eye, normalized to [0,1] on each axis.
    0.5, 0.5 = iris exactly centered in eye box.
    <0.5 x   = looking left (from viewer's perspective)
    >0.5 x   = looking right
    """
    iris = _iris_center(landmarks, iris_idx, img_w, img_h)

    c1 = landmarks.landmark[corner_ids[0]]
    c2 = landmarks.landmark[corner_ids[1]]
    c1_px = np.array([c1.x * img_w, c1.y * img_h])
    c2_px = np.array([c2.x * img_w, c2.y * img_h])

    eye_x_min, eye_x_max = sorted([c1_px[0], c2_px[0]])
    eye_y_center = (c1_px[1] + c2_px[1]) / 2

    # Approximate eye height as ~35% of eye width
    eye_width  = eye_x_max - eye_x_min
    eye_height = eye_width * 0.35
    eye_y_min  = eye_y_center - eye_height / 2
    eye_y_max  = eye_y_center + eye_height / 2

    if eye_width < 1e-6:
        return (0.5, 0.5)

    norm_x = (iris[0] - eye_x_min) / eye_width
    norm_y = (iris[1] - eye_y_min) / (eye_y_max - eye_y_min + 1e-6)
    return (float(np.clip(norm_x, 0, 1)), float(np.clip(norm_y, 0, 1)))


def compute_gaze_features(landmarks, img_w, img_h):
    """
    Return averaged iris (norm_x, norm_y) across both eyes.
    Used both during calibration and live inference.
    """
    lx, ly = _normalized_iris_position(landmarks, LEFT_IRIS_IDX,
                                        LEFT_EYE_CORNERS, img_w, img_h)
    rx, ry = _normalized_iris_position(landmarks, RIGHT_IRIS_IDX,
                                        RIGHT_EYE_CORNERS, img_w, img_h)
    return ((lx + rx) / 2, (ly + ry) / 2)


# ── Calibration data class ──────────────────────────────────────────────────

class CalibrationProfile:
    """
    Stores calibrated ranges for one student.
    Persists to JSON.
    """

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.yaw_range      = None   # [min, max]
        self.pitch_range    = None
        self.iris_x_range   = None
        self.iris_y_range   = None
        self.ear_baseline   = None
        self.ear_threshold  = EAR_DEFAULT_THRESHOLD
        self.created_at     = None
        self.samples_per_corner = 0

    # ── serialization ────────────────────────────────────────────────
    def to_dict(self):
        return {
            "student_id": self.student_id,
            "yaw_range": self.yaw_range,
            "pitch_range": self.pitch_range,
            "iris_x_range": self.iris_x_range,
            "iris_y_range": self.iris_y_range,
            "ear_baseline": self.ear_baseline,
            "ear_threshold": self.ear_threshold,
            "created_at": self.created_at,
            "samples_per_corner": self.samples_per_corner
        }

    def save(self):
        os.makedirs(CALIB_DIR, exist_ok=True)
        path = os.path.join(CALIB_DIR, f"{self.student_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, student_id: str):
        path = os.path.join(CALIB_DIR, f"{student_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        prof = cls(student_id)
        for k, v in data.items():
            setattr(prof, k, v)
        return prof

    # ── build from raw samples ──────────────────────────────────────
    def build_from_samples(self, corner_samples: dict, all_ears: list,
                        margin_deg: float = 12.0, margin_iris: float = 0.10):
        """
        corner_samples: {"top_left": [(yaw,pitch,irx,iry), ...],
                         "top_right": [...], ...}
        all_ears: list of EAR values recorded during calibration (eyes open)
        margin_*: safety padding around observed range
        """
        yaws, pitches, irxs, irys = [], [], [], []
        min_samples = min(len(v) for v in corner_samples.values())
        self.samples_per_corner = min_samples

        for corner, samples in corner_samples.items():
            for y, p, ix, iy in samples:
                yaws.append(y); pitches.append(p)
                irxs.append(ix); irys.append(iy)

        self.yaw_range    = [float(min(yaws) - margin_deg),
                             float(max(yaws) + margin_deg)]
        self.pitch_range  = [float(min(pitches) - margin_deg),
                             float(max(pitches) + margin_deg)]
        self.iris_x_range = [float(min(irxs) - margin_iris),
                             float(max(irxs) + margin_iris)]
        self.iris_y_range = [float(min(irys) - margin_iris),
                             float(max(irys) + margin_iris)]

        # EAR baseline = average of top quartile (assumes eyes mostly open)
        arr = np.array(all_ears)
        top_q = arr[arr >= np.percentile(arr, 50)]
        self.ear_baseline  = float(np.mean(top_q))
        self.ear_threshold = float(self.ear_baseline * EAR_CALIB_MULTIPLIER)

        self.created_at = datetime.utcnow().isoformat()

    # ── live classification ─────────────────────────────────────────
    def is_on_screen(self, yaw, pitch, iris_x, iris_y) -> bool:
        """
        Return True iff head pose AND iris position both fall in the
        calibrated on-screen zone.
        """
        if self.yaw_range is None:  # not calibrated
            return True  # fail open — don't penalize without data

        pose_ok = (self.yaw_range[0]   <= yaw   <= self.yaw_range[1] and
                   self.pitch_range[0] <= pitch <= self.pitch_range[1])
        iris_ok = (self.iris_x_range[0] <= iris_x <= self.iris_x_range[1] and
                   self.iris_y_range[0] <= iris_y <= self.iris_y_range[1])
        return pose_ok and iris_ok