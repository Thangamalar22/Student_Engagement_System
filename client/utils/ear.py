# client/utils/ear.py
# -------------------------------------------------------
# Eye Aspect Ratio (EAR)
# Reference: Soukupová & Čech, "Real-Time Eye Blink Detection
#            using Facial Landmarks" (2016)
#
# EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
#
# where p1..p6 are 6 landmarks around one eye:
#   p1 = left corner
#   p2, p3 = top
#   p4 = right corner
#   p5, p6 = bottom
#
# When the eye is open,  EAR ≈ 0.25 - 0.35
# When the eye is closed, EAR drops to  ~ 0.10 or below
# -------------------------------------------------------

import numpy as np


def _landmark_to_np(landmark, img_w: int, img_h: int) -> np.ndarray:
    """Convert a MediaPipe normalized landmark to a pixel (x,y) numpy vector."""
    return np.array([landmark.x * img_w, landmark.y * img_h], dtype=np.float32)


def compute_ear(landmarks, eye_indices: list, img_w: int, img_h: int) -> float:
    """
    Compute Eye Aspect Ratio for one eye.

    Parameters
    ----------
    landmarks   : mediapipe face_landmarks (has .landmark list of 478 items)
    eye_indices : list of 6 landmark indices in the p1..p6 order above
    img_w, img_h: frame dimensions in pixels (needed to de-normalize)

    Returns
    -------
    float : EAR value. Returns 0.0 if any landmark is invalid.
    """
    try:
        pts = [
            _landmark_to_np(landmarks.landmark[idx], img_w, img_h)
            for idx in eye_indices
        ]
    except (IndexError, AttributeError):
        return 0.0

    p1, p2, p3, p4, p5, p6 = pts

    # Vertical distances between top/bottom eyelid pairs
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)

    # Horizontal eye width
    horizontal = np.linalg.norm(p1 - p4)

    # Guard against divide-by-zero if landmarks collapse
    if horizontal < 1e-6:
        return 0.0

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return float(ear)


def compute_both_ears(landmarks, left_idx, right_idx,
                       img_w: int, img_h: int) -> tuple:
    """
    Convenience wrapper that returns (left_ear, right_ear, avg_ear).
    Using the average is more robust than a single eye.
    """
    left  = compute_ear(landmarks, left_idx,  img_w, img_h)
    right = compute_ear(landmarks, right_idx, img_w, img_h)
    avg   = (left + right) / 2.0
    return left, right, avg