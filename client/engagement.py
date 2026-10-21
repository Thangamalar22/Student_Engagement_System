# client/engagement.py
# -------------------------------------------------------
# TASK 4 — Engagement Score Fusion + Temporal Smoothing
#
# Computes an ESTIMATED engagement score (0–100) per student
# by fusing four independent signal components:
#
#   score = gaze_component   * WEIGHT_GAZE      (default 0.40)
#         + eye_component    * WEIGHT_EAR       (default 0.25)
#         + head_component   * WEIGHT_HEAD_POSE (default 0.15)
#         + emotion_component* WEIGHT_EMOTION   (default 0.20)
#
# Each component is normalised to [0, 100] before weighting.
# The final score is clamped to [0, 100].
#
# IMPORTANT SCIENTIFIC LIMITATION
# ─────────────────────────────────────────────────────────
# This is a VISUAL BEHAVIOURAL ESTIMATE, not a ground-truth
# measure of cognitive engagement.  It cannot detect internal
# mental states.  Results should be treated as indicative only.
#
# Emotion is treated as ONE weak signal.  The mapping is a
# conservative classroom heuristic (see EMOTION_WEIGHTS below)
# and does NOT make medical or psychological claims.
# -------------------------------------------------------

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from collections import deque
from typing import Dict, Optional, Tuple
import time

from core.config import (
    WEIGHT_GAZE,
    WEIGHT_EAR,
    WEIGHT_HEAD_POSE,
    WEIGHT_EMOTION,
    SCORE_WINDOW_SECONDS,
    EAR_CLOSED_THRESHOLD,
    PERCLOS_WARNING_THRESHOLD,
    PERCLOS_DROWSY_THRESHOLD,
)

# ── Sanity-check that weights sum to 1.0 ────────────────────────────────────
_TOTAL_WEIGHT = WEIGHT_GAZE + WEIGHT_EAR + WEIGHT_HEAD_POSE + WEIGHT_EMOTION
assert abs(_TOTAL_WEIGHT - 1.0) < 1e-6, (
    f"Engagement weights must sum to 1.0, got {_TOTAL_WEIGHT:.4f}. "
    "Check WEIGHT_GAZE / WEIGHT_EAR / WEIGHT_HEAD_POSE / WEIGHT_EMOTION in config.py"
)

# ── Emotion → engagement mapping ─────────────────────────────────────────────
# Conservative classroom heuristic.
# These values (0–1) represent a plausible CONTRIBUTION to engagement,
# not a clinical judgment.
#
# Rationale (documented):
#   neutral   → 0.65  (baseline; not disengaged, not highly engaged)
#   happiness → 0.80  (positive affect, likely attentive)
#   surprise  → 0.70  (alert; could be confused or intrigued)
#   sadness   → 0.35  (low arousal; possible disengagement)
#   fear      → 0.40  (high arousal but negative; attention present)
#   disgust   → 0.25  (aversion; likely disengaged from content)
#   anger     → 0.30  (high arousal but negative affect)
#   contempt  → 0.25  (dismissive; likely disengaged)
#
# Deliberate design choices:
#   - "happiness" is NOT assumed = "engaged" (1.0).  A student laughing
#     at something off-screen scores 0.80, not 1.0.
#   - "neutral" is NOT penalised.  Focused study often looks neutral.
#   - Confidence modulates the contribution: low-confidence predictions
#     are blended toward neutral (0.65).
EMOTION_WEIGHTS: Dict[str, float] = {
    "neutral":   0.65,
    "happiness": 0.80,
    "surprise":  0.70,
    "sadness":   0.35,
    "fear":      0.40,
    "disgust":   0.25,
    "anger":     0.30,
    "contempt":  0.25,
}
_EMOTION_NEUTRAL_BASELINE = 0.65   # fallback when label is unknown


# ── Head-pose engagement drop-off ────────────────────────────────────────────
# Soft penalty based on how far the head has turned/tilted.
# Within ±15 deg yaw and ±10 deg pitch → full score.
# Beyond ±45 deg yaw or ±30 deg pitch  → score approaches 0.
_HEAD_YAW_THRESHOLD  = 15.0   # deg — comfortable gaze range
_HEAD_YAW_MAX        = 45.0   # deg — large turn → face away
_HEAD_PITCH_LOOK_DOWN = 20.0  # deg — matching HEAD_DOWN_PITCH_THRESHOLD
_HEAD_PITCH_MAX      = 40.0   # deg — extreme head-down


# ═══════════════════════════════════════════════════════════════════════════
# Component scorers (each returns float in [0, 100])
# ═══════════════════════════════════════════════════════════════════════════

def _gaze_component(gaze_on_screen: bool) -> float:
    """
    Binary: on-screen = 100, off-screen = 0.
    When no calibration is available gaze_on_screen should be passed as True
    (i.e. we don't penalise for missing calibration data).
    """
    return 100.0 if gaze_on_screen else 0.0


def _eye_component(ear: float, perclos: float, drowsiness_state: str) -> float:
    """
    Combines three eye-related signals into one 0-100 component.

    Logic:
      - DROWSY state            → 0–20   (strong penalty)
      - MILD_DROWSINESS state   → 30–50  (moderate penalty)
      - ALERT state             → 60–100, graded by perclos and EAR

    PERCLOS reduces the base score continuously:
      score = base * (1 - perclos_penalty)
    where perclos_penalty ramps from 0 → 1 over [0, PERCLOS_DROWSY_THRESHOLD].
    """
    # State-based floor/ceiling
    if drowsiness_state == "DROWSY":
        base = 15.0
    elif drowsiness_state == "MILD_DROWSINESS":
        base = 45.0
    elif drowsiness_state == "NOTE_TAKING":
        base = 75.0  # Attentive student actively writing/reading notes
    else:  # ALERT
        # EAR above threshold = good; partial closure gives partial score
        if ear <= 0:
            base = 70.0  # no EAR data — be conservative but not zero
        else:
            ear_ratio = min(ear / 0.35, 1.0)   # 0.35 ≈ typical open-eye EAR
            base = 60.0 + 40.0 * ear_ratio      # 60–100

    # PERCLOS continuous penalty (clips at PERCLOS_DROWSY_THRESHOLD)
    perclos_clamped = min(perclos, PERCLOS_DROWSY_THRESHOLD)
    perclos_penalty = perclos_clamped / PERCLOS_DROWSY_THRESHOLD  # 0–1
    score = base * (1.0 - 0.5 * perclos_penalty)  # max 50% reduction from perclos

    return max(0.0, min(100.0, score))


def _head_pose_component(yaw: float, pitch: float, drowsiness_state: str = "ALERT") -> float:
    """
    Score based on head orientation.

    Yaw (left-right) and pitch (up-down) each contribute a soft penalty.
    Within comfortable thresholds → full score.
    Beyond maximum → 0.

    Formula:
      yaw_factor   = 1 - clamp((|yaw| - YAW_THRESHOLD) / (YAW_MAX - YAW_THRESHOLD), 0, 1)
      pitch_factor = 1 - clamp((pitch - PITCH_THRESHOLD) / (PITCH_MAX - PITCH_THRESHOLD), 0, 1)
      score        = 100 * yaw_factor * pitch_factor
    """
    # Yaw factor (symmetric around 0)
    abs_yaw = abs(yaw)
    if abs_yaw <= _HEAD_YAW_THRESHOLD:
        yaw_factor = 1.0
    elif abs_yaw >= _HEAD_YAW_MAX:
        yaw_factor = 0.0
    else:
        yaw_factor = 1.0 - (abs_yaw - _HEAD_YAW_THRESHOLD) / (_HEAD_YAW_MAX - _HEAD_YAW_THRESHOLD)

    # Pitch factor (positive pitch = looking down in our convention)
    if pitch <= _HEAD_PITCH_LOOK_DOWN:
        pitch_factor = 1.0
    elif pitch >= _HEAD_PITCH_MAX:
        pitch_factor = 0.0
    else:
        pitch_factor = 1.0 - (pitch - _HEAD_PITCH_LOOK_DOWN) / (_HEAD_PITCH_MAX - _HEAD_PITCH_LOOK_DOWN)

    # If the student is actively note-taking / desk focusing, floor the pitch penalty at 0.70
    if drowsiness_state == "NOTE_TAKING":
        pitch_factor = max(0.70, pitch_factor)

    score = 100.0 * yaw_factor * pitch_factor
    return max(0.0, min(100.0, score))


def _emotion_component(emotion_label: str, emotion_confidence: float) -> float:
    """
    Maps emotion label to an engagement contribution in [0, 100].

    Low-confidence predictions are blended toward neutral (0.65) so that
    unreliable emotion readings don't dominate the score:
      blended = confidence * weight + (1 - confidence) * NEUTRAL_BASELINE
    """
    weight = EMOTION_WEIGHTS.get(emotion_label, _EMOTION_NEUTRAL_BASELINE)
    # Blend toward neutral when confidence is low
    blended = emotion_confidence * weight + (1.0 - emotion_confidence) * _EMOTION_NEUTRAL_BASELINE
    return max(0.0, min(100.0, blended * 100.0))


# ═══════════════════════════════════════════════════════════════════════════
# Engagement score fusion
# ═══════════════════════════════════════════════════════════════════════════

def compute_raw_engagement(
    gaze_on_screen: bool,
    ear: float,
    perclos: float,
    drowsiness_state: str,
    yaw: float,
    pitch: float,
    emotion_label: str,
    emotion_confidence: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Fuse all signals into a single engagement ESTIMATE.

    Returns:
        raw_score (float): 0–100
        components (dict):  individual component scores for transparency
    """
    g = _gaze_component(gaze_on_screen)
    e = _eye_component(ear, perclos, drowsiness_state)
    h = _head_pose_component(yaw, pitch, drowsiness_state)
    m = _emotion_component(emotion_label, emotion_confidence)

    raw = (
        g * WEIGHT_GAZE
        + e * WEIGHT_EAR
        + h * WEIGHT_HEAD_POSE
        + m * WEIGHT_EMOTION
    )
    raw = max(0.0, min(100.0, raw))

    components = {
        "gaze_component":    round(g, 1),
        "eye_component":     round(e, 1),
        "head_component":    round(h, 1),
        "emotion_component": round(m, 1),
    }
    return raw, components


# ═══════════════════════════════════════════════════════════════════════════
# Per-student temporal smoother
# ═══════════════════════════════════════════════════════════════════════════

class StudentEngagementState:
    """
    Maintains a timestamp-based rolling window of raw engagement scores
    for ONE student.  Returns both the latest raw score and a time-weighted
    smoothed score.
    """

    def __init__(self, student_id, window_seconds: float = SCORE_WINDOW_SECONDS):
        self.student_id = student_id
        self.window_seconds = window_seconds
        # deque of (timestamp, raw_score)
        self._history: deque = deque()
        self.last_raw: float = 0.0
        self.last_smoothed: float = 0.0

    def update(self, raw_score: float, timestamp: float) -> Tuple[float, float]:
        """
        Add a new sample, prune the window, compute smoothed score.

        Smoothing uses a simple time-weighted average:
          smoothed = Σ(score_i * dt_i) / Σ(dt_i)
        where dt_i is the duration each score was "held" (interval to next sample).

        Returns (raw_score, smoothed_score).
        """
        self._history.append((timestamp, raw_score))

        # Prune samples older than the window
        cutoff = timestamp - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        # Compute time-weighted average if we have at least 2 samples
        if len(self._history) >= 2:
            total_weight = 0.0
            weighted_sum = 0.0
            for i in range(1, len(self._history)):
                t_prev, s_prev = self._history[i - 1]
                t_curr, _      = self._history[i]
                dt = t_curr - t_prev
                if dt > 0:
                    weighted_sum += s_prev * dt
                    total_weight += dt
            if total_weight > 0:
                smoothed = weighted_sum / total_weight
            else:
                smoothed = raw_score
        else:
            smoothed = raw_score  # only one sample — raw == smoothed

        self.last_raw      = raw_score
        self.last_smoothed = max(0.0, min(100.0, smoothed))
        return self.last_raw, self.last_smoothed


class EngagementMonitor:
    """
    Session-level manager.  Maintains one StudentEngagementState per
    student_id.  Histories are completely isolated — student_id keys
    are never mixed.
    """

    def __init__(self, window_seconds: float = SCORE_WINDOW_SECONDS):
        self.window_seconds = window_seconds
        self._states: Dict[str, StudentEngagementState] = {}

    def _get_state(self, student_id) -> StudentEngagementState:
        if student_id not in self._states:
            self._states[student_id] = StudentEngagementState(
                student_id, self.window_seconds
            )
        return self._states[student_id]

    def update(
        self,
        student_id,
        timestamp: float,
        gaze_on_screen: bool,
        ear: float,
        perclos: float,
        drowsiness_state: str,
        yaw: float,
        pitch: float,
        emotion_label: str,
        emotion_confidence: float,
    ) -> Dict:
        """
        Compute and record the engagement estimate for one student in one frame.

        Returns a dict with:
          raw_engagement_score    (float, 0–100)
          smoothed_engagement_score (float, 0–100)
          components              (dict of sub-scores for debugging)
        """
        raw, components = compute_raw_engagement(
            gaze_on_screen=gaze_on_screen,
            ear=ear,
            perclos=perclos,
            drowsiness_state=drowsiness_state,
            yaw=yaw,
            pitch=pitch,
            emotion_label=emotion_label,
            emotion_confidence=emotion_confidence,
        )

        state = self._get_state(student_id)
        raw_score, smoothed_score = state.update(raw, timestamp)

        return {
            "raw_engagement_score":      round(raw_score, 1),
            "smoothed_engagement_score": round(smoothed_score, 1),
            "components":                components,
        }

    def get_all_scores(self) -> Dict:
        """Return latest raw + smoothed scores for every tracked student."""
        return {
            sid: {
                "raw":      state.last_raw,
                "smoothed": state.last_smoothed,
            }
            for sid, state in self._states.items()
        }
