# core/alert_manager.py
# -------------------------------------------------------
# TASK 5 — Per-student Disengagement Alert Logic
#
# Fires ONE alert when a student's smoothed engagement score
# stays continuously below ENGAGEMENT_ALERT_THRESHOLD for
# more than ENGAGEMENT_ALERT_DURATION_SECONDS.
#
# When the score recovers above the threshold the timer resets,
# so a new alert can fire if disengagement recurs.
#
# Design choices:
#   - No alert is fired on every frame (rate-limited).
#   - Separate state object per student_id — never mixed.
#   - Alert is passed to a callback so the caller decides
#     how to store / display it (keeps this module testable).
# -------------------------------------------------------

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
from typing import Callable, Dict, Optional

from core.config import (
    ENGAGEMENT_ALERT_THRESHOLD,
    ENGAGEMENT_ALERT_DURATION_SECONDS,
    STUDENT_NUDGE_THRESHOLD,
    STUDENT_NUDGE_DURATION_SECONDS,
    STUDENT_NUDGE_DISPLAY_SECONDS,
    STUDENT_NUDGE_COOLDOWN_SECONDS,
)


class _StudentAlertState:
    """Tracks the continuous low-engagement interval for one student."""

    def __init__(self):
        self.below_since: Optional[float] = None   # timestamp when score went low
        self.alert_fired: bool = False              # has an alert already been fired
                                                    # for the current dip?

    def update(
        self,
        score: float,
        timestamp: float,
        threshold: float,
        duration_required: float,
    ) -> Optional[Dict]:
        """
        Returns a dict describing an alert if one should be fired,
        otherwise returns None.

        Alert dict keys:
          reason    (str)
          duration  (float, seconds below threshold)
          score     (float, current smoothed score)
          timestamp (float)
        """
        if score < threshold:
            # Score is below threshold
            if self.below_since is None:
                # First frame below threshold — start the clock
                self.below_since = timestamp
                self.alert_fired = False

            elapsed = timestamp - self.below_since

            if elapsed >= duration_required and not self.alert_fired:
                # Threshold crossed — fire once
                self.alert_fired = True
                return {
                    "reason":    (
                        f"Smoothed engagement {score:.0f} below "
                        f"threshold {threshold:.0f} for {elapsed:.0f}s"
                    ),
                    "duration":  elapsed,
                    "score":     score,
                    "timestamp": timestamp,
                }
        else:
            # Score recovered — reset
            self.below_since = None
            self.alert_fired = False

        return None


class AlertManager:
    """
    Session-level manager.  Maintains one _StudentAlertState per student.

    Usage:
        mgr = AlertManager(on_alert=lambda sid, lbl, d: db.log_alert(...))
        # every inference frame:
        mgr.update("Student 01", label="Student 01", score=38.5, timestamp=now)
    """

    def __init__(
        self,
        threshold: float = ENGAGEMENT_ALERT_THRESHOLD,
        duration_seconds: float = ENGAGEMENT_ALERT_DURATION_SECONDS,
        on_alert: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        threshold        : smoothed score below which the timer runs
        duration_seconds : continuous seconds needed before alert fires
        on_alert         : optional callback(student_id, label, alert_dict)
                           called whenever an alert fires
        """
        self.threshold        = threshold
        self.duration_seconds = duration_seconds
        self.on_alert         = on_alert
        self._states: Dict[str, _StudentAlertState] = {}

    def _get_state(self, student_id) -> _StudentAlertState:
        key = str(student_id)
        if key not in self._states:
            self._states[key] = _StudentAlertState()
        return self._states[key]

    def update(
        self,
        student_id,
        label: str,
        score: float,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict]:
        """
        Feed the current smoothed engagement score for one student.

        Returns the alert dict if one fired this frame, else None.
        The on_alert callback (if set) is also called with the dict.
        """
        if timestamp is None:
            timestamp = time.time()

        state = self._get_state(student_id)
        alert = state.update(
            score, timestamp,
            self.threshold, self.duration_seconds
        )

        if alert is not None and self.on_alert is not None:
            self.on_alert(student_id, label, alert)

        return alert

    def get_low_since(self, student_id) -> Optional[float]:
        """Return the timestamp when the score went low, or None if alert."""
        return self._states.get(str(student_id), _StudentAlertState()).below_since


class _StudentNudgeState:
    """Tracks low-engagement interval and cooldown for private student nudges."""

    def __init__(self):
        self.below_since: Optional[float] = None
        self.last_nudge_time: float = 0.0
        self.active_until: float = 0.0
        self.active_message: str = ""

    def update(
        self,
        score: float,
        timestamp: float,
        threshold: float,
        duration_required: float,
        display_duration: float,
        cooldown: float,
    ) -> Optional[Dict]:
        if score < threshold:
            if self.below_since is None:
                self.below_since = timestamp

            elapsed = timestamp - self.below_since
            can_nudge = (timestamp - self.last_nudge_time) >= cooldown

            if elapsed >= duration_required and can_nudge:
                self.last_nudge_time = timestamp
                self.active_until = timestamp + display_duration
                self.active_message = "Refocus on screen  *  Take a deep breath"
                return {
                    "message": self.active_message,
                    "score": score,
                    "duration": elapsed,
                    "expires_at": self.active_until,
                    "timestamp": timestamp,
                }
        else:
            self.below_since = None

        return None

    def get_active_nudge(self, timestamp: float) -> Optional[str]:
        if timestamp < self.active_until:
            return self.active_message
        return None


class StudentNudgeManager:
    """
    Manages private student-facing nudges for closed-loop intervention.
    Fires before teacher alert (e.g. at 25s below 45 vs teacher alert at 60s below 40).
    """

    def __init__(
        self,
        threshold: float = STUDENT_NUDGE_THRESHOLD,
        duration_seconds: float = STUDENT_NUDGE_DURATION_SECONDS,
        display_seconds: float = STUDENT_NUDGE_DISPLAY_SECONDS,
        cooldown_seconds: float = STUDENT_NUDGE_COOLDOWN_SECONDS,
        on_nudge: Optional[Callable] = None,
    ):
        self.threshold = threshold
        self.duration_seconds = duration_seconds
        self.display_seconds = display_seconds
        self.cooldown_seconds = cooldown_seconds
        self.on_nudge = on_nudge
        self._states: Dict[str, _StudentNudgeState] = {}

    def _get_state(self, student_id) -> _StudentNudgeState:
        key = str(student_id)
        if key not in self._states:
            self._states[key] = _StudentNudgeState()
        return self._states[key]

    def update(
        self,
        student_id,
        label: str,
        score: float,
        timestamp: Optional[float] = None,
    ) -> Optional[Dict]:
        if timestamp is None:
            timestamp = time.time()

        state = self._get_state(student_id)
        nudge = state.update(
            score, timestamp,
            self.threshold, self.duration_seconds,
            self.display_seconds, self.cooldown_seconds
        )
        if nudge is not None and self.on_nudge is not None:
            self.on_nudge(student_id, label, nudge)
        return nudge

    def get_active_nudge(self, student_id, timestamp: Optional[float] = None) -> Optional[str]:
        if timestamp is None:
            timestamp = time.time()
        return self._get_state(student_id).get_active_nudge(timestamp)

