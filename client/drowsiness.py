import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from collections import deque
from core.config import (
    EAR_CLOSED_THRESHOLD,
    EYE_CLOSURE_MIN_SECONDS,
    PERCLOS_WINDOW_SECONDS,
    PERCLOS_WARNING_THRESHOLD,
    PERCLOS_DROWSY_THRESHOLD,
    HEAD_DOWN_PITCH_THRESHOLD,
    NOTE_TAKING_PITCH_MAX,
    NOTE_TAKING_PERCLOS_MAX
)

class StudentDrowsinessState:
    def __init__(self, student_id):
        self.student_id = student_id
        # Queue stores tuples of (timestamp, is_closed)
        self.history = deque()
        self.eyes_closed_start = None
        self.blink_count = 0
        self.last_state = "ALERT"

    def update(self, ear_avg, pitch, timestamp, ear_threshold=None):
        closed_thresh = ear_threshold if ear_threshold is not None else EAR_CLOSED_THRESHOLD
        is_closed = ear_avg < closed_thresh
        
        # Add current frame info to history
        self.history.append((timestamp, is_closed))
        
        # Remove history outside the PERCLOS rolling window
        while self.history and timestamp - self.history[0][0] > PERCLOS_WINDOW_SECONDS:
            self.history.popleft()
            
        # Calculate PERCLOS based on timestamps
        if not self.history:
            perclos = 0.0
        else:
            total_duration = self.history[-1][0] - self.history[0][0]
            if total_duration <= 0:
                perclos = 0.0
            else:
                closed_duration = 0.0
                for i in range(1, len(self.history)):
                    t_prev, c_prev = self.history[i-1]
                    t_curr, c_curr = self.history[i]
                    # If eyes were closed during this interval
                    if c_prev:
                        closed_duration += (t_curr - t_prev)
                perclos = closed_duration / total_duration
                
        # Calculate eye closure duration and detect blinks
        eye_closure_duration = 0.0
        blink = False
        
        if is_closed:
            if self.eyes_closed_start is None:
                self.eyes_closed_start = timestamp
            eye_closure_duration = timestamp - self.eyes_closed_start
        else:
            if self.eyes_closed_start is not None:
                duration = timestamp - self.eyes_closed_start
                # If closed for less than the threshold, it's a blink
                if duration < EYE_CLOSURE_MIN_SECONDS:
                    blink = True
                    self.blink_count += 1
                self.eyes_closed_start = None

        # Check head down indication vs note-taking
        head_down = False
        note_taking = False
        if pitch is not None and pitch > HEAD_DOWN_PITCH_THRESHOLD:
            # If pitch is looking downward but eyes are open (low PERCLOS & not closed),
            # classify as NOTE_TAKING / desk focus rather than drowsiness
            if pitch <= NOTE_TAKING_PITCH_MAX and perclos <= NOTE_TAKING_PERCLOS_MAX and not is_closed:
                note_taking = True
            else:
                head_down = True

        # Determine overall state
        state = "ALERT"
        if perclos >= PERCLOS_DROWSY_THRESHOLD or eye_closure_duration >= EYE_CLOSURE_MIN_SECONDS:
            state = "DROWSY"
        elif head_down or perclos >= PERCLOS_WARNING_THRESHOLD:
            state = "MILD_DROWSINESS"
        elif note_taking:
            state = "NOTE_TAKING"

        self.last_state = state

        return {
            "student_id": self.student_id,
            "ear": ear_avg,
            "eyes_closed": is_closed,
            "eye_closure_duration": eye_closure_duration,
            "blink": blink,
            "blink_rate": self.blink_count,  # could be over time, but raw count is ok for now
            "perclos": perclos,
            "head_down": head_down,
            "note_taking": note_taking,
            "drowsiness_state": state
        }

class DrowsinessMonitor:
    def __init__(self):
        self.students = {}

    def update(self, student_id, ear_avg, pitch, timestamp=None, ear_threshold=None):
        if timestamp is None:
            timestamp = time.time()

        if student_id not in self.students:
            self.students[student_id] = StudentDrowsinessState(student_id)

        return self.students[student_id].update(ear_avg, pitch, timestamp, ear_threshold=ear_threshold)

if __name__ == "__main__":
    # Small test for PERCLOS calculation
    import random
    monitor = DrowsinessMonitor()
    student_id = "test_student"
    
    print("Testing PERCLOS calculation...")
    
    # Simulate 30 seconds where eyes are closed for the first 15 seconds, and open for the last 15
    # Expect PERCLOS to be around 50%
    start_time = time.time()
    for i in range(30):
        t = start_time + i
        ear = 0.1 if i < 15 else 0.3 # Closed if < 0.20
        res = monitor.update(student_id, ear, pitch=0.0, timestamp=t)
        
    print(f"Final PERCLOS after 30s (15s closed, 15s open): {res['perclos']*100:.1f}%")
    print(f"Final State: {res['drowsiness_state']}")
    assert abs(res['perclos'] - 0.5) < 0.05
    print("Test passed.")
