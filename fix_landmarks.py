import sys, os, time, sqlite3
sys.path.insert(0, '.')

from core import db
from core.config import DB_PATH, ACTIVE_WINDOW_SECONDS

print("=== TASK 6 FIX A TEST ===")

db.init_db()

# 1. Test Ended Session
ended_session_id = db.create_session(notes="Test Ended Session")
db.log_features(
    session_id=ended_session_id,
    label="Student 01",
    timestamp=time.time() - 30,
    engagement_score=75.0,
    smoothed_engagement_score=75.0
)
db.log_features(
    session_id=ended_session_id,
    label="Student 02",
    timestamp=time.time() - 20,
    engagement_score=60.0,
    smoothed_engagement_score=60.0
)
db.close_session(ended_session_id)

ended_session = next(s for s in db.list_sessions() if s["id"] == ended_session_id)
is_live_ended = ended_session["ended_at"] is None
students_latest_ended = db.get_all_student_latest(ended_session_id)
observed_ended = len(students_latest_ended)

if not is_live_ended:
    active_ended = 0
else:
    now = time.time()
    active_ended = sum(
        1 for s in students_latest_ended
        if s.get("timestamp") and (now - s["timestamp"]) <= ACTIVE_WINDOW_SECONDS
    )

print(f"Ended Session #{ended_session_id}:")
print(f"  Active Students: {active_ended} (Expected: 0)")
print(f"  Students Observed: {observed_ended} (Expected: 2)")

assert active_ended == 0, "Ended session must have 0 active students!"
assert observed_ended == 2, "Ended session must reflect observed student count!"

# 2. Test Live Session with stale and active timestamps
live_session_id = db.create_session(notes="Test Live Session")
now = time.time()

# Student 01 is active (timestamp within last 15s)
db.log_features(
    session_id=live_session_id,
    label="Student 01",
    timestamp=now - 5,
    engagement_score=80.0,
    smoothed_engagement_score=80.0
)
# Student 02 is stale / inactive (timestamp 40s ago)
db.log_features(
    session_id=live_session_id,
    label="Student 02",
    timestamp=now - 40,
    engagement_score=50.0,
    smoothed_engagement_score=50.0
)

live_session = next(s for s in db.list_sessions() if s["id"] == live_session_id)
is_live_live = live_session["ended_at"] is None
students_latest_live = db.get_all_student_latest(live_session_id)
observed_live = len(students_latest_live)

if not is_live_live:
    active_live = 0
else:
    active_live = sum(
        1 for s in students_latest_live
        if s.get("timestamp") and (now - s["timestamp"]) <= ACTIVE_WINDOW_SECONDS
    )

print(f"\nLive Session #{live_session_id}:")
print(f"  Active Students: {active_live} (Expected: 1)")
print(f"  Students Observed: {observed_live} (Expected: 2)")

assert active_live == 1, "Live session should only count student active in last 15s!"
assert observed_live == 2, "Live session should reflect total observed track IDs!"

print("\n=== ALL TESTS PASSED SUCCESSFULLY ===")
