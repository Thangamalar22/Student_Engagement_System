# core/db.py
# -------------------------------------------------------
# TASK 5 — SQLite Storage Layer
#
# Manages all persistent data for the engagement system.
# Uses parameterised SQL throughout — no string formatting
# of user-supplied values.
#
# PRIVACY POLICY (enforced here):
#   - No raw frames, face images, screenshots, or video are
#     stored.  Only numerical / categorical features.
#   - All INSERT statements use Python DB-API placeholders (?).
#
# Tables
# ──────
#   students        – one row per tracked student label
#   sessions        – one row per webcam recording session
#   engagement_logs – per-frame feature record (no image data)
#   alerts          – fired when engagement stays low ≥ threshold
# -------------------------------------------------------

import sqlite3
import os
import sys
import time
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.config import DB_PATH


# ── Internal helpers ─────────────────────────────────────────────────────────

def _ensure_data_dir():
    """Create the data/ directory if it does not exist."""
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)


@contextmanager
def _conn():
    """Yield a connection with row_factory set, auto-commit on success."""
    _ensure_data_dir()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row          # allows dict-like access
    con.execute("PRAGMA journal_mode=WAL") # safe for concurrent readers
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Table creation ────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT    NOT NULL UNIQUE,   -- e.g. "Student 01"
    created_at  REAL    NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  REAL    NOT NULL,
    ended_at    REAL,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS engagement_logs (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id               INTEGER NOT NULL REFERENCES sessions(id),
    student_id               INTEGER NOT NULL REFERENCES students(id),
    timestamp                REAL    NOT NULL,
    -- Gaze
    gaze_score               REAL,
    gaze_on_screen           INTEGER,        -- 0/1 boolean
    -- Eye / drowsiness
    ear                      REAL,
    blink                    INTEGER,        -- 0/1
    perclos                  REAL,
    drowsiness_score         REAL,
    drowsiness_state         TEXT,
    -- Head pose
    head_pitch               REAL,
    head_yaw                 REAL,
    head_roll                REAL,
    -- Emotion
    emotion                  TEXT,
    emotion_confidence       REAL,
    -- Engagement
    engagement_score         REAL,
    smoothed_engagement_score REAL
);

CREATE INDEX IF NOT EXISTS idx_eng_session_student
    ON engagement_logs (session_id, student_id);

CREATE INDEX IF NOT EXISTS idx_eng_timestamp
    ON engagement_logs (timestamp);

CREATE TABLE IF NOT EXISTS alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    student_id  INTEGER NOT NULL REFERENCES students(id),
    timestamp   REAL    NOT NULL,
    reason      TEXT    NOT NULL,
    duration    REAL    NOT NULL,   -- seconds below threshold
    score       REAL    NOT NULL    -- smoothed score at alert time
);
"""


def init_db() -> None:
    """
    Create all tables if they don't already exist.
    Safe to call multiple times (uses IF NOT EXISTS).
    """
    with _conn() as con:
        con.executescript(_SCHEMA)
    print(f"[DB] Initialised at {os.path.abspath(DB_PATH)}")


# ── Session management ────────────────────────────────────────────────────────

def create_session(notes: str = "") -> int:
    """
    Open a new recording session and return its integer ID.
    Call once at the start of each webcam run.
    """
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO sessions (started_at, notes) VALUES (?, ?)",
            (time.time(), notes)
        )
        return cur.lastrowid


def close_session(session_id: int) -> None:
    """Mark a session as finished."""
    with _conn() as con:
        con.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (time.time(), session_id)
        )


# ── Student registry ──────────────────────────────────────────────────────────

def _get_or_create_student(con: sqlite3.Connection, label: str) -> int:
    """
    Return the integer student pk for a given label string,
    inserting a new row if this is the first time we see it.
    """
    row = con.execute(
        "SELECT id FROM students WHERE label = ?", (label,)
    ).fetchone()
    if row:
        return row["id"]
    cur = con.execute(
        "INSERT INTO students (label) VALUES (?)", (label,)
    )
    return cur.lastrowid


# ── Logging ───────────────────────────────────────────────────────────────────

def log_features(
    session_id: int,
    label: str,          # "Student 01" / "Student 02" etc.
    timestamp: float,
    *,
    gaze_score: float       = 0.0,
    gaze_on_screen: bool    = False,
    ear: float              = 0.0,
    blink: bool             = False,
    perclos: float          = 0.0,
    drowsiness_score: float = 0.0,
    drowsiness_state: str   = "ALERT",
    head_pitch: float       = 0.0,
    head_yaw: float         = 0.0,
    head_roll: float        = 0.0,
    emotion: str            = "neutral",
    emotion_confidence: float = 0.0,
    engagement_score: float  = 0.0,
    smoothed_engagement_score: float = 0.0,
) -> int:
    """
    Insert one engagement log row.  Returns the new row id.

    NOTE: no image data is accepted by this function — by design.
    """
    with _conn() as con:
        student_pk = _get_or_create_student(con, label)
        cur = con.execute(
            """
            INSERT INTO engagement_logs (
                session_id, student_id, timestamp,
                gaze_score, gaze_on_screen,
                ear, blink, perclos, drowsiness_score, drowsiness_state,
                head_pitch, head_yaw, head_roll,
                emotion, emotion_confidence,
                engagement_score, smoothed_engagement_score
            ) VALUES (
                ?,?,?,  ?,?,  ?,?,?,?,?,  ?,?,?,  ?,?,  ?,?
            )
            """,
            (
                session_id, student_pk, timestamp,
                gaze_score, int(gaze_on_screen),
                ear, int(blink), perclos, drowsiness_score, drowsiness_state,
                head_pitch, head_yaw, head_roll,
                emotion, emotion_confidence,
                engagement_score, smoothed_engagement_score,
            )
        )
        return cur.lastrowid


# Alias used in the pipeline
log_engagement = log_features


def log_alert(
    session_id: int,
    label: str,
    timestamp: float,
    reason: str,
    duration: float,
    score: float,
) -> int:
    """Insert one alert row. Returns new row id."""
    with _conn() as con:
        student_pk = _get_or_create_student(con, label)
        cur = con.execute(
            """
            INSERT INTO alerts (session_id, student_id, timestamp, reason, duration, score)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, student_pk, timestamp, reason, duration, score)
        )
        return cur.lastrowid


# ── Query helpers ─────────────────────────────────────────────────────────────

def get_latest_student_scores(session_id: int) -> List[Dict]:
    """
    Return the most recent smoothed_engagement_score for every student
    in the given session.
    """
    with _conn() as con:
        rows = con.execute(
            """
            SELECT s.label, el.smoothed_engagement_score, el.timestamp
            FROM engagement_logs el
            JOIN students s ON s.id = el.student_id
            WHERE el.session_id = ?
              AND el.timestamp = (
                  SELECT MAX(el2.timestamp)
                  FROM engagement_logs el2
                  WHERE el2.session_id = el.session_id
                    AND el2.student_id = el.student_id
              )
            ORDER BY s.label
            """,
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_class_average(session_id: int) -> Optional[float]:
    """
    Return the average smoothed_engagement_score across all students
    for the most recent log entry per student in this session.
    """
    latest = get_latest_student_scores(session_id)
    if not latest:
        return None
    total = sum(r["smoothed_engagement_score"] for r in latest)
    return round(total / len(latest), 1)


def get_student_history(
    session_id: int,
    label: str,
    limit: int = 100,
) -> List[Dict]:
    """
    Return the last `limit` log rows for a specific student in this session,
    ordered oldest-first.
    """
    with _conn() as con:
        rows = con.execute(
            """
            SELECT el.*
            FROM engagement_logs el
            JOIN students s ON s.id = el.student_id
            WHERE el.session_id = ? AND s.label = ?
            ORDER BY el.timestamp DESC
            LIMIT ?
            """,
            (session_id, label, limit)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_alerts(
    session_id: Optional[int] = None,
    label: Optional[str]      = None,
) -> List[Dict]:
    """
    Return alerts, optionally filtered by session and/or student label.
    """
    clauses = []
    params: list = []
    if session_id is not None:
        clauses.append("a.session_id = ?")
        params.append(session_id)
    if label is not None:
        clauses.append("s.label = ?")
        params.append(label)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    with _conn() as con:
        rows = con.execute(
            f"""
            SELECT a.*, s.label AS student_label
            FROM alerts a
            JOIN students s ON s.id = a.student_id
            {where}
            ORDER BY a.timestamp
            """,
            params
        ).fetchall()
    return [dict(r) for r in rows]


def list_sessions() -> List[Dict]:
    """Return all sessions ordered newest-first."""
    with _conn() as con:
        rows = con.execute(
            """
            SELECT s.id, s.started_at, s.ended_at, s.notes,
                   COUNT(DISTINCT el.student_id) AS student_count,
                   COUNT(el.id)                  AS log_count
            FROM sessions s
            LEFT JOIN engagement_logs el ON el.session_id = s.id
            GROUP BY s.id
            ORDER BY s.started_at DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_class_trend(session_id: int, minutes: int = 10) -> List[Dict]:
    """
    Return per-minute average smoothed_engagement_score for the whole class
    over the last `minutes` minutes of a session.
    Used for the class-average trend line.
    """
    with _conn() as con:
        rows = con.execute(
            """
            SELECT
                CAST(timestamp / 60 AS INTEGER) * 60 AS minute_bucket,
                AVG(smoothed_engagement_score)        AS avg_score
            FROM engagement_logs
            WHERE session_id = ?
            GROUP BY minute_bucket
            ORDER BY minute_bucket DESC
            LIMIT ?
            """,
            (session_id, minutes)
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def get_all_student_latest(session_id: int) -> List[Dict]:
    """
    Return the most recent full row for every student in a session.
    Used to populate individual student cards.
    """
    with _conn() as con:
        rows = con.execute(
            """
            SELECT s.label,
                   el.timestamp,
                   el.gaze_score,
                   el.gaze_on_screen,
                   el.ear,
                   el.perclos,
                   el.drowsiness_state,
                   el.head_pitch,
                   el.head_yaw,
                   el.emotion,
                   el.emotion_confidence,
                   el.engagement_score,
                   el.smoothed_engagement_score
            FROM engagement_logs el
            JOIN students s ON s.id = el.student_id
            WHERE el.session_id = ?
              AND el.timestamp = (
                  SELECT MAX(el2.timestamp)
                  FROM engagement_logs el2
                  WHERE el2.session_id = el.session_id
                    AND el2.student_id = el.student_id
              )
            ORDER BY s.label
            """,
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]

