# dashboard/app.py
# -------------------------------------------------------
# TASK 6 — Streamlit Teacher Dashboard
#
# Data source: SQLite only (core/db.py).
# This file NEVER accesses webcam, video, or image data.
#
# Run:
#   streamlit run dashboard/app.py
# -------------------------------------------------------

import sys
import os
import time
import datetime
import sqlite3

# Ensure project root is on path so core/ is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import inspect
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from core import db
import core.config as config

DB_PATH = getattr(config, "DB_PATH", "data/engagement.db")
ENGAGEMENT_ALERT_THRESHOLD = getattr(config, "ENGAGEMENT_ALERT_THRESHOLD", 40)
ACTIVE_WINDOW_SECONDS = getattr(config, "ACTIVE_WINDOW_SECONDS", 15)

# Version-tolerant width support for Streamlit 1.40+
_ST_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters

def _plot(fig):
    if _ST_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)

def _df(df_data, **kwargs):
    if _ST_HAS_WIDTH:
        st.dataframe(df_data, width="stretch", **kwargs)
    else:
        st.dataframe(df_data, use_container_width=True, **kwargs)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Engagement Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _category(score: float) -> tuple:
    """Return (label, colour) for a given score."""
    if score >= 80:
        return "High",     "#27ae60"
    elif score >= 60:
        return "Moderate", "#2980b9"
    elif score >= 40:
        return "Low",      "#e67e22"
    else:
        return "Critical", "#c0392b"


def _fmt_ts(ts: float) -> str:
    """Unix timestamp → human-readable string."""
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    except Exception:
        return "—"


def _fmt_dt(ts: float) -> str:
    try:
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d  %H:%M:%S")
    except Exception:
        return "—"


def _db_exists() -> bool:
    return os.path.exists(DB_PATH)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Engagement System")
    st.caption("Teacher Dashboard — v1.1")

    st.divider()

    if not _db_exists():
        st.error(f"Database not found:\n`{os.path.abspath(DB_PATH)}`\n\n"
                 "Start the webcam pipeline first:\n```\npython client/landmarks.py\n```")
        st.stop()

    db.init_db()   # safe to call repeatedly; creates tables if missing

    # Session selector
    sessions = db.list_sessions()
    if not sessions:
        st.warning("No sessions recorded yet.\nRun `python client/landmarks.py` to start capturing.")
        st.stop()

    session_labels = {
        s["id"]: (
            f"Session {s['id']}  —  "
            f"{_fmt_dt(s['started_at'])}  "
            f"({'Live' if s['ended_at'] is None else 'Ended'})"
        )
        for s in sessions
    }
    selected_id = st.selectbox(
        "Session",
        options=list(session_labels.keys()),
        format_func=lambda sid: session_labels[sid],
    )
    selected_session = next(s for s in sessions if s["id"] == selected_id)

    st.divider()

    # Auto-refresh
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_secs = st.slider("Refresh every (s)", 2, 30, 5, disabled=not auto_refresh)

    st.divider()
    st.caption(
        "⚠️ **Disclaimer**: Engagement scores are behavioural "
        "ESTIMATES based on gaze, posture, and facial expression. "
        "They do **not** measure cognitive engagement and should "
        "not be used for assessment or clinical decisions."
    )
    st.caption(f"DB: `{os.path.abspath(DB_PATH)}`")


# ── Load data ─────────────────────────────────────────────────────────────────
students_latest = db.get_all_student_latest(selected_id)
class_avg       = db.get_class_average(selected_id)
class_trend     = db.get_class_trend(selected_id, minutes=30)
alerts          = db.get_alerts(session_id=selected_id)

is_live = selected_session["ended_at"] is None

observed_students_count = len(students_latest)

if not is_live:
    active_students_count = 0
else:
    now = time.time()
    active_students_count = sum(
        1 for s in students_latest
        if s.get("timestamp") and (now - s["timestamp"]) <= ACTIVE_WINDOW_SECONDS
    )


# ── Header ────────────────────────────────────────────────────────────────────
col_h1, col_h2, col_h3, col_h4 = st.columns([2.5, 1.2, 1.2, 1.2])
with col_h1:
    status_badge = "🟢 **LIVE**" if is_live else "⚫ **Ended**"
    st.title(f"Session {selected_id}  {status_badge}")
    st.caption(
        f"Started: {_fmt_dt(selected_session['started_at'])}  |  "
        f"{'Running' if is_live else 'Ended: ' + _fmt_dt(selected_session['ended_at'])}"
    )
with col_h2:
    st.metric(
        "Active Students",
        active_students_count,
        help=f"Active in last {ACTIVE_WINDOW_SECONDS}s. Always 0 for ended sessions."
    )
with col_h3:
    st.metric(
        "Students Observed",
        observed_students_count,
        help="Total temporary face-tracking IDs recorded in this session."
    )
with col_h4:
    if class_avg is not None:
        cat_lbl, cat_col = _category(class_avg)
        st.metric("Class Average", f"{class_avg:.0f} / 100", delta=cat_lbl)
    else:
        st.metric("Class Average", "—")


st.divider()


# ── Class average trend ───────────────────────────────────────────────────────
st.subheader("📈 Class Engagement Trend")

if class_trend:
    df_trend = pd.DataFrame(class_trend)
    df_trend["time"] = df_trend["minute_bucket"].apply(
        lambda t: datetime.datetime.fromtimestamp(t).strftime("%H:%M")
    )

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=df_trend["time"],
        y=df_trend["avg_score"],
        mode="lines+markers",
        line=dict(color="#2980b9", width=2),
        marker=dict(size=6),
        name="Class Avg",
        hovertemplate="%{x}<br>Score: %{y:.1f}<extra></extra>",
    ))
    fig_trend.add_hline(
        y=ENGAGEMENT_ALERT_THRESHOLD,
        line_dash="dash",
        line_color="#e74c3c",
        annotation_text=f"Alert threshold ({ENGAGEMENT_ALERT_THRESHOLD})",
        annotation_position="bottom right",
    )
    fig_trend.update_layout(
        yaxis=dict(range=[0, 100], title="Engagement Estimate (0–100)"),
        xaxis_title="Time",
        height=260,
        margin=dict(l=0, r=0, t=20, b=0),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
    )
    _plot(fig_trend)
else:
    st.info("No trend data yet — data accumulates after the first minute of recording.")

st.divider()


# ── Individual student cards ──────────────────────────────────────────────────
st.subheader("👤 Student Cards (Session Track IDs)")
st.caption(
    "Cards show the **latest** feature snapshot per temporary tracking ID (e.g. Student 01, Student 02). "
    "Tracking IDs represent session-level face tracks and do not guarantee unique human identity. "
    "Scores are **behavioural estimates**, not ground-truth engagement."
)

if not students_latest:
    st.info("No student data in this session yet.")
else:
    # Lay cards out in rows of 3
    cols_per_row = 3
    for row_start in range(0, len(students_latest), cols_per_row):
        row_students = students_latest[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)

        for col, s in zip(cols, row_students):
            label     = s["label"]
            raw       = s.get("engagement_score", 0.0) or 0.0
            smoothed  = s.get("smoothed_engagement_score", 0.0) or 0.0
            cat_lbl, cat_col = _category(smoothed)
            gaze_on   = bool(s.get("gaze_on_screen", 0))
            ear       = s.get("ear") or 0.0
            perclos   = s.get("perclos") or 0.0
            d_state   = s.get("drowsiness_state") or "—"
            emotion   = s.get("emotion") or "—"
            emo_conf  = s.get("emotion_confidence") or 0.0
            pitch     = s.get("head_pitch") or 0.0
            yaw       = s.get("head_yaw") or 0.0
            ts        = s.get("timestamp")
            is_student_active = is_live and bool(ts) and ((time.time() - ts) <= ACTIVE_WINDOW_SECONDS)
            status_tag = "🟢 Active" if is_student_active else ("⚫ Inactive" if is_live else "⚫ Session Ended")

            with col:
                st.markdown(
                    f"""
                    <div style="border:1px solid {cat_col};border-radius:10px;
                                padding:14px 16px;margin-bottom:8px;">
                      <h3 style="margin:0;color:{cat_col};">{label} <span style="font-size:12px;color:#aaa;font-weight:normal;">({status_tag})</span></h3>
                      <p style="margin:2px 0;font-size:13px;color:#aaa;">
                        Updated: {_fmt_ts(ts) if ts else '—'}
                      </p>
                      <hr style="border-color:#333;margin:8px 0;">
                      <table style="width:100%;font-size:13px;border-collapse:collapse;">
                        <tr>
                          <td style="color:#aaa;">Engagement (raw)</td>
                          <td style="text-align:right;font-weight:bold;">{raw:.0f} / 100</td>
                        </tr>
                        <tr>
                          <td style="color:#aaa;">Engagement (smoothed)</td>
                          <td style="text-align:right;font-weight:bold;color:{cat_col};">
                            {smoothed:.0f} / 100
                          </td>
                        </tr>
                        <tr>
                          <td style="color:#aaa;">Category</td>
                          <td style="text-align:right;color:{cat_col};">{cat_lbl}</td>
                        </tr>
                        <tr>
                          <td style="color:#aaa;">Gaze on screen</td>
                          <td style="text-align:right;">{"✅ Yes" if gaze_on else "❌ No"}</td>
                        </tr>
                        <tr>
                          <td style="color:#aaa;">EAR</td>
                          <td style="text-align:right;">{ear:.2f}</td>
                        </tr>
                        <tr>
                          <td style="color:#aaa;">PERCLOS</td>
                          <td style="text-align:right;">{perclos*100:.1f}%</td>
                        </tr>
                        <tr>
                          <td style="color:#aaa;">Drowsiness</td>
                          <td style="text-align:right;">{d_state}</td>
                        </tr>
                        <tr>
                          <td style="color:#aaa;">Emotion</td>
                          <td style="text-align:right;">{emotion} ({emo_conf:.0%})</td>
                        </tr>
                        <tr>
                          <td style="color:#aaa;">Head yaw / pitch</td>
                          <td style="text-align:right;">{yaw:+.1f}° / {pitch:+.1f}°</td>
                        </tr>
                      </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Fill empty columns in the last partial row
        for empty_col in cols[len(row_students):]:
            empty_col.empty()

st.divider()


# ── Individual student history charts ─────────────────────────────────────────
st.subheader("📉 Individual Student History")

if students_latest:
    student_names = [s["label"] for s in students_latest]
    chosen = st.selectbox("Choose student", student_names, key="hist_select")

    history = db.get_student_history(selected_id, chosen, limit=150)

    if history:
        df_hist = pd.DataFrame(history)
        df_hist["time"] = df_hist["timestamp"].apply(
            lambda t: datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")
        )

        tab_eng, tab_eye, tab_head = st.tabs(
            ["Engagement", "Eye Signals", "Head Pose"]
        )

        with tab_eng:
            fig_e = go.Figure()
            fig_e.add_trace(go.Scatter(
                x=df_hist["time"], y=df_hist["engagement_score"],
                name="Raw", line=dict(color="#5dade2", width=1, dash="dot"),
                hovertemplate="%{x}<br>Raw: %{y:.1f}<extra></extra>",
            ))
            fig_e.add_trace(go.Scatter(
                x=df_hist["time"], y=df_hist["smoothed_engagement_score"],
                name="Smoothed", line=dict(color="#27ae60", width=2),
                hovertemplate="%{x}<br>Smoothed: %{y:.1f}<extra></extra>",
            ))
            fig_e.add_hline(y=ENGAGEMENT_ALERT_THRESHOLD, line_dash="dash",
                            line_color="#e74c3c",
                            annotation_text="Alert threshold")
            fig_e.add_hrect(y0=80, y1=100, fillcolor="#27ae60", opacity=0.05, line_width=0)
            fig_e.add_hrect(y0=60, y1=80, fillcolor="#2980b9", opacity=0.05, line_width=0)
            fig_e.add_hrect(y0=40, y1=60, fillcolor="#e67e22", opacity=0.05, line_width=0)
            fig_e.add_hrect(y0=0,  y1=40, fillcolor="#c0392b", opacity=0.05, line_width=0)
            fig_e.update_layout(
                yaxis=dict(range=[0, 100], title="Score"),
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fafafa",
            )
            _plot(fig_e)

        with tab_eye:
            fig_eye = go.Figure()
            fig_eye.add_trace(go.Scatter(
                x=df_hist["time"], y=df_hist["ear"],
                name="EAR", line=dict(color="#f39c12", width=2),
            ))
            fig_eye.add_trace(go.Scatter(
                x=df_hist["time"], y=df_hist["perclos"],
                name="PERCLOS", line=dict(color="#e74c3c", width=2),
                yaxis="y2",
            ))
            fig_eye.update_layout(
                yaxis=dict(title="EAR"),
                yaxis2=dict(title="PERCLOS", overlaying="y", side="right",
                            range=[0, 1]),
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fafafa",
            )
            _plot(fig_eye)

        with tab_head:
            fig_head = go.Figure()
            fig_head.add_trace(go.Scatter(
                x=df_hist["time"], y=df_hist["head_yaw"],
                name="Yaw", line=dict(color="#9b59b6", width=2),
            ))
            fig_head.add_trace(go.Scatter(
                x=df_hist["time"], y=df_hist["head_pitch"],
                name="Pitch", line=dict(color="#1abc9c", width=2),
            ))
            fig_head.add_hline(y=0, line_dash="dot", line_color="#555")
            fig_head.update_layout(
                yaxis_title="Degrees",
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="#fafafa",
            )
            _plot(fig_head)
    else:
        st.info(f"No history for {chosen} in this session yet.")
else:
    st.info("No students detected in this session yet.")

st.divider()


# ── Alerts ────────────────────────────────────────────────────────────────────
st.subheader("🚨 Alerts")

if alerts:
    alert_data = [
        {
            "Time":     _fmt_ts(a["timestamp"]),
            "Student":  a["student_label"],
            "Score":    f"{a['score']:.0f}",
            "Duration": f"{a['duration']:.0f}s",
            "Reason":   a["reason"],
        }
        for a in reversed(alerts)   # newest first
    ]
    df_alerts = pd.DataFrame(alert_data)
    _df(
        df_alerts,
        hide_index=True,
        column_config={
            "Score":    st.column_config.NumberColumn("Score", format="%.0f"),
            "Reason":   st.column_config.TextColumn("Reason", width="large"),
        },
    )
else:
    st.success("✅ No alerts for this session.")

st.divider()


# ── Session history table ─────────────────────────────────────────────────────
with st.expander("🗂️ All Sessions"):
    all_sessions = db.list_sessions()
    if all_sessions:
        df_sess = pd.DataFrame([
            {
                "ID":        s["id"],
                "Started":   _fmt_dt(s["started_at"]),
                "Ended":     _fmt_dt(s["ended_at"]) if s["ended_at"] else "Live",
                "Students":  s["student_count"],
                "Log rows":  s["log_count"],
                "Notes":     s["notes"] or "",
            }
            for s in all_sessions
        ])
        _df(df_sess, hide_index=True)
    else:
        st.info("No sessions recorded yet.")

st.caption(
    "⚠️ **Important**: All engagement scores shown here are **behavioural estimates** "
    "derived from gaze, eye-closure, head orientation, and facial expression. "
    "They do not represent a ground-truth measure of cognitive engagement, "
    "attention, or learning. Do not use for grading or medical purposes."
)

# ── Auto-refresh execution ────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_secs)
    st.rerun()

