# Student Engagement & Attention Analytics System

A privacy-conscious, real-time engagement and attention analytics system designed for online classrooms and video lectures. The system processes video feeds locally, estimates gaze direction, eye closure (EAR / PERCLOS), head pose, and facial expressions, and computes a temporal-smoothed engagement score (0–100) per student displayed on an interactive teacher dashboard.

> ⚠️ **Scientific & Ethical Disclaimer**: All scores are **behavioural estimates** derived from visible cues. They do **not** represent a ground-truth measure of cognitive absorption, comprehension, or intelligence, and must never be used for formal grading, discipline, or clinical diagnosis.

---

## Key Differentiators

1. **Privacy-First Architecture**: Raw video streams never leave the student's device. MediaPipe landmarking and emotion CNN inference run locally on the edge/client; only sanitized numerical feature vectors (EAR, gaze state, Euler angles, emotion label, confidence) are sent to the server or database.
2. **Per-Student Gaze & EAR Calibration**: Uses an interactive 4-point gaze calibration routine to establish personalized pitch, yaw, iris boundaries, and baseline eye openness rather than applying arbitrary global thresholds.
3. **Dual-Signal Fusion**: Combines physiological/attention signals (EAR, PERCLOS, head pose, on-screen gaze) with affective/emotional signals (FER+ CNN classifier) into a single balanced engagement score.
4. **Temporal Smoothing**: Employs a time-weighted rolling window average ($\Delta t$-weighted) over a 7-second sliding horizon to filter out instantaneous noise, saccades, and frame blinks.
5. **Closed-Loop Intervention**: 
   - **Student Private Nudge**: When a student's smoothed score drops below 45 for $>25$ seconds, a gentle visual reminder (*"Refocus on screen • Take a breath"*) appears privately on the student's screen.
   - **Teacher Alert**: If disengagement persists continuously for $\ge 60$ seconds below 40, an alert is logged to SQLite and surfaced on the teacher dashboard.

---

## Build Phases & Architecture

| Phase | Description | Key Modules |
|---|---|---|
| **Phase 1** | **Face & Landmark Pipeline**: MediaPipe Face Mesh at 5 FPS, 468+10 iris landmarks, multi-face tracking | `client/landmarks.py`, `client/face_tracker.py` |
| **Phase 2** | **Attention Signals**: Soukupová & Čech EAR, OpenCV `solvePnP` head pose, 4-point calibration routine, note-taking heuristic | `client/utils/ear.py`, `client/utils/head_pose.py`, `client/utils/calibration.py`, `client/drowsiness.py`, `client/attention.py` |
| **Phase 3** | **Emotion Classification**: ONNX FER+ CNN model with OpenCV DNN (64×64 grayscale ROI), valence weighting | `models/emotion_ferplus.onnx`, `client/utils/emotion_classifier.py`, `client/emotion.py` |
| **Phase 4** | **Score Fusion & Temporal Smoothing**: Weighted fusion formula with rolling time-window smoothing | `client/engagement.py`, `core/config.py` |
| **Phase 5** | **Teacher Dashboard & Closed-Loop Alerts**: Streamlit live dashboard, class average trend line, historical charts, alert log | `dashboard/app.py`, `core/db.py`, `core/alert_manager.py` |
| **Phase 6** | **Privacy Architecture**: Strict feature whitelist sanitization, decoupled client-server data flow | `client/feature_extractor.py`, `core/config.py` |
| **Phase 7** | **Multi-Student Simulation Demo**: Interleaved round-robin simulation on video clips, webcam recorder utility | `demo/multi_student_sim.py`, `demo/record_webcam.py`, `demo/generate_test_videos.py` |

---

## Directory Structure

```text
Student_Engagement_System/
├── client/
│   ├── landmarks.py          # Main live webcam client pipeline
│   ├── attention.py          # Standalone attention & calibration tool
│   ├── face_tracker.py       # Multi-face spatial bounding-box tracker
│   ├── drowsiness.py         # EAR + PERCLOS + note-taking detection
│   ├── emotion.py            # Emotion classifier wrapper & exports
│   ├── engagement.py         # Multi-signal fusion & rolling-average smoother
│   ├── feature_extractor.py  # Privacy-safe dataclass & whitelist sanitizer
│   └── utils/
│       ├── ear.py            # Eye Aspect Ratio calculation
│       ├── head_pose.py      # OpenCV solvePnP head pose (pitch/yaw/roll)
│       ├── calibration.py    # 4-point gaze calibration profile manager
│       └── emotion_classifier.py  # OpenCV DNN ONNX FER+ classifier
├── core/
│   ├── config.py             # Central tunable parameters & weights
│   ├── db.py                 # SQLite storage layer (WAL mode, parameterized)
│   └── alert_manager.py      # Teacher alerts & student private nudges
├── dashboard/
│   └── app.py                # Streamlit teacher analytics dashboard
├── demo/
│   ├── multi_student_sim.py  # Multi-student video demo simulator
│   ├── record_webcam.py      # Webcam recorder for creating student clips
│   └── generate_test_videos.py  # Synthetic test clip generator
├── models/
│   └── emotion_ferplus.onnx  # Pretrained FER+ emotion CNN model
├── data/
│   ├── engagement.db         # SQLite database (auto-generated)
│   └── calibration/          # Calibrated student profiles (.json)
└── videos/
    ├── README.md             # Guide to adding demo student video clips
    ├── student01.mp4         # Sample video clip (Student 01)
    └── student02.mp4         # Sample video clip (Student 02)
```

---

## Quick Start Guide

### 1. Environment Setup

```bash
# Create and activate virtual environment (Python 3.10 or 3.11 recommended)
python -m venv .venv311
.venv311\Scripts\activate   # Windows
# source .venv311/bin/activate  # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. (Optional) Run 4-Point Gaze Calibration

Calibrate your screen before starting a session for maximum gaze accuracy:

```bash
python client/attention.py --student student01 --calibrate
```
*Follow the on-screen target as it animates across the 4 corners of your screen. Calibration ranges will be saved to `data/calibration/student01.json`.*

### 3. Run Live Webcam Client

```bash
# Run live webcam tracking bound to student01's profile
python client/landmarks.py --student student01
```
- Press `q` to quit.
- Press `s` to save an annotated screenshot in `data/`.

### 4. Launch Teacher Dashboard

In a separate terminal window:

```bash
streamlit run dashboard/app.py
```
Open `http://localhost:8501` in your browser. The dashboard shows:
- **Active Students** and **Class Average Engagement Score**.
- **Class Engagement Trend** line (30-minute window with alert threshold).
- **Individual Student Cards** with color-coded borders (Green $\ge 80$, Blue $60-79$, Orange $40-59$, Red $<40$).
- **Multi-Tab Historical Charts**: Engagement (Raw vs Smoothed), Eye Signals (EAR vs PERCLOS), and Head Orientation (Yaw & Pitch).
- **Alerts Table** listing timestamps, student IDs, duration below threshold, and trigger reasons.

---

## Multi-Student Presentation Demo

If you are presenting to an examiner panel and cannot connect multiple live webcams simultaneously, use the multi-student simulator:

### 1. Record Clips for Each Student
```bash
python demo/record_webcam.py --output videos/student01.mp4 --duration 30
python demo/record_webcam.py --output videos/student02.mp4 --duration 30
```

### 2. Run the Multi-Student Simulator
```bash
# Headless simulation (loops through all student clips in parallel)
python demo/multi_student_sim.py

# Or with live OpenCV preview window:
python demo/multi_student_sim.py --display
```

### 3. View Aggregated Analytics
Keep `streamlit run dashboard/app.py` running in your browser to view all simulated students in the same class session in real-time.

---

## Configuration (`core/config.py`)

All core thresholds, feature weights, and timings can be tuned in [`core/config.py`](core/config.py):

| Parameter | Default | Purpose |
|---|---|---|
| `TARGET_FPS` | `5` | Video inference throttle rate (saves CPU compute) |
| `MAX_FACES` | `5` | Maximum simultaneous faces tracked per frame |
| `WEIGHT_GAZE` | `0.40` | Fusion weight for on-screen gaze |
| `WEIGHT_EAR` | `0.25` | Fusion weight for eye openness / drowsiness |
| `WEIGHT_HEAD_POSE` | `0.15` | Fusion weight for head orientation |
| `WEIGHT_EMOTION` | `0.20` | Fusion weight for facial emotion valence |
| `SCORE_WINDOW_SECONDS` | `7.0` | Rolling temporal smoothing window |
| `PERCLOS_WINDOW_SECONDS` | `30.0` | Window for eye closure percentage |
| `ENGAGEMENT_ALERT_THRESHOLD` | `40.0` | Score below this triggers teacher alert |
| `ENGAGEMENT_ALERT_DURATION_SECONDS` | `60.0` | Continuous low engagement required for teacher alert |
| `STUDENT_NUDGE_THRESHOLD` | `45.0` | Score below this triggers student private reminder |
| `STUDENT_NUDGE_DURATION_SECONDS` | `25.0` | Duration below threshold before private nudge fires |

---

## Privacy Policy & Architectural Enforcement

This system adheres to strict privacy-by-design standards:
- **No Video Streaming**: Raw video frames and cropped face images are kept exclusively in volatile memory on the client machine and discarded immediately after feature extraction.
- **Whitelist Enforcement**: [`client/feature_extractor.py`](client/feature_extractor.py) sanitizes every outgoing payload against `ALLOWED_FEATURE_KEYS`. No raw image data, unapproved tensors, or biometric identities are stored.
- **Storage Protection**: The SQLite schema in [`core/db.py`](core/db.py) stores only timestamps, anonymized session IDs, and numeric metrics.

---

## Limitations & Honest Academic Disclaimers

1. **Behavioral Proxy vs. Cognitive Engagement**: The engagement score is an automated behavioral estimate. A student looking at the screen may be distracted, and a student looking down may be actively writing lecture notes. The note-taking heuristic minimizes false positives, but cannot infer mental thought processes.
2. **Camera Intrinsics**: `solvePnP` head pose uses a standard 3D anthropometric face model with focal length approximated from image width. Optical distortion is assumed to be minimal.
3. **Expression Classification**: Facial emotion detection relies on the FER+ dataset. Accuracy may be affected by extreme camera angles, spectacles, or low-light conditions; confidence blending ensures unconfident emotion classifications gently converge to a neutral score (0.65).
