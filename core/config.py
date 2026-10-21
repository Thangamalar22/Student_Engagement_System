# core/config.py
# -------------------------------------------------------
# Central configuration for the engagement system.
# Edit these values to tune behaviour without touching logic.
# -------------------------------------------------------

# --- Capture ---
TARGET_FPS = 5
WEBCAM_INDEX = 0

# --- MediaPipe Face Mesh ---
MAX_FACES = 5
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE  = 0.5
REFINE_LANDMARKS     = True

# --- Display ---
DISPLAY_MODE       = "clean"  # "clean" or "debug"
LANDMARK_COLOR     = (220, 220, 220)  # Subtle light gray (BGR)
CONNECTION_COLOR   = (220, 220, 220)
LANDMARK_RADIUS    = 1
FONT               = 0  # cv2.FONT_HERSHEY_SIMPLEX

# --- Drowsiness Detection ---
EAR_CLOSED_THRESHOLD = 0.20
EYE_CLOSURE_MIN_SECONDS = 1.0
PERCLOS_WINDOW_SECONDS = 30
PERCLOS_WARNING_THRESHOLD = 0.15
PERCLOS_DROWSY_THRESHOLD = 0.30
HEAD_DOWN_PITCH_THRESHOLD = 20.0

# --- Engagement Score (Phase 4) ---
SCORE_WINDOW_SECONDS = 7
EAR_THRESHOLD        = 0.20
GAZE_OFF_PENALTY     = 40
EAR_PENALTY          = 30

WEIGHT_GAZE      = 0.40
WEIGHT_EAR       = 0.25
WEIGHT_HEAD_POSE = 0.15
WEIGHT_EMOTION   = 0.20

# --- Alert thresholds (Phase 5) ---
ALERT_SCORE_THRESHOLD  = 35    # kept for backward compat
ALERT_DURATION_SECONDS = 60    # kept for backward compat

ENGAGEMENT_ALERT_THRESHOLD         = 40   # smoothed score below this triggers alert
ENGAGEMENT_ALERT_DURATION_SECONDS  = 60   # continuous seconds below threshold needed

# --- Student Nudge (Closed-loop Intervention) ---
STUDENT_NUDGE_THRESHOLD          = 45.0  # smoothed score below this triggers private nudge
STUDENT_NUDGE_DURATION_SECONDS   = 25.0  # seconds below threshold before nudge shows
STUDENT_NUDGE_DISPLAY_SECONDS    = 8.0   # seconds the banner remains visible
STUDENT_NUDGE_COOLDOWN_SECONDS   = 60.0  # minimum seconds between consecutive nudges

# --- Note-Taking / Desk Focus Heuristic ---
NOTE_TAKING_PITCH_MIN            = 12.0  # degrees pitch down (looking at notebook/desk)
NOTE_TAKING_PITCH_MAX            = 35.0  # extreme pitch down (>35) is considered head dropped
NOTE_TAKING_PERCLOS_MAX          = 0.15  # PERCLOS must be low (eyes open, not asleep)

# --- Database (Phase 5) ---
DB_PATH = "data/engagement.db"
ACTIVE_WINDOW_SECONDS = 15


# --- Privacy (Phase 6) ---
ALLOWED_FEATURE_KEYS = [
    "timestamp", "student_id", "ear", "ear_left", "ear_right",
    "blink", "perclos", "drowsiness_state",
    "pitch", "yaw", "roll", "gaze_on_screen",
    "emotion_label", "emotion_confidence", "engagement_score"
]

# =====================================================
# --- Phase 2: Attention Signals ---
# =====================================================

# Eye landmarks for EAR calculation (Soukupová & Čech, 2016)
# 6 points per eye: horizontal corners + top/bottom pairs
EAR_LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]  # p1..p6
EAR_RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]  # p1..p6

# Iris center landmarks (available when refine_landmarks=True)
LEFT_IRIS_IDX  = [474, 475, 476, 477]   # 4 iris boundary points
RIGHT_IRIS_IDX = [469, 470, 471, 472]

# Eye corner landmarks (for computing iris position within eye)
LEFT_EYE_CORNERS  = (362, 263)   # inner, outer
RIGHT_EYE_CORNERS = (33, 133)    # outer, inner

# Head pose — landmark IDs corresponding to the 3D model points
# Order: Nose tip, Chin, Left eye outer, Right eye outer, Left mouth, Right mouth
POSE_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# Head pose — 3D model points (in mm, approximate generic face)
HEAD_POSE_3D_MODEL = [
    (0.0,    0.0,    0.0),      # Nose tip           (landmark 1)
    (0.0,   -63.6,  -12.5),     # Chin               (landmark 152)
    (-43.3,  32.7,  -26.0),     # Left eye outer     (landmark 263)
    (43.3,   32.7,  -26.0),     # Right eye outer    (landmark 33)
    (-28.9, -28.9,  -24.1),     # Left mouth corner  (landmark 287)
    (28.9,  -28.9,  -24.1),     # Right mouth corner (landmark 57)
]

# Calibration
CALIB_POINTS = ["center", "top_left", "top_right", "bottom_left", "bottom_right"]
CALIB_SECONDS_PER_POINT = 2.0
CALIB_DIR = "data/calibration"

# EAR thresholds (used as fallback if no calibration exists)
EAR_DEFAULT_THRESHOLD = 0.20
EAR_CALIB_MULTIPLIER  = 0.75   # Threshold = 75% of calibrated baseline
# =====================================================
# --- Phase 3: Emotion Classification ---
# =====================================================

# HuggingFace model checkpoint (MobileNetV2 fine-tuned on FER2013, ~9MB)
EMOTION_MODEL_NAME = "dima806/facial_emotions_image_detection"

# Canonical FER+ 8 classes (as produced by models/emotion_ferplus.onnx)
FER_CLASSES = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"
]

# Canonical mapping from FER+ classes -> (Classroom Engagement State, Valence Weight)
FER_ENGAGEMENT_MAPPING = {
    "neutral":   ("Engaged",     0.65),  # Attentive baseline
    "happiness": ("Engaged",     0.80),  # Positive affect, attentive
    "surprise":  ("Intrigued",   0.70),  # High arousal / alert
    "fear":      ("Confused",    0.40),  # Negative affect, attentive
    "sadness":   ("Bored",       0.35),  # Low arousal, disengagement
    "disgust":   ("Distracted",  0.25),  # Content aversion
    "anger":     ("Frustrated",  0.30),  # High arousal negative
    "contempt":  ("Distracted",  0.25),  # Disengaged
}

# Backward compatibility mapping for FER2013 7-class nomenclature
FER2013_TO_CLASSROOM = {
    "happy":    ("Engaged",   0.80),
    "neutral":  ("Neutral",   0.65),
    "surprise": ("Intrigued", 0.70),
    "sad":      ("Bored",     0.35),
    "fear":     ("Confused",  0.40),
    "angry":    ("Frustrated",0.30),
    "disgust":  ("Distracted",0.25),
}