# core/config.py
# -------------------------------------------------------
# Central configuration for the engagement system.
# Edit these values to tune behaviour without touching logic.
# -------------------------------------------------------

# --- Capture ---
TARGET_FPS = 5
WEBCAM_INDEX = 0

# --- MediaPipe Face Mesh ---
MAX_FACES = 1
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE  = 0.5
REFINE_LANDMARKS     = True

# --- Display ---
LANDMARK_COLOR     = (0, 255, 0)
CONNECTION_COLOR   = (0, 128, 255)
LANDMARK_RADIUS    = 1
FONT               = 0  # cv2.FONT_HERSHEY_SIMPLEX

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
ALERT_SCORE_THRESHOLD  = 35
ALERT_DURATION_SECONDS = 60

# --- Privacy (Phase 6) ---
ALLOWED_FEATURE_KEYS = [
    "timestamp", "student_id", "ear_left", "ear_right",
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

# Mapping from FER2013 base classes -> Classroom Engagement States
FER2013_TO_CLASSROOM = {
    "happy":    ("Engaged",   1.00),  # (Classroom Label, Engagement Multiplier 0..1)
    "neutral":  ("Neutral",   0.75),
    "surprise": ("Intrigued", 0.90),
    "sad":      ("Bored",     0.30),
    "fear":     ("Confused",  0.40),
    "angry":    ("Frustrated",0.20),
    "disgust":  ("Distracted",0.20),
}