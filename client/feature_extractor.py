from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from core.config import ALLOWED_FEATURE_KEYS

@dataclass
class StudentFeatures:
    student_id: str
    timestamp: float
    face_detected: bool
    ear: float = 0.0
    blink: bool = False
    perclos: float = 0.0
    drowsiness_state: str = "ALERT"
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    gaze_x: float = 0.5
    gaze_y: float = 0.5
    gaze_on_screen: bool = False
    emotion_label: str = "neutral"
    emotion_confidence: float = 0.0
    engagement_score: float = 0.0

def sanitize_features(raw_features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove any keys from the dictionary that are not in ALLOWED_FEATURE_KEYS.
    This ensures no raw image data, crops, or unexpected arrays are leaked.
    """
    sanitized = {}
    for key, value in raw_features.items():
        if key in ALLOWED_FEATURE_KEYS:
            sanitized[key] = value
    return sanitized

def extract_features(
    student_id: str,
    timestamp: float,
    face_detected: bool,
    ear: float = 0.0,
    blink: bool = False,
    perclos: float = 0.0,
    drowsiness_state: str = "ALERT",
    pose_result: Optional[Dict[str, Any]] = None,
    gaze_xy: tuple = (0.5, 0.5),
    gaze_on_screen: bool = False,
    emotion_result: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Builds a structured feature set and sanitizes it according to privacy rules.
    """
    pitch = pose_result['pitch'] if pose_result and pose_result.get('success') else 0.0
    yaw = pose_result['yaw'] if pose_result and pose_result.get('success') else 0.0
    roll = pose_result['roll'] if pose_result and pose_result.get('success') else 0.0
    
    emo_label = emotion_result['raw_emotion'] if emotion_result else "neutral"
    emo_conf = emotion_result['confidence'] if emotion_result else 0.0
    
    features = StudentFeatures(
        student_id=student_id,
        timestamp=timestamp,
        face_detected=face_detected,
        ear=ear,
        blink=blink,
        perclos=perclos,
        drowsiness_state=drowsiness_state,
        pitch=pitch,
        yaw=yaw,
        roll=roll,
        gaze_x=gaze_xy[0],
        gaze_y=gaze_xy[1],
        gaze_on_screen=gaze_on_screen,
        emotion_label=emo_label,
        emotion_confidence=emo_conf
    )
    
    # Convert to dict
    raw_dict = asdict(features)
    
    # Return sanitized
    return sanitize_features(raw_dict)
