# client/utils/emotion_classifier.py
# -------------------------------------------------------
# Emotion Classification using OpenCV DNN and ONNX FER+ Model.
# -------------------------------------------------------

import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
try:
    from core.config import FER_CLASSES, FER_ENGAGEMENT_MAPPING
    ENGAGEMENT_MAPPING = FER_ENGAGEMENT_MAPPING
except ImportError:
    FER_CLASSES = [
        "neutral", "happiness", "surprise", "sadness",
        "anger", "disgust", "fear", "contempt"
    ]
    ENGAGEMENT_MAPPING = {
        "neutral":   ("Engaged",     0.65),
        "happiness": ("Engaged",     0.80),
        "surprise":  ("Intrigued",   0.70),
        "fear":      ("Confused",    0.40),
        "sadness":   ("Bored",       0.35),
        "disgust":   ("Distracted",  0.25),
        "anger":     ("Frustrated",  0.30),
        "contempt":  ("Distracted",  0.25),
    }

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "models", "emotion_ferplus.onnx")


class EmotionClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.net = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[Emotion] Warning: Model file not found at {self.model_path}")
            return
        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print(f"[Emotion] Loaded model from {self.model_path}")
        except Exception as e:
            print(f"[Emotion] Error loading ONNX model: {e}")
            self.net = None

    def extract_face_roi(self, frame: np.ndarray, landmarks, img_w: int, img_h: int, margin: float = 0.15):
        xs = [lm.x * img_w for lm in landmarks.landmark]
        ys = [lm.y * img_h for lm in landmarks.landmark]

        min_x, max_x = int(min(xs)), int(max(xs))
        min_y, max_y = int(min(ys)), int(max(ys))

        w_box = max_x - min_x
        h_box = max_y - min_y

        pad_x = int(w_box * margin)
        pad_y = int(h_box * margin)

        x1 = max(0, min_x - pad_x)
        y1 = max(0, min_y - pad_y)
        x2 = min(img_w, max_x + pad_x)
        y2 = min(img_h, max_y + pad_y)

        if x2 - x1 < 10 or y2 - y1 < 10:
            return None, (0, 0, 0, 0)

        face_crop = frame[y1:y2, x1:x2]
        return face_crop, (x1, y1, x2 - x1, y2 - y1)

    def predict(self, face_crop: np.ndarray) -> dict:
        if self.net is None or face_crop is None or face_crop.size == 0:
            return {
                "raw_emotion": "neutral",
                "confidence": 0.0,
                "engagement_state": "Neutral",
                "valence_weight": 0.5,
                "probs": {}
            }

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)

        blob = cv2.dnn.blobFromImage(
            resized, scalefactor=1.0, size=(64, 64),
            mean=(0, 0, 0), swapRB=False, crop=False
        )

        self.net.setInput(blob)
        preds = self.net.forward()[0]

        exp_preds = np.exp(preds - np.max(preds))
        probs = exp_preds / np.sum(exp_preds)

        top_idx = int(np.argmax(probs))
        raw_label = FER_CLASSES[top_idx]
        conf = float(probs[top_idx])

        eng_state, valence_weight = ENGAGEMENT_MAPPING.get(raw_label, ("Neutral", 0.5))

        return {
            "raw_emotion": raw_label,
            "confidence": conf,
            "engagement_state": eng_state,
            "valence_weight": valence_weight,
            "probs": {FER_CLASSES[i]: float(probs[i]) for i in range(len(FER_CLASSES))}
        }