# client/emotion.py
# -------------------------------------------------------
# Phase 3: Emotion Classification Wrapper
#
# Provides access to the ONNX FER+ emotion classifier
# and classroom engagement mappings.
# -------------------------------------------------------

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from client.utils.emotion_classifier import (
    EmotionClassifier,
    FER_CLASSES,
    ENGAGEMENT_MAPPING,
    MODEL_PATH
)

__all__ = [
    "EmotionClassifier",
    "FER_CLASSES",
    "ENGAGEMENT_MAPPING",
    "MODEL_PATH"
]
