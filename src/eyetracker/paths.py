"""Where the app keeps its model files and its per-user data."""

from __future__ import annotations

import os
from pathlib import Path

import platformdirs

APP_NAME = "EyeTracker"
LANDMARK_MODEL = "face_landmarks_detector.tflite"
DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"


def models_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


def config_dir() -> Path:
    """Per-user data folder; `EYETRACKER_CONFIG_DIR` overrides it. Created on access."""
    override = os.environ.get("EYETRACKER_CONFIG_DIR")
    path = (
        Path(override) if override else Path(platformdirs.user_data_dir(APP_NAME, appauthor=False))
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def missing_models() -> list[str]:
    folder = models_dir()
    return [name for name in (LANDMARK_MODEL, DETECTOR_MODEL) if not (folder / name).is_file()]
