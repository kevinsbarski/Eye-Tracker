"""Find the face and its 478 landmarks (irises included) in a camera frame.

YuNet finds the face when nothing is being tracked; after that the landmark network's own
output decides where to look in the next frame. Both networks run in OpenCV's DNN module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

INPUT_SIZE = 256
DETECTION_SCALE = 1.6  # crop side relative to the larger side of the YuNet box
TRACKING_SCALE = 1.5  # crop side relative to the larger side of the landmark bounding box
PRESENCE_THRESHOLD = 0.5


@dataclass(frozen=True)
class Roi:
    cx: float
    cy: float
    side: float


@dataclass
class FaceResult:
    landmarks: np.ndarray  # (478, 3) in frame pixels; z scaled like x
    presence: float


def roi_from_box(x: float, y: float, w: float, h: float, scale: float) -> Roi:
    return Roi(x + w / 2, y + h / 2, scale * max(w, h))


def crop_transform(roi: Roi, size: int = INPUT_SIZE) -> np.ndarray:
    """2x3 affine matrix mapping the ROI square in the frame onto the network's input square."""
    k = size / roi.side
    x0, y0 = roi.cx - roi.side / 2, roi.cy - roi.side / 2
    return np.array([[k, 0.0, -x0 * k], [0.0, k, -y0 * k]], dtype=np.float64)


def landmarks_to_frame(pts: np.ndarray, roi: Roi, size: int = INPUT_SIZE) -> np.ndarray:
    k = roi.side / size
    out = np.empty_like(pts, dtype=np.float64)
    out[:, 0] = pts[:, 0] * k + roi.cx - roi.side / 2
    out[:, 1] = pts[:, 1] * k + roi.cy - roi.side / 2
    out[:, 2] = pts[:, 2] * k
    return out


def _load_landmark_net(path: Path):
    # OpenCV 5's new graph engine exposes only one of the three outputs; the classic one has all.
    if hasattr(cv2.dnn, "ENGINE_CLASSIC"):
        return cv2.dnn.readNetFromTFLite(str(path), engine=cv2.dnn.ENGINE_CLASSIC)
    return cv2.dnn.readNetFromTFLite(str(path))


class FaceMesh:
    def __init__(self, landmark_path: Path, detector_path: Path) -> None:
        self._net = _load_landmark_net(landmark_path)
        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", (640, 480), 0.7, 0.3, 5000
        )
        self._detector_size: tuple[int, int] = (640, 480)
        self._roi: Roi | None = None

    def reset(self) -> None:
        self._roi = None

    def process(self, frame_bgr: np.ndarray) -> FaceResult | None:
        if self._roi is None:
            self._roi = self._detect(frame_bgr)
            if self._roi is None:
                return None

        roi = self._roi
        crop = cv2.warpAffine(
            frame_bgr,
            crop_transform(roi),
            (INPUT_SIZE, INPUT_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        blob = cv2.dnn.blobFromImage(crop, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True)
        self._net.setInput(blob)
        raw, logit = self._net.forward(["Identity", "Identity_1"])
        presence = 1.0 / (1.0 + math.exp(-float(np.ravel(logit)[0])))
        if presence < PRESENCE_THRESHOLD:
            self._roi = None
            return None

        landmarks = landmarks_to_frame(raw.reshape(478, 3), roi)
        xs, ys = landmarks[:, 0], landmarks[:, 1]
        self._roi = roi_from_box(
            xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min(), TRACKING_SCALE
        )
        return FaceResult(landmarks=landmarks, presence=presence)

    def _detect(self, frame_bgr: np.ndarray) -> Roi | None:
        h, w = frame_bgr.shape[:2]
        if (w, h) != self._detector_size:
            self._detector.setInputSize((w, h))
            self._detector_size = (w, h)
        _, faces = self._detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4]
        return roi_from_box(float(x), float(y), float(fw), float(fh), DETECTION_SCALE)
