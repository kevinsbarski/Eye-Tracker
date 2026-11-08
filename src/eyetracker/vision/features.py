"""Turn 478 face landmarks into the handful of numbers the gaze model needs.

Landmarks are in frame pixels (x right, y down) with z in the same units, smaller z = closer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# MediaPipe face-mesh landmark indices.
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133  # the eye on the image's left
LEFT_EYE_INNER, LEFT_EYE_OUTER = 362, 263  # the eye on the image's right
RIGHT_LID_TOP, RIGHT_LID_BOTTOM = 159, 145
LEFT_LID_TOP, LEFT_LID_BOTTOM = 386, 374
RIGHT_IRIS, LEFT_IRIS = 468, 473
FOREHEAD, CHIN = 10, 152

GAZE_DIMS = 4  # yaw, pitch, iris_h, iris_v: the features allowed into nonlinear terms
FEATURE_DIMS = 8


@dataclass(frozen=True)
class GazeFeatures:
    yaw: float  # degrees, > 0 when the face turns toward image +x
    pitch: float  # degrees, > 0 when the face turns up
    roll: float  # degrees, slope of the outer-eye-corner line
    iris_h: float  # mean iris position along the eye, 0 = centred, > 0 toward image +x
    iris_v: float  # mean iris offset across the eye, in eye widths, > 0 = down
    face_x: float  # eye midpoint / frame width
    face_y: float  # eye midpoint / frame height
    face_scale: float  # outer-eye-corner distance / frame width
    eye_openness: float  # lid gap / eye width; small means a blink

    def vector(self) -> np.ndarray:
        return np.array(
            [
                self.yaw,
                self.pitch,
                self.iris_h,
                self.iris_v,
                self.roll,
                self.face_x,
                self.face_y,
                self.face_scale,
            ],
            dtype=float,
        )


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _head_angles(lm: np.ndarray) -> tuple[float, float, float]:
    x_axis = _unit(lm[LEFT_EYE_OUTER] - lm[RIGHT_EYE_OUTER])
    down = lm[CHIN] - lm[FOREHEAD]
    down = _unit(down - np.dot(down, x_axis) * x_axis)
    forward = np.cross(down, x_axis)  # points at the camera for a frontal face
    yaw = math.degrees(math.atan2(forward[0], -forward[2]))
    pitch = math.degrees(math.atan2(-forward[1], math.hypot(forward[0], forward[2])))
    roll = math.degrees(math.atan2(x_axis[1], x_axis[0]))
    return yaw, pitch, roll


def _iris_in_eye(lm: np.ndarray, left_corner: int, right_corner: int, iris: int):
    """Iris position along (fraction) and across (eye widths) the eye's corner line, in 2-D."""
    a, b, p = lm[left_corner, :2], lm[right_corner, :2], lm[iris, :2]
    axis = b - a
    width = float(np.linalg.norm(axis))
    u = axis / width
    normal = np.array([-u[1], u[0]])  # image "down" for a level eye
    rel = p - a
    return float(np.dot(rel, u) / width), float(np.dot(rel, normal) / width), width


def features_from_landmarks(lm: np.ndarray, frame_w: int, frame_h: int) -> GazeFeatures:
    yaw, pitch, roll = _head_angles(lm)

    r_along, r_across, r_width = _iris_in_eye(lm, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, RIGHT_IRIS)
    l_along, l_across, l_width = _iris_in_eye(lm, LEFT_EYE_INNER, LEFT_EYE_OUTER, LEFT_IRIS)

    r_gap = float(np.linalg.norm(lm[RIGHT_LID_TOP, :2] - lm[RIGHT_LID_BOTTOM, :2]))
    l_gap = float(np.linalg.norm(lm[LEFT_LID_TOP, :2] - lm[LEFT_LID_BOTTOM, :2]))

    outer_r, outer_l = lm[RIGHT_EYE_OUTER, :2], lm[LEFT_EYE_OUTER, :2]
    mid = (outer_r + outer_l) / 2

    return GazeFeatures(
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        iris_h=(r_along + l_along) / 2 - 0.5,
        iris_v=(r_across + l_across) / 2,
        face_x=float(mid[0]) / frame_w,
        face_y=float(mid[1]) / frame_h,
        face_scale=float(np.linalg.norm(outer_l - outer_r)) / frame_w,
        eye_openness=(r_gap / r_width + l_gap / l_width) / 2,
    )
