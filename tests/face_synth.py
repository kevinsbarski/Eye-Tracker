"""A tiny synthetic face: just the landmark rows the feature code reads, posed on demand.

Camera coordinates: x right, y down, z away from the camera. The canonical face looks straight
at the camera. Rows not listed stay zero.
"""

from __future__ import annotations

import math

import numpy as np

EYE_WIDTH = 30.0

CANONICAL = {
    33: (-45, -35, 10),
    133: (-15, -35, 5),
    362: (15, -35, 5),
    263: (45, -35, 10),
    159: (-30, -42, 3),
    145: (-30, -28, 3),
    386: (30, -42, 3),
    374: (30, -28, 3),
    468: (-30, -35, 3),
    473: (30, -35, 3),
    1: (0, 5, -30),
    10: (0, -90, 5),
    152: (0, 85, 5),
    61: (-25, 50, 0),
    291: (25, 50, 0),
}


def _rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    # yaw: forward (0,0,-1) turns toward +x
    ry = np.array([[math.cos(y), 0, -math.sin(y)], [0, 1, 0], [math.sin(y), 0, math.cos(y)]])
    # pitch: forward turns toward -y (face looks up)
    rx = np.array([[1, 0, 0], [0, math.cos(p), math.sin(p)], [0, -math.sin(p), math.cos(p)]])
    # roll: the image-right eye (263) moves down
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return ry @ rx @ rz


def synth_landmarks(
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    iris_shift: tuple[float, float] = (0.0, 0.0),
    center: tuple[float, float] = (320.0, 240.0),
    scale: float = 1.0,
) -> np.ndarray:
    pts = np.zeros((478, 3))
    for idx, xyz in CANONICAL.items():
        pts[idx] = xyz
    shift = np.array([iris_shift[0] * EYE_WIDTH, iris_shift[1] * EYE_WIDTH, 0.0])
    pts[468] += shift
    pts[473] += shift
    used = list(CANONICAL)
    pts[used] = pts[used] @ _rotation(yaw, pitch, roll).T * scale
    pts[used, 0] += center[0]
    pts[used, 1] += center[1]
    return pts
