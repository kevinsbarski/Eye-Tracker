"""Personal gaze model: ridge regression from face features to a point on the desktop.

Each screen axis has its own small model built only from the features that physically drive it:

- x (left/right) from head yaw and horizontal iris position;
- y (up/down) from head pitch, vertical iris position and yaw (looking far to the side changes
  the measured pitch of a point at the same height).

Head position, distance and roll are deliberately left out. A 30-second calibration hardly varies
them, so a fit that is allowed to use them learns accidental weights. On the first real desk,
those weights moved the estimate by hundreds of points whenever the user leaned or looked at the
keyboard, which pulled the right-hand monitor's estimate back toward the laptop screen.

Degree (1 or 2) and regularisation are picked per axis by leave-one-point-out cross-validation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from eyetracker.geometry import Display, nearest_display
from eyetracker.vision.features import FEATURE_DIMS

log = logging.getLogger(__name__)

# indices into GazeFeatures.vector(): [yaw, pitch, iris_h, iris_v, roll, face_x, face_y, face_scale]
X_FEATURES = (0, 2)  # yaw, iris_h
Y_FEATURES = (1, 3, 0)  # pitch, iris_v, yaw
MODEL_VERSION = 2


class CalibrationFailed(Exception):
    """Not enough usable calibration data to fit a model."""


def expand(z: np.ndarray, degree: int) -> np.ndarray:
    """Degree 1: the columns as they are. Degree 2: plus every square and pairwise product."""
    if degree == 1:
        return z
    if degree != 2:
        raise ValueError(f"unsupported degree {degree}")
    k = z.shape[1]
    products = [z[:, i] * z[:, j] for i, j in combinations(range(k), 2)]
    return (
        np.hstack([z, z**2, *(p[:, None] for p in products)]) if products else np.hstack([z, z**2])
    )


class AxisModel:
    """Ridge regression for one screen coordinate from a chosen subset of features."""

    def __init__(self, indices, mean, std, degree, alpha, coef, intercept) -> None:
        self.indices = tuple(int(i) for i in indices)
        self.mean = np.asarray(mean, float)
        self.std = np.asarray(std, float)
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.coef = np.asarray(coef, float)
        self.intercept = float(intercept)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, indices, degree: int, alpha: float) -> AxisModel:
        Z = np.asarray(X, float)[:, list(indices)]
        mean = Z.mean(axis=0)
        std = Z.std(axis=0)
        std[std < 1e-12] = 1.0
        A = expand((Z - mean) / std, degree)
        a_mean = A.mean(axis=0)
        y = np.asarray(y, float)
        y_mean = float(y.mean())
        Ac = A - a_mean
        coef = np.linalg.solve(Ac.T @ Ac + alpha * np.eye(A.shape[1]), Ac.T @ (y - y_mean))
        return cls(indices, mean, std, degree, alpha, coef, y_mean - float(a_mean @ coef))

    def predict(self, X: np.ndarray) -> np.ndarray:
        Z = np.asarray(X, float)[:, list(self.indices)]
        return expand((Z - self.mean) / self.std, self.degree) @ self.coef + self.intercept

    def to_dict(self) -> dict:
        return {
            "indices": list(self.indices),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "degree": self.degree,
            "alpha": self.alpha,
            "coef": self.coef.tolist(),
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AxisModel:
        return cls(
            d["indices"], d["mean"], d["std"], d["degree"], d["alpha"], d["coef"], d["intercept"]
        )


class GazeModel:
    def __init__(self, x: AxisModel, y: AxisModel) -> None:
        self.x = x
        self.y = y

    @classmethod
    def fit(cls, X: np.ndarray, Y: np.ndarray, degree: int = 1, alpha: float = 1.0) -> GazeModel:
        Y = np.asarray(Y, float)
        return cls(
            AxisModel.fit(X, Y[:, 0], X_FEATURES, degree, alpha),
            AxisModel.fit(X, Y[:, 1], Y_FEATURES, degree, alpha),
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, float)
        single = x.ndim == 1
        rows = x.reshape(1, -1) if single else x
        out = np.column_stack([self.x.predict(rows), self.y.predict(rows)])
        return out[0] if single else out

    def to_dict(self) -> dict:
        return {"version": MODEL_VERSION, "x": self.x.to_dict(), "y": self.y.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> GazeModel:
        if not isinstance(d, dict) or d.get("version") != MODEL_VERSION:
            raise ValueError("calibration was saved by an older version; please calibrate again")
        model = cls(AxisModel.from_dict(d["x"]), AxisModel.from_dict(d["y"]))
        if max(model.x.indices + model.y.indices) >= FEATURE_DIMS:
            raise ValueError("model was saved for a different feature set")
        return model


@dataclass(frozen=True)
class CVReport:
    error_pt: float  # mean distance between held-out prediction and target, in points
    display_accuracy: float  # share of held-out samples landing nearest the right display
    n_points: int
    degree_x: int = 1
    alpha_x: float = 1.0
    degree_y: int = 1
    alpha_y: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


def _cross_validate_axis(X, y, groups, indices, degree, alpha) -> np.ndarray:
    """Held-out predictions for every sample, leaving one calibration point out at a time."""
    preds = np.empty_like(y)
    for g in np.unique(groups):
        test = groups == g
        model = AxisModel.fit(X[~test], y[~test], indices, degree, alpha)
        preds[test] = model.predict(X[test])
    return preds


def _best_axis(X, y, groups, indices, degrees, alphas):
    best = None
    for degree in degrees:
        for alpha in alphas:
            preds = _cross_validate_axis(X, y, groups, indices, degree, alpha)
            error = float(np.abs(preds - y).mean())
            if best is None or error < best[0]:
                best = (error, degree, alpha, preds)
    return best[1], best[2], best[3]


def fit_with_cv(
    X: np.ndarray,
    Y: np.ndarray,
    groups: Sequence[int],
    displays: Sequence[Display],
    degrees: Sequence[int] = (1, 2),
    alphas: Sequence[float] = (0.01, 0.1, 1, 10, 100),
) -> tuple[GazeModel, CVReport]:
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    groups_arr = np.asarray(groups)
    n_groups = len(np.unique(groups_arr)) if len(groups_arr) else 0
    if n_groups < 3:
        raise CalibrationFailed(f"need at least 3 calibration points, got {n_groups}")

    degree_x, alpha_x, px = _best_axis(X, Y[:, 0], groups_arr, X_FEATURES, degrees, alphas)
    degree_y, alpha_y, py = _best_axis(X, Y[:, 1], groups_arr, Y_FEATURES, degrees, alphas)
    preds = np.column_stack([px, py])
    error = float(np.linalg.norm(preds - Y, axis=1).mean())

    if displays:
        hits = [
            nearest_display(displays, *p).id == nearest_display(displays, *t).id
            for p, t in zip(preds, Y, strict=True)
        ]
        accuracy = float(np.mean(hits))
        _log_per_display(displays, preds, Y, hits)
    else:
        accuracy = 0.0
    model = GazeModel(
        AxisModel.fit(X, Y[:, 0], X_FEATURES, degree_x, alpha_x),
        AxisModel.fit(X, Y[:, 1], Y_FEATURES, degree_y, alpha_y),
    )
    report = CVReport(error, accuracy, n_groups, degree_x, alpha_x, degree_y, alpha_y)
    return model, report


def _log_per_display(displays, preds, targets, hits) -> None:
    """Which display calibrated worst; the first thing to look at when switching feels off."""
    truth = [nearest_display(displays, *t).id for t in targets]
    for d in sorted(displays, key=lambda d: d.x):
        rows = [i for i, tid in enumerate(truth) if tid == d.id]
        if not rows:
            continue
        errors = np.linalg.norm(preds[rows] - targets[rows], axis=1)
        dx = float(np.mean(preds[rows, 0] - targets[rows, 0]))
        log.info(
            "calibration display %s: correct-monitor rate %.0f%%, mean error %.0f pt, "
            "mean horizontal bias %+.0f pt (%d samples)",
            d.id,
            100 * float(np.mean([hits[i] for i in rows])),
            float(errors.mean()),
            dx,
            len(rows),
        )
