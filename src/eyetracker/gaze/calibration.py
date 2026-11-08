"""Calibration: which dots to show, and which camera samples count for each dot.

The session is pure logic driven by timestamps, so the window that draws the dots stays thin.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from eyetracker.config import Tuning
from eyetracker.gaze.model import CalibrationFailed, CVReport, GazeModel, fit_with_cv
from eyetracker.geometry import Display
from eyetracker.vision.features import FEATURE_DIMS, GazeFeatures

# centre first, then the corners clockwise from top-left
_PATTERN = ((0.5, 0.5), (0, 0), (1, 0), (1, 1), (0, 1))


@dataclass(frozen=True)
class CalPoint:
    index: int
    display_id: int
    x: float
    y: float


def calibration_points(displays: Sequence[Display], inset: float = 0.15) -> list[CalPoint]:
    points: list[CalPoint] = []
    for d in sorted(displays, key=lambda d: (d.x, d.y)):
        for fx, fy in _PATTERN:
            rx = 0.5 if fx == 0.5 else inset + fx * (1 - 2 * inset)
            ry = 0.5 if fy == 0.5 else inset + fy * (1 - 2 * inset)
            points.append(CalPoint(len(points), d.id, d.x + rx * d.width, d.y + ry * d.height))
    return points


class CalibrationSession:
    def __init__(self, points: Sequence[CalPoint], tuning: Tuning = Tuning()) -> None:
        self.points = list(points)
        self.tuning = tuning
        self.skipped: list[int] = []
        self._idx = 0
        self._point_start = 0.0
        self._retried = False
        self._pending: list[np.ndarray] = []
        self._X: list[np.ndarray] = []
        self._Y: list[tuple[float, float]] = []
        self._groups: list[int] = []

    @property
    def done(self) -> bool:
        return self._idx >= len(self.points)

    @property
    def current(self) -> CalPoint | None:
        return None if self.done else self.points[self._idx]

    def start(self, now: float) -> None:
        self._idx = 0
        self._begin_point(now)

    def in_settle(self, now: float) -> bool:
        return now - self._point_start < self.tuning.settle_s

    def settle_progress(self, now: float) -> float:
        """0 when the current dot appears, 1 once samples start counting."""
        return min(max((now - self._point_start) / self.tuning.settle_s, 0.0), 1.0)

    def _window_end(self) -> float:
        return self._point_start + self.tuning.settle_s + self.tuning.collect_s

    def add_sample(self, now: float, f: GazeFeatures) -> None:
        if self.done or self.in_settle(now) or now > self._window_end():
            return
        if f.eye_openness < self.tuning.blink_openness:
            return
        self._pending.append(f.vector())

    def tick(self, now: float) -> None:
        if self.done or now < self._window_end():
            return
        point = self.points[self._idx]
        if len(self._pending) >= self.tuning.min_samples:
            self._X.extend(self._pending)
            self._Y.extend([(point.x, point.y)] * len(self._pending))
            self._groups.extend([point.index] * len(self._pending))
        elif not self._retried:
            self._retried = True
            self._pending = []
            self._point_start = now
            return
        else:
            self.skipped.append(point.index)
        self._idx += 1
        self._begin_point(now)

    def _begin_point(self, now: float) -> None:
        self._point_start = now
        self._retried = False
        self._pending = []

    def dataset(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        X = np.array(self._X, float).reshape(-1, FEATURE_DIMS)
        Y = np.array(self._Y, float).reshape(-1, 2)
        return X, Y, list(self._groups)


NO_FACE_MESSAGE = "Couldn't see your face clearly enough."


def describe_display(display: Display, displays: Sequence[Display]) -> str:
    """A name a person recognises: "the left screen", "the middle screen", …"""
    ordered = sorted(displays, key=lambda d: (d.x, d.y))
    position = next(i for i, d in enumerate(ordered) if d.id == display.id)
    names = {
        1: ["the screen"],
        2: ["the left screen", "the right screen"],
        3: ["the left screen", "the middle screen", "the right screen"],
    }.get(len(ordered))
    if names:
        return names[position]
    return f"screen {position + 1} of {len(ordered)} from the left"


def fit_session(
    session: CalibrationSession, displays: Sequence[Display]
) -> tuple[GazeModel, CVReport]:
    """Fit the model, refusing to grade a display the camera never saw the user look at."""
    X, Y, groups = session.dataset()
    if not groups:
        raise CalibrationFailed(NO_FACE_MESSAGE)
    seen = set(groups)
    for display in displays:
        usable = sum(1 for p in session.points if p.display_id == display.id and p.index in seen)
        if usable < session.tuning.min_targets_per_display:
            raise CalibrationFailed(
                f"Couldn't see your face while you looked at {describe_display(display, displays)}."
            )
    return fit_with_cv(X, Y, groups, displays)


def save_samples(session: CalibrationSession, displays: Sequence[Display], path) -> None:
    """Debug aid: the raw calibration numbers (features, dot positions), never any images."""
    import json
    from pathlib import Path

    X, Y, groups = session.dataset()
    data = {
        "displays": [[d.id, d.x, d.y, d.width, d.height] for d in displays],
        "features": X.round(5).tolist(),
        "targets": Y.round(1).tolist(),
        "groups": groups,
        "skipped": session.skipped,
    }
    Path(path).write_text(json.dumps(data))
