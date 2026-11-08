"""Stand-ins for the camera-facing and OS-facing parts, so the controller can be tested alone."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from eyetracker.engine.decision import InputIdle
from eyetracker.vision.features import GazeFeatures


class FakePlatform:
    def __init__(self, displays, cursor=(100.0, 100.0)):
        self._displays = list(displays)
        self.cursor = cursor
        self.idle = InputIdle(99.0, 99.0)
        self.warps: list[tuple[float, float]] = []
        self.focused: list[int] = []
        self.focus_raises = False
        self.trusted = True

    def set_displays(self, displays):
        self._displays = list(displays)

    def displays(self):
        return list(self._displays)

    def cursor_position(self):
        return self.cursor

    def warp_cursor(self, x, y):
        self.warps.append((x, y))
        self.cursor = (x, y)

    def input_idle(self):
        return self.idle

    def focus_frontmost_window_on(self, display):
        if self.focus_raises:
            raise RuntimeError("accessibility said no")
        self.focused.append(display.id)
        return True

    def accessibility_trusted(self, prompt=False):
        return self.trusted


class FakeModel:
    """Predicts the screen point straight from the first two features (yaw, pitch)."""

    def predict(self, v):
        return np.asarray(v, float)[:2]


class FakeStore:
    def __init__(self, models=None, default=None):
        self.models = dict(models or {})
        self.default = default
        self.saved = []

    def load(self, signature):
        return self.models.get(signature, self.default)

    def save(self, signature, model, report, camera_name):
        self.saved.append((signature, model, report, camera_name))
        self.models[signature] = model


@dataclass
class Obs:
    t: float
    features: GazeFeatures | None


def feat(x: float, y: float, openness: float = 0.3) -> GazeFeatures:
    """Features that FakeModel maps to the screen point (x, y)."""
    return GazeFeatures(x, y, 0.0, 0.0, 0.0, 0.5, 0.4, 0.16, openness)
