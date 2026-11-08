"""Glue between what the camera sees and what the desktop does.

Runs on the Qt main thread: observation -> gaze model -> smoothing -> switch decision -> cursor
warp and window focus.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from eyetracker.config import Tuning
from eyetracker.engine.decision import SwitchDecider
from eyetracker.gaze.filters import PointFilter
from eyetracker.geometry import Display, display_at, layout_signature, nearest_display

log = logging.getLogger(__name__)

DISPLAY_REFRESH_S = 2.0
STALE_OBSERVATION_S = 0.25  # older frames (a backlog after a stall) are dropped, not replayed


class Controller(QObject):
    gaze_updated = Signal(object)  # (x, y) or None
    features_observed = Signal(object, float)  # GazeFeatures, timestamp
    switched = Signal(int)  # target display id
    state_changed = Signal(str)  # tracking | paused | not_calibrated | single_display

    def __init__(
        self,
        platform,
        store,
        tuning: Tuning = Tuning(),
        clock: Callable[[], float] = time.monotonic,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._platform = platform
        self._store = store
        self._tuning = tuning
        self._clock = clock
        self._displays: list[Display] = []
        self._signature: str | None = None
        self._model = None
        self._paused = False
        self._calibrating = False
        self._memory: dict[int, tuple[float, float]] = {}
        self._filter = PointFilter()
        self._decider = SwitchDecider(tuning)
        self._last_refresh = float("-inf")
        self._state = ""
        self.refresh_displays()
        self._state = self._compute_state()

    # --- public state -------------------------------------------------------------------------

    @property
    def displays(self) -> list[Display]:
        return list(self._displays)

    @property
    def signature(self) -> str | None:
        return self._signature

    @property
    def model(self):
        return self._model

    @property
    def state(self) -> str:
        return self._state

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self._restart_tracking()

    def set_calibrating(self, calibrating: bool) -> None:
        self._calibrating = calibrating
        self._restart_tracking()

    def set_model(self, model) -> None:
        self._model = model
        self._restart_tracking()

    def refresh_displays(self) -> None:
        self._last_refresh = self._clock()
        try:
            displays = self._platform.displays()
        except Exception:
            log.exception("could not read the display layout")
            return
        signature = layout_signature(displays)
        self._displays = displays
        if signature == self._signature:
            return
        log.info("display layout is now %s", signature)
        self._signature = signature
        self._model = self._store.load(signature) if len(displays) >= 2 else None
        ids = {d.id for d in displays}
        self._memory = {k: v for k, v in self._memory.items() if k in ids}
        self._restart_tracking()

    # --- the per-frame path -------------------------------------------------------------------

    def on_observation(self, obs) -> None:
        if self._clock() - self._last_refresh >= DISPLAY_REFRESH_S:
            self.refresh_displays()
        if self._clock() - obs.t > STALE_OBSERVATION_S:
            return

        features = obs.features
        if features is not None:
            self.features_observed.emit(features, obs.t)

        if self._state != "tracking" or self._calibrating:
            return

        point = raw = None
        if features is not None and features.eye_openness >= self._tuning.blink_openness:
            x, y = self._model.predict(features.vector())
            raw = (float(x), float(y))
            point = self._filter(obs.t, raw)
        self.gaze_updated.emit(point)

        cursor = self._platform.cursor_position()
        current = display_at(self._displays, *cursor) or nearest_display(self._displays, *cursor)
        self._memory[current.id] = cursor  # "where you left it", however you left
        idle = self._platform.input_idle()
        target_id = self._decider.update(obs.t, point, self._displays, current.id, idle)
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "trace t=%.3f lag=%.0fms face=%s yaw=%s iris_h=%s raw=%s gaze=%s cur=%s "
                "reason=%s mouse_idle=%.1f key_idle=%.1f f=%s",
                obs.t,
                (self._clock() - obs.t) * 1000,
                "no" if features is None else ("blink" if raw is None else "yes"),
                _fmt(features.yaw if features else None),
                _fmt(features.iris_h if features else None, 3),
                _fmt_point(raw),
                _fmt_point(point),
                current.id,
                self._decider.last_reason,
                idle.seconds_since_mouse,
                idle.seconds_since_key,
                "-"
                if features is None
                else "[" + ",".join(f"{v:.4f}" for v in features.vector()) + "]",
            )
        if target_id is not None:
            target = next(d for d in self._displays if d.id == target_id)
            self._switch(current, target, cursor)

    def _switch(self, source: Display, target: Display, cursor: tuple[float, float]) -> None:
        # The layout is otherwise re-read every 2 s; a monitor unplugged in between must not
        # receive the cursor. Switches are rare, so one extra query each is cheap.
        try:
            live_signature = layout_signature(self._platform.displays())
        except Exception:
            log.exception("could not re-check the display layout")
            live_signature = self._signature
        if live_signature != self._signature:
            self.refresh_displays()
            return
        self._memory[source.id] = cursor
        destination = self._memory.get(target.id)
        if destination is None or not target.contains(*destination):
            destination = target.center
        started = time.perf_counter()
        self._platform.warp_cursor(*destination)
        focused = None
        try:
            focused = self._platform.focus_frontmost_window_on(target)
        except Exception:
            log.exception("could not focus a window on display %s", target.id)
        log.info(
            "switch %s -> %s after %.2f s of steady gaze; warp+focus took %.0f ms (focused=%s)",
            source.id,
            target.id,
            getattr(self._decider, "last_dwell", float("nan")),
            (time.perf_counter() - started) * 1000,
            focused,
        )
        self.switched.emit(target.id)

    # --- helpers ------------------------------------------------------------------------------

    def _compute_state(self) -> str:
        if self._paused:
            return "paused"
        if len(self._displays) < 2:
            return "single_display"
        if self._model is None:
            return "not_calibrated"
        return "tracking"

    def _restart_tracking(self) -> None:
        self._filter.reset()
        self._decider.reset()
        state = self._compute_state()
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)


def _fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _fmt_point(point: tuple[float, float] | None) -> str:
    return "-" if point is None else f"({point[0]:.0f},{point[1]:.0f})"
