"""When does a gaze become a monitor switch? Pure logic: time in, decision out."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from eyetracker.config import Tuning
from eyetracker.geometry import Display, nearest_display


@dataclass(frozen=True)
class InputIdle:
    seconds_since_key: float
    seconds_since_mouse: float  # movement, clicks, drags and scrolls


class SwitchDecider:
    def __init__(self, tuning: Tuning = Tuning()) -> None:
        self.tuning = tuning
        self.reset()

    def reset(self) -> None:
        self._candidate: int | None = None
        self._since = 0.0
        self._last_gaze_t: float | None = None
        self._last_switch = float("-inf")
        self.last_reason = "reset"  # why the latest update did or did not switch (diagnostics)

    def update(
        self,
        now: float,
        gaze: tuple[float, float] | None,
        displays: Sequence[Display],
        current_id: int,
        idle: InputIdle,
    ) -> int | None:
        """Return the id of the display to switch to, or None."""
        t = self.tuning
        stale = self._last_gaze_t is not None and now - self._last_gaze_t > t.gap_reset_s
        if stale:
            self._candidate = None
        if gaze is None:
            self.last_reason = "no_gaze"
            return None
        self._last_gaze_t = now

        target, reason = self._target_for(gaze, displays, current_id)
        if target is not None:
            if idle.seconds_since_mouse < t.mouse_grace_s:
                reason = "mouse"
            elif idle.seconds_since_key < t.typing_grace_s:
                reason = "typing"
            elif now - self._last_switch < t.cooldown_s:
                reason = "cooldown"
        if reason is not None:
            self.last_reason = reason
            self._candidate = None
            return None
        if target != self._candidate:
            self._candidate = target
            self._since = now
        held = now - self._since
        if held >= t.dwell_s:
            self.last_dwell = held
            self._candidate = None
            self._last_switch = now
            self.last_reason = "switch"
            return target
        self.last_reason = f"dwell {held:.2f}/{t.dwell_s:.2f}"
        return None

    def _target_for(
        self, gaze: tuple[float, float], displays: Sequence[Display], current_id: int
    ) -> tuple[int | None, str | None]:
        """(target display id, None) or (None, reason it is not a target)."""
        current = next((d for d in displays if d.id == current_id), None)
        if current is None or not displays:
            return None, "unknown_display"
        nearest = nearest_display(displays, *gaze)
        to_nearest = nearest.distance_to(*gaze)
        if to_nearest > self.tuning.away_frac * nearest.diagonal:
            return None, "away"  # looking at the phone, the desk, out of the window
        if nearest.id == current.id:
            return None, "same_display"
        margin = self.tuning.hysteresis_frac * min(current.shorter_side, nearest.shorter_side)
        if current.distance_to(*gaze) - to_nearest < margin:
            return None, "hysteresis"
        return nearest.id, None
