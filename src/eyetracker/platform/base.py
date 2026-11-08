"""What the app needs from the operating system, independent of which one it is."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from eyetracker.engine.decision import InputIdle
from eyetracker.geometry import Display

__all__ = ["InputIdle", "Platform", "WindowInfo", "pick_window_on_display"]

MIN_WINDOW_SIDE = 50.0


@dataclass(frozen=True)
class WindowInfo:
    pid: int
    window_id: int
    x: float
    y: float
    width: float
    height: float
    layer: int = 0

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


class Platform(Protocol):
    def displays(self) -> list[Display]: ...

    def cursor_position(self) -> tuple[float, float]: ...

    def warp_cursor(self, x: float, y: float) -> None: ...

    def input_idle(self) -> InputIdle: ...

    def focus_frontmost_window_on(self, display: Display) -> bool: ...

    def accessibility_trusted(self, prompt: bool = False) -> bool: ...


def pick_window_on_display(
    windows: Sequence[WindowInfo], display: Display, own_pid: int
) -> WindowInfo | None:
    """The front-most ordinary window whose centre is on `display` (windows are front-to-back)."""
    for w in windows:
        if w.layer != 0 or w.pid == own_pid:
            continue
        if w.width < MIN_WINDOW_SIDE or w.height < MIN_WINDOW_SIDE:
            continue
        if display.contains(*w.center):
            return w
    return None
