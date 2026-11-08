"""Display rectangles in global display points (origin top-left of the main display, y down)."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Display:
    id: int
    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)

    @property
    def shorter_side(self) -> float:
        return min(self.width, self.height)

    @property
    def diagonal(self) -> float:
        return math.hypot(self.width, self.height)

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.width and self.y <= py < self.y + self.height

    def distance_to(self, px: float, py: float) -> float:
        """0 inside the rectangle, otherwise the Euclidean distance to its nearest edge."""
        dx = max(self.x - px, 0.0, px - (self.x + self.width))
        dy = max(self.y - py, 0.0, py - (self.y + self.height))
        return math.hypot(dx, dy)


def display_at(displays: Iterable[Display], px: float, py: float) -> Display | None:
    for d in displays:
        if d.contains(px, py):
            return d
    return None


def nearest_display(displays: Iterable[Display], px: float, py: float) -> Display:
    return min(displays, key=lambda d: d.distance_to(px, py))


def layout_signature(displays: Iterable[Display]) -> str:
    """A stable text key for a desk layout: every display's rounded x,y,w,h, sorted."""
    rects = sorted((round(d.x), round(d.y), round(d.width), round(d.height)) for d in displays)
    return ";".join(f"{x},{y},{w},{h}" for x, y, w, h in rects)
