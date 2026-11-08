"""A small see-through dot that follows the estimated gaze. For checking calibration by eye."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

SIZE = 24


class GazeDot(QWidget):
    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.Window
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowTransparentForInput
            | Qt.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.resize(SIZE, SIZE)
        self._enabled = False

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self.hide()

    def show_point(self, point: tuple[float, float] | None) -> None:
        if not self._enabled:
            return
        if point is None:
            self.hide()
            return
        self.move(int(point[0] - SIZE / 2), int(point[1] - SIZE / 2))
        if not self.isVisible():
            self.show()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(255, 255, 255, 200))
        painter.setBrush(QColor(255, 60, 60, 140))
        painter.drawEllipse(1, 1, SIZE - 2, SIZE - 2)
