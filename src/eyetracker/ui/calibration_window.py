"""Calibration screens: one plain window per display, a dot to look at, a grade at the end."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence

from PySide6.QtCore import QObject, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from eyetracker.config import Tuning
from eyetracker.gaze.calibration import CalibrationSession, calibration_points, fit_session
from eyetracker.gaze.model import CalibrationFailed
from eyetracker.geometry import Display

log = logging.getLogger(__name__)

INTRO = "Look at each dot as it appears. Turn your head naturally.\nSpace to start, Esc to cancel."
WAITING = "Look at the dot"
RETRY_HINT = "Check the camera preview (Debug menu) and try again."
RESULT_SHOW_MS = 2500
DOT_START_RADIUS = 14.0
DOT_END_RADIUS = 6.0
TICK_MS = 30


def result_text(display_accuracy: float, error_pt: float, skipped: int = 0) -> str:
    text = (
        f"Done. Picks the right monitor {display_accuracy:.0%} of the time "
        f"(average error {error_pt:.0f} pt)."
    )
    if skipped:
        text += f"\n{skipped} dot was skipped" if skipped == 1 else f"\n{skipped} dots were skipped"
        text += " because your face wasn't visible."
    if display_accuracy < 0.9:
        text += "\nThat's low. Try calibrating again from the eye icon in the menu bar."
    return text


class _DotWindow(QWidget):
    key_pressed = Signal(int)

    def __init__(self, display: Display) -> None:
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.display = display
        self.message = ""
        self.dimmed = False
        self.dot: tuple[float, float] | None = None
        self.dot_radius = DOT_END_RADIUS
        self.setGeometry(
            QRect(int(display.x), int(display.y), int(display.width), int(display.height))
        )
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.BlankCursor)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(17, 17, 20))
        if self.message:
            painter.setPen(QColor(140, 140, 150) if self.dimmed else QColor(235, 235, 240))
            font = QFont()
            font.setPointSize(22)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self.message)
        if self.dot is not None:
            x, y = self.dot
            r = self.dot_radius
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(255, 255, 255))
            painter.drawEllipse(int(x - r), int(y - r), int(2 * r), int(2 * r))
            painter.setBrush(QColor(230, 60, 60))
            painter.drawEllipse(int(x - 2), int(y - 2), 4, 4)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt naming
        self.key_pressed.emit(event.key())


class CalibrationWindow(QObject):
    finished = Signal(object, object)  # GazeModel, CVReport
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        displays: Sequence[Display],
        clock: Callable[[], float] = time.monotonic,
        tuning: Tuning = Tuning(),
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.displays = list(displays)
        self.clock = clock
        self.tuning = tuning
        self.session: CalibrationSession | None = None
        self.windows: list[_DotWindow] = []
        self._state = "idle"
        self._timer = QTimer(self)
        self._timer.setInterval(TICK_MS)
        self._timer.timeout.connect(self.tick)

    def begin(self) -> None:
        self.windows = [_DotWindow(d) for d in self.displays]
        for w in self.windows:
            w.key_pressed.connect(self._on_key)
            w.message = INTRO
            w.show()
            w.raise_()
        if self.windows:
            self.windows[0].activateWindow()
            self.windows[0].setFocus()
        self._state = "intro"

    def start(self) -> None:
        if self._state not in ("intro", "idle"):
            return
        points = calibration_points(self.displays, self.tuning.inset)
        self.session = CalibrationSession(points, self.tuning)
        self.session.start(self.clock())
        self._state = "running"
        self._timer.start()
        self._render()

    def cancel(self) -> None:
        if self._state in ("cancelled", "closed"):
            return
        self._timer.stop()
        self._close_windows()
        self._state = "cancelled"
        self.cancelled.emit()

    def on_features(self, features, t: float) -> None:
        if self._state == "running" and self.session is not None:
            self.session.add_sample(t, features)

    def tick(self) -> None:
        if self._state != "running" or self.session is None:
            return
        self.session.tick(self.clock())
        if self.session.done:
            self._finish()
        else:
            self._render()

    # --- internals ----------------------------------------------------------------------------

    def _on_key(self, key: int) -> None:
        if key == Qt.Key_Escape:
            self.cancel()
        elif key == Qt.Key_Space and self._state == "intro":
            self.start()

    def _render(self) -> None:
        point = self.session.current if self.session else None
        now = self.clock()
        for w in self.windows:
            if point is not None and point.display_id == w.display.id:
                w.message = ""
                w.dot = (point.x - w.display.x, point.y - w.display.y)
                progress = self.session.settle_progress(now)
                w.dot_radius = DOT_START_RADIUS + (DOT_END_RADIUS - DOT_START_RADIUS) * progress
            else:
                w.dot = None
                w.message = WAITING
                w.dimmed = True
            w.update()

    def _finish(self) -> None:
        self._timer.stop()
        self._state = "done"
        try:
            model, report = fit_session(self.session, self.displays)
        except CalibrationFailed as exc:
            log.warning("calibration failed: %s", exc)
            message = f"{exc}\n{RETRY_HINT}"
            self._show_final(message)
            self.failed.emit(message.replace("\n", " "))
            return
        log.info("calibration finished: %s (skipped dots: %s)", report, self.session.skipped)
        skipped = len(self.session.skipped)
        self._show_final(result_text(report.display_accuracy, report.error_pt, skipped))
        self.finished.emit(model, report)

    def _show_final(self, text: str) -> None:
        for w in self.windows:
            w.dot = None
            w.dimmed = False
            w.message = text
            w.update()
        QTimer.singleShot(RESULT_SHOW_MS, self._close_windows)

    def _close_windows(self) -> None:
        for w in self.windows:
            w.hide()
            w.deleteLater()
        self.windows = []
        if self._state == "done":
            self._state = "closed"
