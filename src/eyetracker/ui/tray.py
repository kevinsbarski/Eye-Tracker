"""The eye icon in the menu bar and its menu."""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu, QSystemTrayIcon

from eyetracker.ui.icons import eye_icon

log = logging.getLogger(__name__)

STATUS_TEXT = {
    "tracking": "Tracking",
    "paused": "Paused",
    "not_calibrated": "Not calibrated",
    "single_display": "Only one display",
    "camera_unavailable": "Camera unavailable",
}


class Tray(QObject):
    pause_toggled = Signal(bool)
    calibrate_requested = Signal()
    camera_selected = Signal(str)
    preview_toggled = Signal(bool)
    gaze_dot_toggled = Signal(bool)
    accessibility_requested = Signal()
    camera_settings_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._paused = False
        self.menu = QMenu()

        self.status_action = self.menu.addAction("Starting…")
        self.status_action.setEnabled(False)
        self.menu.addSeparator()

        self.pause_action = self.menu.addAction("Pause tracking")
        self.pause_action.triggered.connect(self._on_pause)
        self.calibrate_action = self.menu.addAction("Calibrate…")
        self.calibrate_action.triggered.connect(self.calibrate_requested)

        self.camera_menu = self.menu.addMenu("Camera")
        self._camera_group = QActionGroup(self)
        self._camera_group.setExclusive(True)

        debug = self.menu.addMenu("Debug")
        self.preview_action = debug.addAction("Show camera preview")
        self.preview_action.setCheckable(True)
        self.preview_action.toggled.connect(self.preview_toggled)
        self.gaze_dot_action = debug.addAction("Show gaze dot")
        self.gaze_dot_action.setCheckable(True)
        self.gaze_dot_action.toggled.connect(self.gaze_dot_toggled)

        self.menu.addSeparator()
        self.accessibility_action = self.menu.addAction("Grant Accessibility access…")
        self.accessibility_action.setVisible(False)
        self.accessibility_action.triggered.connect(self.accessibility_requested)
        self.camera_settings_action = self.menu.addAction("Open Camera settings…")
        self.camera_settings_action.setVisible(False)
        self.camera_settings_action.triggered.connect(self.camera_settings_requested)
        self.quit_action = self.menu.addAction("Quit Eye Tracker")
        self.quit_action.triggered.connect(self.quit_requested)

        self.icon = QSystemTrayIcon(eye_icon(False))
        self.icon.setToolTip("Eye Tracker")
        self.icon.setContextMenu(self.menu)

    def show(self) -> None:
        self.icon.show()

    def hide(self) -> None:
        self.icon.hide()

    def set_state(self, state: str) -> None:
        self.status_action.setText(STATUS_TEXT.get(state, state))
        self.icon.setIcon(eye_icon(state == "tracking"))

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        self.pause_action.setText("Resume tracking" if paused else "Pause tracking")

    def set_cameras(self, names: list[str], current: str | None) -> None:
        for action in self.camera_menu.actions():
            self._camera_group.removeAction(action)
        self.camera_menu.clear()
        for name in names:
            action = QAction(name, self.camera_menu)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.triggered.connect(lambda _checked=False, n=name: self.camera_selected.emit(n))
            self._camera_group.addAction(action)
            self.camera_menu.addAction(action)

    def set_accessibility_missing(self, missing: bool) -> None:
        self.accessibility_action.setVisible(missing)

    def set_camera_permission_missing(self, missing: bool) -> None:
        self.camera_settings_action.setVisible(missing)

    def set_preview_checked(self, on: bool) -> None:
        self.preview_action.blockSignals(True)
        self.preview_action.setChecked(on)
        self.preview_action.blockSignals(False)

    def notify(self, title: str, text: str) -> None:
        log.info("%s: %s", title, text)
        self.icon.showMessage(title, text)

    def _on_pause(self) -> None:
        self.set_paused(not self._paused)
        self.pause_toggled.emit(self._paused)
