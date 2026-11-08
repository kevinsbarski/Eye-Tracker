"""Start the menu-bar app and wire the pieces together."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from collections.abc import Callable

from PySide6.QtCore import QObject, QTimer
from PySide6.QtWidgets import QApplication

from eyetracker import paths
from eyetracker.config import Settings, Tuning
from eyetracker.engine.controller import Controller
from eyetracker.gaze.calibration import save_samples
from eyetracker.ui.calibration_window import CalibrationWindow, result_text
from eyetracker.ui.gaze_dot import GazeDot
from eyetracker.ui.preview import PreviewWindow
from eyetracker.ui.tray import Tray
from eyetracker.vision.camera import CameraInfo, choose_camera
from eyetracker.vision.worker import VisionLoop, VisionWorker

log = logging.getLogger("eyetracker")

POLL_MS = 5000  # permissions and cameras are re-checked this often


class EyeTrackerApp(QObject):
    """Owns every long-lived piece. Everything OS- or camera-specific is passed in."""

    def __init__(
        self,
        qapp: QApplication,
        platform,
        settings: Settings,
        store,
        face_mesh,
        list_cameras: Callable[[], list[CameraInfo]],
        camera_factory: Callable[[CameraInfo], Callable],
        camera_permission: Callable[[], str],
        request_camera_access: Callable[[], None],
        tuning: Tuning = Tuning(),
        debug: bool = False,
    ) -> None:
        super().__init__()
        self.debug = debug
        self.qapp = qapp
        self.platform = platform
        self.settings = settings
        self.store = store
        self.tuning = tuning
        self.list_cameras = list_cameras
        self.camera_factory = camera_factory
        self.camera_permission = camera_permission
        self.request_camera_access = request_camera_access
        self.cameras = list_cameras()
        self.camera = choose_camera(self.cameras, settings.camera_name)
        self.camera_status = "starting"
        self.calibration: CalibrationWindow | None = None
        self._paused = False
        self._waiting_for_permission = False
        self._asked_for_access = False

        self.controller = Controller(platform, store, tuning)
        self.tray = Tray()
        self.preview = PreviewWindow()
        self.gaze_dot = GazeDot()
        loop = VisionLoop(self._factory_for(self.camera), face_mesh)
        self.worker = VisionWorker(loop, fps=tuning.fps)

        self.worker.observed.connect(self.controller.on_observation)
        self.worker.observed.connect(self.preview.show_observation)
        self.worker.status_changed.connect(self._on_camera_status)
        self.controller.state_changed.connect(lambda _state: self._refresh_status())
        self.controller.gaze_updated.connect(self.gaze_dot.show_point)

        self.tray.pause_toggled.connect(self._set_paused)
        self.tray.calibrate_requested.connect(self.calibrate)
        self.tray.camera_selected.connect(self._select_camera)
        self.tray.preview_toggled.connect(self._toggle_preview)
        self.tray.gaze_dot_toggled.connect(self.gaze_dot.set_enabled)
        self.tray.accessibility_requested.connect(self._ask_for_accessibility)
        self.tray.camera_settings_requested.connect(self._open_camera_settings)
        self.tray.camera_menu.aboutToShow.connect(self._refresh_cameras)
        self.tray.quit_requested.connect(qapp.quit)
        self.preview.closed.connect(self._on_preview_closed)
        qapp.aboutToQuit.connect(self.shutdown)

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(POLL_MS)
        self._poll_timer.timeout.connect(self.poll)

    # --- lifecycle ----------------------------------------------------------------------------

    def start(self) -> None:
        self.tray.set_cameras(
            [c.name for c in self.cameras], self.camera.name if self.camera else None
        )
        self.tray.show()
        self._start_camera_if_allowed()
        self._refresh_status()
        self._check_accessibility()
        self._poll_timer.start()
        if self.controller.state == "not_calibrated":
            self.tray.notify("Eye Tracker", "Choose Calibrate… from the eye icon to set up.")

    def poll(self) -> None:
        """Every few seconds: pick up permissions granted in System Settings, or a new camera."""
        self._check_accessibility()
        if self.camera is None:
            self._refresh_cameras()
        if self._waiting_for_permission or self.camera_status == "camera_unavailable":
            self._start_camera_if_allowed()

    def shutdown(self) -> None:
        self._poll_timer.stop()
        if self.worker.isRunning():
            self.worker.stop()
        self.gaze_dot.hide()
        self.preview.hide()
        self.tray.hide()

    # --- calibration --------------------------------------------------------------------------

    def calibrate(self) -> None:
        self.controller.refresh_displays()
        if len(self.controller.displays) < 2:
            self.tray.notify("Eye Tracker", "Only one display — nothing to switch.")
            return
        if self._paused or self.camera_status == "camera_unavailable":
            self.tray.notify("Eye Tracker", "The camera is off. Resume tracking first.")
            return
        if self.calibration is not None:
            self.calibration.cancel()
        self.controller.set_calibrating(True)
        activate = getattr(self.platform, "activate_self", None)
        if activate:
            activate()
        self.calibration = CalibrationWindow(self.controller.displays, tuning=self.tuning)
        self.controller.features_observed.connect(self.calibration.on_features)
        self.calibration.finished.connect(self._on_calibrated)
        self.calibration.failed.connect(self._on_calibration_failed)
        self.calibration.cancelled.connect(self._end_calibration)
        self.calibration.begin()

    def _on_calibrated(self, model, report) -> None:
        camera_name = self.camera.name if self.camera else None
        self.store.save(self.controller.signature, model, report, camera_name)
        if self.debug and self.calibration is not None:
            samples = paths.config_dir() / "calibration_samples.json"
            save_samples(self.calibration.session, self.controller.displays, samples)
            log.info("calibration samples saved to %s", samples)
        self.controller.set_model(model)
        self._end_calibration()
        skipped = len(self.calibration.session.skipped) if self.calibration else 0
        text = result_text(report.display_accuracy, report.error_pt, skipped)
        self.tray.notify("Calibration saved", text)

    def _on_calibration_failed(self, message: str) -> None:
        self._end_calibration()
        self.tray.notify("Calibration failed", message)

    def _end_calibration(self) -> None:
        if self.calibration is not None:
            try:
                self.controller.features_observed.disconnect(self.calibration.on_features)
            except (RuntimeError, TypeError):
                pass
        self.controller.set_calibrating(False)

    # --- tray actions -------------------------------------------------------------------------

    def _set_paused(self, paused: bool) -> None:
        self._paused = paused
        self.controller.set_paused(paused)
        if paused:
            if self.worker.isRunning():
                self.worker.stop()  # releases the camera; the light goes out
        else:
            self._start_camera_if_allowed()
        self.tray.set_paused(paused)
        self._refresh_status()

    def _select_camera(self, name: str) -> None:
        camera = next((c for c in self.cameras if c.name == name), None)
        if camera is None:
            return
        self.camera = camera
        self.settings.camera_name = name
        self.settings.save()
        self.worker.set_camera_factory(self._factory_for(camera))
        self.tray.set_cameras([c.name for c in self.cameras], name)

    def _refresh_cameras(self) -> None:
        self.cameras = self.list_cameras()
        if self.camera is None:
            self.camera = choose_camera(self.cameras, self.settings.camera_name)
            if self.camera is not None:
                self.worker.set_camera_factory(self._factory_for(self.camera))
        self.tray.set_cameras(
            [c.name for c in self.cameras], self.camera.name if self.camera else None
        )

    def _start_camera_if_allowed(self) -> None:
        if self._paused or self.worker.isRunning():
            return
        if self.camera is None:
            self.camera_status = "camera_unavailable"
            self._refresh_status()
            return
        permission = self.camera_permission()
        if permission == "authorized":
            self._waiting_for_permission = False
            self.tray.set_camera_permission_missing(False)
            self.camera_status = "starting"
            self.worker.start()
            self._refresh_status()
            return
        self.camera_status = "camera_unavailable"
        if permission == "not_determined":
            if not self._asked_for_access:
                self._asked_for_access = True
                self.request_camera_access()  # macOS shows its prompt; poll() picks up the answer
        else:
            self.tray.set_camera_permission_missing(True)
            if not self._waiting_for_permission:
                self.tray.notify(
                    "Camera access needed",
                    "Allow camera access for your terminal app in System Settings → Privacy & "
                    "Security → Camera. The camera starts as soon as you do.",
                )
        self._waiting_for_permission = True
        self._refresh_status()

    def _open_camera_settings(self) -> None:
        opener = getattr(self.platform, "open_camera_settings", None)
        if opener:
            opener()

    def _toggle_preview(self, on: bool) -> None:
        self.worker.preview = on
        if on:
            self.preview.show()
            self.preview.raise_()
            activate = getattr(self.platform, "activate_self", None)
            if activate:
                activate()
        else:
            self.preview.hide()

    def _on_preview_closed(self) -> None:
        self.worker.preview = False
        self.tray.set_preview_checked(False)

    def _ask_for_accessibility(self) -> None:
        if not self.platform.accessibility_trusted(prompt=True):
            opener = getattr(self.platform, "open_accessibility_settings", None)
            if opener:
                opener()

    def _check_accessibility(self) -> None:
        self.tray.set_accessibility_missing(not self.platform.accessibility_trusted())

    # --- status -------------------------------------------------------------------------------

    def _on_camera_status(self, status: str) -> None:
        self.camera_status = status
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self.camera_status == "camera_unavailable" and not self._paused:
            self.tray.set_state("camera_unavailable")
        else:
            self.tray.set_state(self.controller.state)

    def _factory_for(self, camera: CameraInfo | None) -> Callable:
        if camera is None:
            return _NoCamera
        return self.camera_factory(camera)


class _NoCamera:
    def open(self) -> bool:
        return False

    def read(self):
        return None

    def release(self) -> None:
        pass


def install_quit_on_signals(qapp) -> QTimer:
    """Make Ctrl+C in the terminal (and SIGTERM) quit cleanly, releasing the camera.

    Python only runs signal handlers between bytecodes, and Qt's event loop is C++, so a cheap
    timer wakes the interpreter a few times a second.
    """
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_args: qapp.quit())
    timer = QTimer()
    timer.setInterval(250)
    timer.timeout.connect(lambda: None)
    timer.start()
    return timer


def _configure_logging(debug: bool) -> None:
    """INFO to the terminal; with --debug, a per-frame trace also goes to debug.log.

    The trace holds numbers only (gaze points, head angles, timings), never images.
    """
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    terminal = logging.StreamHandler()
    terminal.setLevel(logging.INFO)
    terminal.setFormatter(fmt)
    root.addHandler(terminal)
    if debug:
        trace_file = paths.config_dir() / "debug.log"
        to_file = logging.FileHandler(trace_file, mode="w", encoding="utf-8")
        to_file.setLevel(logging.DEBUG)
        to_file.setFormatter(fmt)
        root.addHandler(to_file)
        logging.getLogger("eyetracker").info("detailed trace: %s", trace_file)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="eyetracker", description=__doc__)
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.debug)
    missing = paths.missing_models()
    if missing:
        print(
            "Face models are missing (" + ", ".join(missing) + ").\n"
            "Run once: uv run python scripts/fetch_models.py",
            file=sys.stderr,
        )
        return 1
    if sys.platform != "darwin":
        print("This version runs on macOS only; Windows and Linux are planned.", file=sys.stderr)
        return 1

    import cv2

    from eyetracker.gaze.store import CalibrationStore
    from eyetracker.platform.macos import MacPlatform
    from eyetracker.vision.camera import (
        DeviceCamera,
        camera_permission,
        list_cameras,
        request_camera_access,
    )
    from eyetracker.vision.face_mesh import FaceMesh

    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

    qapp = QApplication(sys.argv[:1])
    qapp.setApplicationName("Eye Tracker")
    qapp.setQuitOnLastWindowClosed(False)
    platform = MacPlatform()
    platform.hide_dock_icon()

    models = paths.models_dir()
    eye = EyeTrackerApp(
        qapp,
        platform,
        settings=Settings.load(),
        store=CalibrationStore(),
        face_mesh=FaceMesh(models / paths.LANDMARK_MODEL, models / paths.DETECTOR_MODEL),
        list_cameras=list_cameras,
        camera_factory=lambda info: lambda: DeviceCamera(info.unique_id),
        camera_permission=camera_permission,
        request_camera_access=request_camera_access,
        debug=args.debug,
    )
    eye.start()
    _signal_timer = install_quit_on_signals(qapp)  # noqa: F841 - must stay alive
    log.info("Eye Tracker is running; look for the eye icon in the menu bar (Ctrl+C quits)")
    return qapp.exec()
