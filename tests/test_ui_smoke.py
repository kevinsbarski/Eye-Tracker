import numpy as np
import pytest
from face_synth import synth_landmarks
from fakes import FakeModel, FakePlatform, FakeStore, feat

from eyetracker.config import Settings
from eyetracker.geometry import Display
from eyetracker.ui.gaze_dot import GazeDot
from eyetracker.ui.icons import eye_icon
from eyetracker.ui.preview import PreviewWindow
from eyetracker.ui.tray import Tray
from eyetracker.vision.camera import CameraInfo
from eyetracker.vision.worker import Observation

A = Display(1, 0, 0, 1440, 900)
B = Display(2, 1440, 0, 1920, 1080)


@pytest.fixture(autouse=True)
def _qt(qapp):
    yield


def test_eye_icon_has_pixels():
    assert not eye_icon(True).isNull()
    assert not eye_icon(False).isNull()


def test_tray_signals_and_states():
    tray = Tray()
    paused = []
    tray.pause_toggled.connect(paused.append)
    tray.pause_action.trigger()
    tray.pause_action.trigger()
    assert paused == [True, False]
    tray.set_state("not_calibrated")
    assert tray.status_action.text() == "Not calibrated"
    tray.set_state("camera_unavailable")
    assert tray.status_action.text() == "Camera unavailable"
    tray.set_accessibility_missing(True)
    assert tray.accessibility_action.isVisible()
    tray.set_accessibility_missing(False)
    assert not tray.accessibility_action.isVisible()


def test_tray_camera_menu():
    tray = Tray()
    picked = []
    tray.camera_selected.connect(picked.append)
    tray.set_cameras(["MacBook Air Camera", "iPhone Camera"], "MacBook Air Camera")
    actions = tray.camera_menu.actions()
    assert [a.text() for a in actions] == ["MacBook Air Camera", "iPhone Camera"]
    assert actions[0].isChecked() and not actions[1].isChecked()
    actions[1].trigger()
    assert picked == ["iPhone Camera"]


def test_gaze_dot_moves_and_hides():
    dot = GazeDot()
    dot.show_point((500, 300))
    assert not dot.isVisible()  # disabled by default
    dot.set_enabled(True)
    dot.show_point((500, 300))
    assert dot.isVisible()
    center = dot.geometry().center()
    assert abs(center.x() - 500) <= 1 and abs(center.y() - 300) <= 1
    dot.show_point(None)
    assert not dot.isVisible()


def test_preview_renders_observation():
    preview = PreviewWindow()
    preview.show()
    frame = np.zeros((480, 640, 3), np.uint8)
    obs = Observation(1.0, feat(1.0, 2.0), landmarks=synth_landmarks(), frame=frame)
    preview.show_observation(obs)
    assert not preview.image_label.pixmap().isNull()
    assert "yaw" in preview.text_label.text()
    preview.show_observation(Observation(2.0, None, frame=frame))
    assert "No face" in preview.text_label.text()


class _Mesh:
    def process(self, frame):
        return None


class _Camera:
    def open(self):
        return False

    def read(self):
        return None

    def release(self):
        pass


class _Env:
    """What the app sees of the outside world, changeable mid-test."""

    def __init__(self, permission="authorized"):
        self.permission = permission
        self.access_requests = 0
        self.cameras = [CameraInfo(0, "MacBook Air Camera", True, unique_id="builtin")]

    def request_access(self):
        self.access_requests += 1


def _make_app(qapp, granted=True, displays=(A, B), store=None, env=None):
    from eyetracker.app import EyeTrackerApp

    env = env or _Env("authorized" if granted else "denied")
    platform = FakePlatform(displays)
    eye = EyeTrackerApp(
        qapp,
        platform,
        settings=Settings(),
        store=store or FakeStore(),
        face_mesh=_Mesh(),
        list_cameras=lambda: list(env.cameras),
        camera_factory=lambda info: _Camera,
        camera_permission=lambda: env.permission,
        request_camera_access=env.request_access,
    )
    eye.env = env
    return eye


def test_app_without_camera_permission_reports_unavailable(qapp):
    eye = _make_app(qapp, granted=False)
    eye.start()
    assert eye.tray.status_action.text() == "Camera unavailable"
    assert eye.tray.camera_settings_action.isVisible()
    assert not eye.worker.isRunning()
    eye.shutdown()


def test_app_starts_camera_once_permission_is_granted(qapp):
    eye = _make_app(qapp, granted=False)
    eye.start()
    assert not eye.worker.isRunning()
    eye.env.permission = "authorized"  # the user flipped the switch in System Settings
    eye.poll()
    assert eye.worker.isRunning()
    assert not eye.tray.camera_settings_action.isVisible()
    eye.shutdown()


def test_app_asks_for_camera_access_once_when_undecided(qapp):
    env = _Env("not_determined")
    eye = _make_app(qapp, env=env)
    eye.start()
    eye.poll()
    assert env.access_requests == 1
    assert eye.tray.status_action.text() == "Camera unavailable"
    eye.shutdown()


def test_camera_menu_refreshes_when_opened(qapp):
    eye = _make_app(qapp)
    eye.start()
    eye.env.cameras = eye.env.cameras + [CameraInfo(1, "iPhone Camera", False, "phone")]
    eye.tray.camera_menu.aboutToShow.emit()
    names = [a.text() for a in eye.tray.camera_menu.actions()]
    assert names == ["MacBook Air Camera", "iPhone Camera"]
    eye.shutdown()


def test_app_calibrate_opens_and_cancels(qapp):
    eye = _make_app(qapp)
    eye.start()
    assert eye.tray.status_action.text() == "Not calibrated"
    eye.tray.calibrate_action.trigger()
    assert eye.calibration is not None and eye.controller._calibrating
    eye.calibration.cancel()
    assert not eye.controller._calibrating
    eye.shutdown()
    assert not eye.worker.isRunning()


def test_app_finished_calibration_is_saved_and_used(qapp):
    store = FakeStore()
    eye = _make_app(qapp, store=store)
    eye.start()
    eye.tray.calibrate_action.trigger()
    from eyetracker.gaze.model import CVReport

    eye.calibration.finished.emit(
        FakeModel(), CVReport(error_pt=50.0, display_accuracy=0.97, n_points=10)
    )
    assert store.saved and store.saved[0][3] == "MacBook Air Camera"
    assert eye.controller.state == "tracking"
    eye.shutdown()


def test_app_single_display_does_not_calibrate(qapp):
    eye = _make_app(qapp, displays=(A,))
    eye.start()
    eye.tray.calibrate_action.trigger()
    assert eye.calibration is None
    assert eye.tray.status_action.text() == "Only one display"
    eye.shutdown()


def test_ctrl_c_and_sigterm_quit_the_app(qapp):
    import os
    import signal

    from eyetracker.app import install_quit_on_signals

    class FakeApp:
        quits = 0

        def quit(self):
            self.quits += 1

    fake = FakeApp()
    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
    timer = install_quit_on_signals(fake)
    try:
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGTERM)
        assert fake.quits == 2
        assert timer.isActive()
    finally:
        timer.stop()
        for s, handler in old.items():
            signal.signal(s, handler)
