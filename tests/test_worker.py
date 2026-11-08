import time

import numpy as np
import pytest
from face_synth import synth_landmarks

from eyetracker.vision.camera import CameraInfo, choose_camera
from eyetracker.vision.face_mesh import FaceResult
from eyetracker.vision.worker import VisionLoop, VisionWorker


class FakeCamera:
    def __init__(self, opens=True, fail_first_read=False):
        self.opens = opens
        self.fail_first_read = fail_first_read
        self.reads = 0
        self.released = False

    def open(self):
        return self.opens

    def read(self):
        self.reads += 1
        if self.fail_first_read and self.reads == 1:
            return None
        return np.zeros((480, 640, 3), np.uint8)

    def release(self):
        self.released = True


class FakeMesh:
    def __init__(self, face=True):
        self.face = face

    def process(self, frame):
        if not self.face:
            return None
        return FaceResult(landmarks=synth_landmarks(), presence=0.99)


class Factory:
    def __init__(self, **camera_kwargs):
        self.kwargs = camera_kwargs
        self.made: list[FakeCamera] = []

    def __call__(self):
        cam = FakeCamera(**self.kwargs)
        self.made.append(cam)
        return cam


def test_choose_camera_prefers_builtin_over_iphone():
    cams = [CameraInfo(0, "iPhone Camera", False), CameraInfo(1, "MacBook Air Camera", True)]
    assert choose_camera(cams, None).index == 1
    assert choose_camera(cams, "iPhone Camera").index == 0
    assert choose_camera(cams, "gone").index == 1
    assert choose_camera([CameraInfo(0, "USB cam", False)], None).index == 0
    assert choose_camera([], None) is None


def test_step_emits_features_when_face_found():
    loop = VisionLoop(Factory(), FakeMesh(), clock=lambda: 12.5)
    obs = loop.step()
    assert obs.t == 12.5
    assert obs.features is not None and obs.features.yaw == pytest.approx(0, abs=0.5)
    assert loop.status == "running"


def test_step_emits_none_features_without_face():
    obs = VisionLoop(Factory(), FakeMesh(face=False)).step()
    assert obs is not None and obs.features is None


def test_frame_attached_only_when_preview():
    loop = VisionLoop(Factory(), FakeMesh())
    first = loop.step()
    assert first.frame is None and first.landmarks is None
    loop.preview = True
    second = loop.step()
    assert second.frame is not None and second.landmarks is not None


def test_step_reports_unavailable_when_camera_will_not_open():
    loop = VisionLoop(Factory(opens=False), FakeMesh())
    assert loop.step() is None
    assert loop.status == "camera_unavailable"


def test_read_failure_reopens_camera():
    factory = Factory(fail_first_read=True)
    loop = VisionLoop(factory, FakeMesh())
    assert loop.step() is None
    assert factory.made[0].released
    factory.kwargs = {}  # the next camera behaves
    assert loop.step() is not None
    assert len(factory.made) == 2


def test_close_releases_camera():
    factory = Factory()
    loop = VisionLoop(factory, FakeMesh())
    loop.step()
    loop.close()
    assert factory.made[0].released


def test_worker_thread_emits_and_stops(qapp):
    worker = VisionWorker(VisionLoop(Factory(), FakeMesh()), fps=60)
    seen = []
    worker.observed.connect(seen.append)
    worker.start()
    deadline = time.monotonic() + 3
    while len(seen) < 3 and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    worker.stop()
    assert len(seen) >= 3
    assert not worker.isRunning()


def test_resolve_index_follows_the_device_not_the_number():
    from eyetracker.vision.camera import resolve_index

    builtin = CameraInfo(0, "MacBook Air Camera", True, unique_id="6C70")
    assert resolve_index("6C70", [builtin]) == 0
    # an iPhone with a lower uniqueID shows up: the built-in camera is now index 1
    later = [
        CameraInfo(0, "iphone", False, unique_id="0AAA"),
        CameraInfo(1, "MacBook Air Camera", True, unique_id="6C70"),
    ]
    assert resolve_index("6C70", later) == 1
    assert resolve_index("gone", [builtin]) is None


def test_device_camera_fails_to_open_when_device_is_gone():
    from eyetracker.vision.camera import DeviceCamera

    cam = DeviceCamera("6C70", lister=lambda: [CameraInfo(0, "iphone", False, unique_id="0AAA")])
    assert cam.open() is False
    assert cam.read() is None
    cam.release()
