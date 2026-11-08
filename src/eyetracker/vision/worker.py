"""The vision thread: camera frame in, gaze features out, as fast as the frame-rate cap allows."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QThread, Signal

from eyetracker.vision.features import GazeFeatures, features_from_landmarks

log = logging.getLogger(__name__)

RETRY_CAMERA_S = 5.0


@dataclass
class Observation:
    t: float
    features: GazeFeatures | None
    landmarks: np.ndarray | None = None  # only while the preview is open
    frame: np.ndarray | None = None  # only while the preview is open; never written anywhere


class VisionLoop:
    """One step = one frame. No threads in here, so it can be tested directly."""

    def __init__(
        self, camera_factory: Callable, face_mesh, clock: Callable[[], float] = time.monotonic
    ):
        self.camera_factory = camera_factory
        self.face_mesh = face_mesh
        self.clock = clock
        self.preview = False
        self.status = "starting"
        self._camera = None

    def step(self) -> Observation | None:
        if self._camera is None:
            camera = self.camera_factory()
            if not camera.open():
                self.status = "camera_unavailable"
                return None
            self._camera = camera
            self.status = "running"
            reset = getattr(self.face_mesh, "reset", None)
            if reset:
                reset()

        frame = self._camera.read()
        if frame is None:
            log.warning("camera stopped delivering frames; reopening it")
            self.close()
            return None

        result = self.face_mesh.process(frame)
        features = None
        if result is not None:
            h, w = frame.shape[:2]
            features = features_from_landmarks(result.landmarks, w, h)
        return Observation(
            t=self.clock(),
            features=features,
            landmarks=result.landmarks if (self.preview and result is not None) else None,
            frame=frame if self.preview else None,
        )

    def close(self) -> None:
        if self._camera is not None:
            self._camera.release()
            self._camera = None


class VisionWorker(QThread):
    observed = Signal(object)  # Observation
    status_changed = Signal(str)  # running | camera_unavailable

    def __init__(self, loop: VisionLoop, fps: int = 15, parent=None) -> None:
        super().__init__(parent)
        self._loop = loop
        self._fps = fps
        self._stop = threading.Event()
        self._pending_factory: Callable | None = None
        self._lock = threading.Lock()

    @property
    def preview(self) -> bool:
        return self._loop.preview

    @preview.setter
    def preview(self, value: bool) -> None:
        self._loop.preview = value

    def set_camera_factory(self, factory: Callable) -> None:
        """Switch cameras; the old one is released on the vision thread."""
        with self._lock:
            self._pending_factory = factory

    def stop(self) -> None:
        self._stop.set()
        self.wait()

    def start(self, *args, **kwargs) -> None:
        self._stop.clear()
        super().start(*args, **kwargs)

    def run(self) -> None:
        period = 1.0 / self._fps
        last_status = None
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                with self._lock:
                    factory, self._pending_factory = self._pending_factory, None
                if factory is not None:
                    self._loop.close()
                    self._loop.camera_factory = factory
                try:
                    obs = self._loop.step()
                except Exception:
                    log.exception("vision step failed")
                    obs = None
                if self._loop.status != last_status:
                    last_status = self._loop.status
                    self.status_changed.emit(last_status)
                if obs is not None:
                    self.observed.emit(obs)
                if self._loop.status == "camera_unavailable":
                    self._stop.wait(RETRY_CAMERA_S)
                else:
                    self._stop.wait(max(0.0, period - (time.monotonic() - started)))
        finally:
            self._loop.close()
