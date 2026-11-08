"""Which cameras exist, which one to use, and a thin wrapper to read frames from it."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraInfo:
    index: int  # OpenCV's index for this camera, valid only for the moment it was listed
    name: str
    builtin: bool
    unique_id: str = ""  # AVFoundation's stable device id


def list_cameras() -> list[CameraInfo]:
    """Cameras in OpenCV's index order.

    OpenCV's AVFoundation backend takes the video devices plus the muxed devices and sorts them by
    uniqueID (modules/videoio/src/cap_avfoundation_mac.mm, 4.x and 5.x), so we do the same.
    """
    if sys.platform != "darwin":
        return []
    import AVFoundation as AV

    devices = list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeVideo))
    devices += list(AV.AVCaptureDevice.devicesWithMediaType_(AV.AVMediaTypeMuxed))
    devices.sort(key=lambda d: str(d.uniqueID()))
    builtin_type = str(AV.AVCaptureDeviceTypeBuiltInWideAngleCamera)
    return [
        CameraInfo(
            i, str(d.localizedName()), str(d.deviceType()) == builtin_type, str(d.uniqueID())
        )
        for i, d in enumerate(devices)
    ]


def choose_camera(cameras: Sequence[CameraInfo], preferred_name: str | None) -> CameraInfo | None:
    """The saved camera if it is still there, else the first built-in one, else the first one."""
    if preferred_name:
        for cam in cameras:
            if cam.name == preferred_name:
                return cam
    for cam in cameras:
        if cam.builtin:
            return cam
    return cameras[0] if cameras else None


def resolve_index(unique_id: str, cameras: Sequence[CameraInfo]) -> int | None:
    """OpenCV's current index for a device, or None when it is no longer connected."""
    for cam in cameras:
        if cam.unique_id == unique_id:
            return cam.index
    return None


_STATUS = {0: "not_determined", 1: "restricted", 2: "denied", 3: "authorized"}


def camera_permission() -> str:
    if sys.platform != "darwin":
        return "authorized"
    import AVFoundation as AV

    status = AV.AVCaptureDevice.authorizationStatusForMediaType_(AV.AVMediaTypeVideo)
    return _STATUS.get(int(status), "denied")


def request_camera_access() -> None:
    """Ask macOS for camera access without waiting; macOS shows its own prompt.

    The answer is picked up later by polling camera_permission(), so the menu bar never freezes.
    """
    if sys.platform != "darwin" or camera_permission() != "not_determined":
        return
    import AVFoundation as AV

    def handler(granted):
        log.info("camera access %s", "granted" if granted else "refused")

    AV.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AV.AVMediaTypeVideo, handler)


class OpenCVCamera:
    def __init__(self, index: int, width: int = 640, height: int = 480) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None

    def open(self) -> bool:
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            log.warning("camera %s could not be opened", self.index)
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        return True

    def read(self) -> np.ndarray | None:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class DeviceCamera:
    """A camera remembered by device, not by number.

    OpenCV only takes an index, and indexes shift when an iPhone (Continuity Camera) comes or
    goes. So every open looks the device up again; if it is gone, open() fails instead of
    silently opening some other camera.
    """

    def __init__(
        self,
        unique_id: str,
        lister: Callable[[], list[CameraInfo]] = list_cameras,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.unique_id = unique_id
        self._lister = lister
        self._size = (width, height)
        self._inner: OpenCVCamera | None = None

    def open(self) -> bool:
        index = resolve_index(self.unique_id, self._lister())
        if index is None:
            log.warning("camera %s is not connected", self.unique_id)
            return False
        inner = OpenCVCamera(index, *self._size)
        if not inner.open():
            return False
        self._inner = inner
        return True

    def read(self) -> np.ndarray | None:
        return self._inner.read() if self._inner is not None else None

    def release(self) -> None:
        if self._inner is not None:
            self._inner.release()
            self._inner = None
