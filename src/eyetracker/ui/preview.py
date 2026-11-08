"""Debug window: what the camera sees, the landmarks found, and the live feature values."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QFont, QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from eyetracker.vision.features import LEFT_IRIS, RIGHT_IRIS

WIDTH = 640


class PreviewWindow(QWidget):
    closed = Signal()

    def __init__(self) -> None:
        super().__init__(None, Qt.Window)
        self.setWindowTitle("Eye Tracker — camera preview")
        self.image_label = QLabel()
        self.image_label.setFixedSize(WIDTH, WIDTH * 3 // 4)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.text_label = QLabel("Waiting for the camera…")
        font = QFont("Menlo")
        font.setPointSize(11)
        self.text_label.setFont(font)
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)
        layout.addWidget(self.text_label)

    def show_observation(self, obs) -> None:
        if not self.isVisible() or obs.frame is None:
            return
        frame = obs.frame.copy()
        if obs.landmarks is not None:
            for i, (x, y, _z) in enumerate(obs.landmarks):
                iris = i >= RIGHT_IRIS
                color = (40, 40, 230) if iris else (60, 200, 60)
                cv2.circle(frame, (int(x), int(y)), 2 if iris else 1, color, -1)
            for i in (RIGHT_IRIS, LEFT_IRIS):
                x, y = obs.landmarks[i, :2]
                cv2.circle(frame, (int(x), int(y)), 3, (40, 40, 230), -1)
        self.image_label.setPixmap(
            _to_pixmap(frame).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )
        f = obs.features
        if f is None:
            self.text_label.setText("No face found")
        else:
            self.text_label.setText(
                f"yaw {f.yaw:6.1f}°  pitch {f.pitch:6.1f}°  roll {f.roll:6.1f}°\n"
                f"iris h {f.iris_h:+.3f}  v {f.iris_v:+.3f}  open {f.eye_openness:.2f}\n"
                f"face x {f.face_x:.2f}  y {f.face_y:.2f}  scale {f.face_scale:.3f}"
            )

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt naming
        super().closeEvent(event)
        self.closed.emit()


def _to_pixmap(frame_bgr: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    h, w = rgb.shape[:2]
    image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())
