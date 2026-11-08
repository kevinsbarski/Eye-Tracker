"""The menu-bar eye, drawn in code so there are no image files to ship."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap

_SIZE = 44  # 22 pt at 2x


def eye_icon(active: bool) -> QIcon:
    pixmap = QPixmap(_SIZE, _SIZE)
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.transparent)
    s = _SIZE / 2  # logical size
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(0, 0, 0), 1.6)
    painter.setPen(pen)

    outline = QPainterPath()
    outline.moveTo(QPointF(2, s / 2))
    outline.quadTo(QPointF(s / 2, 2), QPointF(s - 2, s / 2))
    outline.quadTo(QPointF(s / 2, s - 2), QPointF(2, s / 2))
    painter.drawPath(outline)

    if active:
        painter.setBrush(QColor(0, 0, 0))
        painter.drawEllipse(QRectF(s / 2 - 3.5, s / 2 - 3.5, 7, 7))
    else:
        painter.drawEllipse(QRectF(s / 2 - 3, s / 2 - 3, 6, 6))
        painter.drawLine(QPointF(4, s - 4), QPointF(s - 4, 4))  # struck through = not tracking
    painter.end()

    icon = QIcon(pixmap)
    icon.setIsMask(True)  # macOS draws it as a template image (follows light/dark menu bar)
    return icon
