"""Small reusable widgets."""

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QImage, QPainter, QPaintEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from nfmimiface.images import RGBImage


def to_qimage(image: RGBImage) -> QImage:
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    # copy(): the QImage must not keep pointing into the numpy buffer
    return QImage(image.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class ImageView(QWidget):
    """Shows an image scaled to fit, keeping its aspect ratio, centred."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, image: RGBImage | None) -> None:
        self._image = None if image is None else to_qimage(image)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        if self._image is None:
            return
        size = self._image.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRect(QPoint(0, 0), size)
        target.moveCenter(self.rect().center())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(target, self._image)
