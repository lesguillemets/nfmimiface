"""Small reusable widgets."""

import numpy as np
from PySide6.QtCore import Property, QPoint, QRect, Qt
from PySide6.QtGui import QImage, QPainter, QPainterPath, QPaintEvent
from PySide6.QtWidgets import QSizePolicy, QWidget

from nfmimiface.images import RGBImage


def to_qimage(image: RGBImage) -> QImage:
    image = np.ascontiguousarray(image)
    h, w = image.shape[:2]
    # copy(): the QImage must not keep pointing into the numpy buffer
    return QImage(image.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class ImageView(QWidget):
    """A panel showing an image scaled to fit, keeping its aspect ratio, centred.

    The panel (background, border) and `radius` come from the theme's stylesheet.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: QImage | None = None
        self._radius = 0.0
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)  # draw the QSS panel
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def _get_radius(self) -> float:
        return self._radius

    def _set_radius(self, radius: float) -> None:
        self._radius = radius
        self.update()

    radius = Property(float, _get_radius, _set_radius)

    def set_image(self, image: RGBImage | None) -> None:
        self._image = None if image is None else to_qimage(image)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        if self._image is None:
            return
        panel = self.rect().adjusted(1, 1, -1, -1)  # inside the 1px border
        size = self._image.size().scaled(panel.size(), Qt.AspectRatioMode.KeepAspectRatio)
        target = QRect(QPoint(0, 0), size)
        target.moveCenter(panel.center())
        clip = QPainterPath()  # only the panel's corners are rounded
        clip.addRoundedRect(panel, self._radius, self._radius)
        painter = QPainter(self)
        painter.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        painter.setClipPath(clip)
        painter.drawImage(target, self._image)
