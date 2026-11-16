"""Frame sources running in background threads.

Every source emits `frame` with an RGB image that is already mirrored (as displayed).
"""

import sys

import cv2
from PySide6.QtCore import QThread, Signal

from nfmimiface.config import CameraConfig
from nfmimiface.images import RGBImage


class FrameSource(QThread):
    frame = Signal(object)  # RGBImage, mirrored

    def stop(self) -> None:
        self.requestInterruption()
        self.wait()


class Camera(FrameSource):
    """A webcam. Raises RuntimeError on creation if it cannot be opened."""

    def __init__(self, config: CameraConfig) -> None:
        super().__init__()
        # On Windows, DirectShow opens much faster than the default backend (MSMF)
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self._capture = cv2.VideoCapture(config.index, backend)
        if not self._capture.isOpened():
            raise RuntimeError(f"カメラ {config.index} を開けませんでした。")
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

    def run(self) -> None:
        while not self.isInterruptionRequested():
            ok, bgr = self._capture.read()  # blocks until the next frame
            if ok:
                self.frame.emit(cv2.flip(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 1))
            else:
                self.msleep(10)
        self._capture.release()


class StillImage(FrameSource):
    """Stand-in for a webcam during development: emits one image, mirrored, at ~30 fps."""

    def __init__(self, image: RGBImage) -> None:
        super().__init__()
        self._frame = cv2.flip(image, 1)

    def run(self) -> None:
        while not self.isInterruptionRequested():
            self.frame.emit(self._frame)
            self.msleep(33)
