"""Application entry point (`uv run nfmimiface`)."""

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

from nfmimiface.config import load_config
from nfmimiface.constants import DEFAULT_EXPERIMENT
from nfmimiface.gui.camera import Camera, FrameSource, StillImage
from nfmimiface.gui.widgets import ImageView
from nfmimiface.images import load_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Facial expression imitation demo.")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--fake-camera", type=Path, metavar="IMAGE", help="show IMAGE instead of the webcam"
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    config = load_config(args.experiment)
    try:
        source: FrameSource = (
            StillImage(load_rgb(args.fake_camera)) if args.fake_camera else Camera(config.camera)
        )
    except RuntimeError as e:
        QMessageBox.critical(None, "エラー", str(e))
        sys.exit(1)

    # Placeholder window until the pages exist: just the live (mirrored) view.
    window = QMainWindow()
    window.setWindowTitle(config.name)
    view = ImageView()
    window.setCentralWidget(view)
    source.frame.connect(view.set_image)

    app.aboutToQuit.connect(source.stop)
    source.start()
    window.resize(1280, 720)
    window.show()
    sys.exit(app.exec())
