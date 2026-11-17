"""Application entry point (`uv run nfmimiface`)."""

import argparse
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from nfmimiface.config import load_config
from nfmimiface.constants import DEFAULT_EXPERIMENT
from nfmimiface.experiment import Experiment
from nfmimiface.face import FaceAnalyzer
from nfmimiface.gui.camera import Camera, FrameSource, StillImage
from nfmimiface.gui.main_window import MainWindow
from nfmimiface.images import load_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Facial expression imitation demo.")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--fake-camera", type=Path, metavar="IMAGE", help="show IMAGE instead of the webcam"
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    try:
        config = load_config(args.experiment)
        analyzer = FaceAnalyzer()
        experiment = Experiment(config, analyzer)  # loads and analyses the stimuli
        source: FrameSource = (
            StillImage(load_rgb(args.fake_camera)) if args.fake_camera else Camera(config.camera)
        )
    except Exception as e:  # any start-up failure: show it and quit
        QMessageBox.critical(None, "起動エラー", f"{type(e).__name__}: {e}")
        sys.exit(1)

    window = MainWindow(experiment, source)
    app.aboutToQuit.connect(source.stop)
    app.aboutToQuit.connect(analyzer.close)
    source.start()
    window.resize(1280, 720)
    window.show()
    sys.exit(app.exec())
