"""Live scores of the webcam face against a target image, for trying out scorers.

    uv run nfmimiface-live-score --img-path TARGET [--scorer NAME ...] [--camera N]

TARGET is an image, or a directory of images (← / → switch between them). Each
`--scorer` NAME is shown with its default params; without any, the candidates of
nfmimiface-compare-scorers are shown. As in the app, the webcam image is mirrored before
analysis; unlike the app, the whole frame is analysed (no crop). Scores update ~10
times per second and are not smoothed.
"""

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from nfmimiface.compare_scorers import CANDIDATES
from nfmimiface.config import CameraConfig, ScorerConfig
from nfmimiface.face import Blendshapes, FaceAnalyzer
from nfmimiface.gui.camera import Camera, FrameSource, StillImage
from nfmimiface.gui.theme import apply_theme
from nfmimiface.gui.widgets import ImageView
from nfmimiface.images import RGBImage, load_rgb
from nfmimiface.scoring import Scorer, make_scorer

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANALYSIS_INTERVAL_MS = 100

type Target = tuple[Path, RGBImage, Blendshapes]


class LiveWindow(QWidget):
    def __init__(
        self,
        targets: list[Target],
        scorers: dict[str, Scorer],
        analyzer: FaceAnalyzer,
        source: FrameSource,
    ) -> None:
        super().__init__()
        self._targets, self._scorers, self._analyzer = targets, scorers, analyzer
        self._index = 0
        self._frame: RGBImage | None = None

        self._target_view, self._camera_view = ImageView(), ImageView()
        self._scores = QLabel()
        font = QFont("monospace", 20)
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        self._scores.setFont(font)
        self._scores.setStyleSheet("font-size: 20pt;")  # the theme sets 16pt on QWidget
        images = QHBoxLayout()
        images.addWidget(self._target_view)
        images.addWidget(self._camera_view)
        layout = QVBoxLayout(self)
        layout.addLayout(images, stretch=1)
        layout.addWidget(self._scores, alignment=Qt.AlignmentFlag.AlignHCenter)

        source.frame.connect(self._on_frame)
        timer = QTimer(self)
        timer.timeout.connect(self._update_scores)
        timer.start(ANALYSIS_INTERVAL_MS)
        self._show_target()

    def _show_target(self) -> None:
        path, image, _ = self._targets[self._index]
        self._target_view.set_image(image)
        self.setWindowTitle(f"{path.name} ({self._index + 1}/{len(self._targets)})")

    def _on_frame(self, frame: RGBImage) -> None:
        self._frame = frame
        self._camera_view.set_image(frame)

    def _update_scores(self) -> None:
        if self._frame is None:
            return
        if (participant := self._analyzer.blendshapes(self._frame)) is None:
            self._scores.setText("no face")
            return
        target = self._targets[self._index][2]
        width = max(map(len, self._scorers))
        lines = [f"{n:<{width}} {s(participant, target):5.1f}" for n, s in self._scorers.items()]
        self._scores.setText("\n".join(lines))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        step = {Qt.Key.Key_Right: 1, Qt.Key.Key_Left: -1}.get(Qt.Key(event.key()))
        if step is None:
            return super().keyPressEvent(event)
        self._index = (self._index + step) % len(self._targets)
        self._show_target()


def load_targets(path: Path, analyzer: FaceAnalyzer) -> list[Target]:
    if path.is_dir():
        paths = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        paths = [path]
    targets: list[Target] = []
    for p in paths:
        image = load_rgb(p)
        if (blendshapes := analyzer.blendshapes(image)) is None:
            print(f"{p}: no face detected, skipped")
        else:
            targets.append((p, image, blendshapes))
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description="Live scores of the webcam face vs a target.")
    parser.add_argument("--img-path", type=Path, required=True, help="image or directory")
    parser.add_argument("--scorer", action="append", help="scorer name (repeatable)")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--fake-camera", type=Path, metavar="IMAGE", help="IMAGE as the webcam")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    apply_theme(app)
    analyzer = FaceAnalyzer()
    if not (targets := load_targets(args.img_path, analyzer)):
        sys.exit(f"no usable target image in {args.img_path}")
    configs = {n: ScorerConfig(n) for n in args.scorer} if args.scorer else CANDIDATES
    scorers = {name: make_scorer(config) for name, config in configs.items()}
    source = (
        StillImage(load_rgb(args.fake_camera))
        if args.fake_camera
        else Camera(CameraConfig(index=args.camera))
    )

    window = LiveWindow(targets, scorers, analyzer, source)
    app.aboutToQuit.connect(source.stop)
    app.aboutToQuit.connect(analyzer.close)
    source.start()
    window.resize(1280, 720)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
