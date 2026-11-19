"""Live scores of the webcam face against a target image, for trying out scorers.

    uv run nfmimiface-live-score --img-path TARGET [--scorer NAME ...] [--camera N] [--radar]
    uv run nfmimiface-live-score --config TOML [--scorer NAME ...] [--camera N] [--radar]

TARGET is an image, or a directory of images (← / → switch between them). Each
`--scorer` NAME is shown with its default params (as a number and a bar); without any,
the candidates of nfmimiface-compare-scorers are shown.

With --config, the target is the first stimulus of the experiment TOML, and each scorer
(every registered one without --scorer) gets the TOML's [scorer] params, minus those it
does not take (printed at startup). The camera also comes from the TOML unless --camera
is given.

--radar adds a radar chart of the blendshapes (as in nfmimiface-radar): the axes are those
the TOML's scorer params select, or all but NON_EXPRESSIVE without --config.

The drop-down above the bars shows all of them or just one. As in the app, the webcam
image is mirrored before analysis; unlike the app, the whole frame is analysed (no crop).
Scores update ~10 times per second and are not smoothed.
"""

import argparse
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from nfmimiface.compare_scorers import CANDIDATES
from nfmimiface.config import CameraConfig, ScorerConfig, load_config
from nfmimiface.face import Blendshapes, FaceAnalyzer
from nfmimiface.gui.camera import Camera, FrameSource, StillImage
from nfmimiface.gui.theme import LIGHT, Theme, apply_theme
from nfmimiface.gui.widgets import ImageView
from nfmimiface.images import RGBImage, load_rgb
from nfmimiface.radar import Radar, axis_names
from nfmimiface.scoring import Scorer, accepted_params, make_scorer, scorer_names

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ANALYSIS_INTERVAL_MS = 100
ALL = "(all)"

type Target = tuple[Path, RGBImage, Blendshapes]


class ScoreBars(QWidget):
    """One row per scorer: name, a 0–100 bar, and the score. `None` means no face."""

    def __init__(self, names: list[str], theme: Theme = LIGHT) -> None:
        super().__init__()
        self._theme = theme
        font = QFont("monospace", 18)
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        self.setFont(font)
        self.setStyleSheet("font-size: 18pt;")  # the theme sets 16pt on QWidget
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.set_names(names)

    def set_names(self, names: list[str]) -> None:
        self._names = names
        self._scores: list[float] | None = []  # nothing to show until the next scores
        self.setFixedHeight(int(len(names) * self.fontMetrics().height() * 1.5))
        self.update()

    def set_scores(self, scores: list[float] | None) -> None:
        self._scores = scores
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        fm = self.fontMetrics()
        if self._scores is None:
            painter.setPen(QColor(self._theme.warning))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "no face")
            return
        if len(self._scores) != len(self._names):
            return
        row_h = self.height() / len(self._names)
        name_w = max(fm.horizontalAdvance(n) for n in self._names) + fm.averageCharWidth()
        value_w = fm.horizontalAdvance(" 100.0")
        bar_x, bar_w = name_w, self.width() - name_w - value_w
        # recessive grid every 20 points
        painter.setPen(QColor(self._theme.border))
        for v in range(0, 101, 20):
            x = bar_x + bar_w * v / 100
            painter.drawLine(int(x), 0, int(x), self.height())
        for i, (name, score) in enumerate(zip(self._names, self._scores, strict=True)):
            row = QRectF(0, i * row_h, self.width(), row_h)
            painter.setPen(QColor(self._theme.text))
            painter.drawText(row, Qt.AlignmentFlag.AlignVCenter, name)
            painter.drawText(
                row, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, f"{score:5.1f}"
            )
            bar_h = row_h * 0.5
            bar = QRectF(bar_x, row.center().y() - bar_h / 2, bar_w * score / 100, bar_h)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(self._theme.accent))
            painter.drawRoundedRect(bar, 2, 2)


class RadarCanvas(FigureCanvasQTAgg):
    """A Radar on a Qt widget. Redrawing the whole chart is slow (~60 ms with 43 axes), so
    the rest is drawn once and saved, and set_participant only redraws its polygon over it
    (blitting)."""

    def __init__(self, names: list[str]) -> None:
        super().__init__(Figure())
        self._radar = Radar(self.figure, names)
        for artist in self._radar.participant_artists:
            artist.set_animated(True)  # left out of full draws
        self._background: Any = None
        self.mpl_connect("draw_event", self._on_draw)

    def set_target(self, values: Mapping[str, float]) -> None:
        self._radar.set_target(values)
        self.draw_idle()  # a full draw, which saves the new background

    def set_participant(self, values: Mapping[str, float] | None) -> None:
        self._radar.set_participant(values)
        if self._background is not None:
            self.restore_region(self._background)
            self._draw_participant()
            self.blit(self.figure.bbox)

    def _on_draw(self, _event: object) -> None:
        self._background = self.copy_from_bbox(self.figure.bbox)
        self._draw_participant()

    def _draw_participant(self) -> None:
        for artist in self._radar.participant_artists:
            self.figure.draw_artist(artist)


class LiveWindow(QWidget):
    def __init__(
        self,
        targets: list[Target],
        scorers: dict[str, Scorer],
        analyzer: FaceAnalyzer,
        source: FrameSource,
        radar_names: list[str] | None = None,  # no radar chart if None
    ) -> None:
        super().__init__()
        self._targets, self._analyzer = targets, analyzer
        self._all_scorers = self._scorers = scorers
        self._index = 0
        self._frame: RGBImage | None = None

        self._target_view, self._camera_view = ImageView(), ImageView()
        self._choice = QComboBox()
        self._choice.addItems([ALL, *scorers])
        self._choice.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keep ← / → for the targets
        self._choice.currentTextChanged.connect(self._choose)
        self._bars = ScoreBars(list(scorers))
        images = QHBoxLayout()
        images.addWidget(self._target_view)
        images.addWidget(self._camera_view)
        main = QVBoxLayout()
        main.addLayout(images, stretch=1)
        main.addWidget(self._choice, alignment=Qt.AlignmentFlag.AlignLeft)
        main.addWidget(self._bars)
        layout = QHBoxLayout(self)
        layout.addLayout(main, stretch=2)
        self._radar = None if radar_names is None else RadarCanvas(radar_names)
        if self._radar is not None:  # to the right, over the whole height
            layout.addWidget(self._radar, stretch=1)

        source.frame.connect(self._on_frame)
        timer = QTimer(self)
        timer.timeout.connect(self._update_scores)
        timer.start(ANALYSIS_INTERVAL_MS)
        self._show_target()

    def _show_target(self) -> None:
        path, image, blendshapes = self._targets[self._index]
        self._target_view.set_image(image)
        self.setWindowTitle(f"{path.name} ({self._index + 1}/{len(self._targets)})")
        if self._radar is not None:
            self._radar.set_target(blendshapes)

    def _choose(self, name: str) -> None:
        all_ = self._all_scorers
        self._scorers = all_ if name == ALL else {name: all_[name]}
        self._bars.set_names(list(self._scorers))
        self._update_scores()

    def _on_frame(self, frame: RGBImage) -> None:
        self._frame = frame
        self._camera_view.set_image(frame)

    def _update_scores(self) -> None:
        if self._frame is None:
            return
        participant = self._analyzer.blendshapes(self._frame)
        if self._radar is not None:
            self._radar.set_participant(participant)  # hidden while there is no face
        if participant is None:
            self._bars.set_scores(None)
            return
        target = self._targets[self._index][2]
        self._bars.set_scores([s(participant, target) for s in self._scorers.values()])

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


def with_params(names: list[str], params: Mapping[str, Any]) -> dict[str, ScorerConfig]:
    """A config per scorer name, with the `params` that scorer takes."""
    configs: dict[str, ScorerConfig] = {}
    for name in names:
        accepted = accepted_params(name)
        if ignored := sorted(params.keys() - accepted):
            print(f"{name}: ignoring {', '.join(ignored)}")
        configs[name] = ScorerConfig(name, {k: v for k, v in params.items() if k in accepted})
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Live scores of the webcam face vs a target.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--img-path", type=Path, help="image or directory")
    target.add_argument("--config", type=Path, help="experiment TOML (first stimulus, params)")
    parser.add_argument("--scorer", action="append", help="scorer name (repeatable)")
    parser.add_argument("--camera", type=int, help="camera index (default: TOML's, else 0)")
    parser.add_argument("--radar", action="store_true", help="add a radar chart")
    parser.add_argument("--fake-camera", type=Path, metavar="IMAGE", help="IMAGE as the webcam")
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        img_path = config.stimuli[0].resolved_path
        configs = with_params(args.scorer or scorer_names(), config.scorer.params)
        radar_names = axis_names(config.scorer.params)
        camera = config.camera
    else:
        img_path = args.img_path
        configs = {n: ScorerConfig(n) for n in args.scorer} if args.scorer else CANDIDATES
        radar_names = axis_names({})
        camera = CameraConfig()
    if args.camera is not None:
        camera = replace(camera, index=args.camera)

    app = QApplication(sys.argv)
    apply_theme(app)
    analyzer = FaceAnalyzer()
    if not (targets := load_targets(img_path, analyzer)):
        sys.exit(f"no usable target image in {img_path}")
    scorers = {name: make_scorer(config) for name, config in configs.items()}
    source = StillImage(load_rgb(args.fake_camera)) if args.fake_camera else Camera(camera)

    window = LiveWindow(targets, scorers, analyzer, source, radar_names if args.radar else None)
    app.aboutToQuit.connect(source.stop)
    app.aboutToQuit.connect(analyzer.close)
    source.start()
    window.resize(1800 if args.radar else 1280, 720)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
