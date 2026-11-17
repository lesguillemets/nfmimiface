"""One page per step type (plus the start page). Pages only display; MainWindow drives them."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from nfmimiface.config import Side
from nfmimiface.experiment import Stimulus, TrialResult
from nfmimiface.gui.widgets import ImageView


class Page(QWidget):
    """A full-window page with one main button, also pressed by Space/Enter."""

    def __init__(self, button_text: str) -> None:
        super().__init__()
        self.button = QPushButton(button_text)
        # No focus: otherwise a focused button would also react to Space by itself.
        self.button.setFocusPolicy(Qt.FocusPolicy.NoFocus)


def _label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    return label


def _captioned(caption: str, view: ImageView) -> QVBoxLayout:
    column = QVBoxLayout()
    column.addWidget(_label(caption))
    column.addWidget(view, 1)
    return column


def _side_by_side(stimulus: QVBoxLayout, face: QVBoxLayout, side: Side) -> QHBoxLayout:
    row = QHBoxLayout()
    for column in (stimulus, face) if side == "left" else (face, stimulus):
        row.addLayout(column, 1)
    return row


class StartPage(Page):
    def __init__(self, title: str) -> None:
        super().__init__("スタート")
        self.live = ImageView()
        self.status = _label()
        layout = QVBoxLayout(self)
        layout.addWidget(_label(title))
        layout.addWidget(self.live, 1)
        layout.addWidget(_label("カメラの正面に座って、「スタート」を押してください。"))
        layout.addWidget(self.status)
        layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignCenter)

    def reset(self) -> None:
        self.status.clear()

    def show_no_face(self) -> None:
        self.status.setText(
            "顔が見つかりませんでした。カメラの正面に座って、もう一度「スタート」を押してください。"
        )


class ImitatePage(Page):
    def __init__(self, side: Side) -> None:
        super().__init__("完了")
        self.instruction = _label()
        self.stimulus = ImageView()
        self.live = ImageView()
        self.status = _label()
        layout = QVBoxLayout(self)
        layout.addWidget(self.instruction)
        layout.addLayout(
            _side_by_side(
                _captioned("お手本", self.stimulus), _captioned("あなた", self.live), side
            ),
            1,
        )
        layout.addWidget(self.status)
        layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignCenter)

    def show_stimulus(self, stimulus: Stimulus) -> None:
        self.instruction.setText(stimulus.instruction)
        self.stimulus.set_image(stimulus.image)
        self.live.set_image(None)
        self.status.clear()

    def show_no_face(self) -> None:
        self.status.setText(
            "顔が見つかりませんでした。顔がカメラに映るようにして、もう一度「完了」を押してください。"
        )


class FeedbackPage(Page):
    def __init__(self, side: Side) -> None:
        super().__init__("次へ")
        self.stimulus = ImageView()
        self.face = ImageView()
        self.score = _label()
        self.message = _label()
        layout = QVBoxLayout(self)
        layout.addLayout(
            _side_by_side(
                _captioned("お手本", self.stimulus), _captioned("あなた", self.face), side
            ),
            1,
        )
        layout.addWidget(self.score)
        layout.addWidget(self.message)
        layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignCenter)

    def show_trial(self, trial: TrialResult) -> None:
        self.stimulus.set_image(trial.stimulus.image)
        self.face.set_image(trial.face)
        self.score.setText(f"{trial.score:.1f} 点")
        self.message.setText(trial.message)


class ClosingPage(Page):
    def __init__(self, message: str) -> None:
        super().__init__("おわる")
        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(_label(message))
        layout.addStretch(1)
        layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignCenter)
