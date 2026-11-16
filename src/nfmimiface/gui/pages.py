"""One page per step type (plus the start page). Pages only display; MainWindow drives them.

Widgets get object names (title, caption, primary, ...) that theme.py styles.
"""

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
        self.button.setObjectName("primary")
        # No focus: otherwise a focused button would also react to Space by itself.
        self.button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.column = QVBoxLayout(self)
        self.column.setContentsMargins(48, 32, 48, 32)
        self.column.setSpacing(16)


def _label(role: str, text: str = "") -> QLabel:
    label = QLabel(text)
    label.setObjectName(role)
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    return label


def _captioned(view: ImageView, caption: str) -> QWidget:
    widget = QWidget()
    column = QVBoxLayout(widget)
    column.setContentsMargins(0, 0, 0, 0)
    column.addWidget(_label("caption", caption))
    column.addWidget(view, 1)  # the image fills the remaining height
    return widget


def _side_by_side(stimulus: QWidget, face: QWidget, side: Side) -> QHBoxLayout:
    """Two columns of equal width, the stimulus on `side`."""
    row = QHBoxLayout()
    row.setSpacing(24)
    for widget in (stimulus, face) if side == "left" else (face, stimulus):
        row.addWidget(widget, 1)
    return row


class StartPage(Page):
    def __init__(self, title: str) -> None:
        super().__init__("スタート")
        self.live = ImageView()
        self.status = _label("status")
        self.column.addWidget(_label("title", title))
        self.column.addWidget(self.live, 1)
        self.column.addWidget(
            _label("hint", "カメラの正面に座って、「スタート」を押してください。")
        )
        self.column.addWidget(self.status)
        self.column.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignCenter)

    def reset(self) -> None:
        self.status.clear()

    def show_no_face(self) -> None:
        self.status.setText(
            "顔が見つかりませんでした。カメラの正面に座って、もう一度「スタート」を押してください。"
        )


class ImitatePage(Page):
    def __init__(self, side: Side) -> None:
        super().__init__("完了")
        self.instruction = _label("instruction")
        self.stimulus = ImageView()
        self.live = ImageView()
        self.status = _label("status")
        self.column.addWidget(self.instruction)
        self.column.addLayout(
            _side_by_side(
                _captioned(self.stimulus, "お手本"), _captioned(self.live, "あなた"), side
            ),
            1,
        )
        self.column.addWidget(self.status)
        self.column.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignCenter)

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
        self.score = _label("score")
        self.message = _label("message")
        self.column.addLayout(
            _side_by_side(
                _captioned(self.stimulus, "お手本"), _captioned(self.face, "あなた"), side
            ),
            1,
        )
        self.column.addWidget(self.score)
        self.column.addWidget(self.message)
        self.column.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignCenter)

    def show_trial(self, trial: TrialResult) -> None:
        self.stimulus.set_image(trial.stimulus.image)
        self.face.set_image(trial.face)
        self.show_score(trial.score)
        self.message.setText(trial.message)

    def show_score(self, score: float) -> None:
        self.score.setText(
            f'<span style="font-size: 64pt; font-weight: 800;">{score:.1f}</span>'
            '<span style="font-size: 28pt; font-weight: 700;"> 点</span>'
        )


class ClosingPage(Page):
    def __init__(self, message: str) -> None:
        super().__init__("おわる")
        self.column.addStretch(1)
        self.column.addWidget(_label("closing", message))
        self.column.addStretch(1)
        self.column.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignCenter)
