"""Main window: switches pages according to the session's current step."""

from typing import assert_never

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QMainWindow, QStackedWidget

from nfmimiface.experiment import Closing, Experiment, Feedback, Imitate, Session
from nfmimiface.gui.camera import FrameSource
from nfmimiface.gui.pages import ClosingPage, FeedbackPage, ImitatePage, Page, StartPage
from nfmimiface.images import RGBImage


class MainWindow(QMainWindow):
    def __init__(self, experiment: Experiment, source: FrameSource) -> None:
        super().__init__()
        config = experiment.config
        self.setWindowTitle(config.name)
        self._experiment = experiment
        self._session: Session | None = None
        self._frame: RGBImage | None = None  # latest camera frame (mirrored)

        self._start_page = StartPage(config.name)
        self._imitate_page = ImitatePage(config.stimulus_side)
        self._feedback_page = FeedbackPage(config.stimulus_side)
        self._closing_page = ClosingPage(config.closing_message)
        self._stack = QStackedWidget()
        for page in (self._start_page, self._imitate_page, self._feedback_page, self._closing_page):
            self._stack.addWidget(page)
        self.setCentralWidget(self._stack)

        self._start_page.button.clicked.connect(self._start)
        self._imitate_page.button.clicked.connect(self._capture)
        self._feedback_page.button.clicked.connect(self._advance)
        self._closing_page.button.clicked.connect(self._end)
        for key in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setAutoRepeat(False)  # holding the key must not skip pages
            shortcut.activated.connect(self._press_main_button)
        source.frame.connect(self._on_frame)

    @property
    def session(self) -> Session:
        assert self._session is not None
        return self._session

    def _on_frame(self, frame: RGBImage) -> None:
        self._frame = frame
        page = self._stack.currentWidget()
        if page is self._start_page:
            self._start_page.live.set_image(frame)
        elif page is self._imitate_page:
            self._imitate_page.live.set_image(self.session.crop.crop(frame))

    def _press_main_button(self) -> None:
        page = self._stack.currentWidget()
        assert isinstance(page, Page)
        page.button.click()

    def _start(self) -> None:
        if self._frame is None:
            return
        self._session = self._experiment.start_session(self._frame)
        if self._session is None:
            self._start_page.show_no_face()
            return
        self._show_step()

    def _capture(self) -> None:
        if self._frame is None:
            return
        if self.session.capture(self._frame) is None:
            self._imitate_page.show_no_face()
            return
        self._advance()

    def _advance(self) -> None:
        self.session.advance()
        self._show_step()

    def _end(self) -> None:
        self._session = None
        self._start_page.reset()
        self._stack.setCurrentWidget(self._start_page)

    def _show_step(self) -> None:
        match self.session.step:
            case Imitate(stimulus=stimulus):
                self._imitate_page.show_stimulus(stimulus)
                self._stack.setCurrentWidget(self._imitate_page)
            case Feedback(trial=trial):
                self._feedback_page.show_trial(
                    self.session.trials[trial], last=self.session.next_is_closing
                )
                self._stack.setCurrentWidget(self._feedback_page)
            case Closing():
                self._stack.setCurrentWidget(self._closing_page)
            case step:
                assert_never(step)
