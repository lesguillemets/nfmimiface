import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest

from nfmimiface.config import ScoreMessage, StimulusConfig, load_config
from nfmimiface.constants import DEFAULT_EXPERIMENT
from nfmimiface.experiment import Closing, Experiment, Feedback, Imitate
from nfmimiface.face import BLENDSHAPE_NAMES, Blendshapes, FaceAnalyzer, Rect
from nfmimiface.images import RGBImage, save_png


class FakeAnalyzer(FaceAnalyzer):
    """Finds a face (or not) in any image, without loading the model."""

    def __init__(self) -> None:
        self.has_face = True

    def face_box(self, image: RGBImage) -> Rect | None:
        return Rect(100, 100, 50, 50) if self.has_face else None

    def blendshapes(self, image: RGBImage) -> Blendshapes | None:
        return Blendshapes(np.full(len(BLENDSHAPE_NAMES), 0.5)) if self.has_face else None


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    stimulus = tmp_path / "stimulus.png"
    save_png(stimulus, np.zeros((40, 30, 3), dtype=np.uint8))
    config = dataclasses.replace(
        load_config(DEFAULT_EXPERIMENT),
        stimuli=(StimulusConfig(stimulus, "a"), StimulusConfig(stimulus, "b")),
        messages=(ScoreMessage(50.0, "good"), ScoreMessage(0.0, "try again")),
    )
    return Experiment(config, FakeAnalyzer(), results_root=tmp_path / "results")


def test_build_steps(experiment: Experiment) -> None:
    steps = experiment.build_steps()
    assert [type(s) for s in steps] == [Imitate, Feedback, Imitate, Feedback, Closing]
    assert steps[3] == Feedback(trial=1)


def test_message_for(experiment: Experiment) -> None:
    assert experiment.message_for(80.0) == experiment.message_for(50.0) == "good"
    assert experiment.message_for(49.9) == experiment.message_for(-1.0) == "try again"


def test_session_records_trials(experiment: Experiment) -> None:
    session = experiment.start_session(FRAME)
    assert session is not None
    assert session.crop.w / session.crop.h == pytest.approx(30 / 40, abs=0.01)

    trial = session.capture(FRAME)
    assert trial is not None
    assert trial.score == pytest.approx(100)
    assert (session.dir / trial.face_file).exists()

    record = json.loads((session.dir / f"{session.id}.json").read_text(encoding="utf-8"))
    assert record["session_id"] == session.id
    assert record["trials"][0]["score"] == trial.score
    assert len(record["stimuli"]) == 2 and len(record["stimuli"][0]["md5"]) == 32


def test_no_face_records_nothing(experiment: Experiment) -> None:
    session = experiment.start_session(FRAME)
    assert session is not None
    assert isinstance(experiment.analyzer, FakeAnalyzer)
    experiment.analyzer.has_face = False
    assert session.capture(FRAME) is None
    assert session.trials == []
    assert experiment.start_session(FRAME) is None


def test_session_ids_are_numbered(experiment: Experiment) -> None:
    machine = experiment.config.machine_name
    other_day = experiment.results_root / "20000101"
    other_day.mkdir(parents=True)
    (other_day / f"{machine}_007_abcde.json").write_text("{}")
    (other_day / "other_099_abcde.json").write_text("{}")  # another machine
    first, second = experiment.start_session(FRAME), experiment.start_session(FRAME)
    assert first is not None and second is not None
    assert re.fullmatch(rf"{machine}_008_[a-z0-9]{{5}}", first.id)
    assert re.fullmatch(rf"{machine}_009_[a-z0-9]{{5}}", second.id)


def test_next_is_closing(experiment: Experiment) -> None:
    session = experiment.start_session(FRAME)
    assert session is not None
    feedbacks = []  # next_is_closing on each Feedback step
    while not isinstance(session.step, Closing):
        if isinstance(session.step, Feedback):
            feedbacks.append(session.next_is_closing)
        session.advance()
    assert feedbacks == [False, True]
