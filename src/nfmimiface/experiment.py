"""Experiment flow: stimuli, steps and sessions. No GUI code here.

See docs/architecture.md for how the classes relate.

A session is a list of steps built by `Experiment.build_steps()`. The GUI shows one page
per step type and calls `Session` methods. To add a kind of step (e.g. capturing a
neutral face before the task): add a step class, include it in `Step`, emit it from
`build_steps()`, and add a page for it in the GUI.

Camera frames passed in here are RGB and already mirrored (as displayed), so the face
crop that is analysed and saved is what the participant saw.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from nfmimiface.config import ExperimentConfig, StimulusConfig
from nfmimiface.constants import RESULTS_ROOT
from nfmimiface.face import Blendshapes, FaceAnalyzer, Rect, crop_rect
from nfmimiface.images import RGBImage, load_rgb, save_png
from nfmimiface.scoring import make_scorer


@dataclass(frozen=True)
class Stimulus:
    path: Path  # as written in the TOML
    md5: str  # of the image file, to tell apart different images saved under the same name
    instruction: str
    image: RGBImage
    blendshapes: Blendshapes


@dataclass(frozen=True)
class Imitate:
    """Show the stimulus beside the live camera; the participant imitates and captures."""

    stimulus: Stimulus


@dataclass(frozen=True)
class Feedback:
    """Show the result of a trial (`trial` is an index into `Session.trials`)."""

    trial: int


@dataclass(frozen=True)
class Closing:
    """Last page of a session."""


type Step = Imitate | Feedback | Closing


@dataclass(frozen=True)
class TrialResult:
    stimulus: Stimulus
    face: RGBImage  # the mirrored crop that was analysed and saved
    face_file: str  # file name within the session's results directory
    captured_at: datetime
    blendshapes: Blendshapes
    score: float
    message: str


class Experiment:
    """Loads stimuli and the scorer once; creates sessions."""

    def __init__(
        self, config: ExperimentConfig, analyzer: FaceAnalyzer, results_root: Path = RESULTS_ROOT
    ) -> None:
        self.config = config
        self.analyzer = analyzer
        self.results_root = results_root
        self.scorer = make_scorer(config.scorer)
        self.stimuli = tuple(self._load_stimulus(s) for s in config.stimuli)

    def build_steps(self) -> list[Step]:
        """Imitate + Feedback for each stimulus in order, then Closing.

        Variants change this, e.g. shuffling with `random.sample(self.stimuli, len(...))`.
        """
        steps: list[Step] = []
        for i, stimulus in enumerate(self.stimuli):
            steps += [Imitate(stimulus), Feedback(trial=i)]
        return [*steps, Closing()]

    def message_for(self, score: float) -> str:
        """Message with the highest min_score not above `score` (the lowest one if none)."""
        messages = sorted(self.config.messages, key=lambda m: m.min_score)
        reached = [m for m in messages if score >= m.min_score]
        return (reached[-1] if reached else messages[0]).text

    def start_session(self, frame: RGBImage) -> "Session | None":
        """New session cropping around the face in `frame`, or None if no face is detected.

        The crop has the aspect ratio of the first stimulus (all are assumed to match).
        """
        box = self.analyzer.face_box(frame)
        if box is None:
            return None
        frame_h, frame_w = frame.shape[:2]
        stim_h, stim_w = self.stimuli[0].image.shape[:2]
        crop = crop_rect(box, frame_w, frame_h, self.config.crop_scale, aspect=stim_w / stim_h)
        return Session(self, crop, started_at=datetime.now())

    def _load_stimulus(self, config: StimulusConfig) -> Stimulus:
        image = load_rgb(config.resolved_path)
        blendshapes = self.analyzer.blendshapes(image)
        if blendshapes is None:
            raise ValueError(f"no face detected in stimulus {config.path}")
        md5 = hashlib.md5(config.resolved_path.read_bytes()).hexdigest()
        return Stimulus(config.path, md5, config.instruction, image, blendshapes)


class Session:
    """One participant's visit. Its JSON record is written at the start and after each trial."""

    def __init__(self, experiment: Experiment, crop: Rect, started_at: datetime) -> None:
        self.experiment = experiment
        self.crop = crop
        self.started_at = started_at
        self.dir = experiment.results_root / f"{started_at:%Y%m%d}"
        self.dir.mkdir(parents=True, exist_ok=True)
        number = len(list(self.dir.glob("*.json"))) + 1  # per machine and day
        self.id = f"{experiment.config.machine_name}-{started_at:%Y%m%d-%H%M%S}-{number:03d}"
        self.steps = experiment.build_steps()
        self.trials: list[TrialResult] = []
        self._position = 0
        self._save()

    @property
    def step(self) -> Step:
        return self.steps[self._position]

    def advance(self) -> None:
        """Move to the next step. Not called after Closing: the session just ends."""
        self._position += 1

    def capture(self, frame: RGBImage) -> TrialResult | None:
        """Crop, analyse, score and save the frame for the current Imitate step.

        Returns None, recording nothing, if no face is detected.
        """
        step = self.step
        assert isinstance(step, Imitate)
        face = self.crop.crop(frame).copy()
        blendshapes = self.experiment.analyzer.blendshapes(face)
        if blendshapes is None:
            return None

        score = self.experiment.scorer(blendshapes, step.stimulus.blendshapes)
        face_file = f"{self.id}_t{len(self.trials) + 1}.png"
        save_png(self.dir / face_file, face)
        trial = TrialResult(
            stimulus=step.stimulus,
            face=face,
            face_file=face_file,
            captured_at=datetime.now(),
            blendshapes=blendshapes,
            score=score,
            message=self.experiment.message_for(score),
        )
        self.trials.append(trial)
        self._save()
        return trial

    def _save(self) -> None:
        config = self.experiment.config
        record: dict[str, Any] = {
            "session_id": self.id,
            "machine_name": config.machine_name,
            "experiment": config.name,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "scorer": {"name": config.scorer.name, "params": config.scorer.params},
            "crop": {"x": self.crop.x, "y": self.crop.y, "w": self.crop.w, "h": self.crop.h},
            "stimuli": [{"path": s.path.as_posix(), "md5": s.md5} for s in self.experiment.stimuli],
            "trials": [
                {
                    "stimulus": t.stimulus.path.as_posix(),
                    "face": t.face_file,
                    "captured_at": t.captured_at.isoformat(timespec="seconds"),
                    "score": t.score,
                    "message": t.message,
                    "blendshapes": dict(t.blendshapes),
                    "target_blendshapes": dict(t.stimulus.blendshapes),
                }
                for t in self.trials
            ],
        }
        text = json.dumps(record, ensure_ascii=False, indent=2)
        (self.dir / f"{self.id}.json").write_text(text, encoding="utf-8")
