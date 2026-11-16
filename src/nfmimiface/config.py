"""Experiment configuration, loaded from a TOML file.

Validation is intentionally minimal: a missing key raises KeyError, which is enough
for a file we write ourselves.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

from nfmimiface.constants import PROJECT_ROOT

type Side = Literal["left", "right"]


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class ScorerConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class StimulusConfig:
    path: Path  # as written in the TOML; relative to the project root
    instruction: str

    @property
    def resolved_path(self) -> Path:
        return PROJECT_ROOT / self.path


@dataclass(frozen=True)
class ScoreMessage:
    min_score: float
    text: str


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    machine_name: str
    stimulus_side: Side
    crop_scale: float  # crop size relative to the detected face box
    closing_message: str
    camera: CameraConfig
    scorer: ScorerConfig
    stimuli: tuple[StimulusConfig, ...]
    messages: tuple[ScoreMessage, ...]


def load_config(path: Path) -> ExperimentConfig:
    with path.open("rb") as f:
        raw = tomllib.load(f)

    side = raw["stimulus_side"]
    if side not in ("left", "right"):
        raise ValueError(f"stimulus_side must be 'left' or 'right', got {side!r}")

    stimuli = tuple(
        StimulusConfig(path=Path(s["path"]), instruction=s["instruction"]) for s in raw["stimuli"]
    )
    messages = tuple(
        ScoreMessage(min_score=float(m["min_score"]), text=m["text"]) for m in raw["messages"]
    )
    if not stimuli or not messages:
        raise ValueError("at least one [[stimuli]] and one [[messages]] entry are required")

    return ExperimentConfig(
        name=raw["name"],
        machine_name=raw["machine_name"],
        stimulus_side=cast(Side, side),
        crop_scale=float(raw["crop_scale"]),
        closing_message=raw["closing_message"],
        camera=CameraConfig(**raw.get("camera", {})),
        scorer=ScorerConfig(name=raw["scorer"]["name"], params=raw["scorer"].get("params", {})),
        stimuli=stimuli,
        messages=messages,
    )
