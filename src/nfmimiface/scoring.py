"""Scoring: how well the participant's blendshapes match the target's.

A `Scorer` is any callable `(participant, target) -> float` returning a score in 0..100.
Scorers are created by *factories* registered under a name; the experiment TOML selects one:

    [scorer]
    name = "cosine"
    params = { exclude = ["_neutral", "eyeBlinkLeft", "eyeBlinkRight"] }

`params` are passed to the factory as keyword arguments.

To add a scorer, define a factory in this module (or in a module imported by it) and
decorate it with `@register`:

    @register("weighted_l1")
    def weighted_l1(weights: dict[str, float]) -> Scorer:
        # Done once, at startup: turn names into indices (unknown names raise KeyError).
        idx = blendshape_indices(weights)
        w = np.array(list(weights.values()))

        # Called per trial: only array arithmetic.
        def score(participant: Blendshapes, target: Blendshapes) -> float:
            diff = np.abs(participant.array[idx] - target.array[idx])
            return float(100 * (1 - (w * diff).sum() / w.sum()))

        return score

Blendshapes are accessed by name in configs and by index (`Blendshapes.array`) in
computations; `blendshape_indices` converts between the two.

Scores are kept unrounded; rounding is a display concern.
"""

from collections.abc import Callable, Sequence

import numpy as np

from nfmimiface.config import ScorerConfig
from nfmimiface.face import BLENDSHAPE_NAMES, Blendshapes, blendshape_indices

type Scorer = Callable[[Blendshapes, Blendshapes], float]
type ScorerFactory = Callable[..., Scorer]

_REGISTRY: dict[str, ScorerFactory] = {}


def register(name: str) -> Callable[[ScorerFactory], ScorerFactory]:
    """Decorator registering a scorer factory under `name`."""

    def decorator(factory: ScorerFactory) -> ScorerFactory:
        if name in _REGISTRY:
            raise ValueError(f"scorer {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def make_scorer(config: ScorerConfig) -> Scorer:
    if config.name not in _REGISTRY:
        raise ValueError(f"unknown scorer {config.name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[config.name](**config.params)


@register("cosine")
def cosine(exclude: Sequence[str] = ("_neutral",)) -> Scorer:
    """100 × cosine similarity of the blendshape vectors, ignoring `exclude`.

    Blendshape scores are non-negative, so the result is already within 0..100.
    """
    keep = np.ones(len(BLENDSHAPE_NAMES), dtype=bool)
    keep[blendshape_indices(exclude)] = False

    def score(participant: Blendshapes, target: Blendshapes) -> float:
        p, t = participant.array[keep], target.array[keep]
        norms = np.linalg.norm(p) * np.linalg.norm(t)
        return float(100 * p @ t / norms) if norms > 0 else 0.0

    return score
