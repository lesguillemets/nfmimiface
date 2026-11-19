"""Scoring: how well the participant's blendshapes match the target's.

A `Scorer` is any callable `(participant, target) -> float` returning a score in 0..100.
Scorers are created by *factories* registered under a name; the experiment TOML selects one:

    [scorer]
    name = "cosine"
    params = { exclude = ["_neutral", "eyeBlinkLeft", "eyeBlinkRight"] }

The vector scorers (cosine, pearson, overlap, shape_strength) take either `exclude`
(blendshapes to ignore) or `include` (the only blendshapes to use); given both, `include`
wins.

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

import inspect
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from nfmimiface.config import ScorerConfig
from nfmimiface.face import BLENDSHAPE_NAMES, Blendshapes, blendshape_indices

type Scorer = Callable[[Blendshapes, Blendshapes], float]
type ScorerFactory = Callable[..., Scorer]

_REGISTRY: dict[str, ScorerFactory] = {}

GAZE = tuple(n for n in BLENDSHAPE_NAMES if n.startswith("eyeLook"))
# Not part of the expression: _neutral is a constant-ish bias term, eyeLook* is gaze direction.
NON_EXPRESSIVE = ("_neutral", *GAZE)


def register(name: str) -> Callable[[ScorerFactory], ScorerFactory]:
    """Decorator registering a scorer factory under `name`."""

    def decorator(factory: ScorerFactory) -> ScorerFactory:
        if name in _REGISTRY:
            raise ValueError(f"scorer {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def _factory(name: str) -> ScorerFactory:
    if name not in _REGISTRY:
        raise ValueError(f"unknown scorer {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def make_scorer(config: ScorerConfig) -> Scorer:
    return _factory(config.name)(**config.params)


def scorer_names() -> list[str]:
    """Names of the registered scorers, in registration order."""
    return list(_REGISTRY)


def accepted_params(name: str) -> set[str]:
    """Names of the params the scorer `name` takes."""
    return set(inspect.signature(_factory(name)).parameters)


@register("cosine")
def cosine(exclude: Sequence[str] = ("_neutral",), include: Sequence[str] | None = None) -> Scorer:
    """100 × cosine similarity of the blendshape vectors, ignoring `exclude`.

    Blendshape scores are non-negative, so the result is already within 0..100.
    """
    keep = _kept(exclude, include)

    def score(participant: Blendshapes, target: Blendshapes) -> float:
        p, t = participant.array[keep], target.array[keep]
        norms = np.linalg.norm(p) * np.linalg.norm(t)
        return float(100 * p @ t / norms) if norms > 0 else 0.0

    return score


def _kept(exclude: Sequence[str], include: Sequence[str] | None) -> npt.NDArray[np.bool_]:
    """Mask of the blendshapes in `include` if given, else of those not in `exclude`."""
    if include is not None:
        keep = np.zeros(len(BLENDSHAPE_NAMES), dtype=bool)
        keep[blendshape_indices(include)] = True
        return keep
    keep = np.ones(len(BLENDSHAPE_NAMES), dtype=bool)
    keep[blendshape_indices(exclude)] = False
    return keep


@register("pearson")
def pearson(
    exclude: Sequence[str] = NON_EXPRESSIVE, include: Sequence[str] | None = None
) -> Scorer:
    """100 × Pearson correlation (negative clipped to 0): cosine after subtracting each
    face's mean, so a general level of activity matching does not count by itself."""
    keep = _kept(exclude, include)

    def score(participant: Blendshapes, target: Blendshapes) -> float:
        p, t = participant.array[keep], target.array[keep]
        p, t = p - p.mean(), t - t.mean()
        norms = np.linalg.norm(p) * np.linalg.norm(t)
        return float(100 * max(p @ t / norms, 0.0)) if norms > 0 else 0.0

    return score


@register("overlap")
def overlap(
    exclude: Sequence[str] = NON_EXPRESSIVE, include: Sequence[str] | None = None
) -> Scorer:
    """100 × Σmin(p, t) / Σmax(p, t) (weighted Jaccard, a.k.a. Ruzicka similarity).

    Unlike cosine, intensity matters: half the target's smile scores about 50, and a
    neutral participant scores low against an expressive target.
    """
    keep = _kept(exclude, include)

    def score(participant: Blendshapes, target: Blendshapes) -> float:
        p, t = participant.array[keep], target.array[keep]
        union = np.maximum(p, t).sum()
        return float(100 * np.minimum(p, t).sum() / union) if union > 0 else 0.0

    return score


def _starting(*prefixes: str) -> tuple[str, ...]:
    return tuple(n for n in BLENDSHAPE_NAMES if n.startswith(prefixes))


REGIONS: dict[str, tuple[str, ...]] = {
    "brows": _starting("brow"),
    "eyes": _starting("eyeBlink", "eyeSquint", "eyeWide"),
    "cheeks_nose": _starting("cheek", "nose"),
    "mouth": _starting("mouth"),
    "jaw": _starting("jaw"),
}


@register("regions")
def regions(full_activity: float = 0.5) -> Scorer:
    """Weighted Jaccard (see `overlap`) per facial region, averaged over regions.

    A region's weight is its activity (Σmax(p, t) over its blendshapes) capped at
    `full_activity`, so a clearly moving region counts fully however many blendshapes it
    has (the mouth has 23, the brows 5) and quiet regions barely count. Gaze is ignored.
    """
    indices = [blendshape_indices(names) for names in REGIONS.values()]

    def score(participant: Blendshapes, target: Blendshapes) -> float:
        total = weights = 0.0
        for idx in indices:
            p, t = participant.array[idx], target.array[idx]
            union = np.maximum(p, t).sum()
            if union > 0:
                weight = min(union, full_activity)
                total += weight * np.minimum(p, t).sum() / union
                weights += weight
        return float(100 * total / weights) if weights > 0 else 0.0

    return score


def _both_sides(name: str) -> tuple[str, ...]:
    return (name,) if name in BLENDSHAPE_NAMES else (f"{name}Left", f"{name}Right")


# Turrisi et al. (2026), "Blendshape features meet action units: a clinical mapping ...",
# Computers in Human Behavior Reports 22, 101125: an expert-consensus mapping from
# MediaPipe blendshapes to FACS action units (Fig. 4a) and from action units to the basic
# emotions (Fig. 4b). Both are followed as published; lateralised action units are
# averaged over both sides. Only the action units used by an emotion are listed.
ACTION_UNITS: dict[int, tuple[str, ...]] = {
    1: _both_sides("browInnerUp"),  # inner brow raiser
    2: _both_sides("browOuterUp"),  # outer brow raiser
    4: _both_sides("browDown"),  # brow lowerer
    5: _both_sides("eyeWide"),  # upper lid raiser
    6: _both_sides("cheekSquint"),  # cheek raiser
    7: _both_sides("eyeSquint"),  # lid tightener
    9: _both_sides("noseSneer"),  # nose wrinkler
    12: _both_sides("mouthSmile"),  # lip corner puller
    15: _both_sides("mouthFrown"),  # lip corner depressor
    17: _both_sides("mouthShrugLower"),  # chin raiser
    20: _both_sides("mouthStretch"),  # lip stretcher
    26: _both_sides("jawOpen"),  # jaw drop
}

# Contempt (one-sided 12+14) is left out: the profile averages both sides, so it cannot
# express a one-sided movement.
EMOTIONS: dict[str, tuple[int, ...]] = {
    "happiness": (6, 12),
    "sadness": (1, 4, 15),
    "surprise": (1, 2, 5, 26),
    "fear": (1, 2, 4, 5, 7, 20, 26),
    "anger": (4, 5, 7),
    "disgust": (9, 15, 17),
}


@register("emotions")
def emotions(action_units: Mapping[str, Sequence[str]] | None = None) -> Scorer:
    """Weighted Jaccard of basic-emotion profiles.

    Each emotion's evidence is the mean intensity of its prototypical action units;
    the two faces' 6-emotion profiles are compared like in `overlap`. Only blendshapes
    used by some prototype matter: it judges *which emotion* is shown, and cannot tell
    apart faces outside the basic emotions (puffed cheeks, pouting, winking, ...).

    `action_units` replaces entries of ACTION_UNITS, keyed by the action unit's number
    as a string (TOML keys are strings), e.g. {"6": ["eyeSquintLeft", "eyeSquintRight"]}.
    An empty list removes that action unit from the emotions.
    """
    overrides = {int(au): tuple(names) for au, names in (action_units or {}).items()}
    mapping = ACTION_UNITS | overrides
    au_idx = {au: blendshape_indices(names) for au, names in mapping.items() if names}
    emotion_aus = [[au for au in aus if au in au_idx] for aus in EMOTIONS.values()]

    def profile(b: Blendshapes) -> npt.NDArray[np.float64]:
        intensity = {au: b.array[idx].mean() for au, idx in au_idx.items()}
        return np.array([np.mean([intensity[au] for au in aus]) for aus in emotion_aus])

    def score(participant: Blendshapes, target: Blendshapes) -> float:
        p, t = profile(participant), profile(target)
        union = np.maximum(p, t).sum()
        return float(100 * np.minimum(p, t).sum() / union) if union > 0 else 0.0

    return score


@register("shape_strength")
def shape_strength(
    strength_weight: float = 0.5,
    exclude: Sequence[str] = NON_EXPRESSIVE,
    include: Sequence[str] | None = None,
) -> Scorer:
    """100 × cosine × (smaller norm / larger norm) ** strength_weight.

    Separates *which* movements (cosine: the direction of the vector) from *how strongly*
    (the ratio of the vectors' lengths). With the default 0.5, a half-strength imitation
    of the right movements scores 71; 0 ignores strength (plain cosine), 1 is strict.
    A quiet participant scores low, since its vector is short.
    """
    keep = _kept(exclude, include)

    def score(participant: Blendshapes, target: Blendshapes) -> float:
        p, t = participant.array[keep], target.array[keep]
        np_, nt = np.linalg.norm(p), np.linalg.norm(t)
        if np_ == 0 or nt == 0:
            return 0.0
        return float(100 * (p @ t / (np_ * nt)) * (min(np_, nt) / max(np_, nt)) ** strength_weight)

    return score
