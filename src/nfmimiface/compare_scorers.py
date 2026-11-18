"""Compare scorers on a labelled set of face images.

    uv run nfmimiface-compare-scorers [DIR] [--mirror] [--matrix] [--edits]

DIR (default data/expressions) holds one subdirectory per expression, e.g.
DIR/smiling/a.jpg. Every image is scored against every other one (as participant,
target). Printed per scorer:

- same / diff: mean score of same-expression / different-expression pairs, with the
  worst case of each (min of same, max of diff).
- auc: probability that a random same pair beats a random different pair
  (0.5 = no better than chance, 1 = perfect separation).
- neutral: if DIR/neutral/ exists, its faces act as participants only (never targets):
  mean (max) score of a neutral face against the expressive targets. Lower is better;
  it shows how much a participant gets for doing nothing.
- mirror (with --mirror): every image is also analysed flipped left-right, as an extra
  participant. mean (min) score of each face against its own mirror image; 100 means
  the scorer ignores which side of the face moves. Pairs from the same file are never
  counted as same/diff pairs.

With --edits, each target is also scored against controlled edits of itself (see
EDITS): a weaker or partial imitation, a quiet face, the gaze moved, ... These show how a
scorer behaves in situations we can reason about, independent of the photos' labels.

Candidates are listed in CANDIDATES below; edit it while developing scorers.
Caveat: the faces belong to different people, whereas a real trial compares one
participant against a stimulus.
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nfmimiface.config import ScorerConfig
from nfmimiface.constants import PROJECT_ROOT
from nfmimiface.face import BLENDSHAPE_NAMES, Blendshapes, FaceAnalyzer, blendshape_indices
from nfmimiface.images import load_rgb
from nfmimiface.scoring import GAZE, NON_EXPRESSIVE, REGIONS, Scorer, make_scorer

CANDIDATES: dict[str, ScorerConfig] = {
    "cosine": ScorerConfig("cosine"),
    "cosine-nogaze": ScorerConfig("cosine", {"exclude": NON_EXPRESSIVE}),
    "pearson": ScorerConfig("pearson"),
    "overlap": ScorerConfig("overlap"),
    "shape_strength": ScorerConfig("shape_strength"),
    "regions": ScorerConfig("regions"),
    "emotions": ScorerConfig("emotions"),
}

NEUTRAL = "neutral"

# A face at rest: small, slightly uneven values (most blendshapes are < 0.02 when inactive).
QUIET = np.random.default_rng(0).uniform(0, 0.02, len(BLENDSHAPE_NAMES))


def _only(names: tuple[str, ...]) -> Callable[[Blendshapes], Blendshapes]:
    """Edit keeping `names` and setting the other blendshapes to QUIET."""
    idx = blendshape_indices(names)

    def edit(b: Blendshapes) -> Blendshapes:
        x = QUIET.copy()
        x[idx] = b.array[idx]
        return Blendshapes(x)

    return edit


def _look_aside(b: Blendshapes) -> Blendshapes:
    x = b.array.copy()
    x[blendshape_indices(GAZE)] = 0
    x[blendshape_indices(["eyeLookOutLeft", "eyeLookInRight"])] = 0.7
    return Blendshapes(x)


# name -> (edit of the target used as the participant, what an ideal scorer would give)
EDITS: dict[str, tuple[Callable[[Blendshapes], Blendshapes], str]] = {
    "weaker": (lambda b: Blendshapes(b.array * 0.5), "partial"),
    "stronger": (lambda b: Blendshapes(np.minimum(b.array * 1.5, 1)), "high"),
    "mouth": (_only(REGIONS["mouth"] + REGIONS["jaw"]), "partial"),
    "upper": (_only(REGIONS["brows"] + REGIONS["eyes"]), "partial"),
    "gaze": (_look_aside, "100"),
    "quiet": (_only(()), "0"),
}


@dataclass(frozen=True)
class Face:
    label: str
    expression: str
    source: Path
    mirrored: bool
    blendshapes: Blendshapes


def load_faces(root: Path, mirror: bool) -> list[Face]:
    faces: list[Face] = []
    with FaceAnalyzer() as analyzer:
        for path in sorted(p for p in root.glob("*/*") if p.is_file()):
            label, image = f"{path.parent.name}/{path.stem}", load_rgb(path)
            for flipped in (False, True) if mirror else (False,):
                blendshapes = analyzer.blendshapes(np.fliplr(image) if flipped else image)
                if blendshapes is None:
                    print(f"{label}: no face detected{' (mirrored)' * flipped}, skipped")
                    continue
                name = label + " (m)" * flipped
                faces.append(Face(name, path.parent.name, path, flipped, blendshapes))
    return faces


def print_matrix(scorer: Scorer, faces: list[Face]) -> None:
    """Rows: participant, columns: target (original images only)."""
    targets = [f for f in faces if not f.mirrored and f.expression != NEUTRAL]
    print(" " * 4 + "".join(f"{j:>6}" for j in range(len(targets))))
    for i, p in enumerate(faces):
        row = "".join(f"{scorer(p.blendshapes, t.blendshapes):6.1f}" for t in targets)
        print(f"{i:>4}{row}  {p.label}")


def summary(scorer: Scorer, faces: list[Face]) -> str:
    def score(p: Face, t: Face) -> float:
        return scorer(p.blendshapes, t.blendshapes)

    targets = [f for f in faces if not f.mirrored and f.expression != NEUTRAL]
    same: list[float] = []
    diff: list[float] = []
    neutral: list[float] = []
    for p in faces:
        for t in targets:
            if p.source == t.source:
                continue
            if p.expression == NEUTRAL:
                neutral.append(score(p, t))
            else:
                (same if p.expression == t.expression else diff).append(score(p, t))
    s, d = np.array(same), np.array(diff)
    auc = ((s[:, None] > d[None, :]).mean() + 0.5 * (s[:, None] == d[None, :]).mean()).item()
    text = (
        f"same {s.mean():5.1f} ({s.min():5.1f})  "
        f"diff {d.mean():5.1f} ({d.max():5.1f})  auc {auc:.2f}"
    )
    if neutral:
        text += f"  neutral {np.mean(neutral):5.1f} ({np.max(neutral):5.1f})"
    originals = {f.source: f for f in faces if not f.mirrored}
    if mirrored := [score(f, originals[f.source]) for f in faces if f.mirrored]:
        text += f"  mirror {np.mean(mirrored):5.1f} ({np.min(mirrored):5.1f})"
    return text


def print_edits(scorers: dict[str, Scorer], faces: list[Face]) -> None:
    """Mean score of each edit (as participant) against the unedited target."""
    targets = [f.blendshapes for f in faces if not f.mirrored and f.expression != NEUTRAL]
    width = max(map(len, scorers))
    print(f"{'edit:':<{width}}" + "".join(f"{e:>9}" for e in EDITS))
    print(f"{'ideal:':<{width}}" + "".join(f"{ideal:>9}" for _, ideal in EDITS.values()))
    for name, scorer in scorers.items():
        means = [np.mean([scorer(edit(t), t) for t in targets]) for edit, _ in EDITS.values()]
        print(f"{name:<{width}}" + "".join(f"{m:9.1f}" for m in means))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scorers on labelled face images.")
    parser.add_argument("root", nargs="?", type=Path, default=PROJECT_ROOT / "data/expressions")
    parser.add_argument("--mirror", action="store_true", help="also use left-right flipped images")
    parser.add_argument("--matrix", action="store_true", help="also print pairwise scores")
    parser.add_argument("--edits", action="store_true", help="also score edits of each target")
    args = parser.parse_args()

    faces = load_faces(args.root, args.mirror)
    scorers = {name: make_scorer(config) for name, config in CANDIDATES.items()}
    width = max(map(len, scorers))
    for name, scorer in scorers.items():
        if args.matrix:
            print(f"== {name}")
            print_matrix(scorer, faces)
        print(f"{name:<{width}}  {summary(scorer, faces)}")
    if args.edits:
        print()
        print_edits(scorers, faces)


if __name__ == "__main__":
    main()
