"""Compare scorers on a labelled set of face images.

    uv run nfmimiface-compare-scorers [DIR] [--matrix]

DIR (default data/expressions) holds one subdirectory per expression, e.g.
DIR/smiling/a.jpg. Every image is scored against every other one (as participant,
target). A good scorer gives same-expression pairs higher scores than different ones;
`auc` is the probability that a random same pair beats a random different pair
(0.5 = no better than chance, 1 = perfect separation).

Candidates are listed in CANDIDATES below; edit it while developing scorers.
Caveat: the faces belong to different people, whereas a real trial compares one
participant against a stimulus.
"""

import argparse
from itertools import product
from pathlib import Path

import numpy as np

from nfmimiface.config import ScorerConfig
from nfmimiface.constants import PROJECT_ROOT
from nfmimiface.face import Blendshapes, FaceAnalyzer
from nfmimiface.images import load_rgb
from nfmimiface.scoring import Scorer, make_scorer

CANDIDATES: dict[str, ScorerConfig] = {
    "cosine": ScorerConfig("cosine"),
}

type Face = tuple[str, str, Blendshapes]  # (label, expression, blendshapes)


def load_faces(root: Path) -> list[Face]:
    faces: list[Face] = []
    with FaceAnalyzer() as analyzer:
        for path in sorted(p for p in root.glob("*/*") if p.is_file()):
            label = f"{path.parent.name}/{path.stem}"
            if (blendshapes := analyzer.blendshapes(load_rgb(path))) is None:
                print(f"{label}: no face detected, skipped")
            else:
                faces.append((label, path.parent.name, blendshapes))
    return faces


def print_matrix(scorer: Scorer, faces: list[Face]) -> None:
    """Rows: participant, columns: target."""
    print(" " * 4 + "".join(f"{j:>6}" for j in range(len(faces))))
    for i, (label, _, p) in enumerate(faces):
        row = "".join(f"{scorer(p, t):6.1f}" for _, _, t in faces)
        print(f"{i:>4}{row}  {label}")


def summary(scorer: Scorer, faces: list[Face]) -> str:
    same: list[float] = []
    diff: list[float] = []
    for (_, ep, p), (_, et, t) in product(faces, repeat=2):
        if p is not t:
            (same if ep == et else diff).append(scorer(p, t))
    s, d = np.array(same), np.array(diff)
    auc = ((s[:, None] > d[None, :]).mean() + 0.5 * (s[:, None] == d[None, :]).mean()).item()
    return (
        f"same {s.mean():5.1f} (min {s.min():5.1f})  "
        f"diff {d.mean():5.1f} (max {d.max():5.1f})  auc {auc:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare scorers on labelled face images.")
    parser.add_argument("root", nargs="?", type=Path, default=PROJECT_ROOT / "data/expressions")
    parser.add_argument("--matrix", action="store_true", help="also print pairwise scores")
    args = parser.parse_args()

    faces = load_faces(args.root)
    width = max(map(len, CANDIDATES))
    for name, config in CANDIDATES.items():
        scorer = make_scorer(config)
        if args.matrix:
            print(f"== {name}")
            print_matrix(scorer, faces)
        print(f"{name:<{width}}  {summary(scorer, faces)}")


if __name__ == "__main__":
    main()
