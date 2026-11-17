"""Command-line tool to inspect blendshapes of images, or score them against a target.

    uv run nfmimiface-blendshapes IMAGE [IMAGE ...]               # print blendshapes
    uv run nfmimiface-blendshapes IMAGE [IMAGE ...] --target T    # score each against T

Scoring uses the scorer configured in the experiment TOML (`--experiment`).
Images are used as they are (not mirrored).
"""

import argparse
from pathlib import Path

from nfmimiface.config import load_config
from nfmimiface.constants import DEFAULT_EXPERIMENT
from nfmimiface.face import Blendshapes, FaceAnalyzer
from nfmimiface.images import load_rgb
from nfmimiface.scoring import make_scorer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print blendshapes of images, or score them against --target."
    )
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--target", type=Path, help="score each image against this one")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    args = parser.parse_args()

    with FaceAnalyzer() as analyzer:

        def analyze(path: Path) -> Blendshapes | None:
            blendshapes = analyzer.blendshapes(load_rgb(path))
            if blendshapes is None:
                print(f"{path}: no face detected")
            return blendshapes

        if args.target is None:
            for path in args.images:
                if (blendshapes := analyze(path)) is not None:
                    print(f"== {path}")
                    for name, value in sorted(blendshapes.items(), key=lambda kv: -kv[1]):
                        print(f"  {name:<20} {value:.3f}")
            return

        config = load_config(args.experiment).scorer
        scorer = make_scorer(config)
        if (target := analyze(args.target)) is None:
            return
        print(f"scorer: {config.name} {config.params}, target: {args.target}")
        for path in args.images:
            if (blendshapes := analyze(path)) is not None:
                print(f"  {scorer(blendshapes, target):6.1f}  {path}")


if __name__ == "__main__":
    main()
