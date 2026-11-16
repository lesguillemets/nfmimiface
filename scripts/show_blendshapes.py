"""Print the face box and blendshape scores of image files.

Usage: uv run scripts/show_blendshapes.py IMAGE [IMAGE ...]
"""

import sys
from pathlib import Path

from nfmimiface.face import FaceAnalyzer
from nfmimiface.images import load_rgb


def main(paths: list[Path]) -> None:
    with FaceAnalyzer() as analyzer:
        for path in paths:
            image = load_rgb(path)
            print(f"== {path} ({image.shape[1]}x{image.shape[0]})")
            blendshapes = analyzer.blendshapes(image)
            if blendshapes is None:
                print("  no face detected")
                continue
            print(f"  face box: {analyzer.face_box(image)}")
            for name, score in sorted(blendshapes.items(), key=lambda kv: -kv[1]):
                print(f"  {name:<20} {score:.3f}")


if __name__ == "__main__":
    main([Path(p) for p in sys.argv[1:]])
