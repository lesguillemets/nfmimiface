"""Image type and file I/O.

Throughout the app, images are RGB uint8 arrays of shape (H, W, 3).
"""

from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

type RGBImage = npt.NDArray[np.uint8]


def load_rgb(path: Path) -> RGBImage:
    # np.fromfile + imdecode instead of cv2.imread, which fails on non-ASCII paths on Windows
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image: {path}")
    return cast(RGBImage, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def save_png(path: Path, image: RGBImage) -> None:
    # imencode + tofile instead of cv2.imwrite, for the same reason as in load_rgb
    ok, buffer = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"cannot encode image for {path}")
    buffer.tofile(path)
