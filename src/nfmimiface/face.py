"""Face analysis with mediapipe's FaceLandmarker: face box and blendshape scores."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import mediapipe as mp
import numpy as np
import numpy.typing as npt
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult

from nfmimiface.constants import MODEL_PATH
from nfmimiface.images import RGBImage

# In the model's output order: BLENDSHAPE_NAMES[i] is the category with index i.
BLENDSHAPE_NAMES: tuple[str, ...] = (
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)  # fmt: skip
_INDEX = {name: i for i, name in enumerate(BLENDSHAPE_NAMES)}


def blendshape_indices(names: Iterable[str]) -> npt.NDArray[np.intp]:
    """Indices of `names` in BLENDSHAPE_NAMES. Raises KeyError on an unknown name."""
    return np.array([_INDEX[n] for n in names], dtype=np.intp)


class Blendshapes(Mapping[str, float]):
    """Immutable mapping of blendshape name -> score (0..1), in BLENDSHAPE_NAMES order."""

    def __init__(self, scores: npt.ArrayLike) -> None:
        array = np.array(scores, dtype=np.float64)
        if array.shape != (len(BLENDSHAPE_NAMES),):
            raise ValueError(f"expected {len(BLENDSHAPE_NAMES)} scores, got shape {array.shape}")
        array.flags.writeable = False
        self._scores = array

    def __getitem__(self, name: str) -> float:
        return float(self._scores[_INDEX[name]])

    def __iter__(self) -> Iterator[str]:
        return iter(BLENDSHAPE_NAMES)

    def __len__(self) -> int:
        return len(BLENDSHAPE_NAMES)

    def __repr__(self) -> str:
        return f"Blendshapes({dict(self)!r})"

    @property
    def array(self) -> npt.NDArray[np.float64]:
        """Read-only array of scores, in BLENDSHAPE_NAMES order (see blendshape_indices)."""
        return self._scores


@dataclass(frozen=True)
class Rect:
    """Pixel rectangle."""

    x: int
    y: int
    w: int
    h: int

    def crop(self, image: RGBImage) -> RGBImage:
        return image[self.y : self.y + self.h, self.x : self.x + self.w]


class FaceAnalyzer:
    """Wraps FaceLandmarker in IMAGE mode. Assumes one face per image (the first is used).

    Call close() (or use as a context manager) when done: mediapipe's own cleanup fails
    if left to interpreter shutdown.
    """

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def face_box(self, image: RGBImage) -> Rect | None:
        """Bounding box of the face landmarks (may extend past the image), or None if no face."""
        result = self._detect(image)
        if not result.face_landmarks:
            return None
        img_h, img_w = image.shape[:2]
        xs = [lm.x * img_w for lm in result.face_landmarks[0]]
        ys = [lm.y * img_h for lm in result.face_landmarks[0]]
        return Rect(int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))

    def blendshapes(self, image: RGBImage) -> Blendshapes | None:
        """Blendshape scores of the face, or None if no face."""
        result = self._detect(image)
        if not result.face_blendshapes:
            return None
        return Blendshapes([c.score for c in result.face_blendshapes[0]])

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _detect(self, image: RGBImage) -> FaceLandmarkerResult:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image))
        return self._landmarker.detect(mp_image)


def crop_rect(face: Rect, frame_w: int, frame_h: int, scale: float, aspect: float) -> Rect:
    """Rectangle centred on `face`, `scale` times the face height, with width/height == aspect.

    Shrunk (keeping the aspect ratio) if larger than the frame, then shifted to lie within it.
    """
    h = face.h * scale
    w = h * aspect
    fit = min(1.0, frame_w / w, frame_h / h)
    w, h = int(w * fit), int(h * fit)
    cx, cy = face.x + face.w / 2, face.y + face.h / 2
    x = int(min(max(cx - w / 2, 0), frame_w - w))
    y = int(min(max(cy - h / 2, 0), frame_h - h))
    return Rect(x, y, w, h)
