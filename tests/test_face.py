import numpy as np
import pytest

from nfmimiface.constants import MODEL_PATH
from nfmimiface.face import (
    BLENDSHAPE_NAMES,
    Blendshapes,
    FaceAnalyzer,
    Rect,
    blendshape_indices,
    crop_rect,
)


def test_blendshapes_by_name() -> None:
    b = Blendshapes(np.arange(len(BLENDSHAPE_NAMES)) / 100)
    assert b["jawOpen"] == 0.25
    assert list(b) == list(BLENDSHAPE_NAMES)


def test_blendshape_indices() -> None:
    assert list(blendshape_indices(["_neutral", "jawOpen"])) == [0, 25]
    with pytest.raises(KeyError):
        blendshape_indices(["_nuetral"])


def test_crop_rect_centred_with_aspect() -> None:
    r = crop_rect(Rect(400, 300, 100, 100), frame_w=1000, frame_h=1000, scale=2.0, aspect=0.75)
    assert (r.w, r.h) == (150, 200)
    assert (r.x + r.w / 2, r.y + r.h / 2) == (450, 350)


def test_crop_rect_stays_within_frame() -> None:
    r = crop_rect(Rect(0, 0, 100, 400), frame_w=640, frame_h=480, scale=2.0, aspect=1.0)
    assert r.h == 480 and r.w == 480
    assert r.x >= 0 and r.y >= 0 and r.x + r.w <= 640 and r.y + r.h <= 480


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model file not present")
def test_no_face_in_blank_image() -> None:
    with FaceAnalyzer() as analyzer:
        assert analyzer.blendshapes(np.zeros((480, 640, 3), dtype=np.uint8)) is None
