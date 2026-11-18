import numpy as np

from nfmimiface.images import qr_image


def test_qr_image() -> None:
    image = qr_image("https://example.com/?id=pc1_0001_abcde", module_px=3)
    assert image.ndim == 3 and image.shape[0] == image.shape[1] and image.shape[2] == 3
    assert set(np.unique(image)) == {0, 255}
    assert (image[:12] == 255).all()  # quiet zone: 4 modules of 3 px
