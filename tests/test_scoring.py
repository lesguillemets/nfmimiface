import numpy as np
import pytest

from nfmimiface.config import ScorerConfig
from nfmimiface.face import BLENDSHAPE_NAMES, Blendshapes
from nfmimiface.scoring import make_scorer

rng = np.random.default_rng(0)


def random_blendshapes() -> Blendshapes:
    return Blendshapes(rng.random(len(BLENDSHAPE_NAMES)))


def test_identical_faces_score_100() -> None:
    b = random_blendshapes()
    assert make_scorer(ScorerConfig("cosine"))(b, b) == pytest.approx(100)


def test_params_reach_the_factory() -> None:
    # Only jawOpen is compared, so otherwise unrelated faces score 100.
    exclude = [n for n in BLENDSHAPE_NAMES if n != "jawOpen"]
    scorer = make_scorer(ScorerConfig("cosine", {"exclude": exclude}))
    assert scorer(random_blendshapes(), random_blendshapes()) == pytest.approx(100)


def test_unknown_scorer() -> None:
    with pytest.raises(ValueError, match="unknown scorer"):
        make_scorer(ScorerConfig("no-such-scorer"))


def test_misspelled_blendshape_name_fails_at_creation() -> None:
    with pytest.raises(KeyError):
        make_scorer(ScorerConfig("cosine", {"exclude": ["_nuetral"]}))


@pytest.mark.parametrize(
    "name", ["cosine", "pearson", "overlap", "shape_strength", "regions", "emotions"]
)
def test_every_scorer_gives_100_for_identical_faces(name: str) -> None:
    b = random_blendshapes()
    assert make_scorer(ScorerConfig(name))(b, b) == pytest.approx(100)


def test_overlap_counts_intensity() -> None:
    b = random_blendshapes()
    half = Blendshapes(b.array / 2)
    assert make_scorer(ScorerConfig("overlap"))(half, b) == pytest.approx(50)


def test_shape_strength_half_strength() -> None:
    b = random_blendshapes()
    scorer = make_scorer(ScorerConfig("shape_strength"))
    assert scorer(Blendshapes(b.array / 2), b) == pytest.approx(100 * 0.5**0.5)
