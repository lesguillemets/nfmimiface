from nfmimiface.face import BLENDSHAPE_NAMES
from nfmimiface.radar import axis_names
from nfmimiface.scoring import NON_EXPRESSIVE


def test_axes_follow_the_scorer_params() -> None:
    assert axis_names({"include": ["jawOpen", "browInnerUp"]}) == ["browInnerUp", "jawOpen"]
    assert axis_names({"exclude": ["_neutral"]}) == list(BLENDSHAPE_NAMES[1:])
    assert not set(axis_names({})) & set(NON_EXPRESSIVE)
