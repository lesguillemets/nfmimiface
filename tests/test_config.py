from nfmimiface.config import load_config
from nfmimiface.constants import DEFAULT_EXPERIMENT


def test_default_experiment_loads() -> None:
    config = load_config(DEFAULT_EXPERIMENT)
    assert config.stimuli
    assert config.messages
    assert config.stimulus_side in ("left", "right")


def test_form_url_for() -> None:
    form = load_config(DEFAULT_EXPERIMENT).form
    assert form is not None
    url = form.url_for("pc1_0001_abcde")
    assert url.endswith("=pc1_0001_abcde") and "{" not in url
