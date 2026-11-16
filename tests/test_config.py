from nfmimiface.config import load_config
from nfmimiface.constants import DEFAULT_EXPERIMENT


def test_default_experiment_loads() -> None:
    config = load_config(DEFAULT_EXPERIMENT)
    assert config.stimuli
    assert config.messages
    assert config.stimulus_side in ("left", "right")
