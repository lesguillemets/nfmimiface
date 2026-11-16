from nfmimiface.config import load_config
from nfmimiface.constants import DEFAULT_EXPERIMENT


def test_default_experiment_loads() -> None:
    config = load_config(DEFAULT_EXPERIMENT)
    assert config.stimuli
    assert config.stimulus_side in ("left", "right")


def test_message_for_picks_highest_reached_threshold() -> None:
    config = load_config(DEFAULT_EXPERIMENT)
    assert config.message_for(100.0) == config.messages[0].text
    assert config.message_for(-1.0) == config.messages[-1].text
    for m in config.messages:
        assert config.message_for(m.min_score) == m.text
