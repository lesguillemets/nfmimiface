"""Static settings. Could later be overridden by command-line arguments."""

from pathlib import Path

# Assumes the package is run from the source tree (`uv run`, editable install): src/nfmimiface/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "data" / "face_landmarker.task"
DEFAULT_EXPERIMENT = PROJECT_ROOT / "experiments" / "default.toml"
RESULTS_ROOT = PROJECT_ROOT / "results"  # one subdirectory per day, e.g. results/20261116/
