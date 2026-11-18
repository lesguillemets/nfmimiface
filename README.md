# nfmimiface

A facial expression imitation demo for an outreach event: a participant imitates a
pictured face in front of a webcam, and the program scores how close the imitation is.
See `project.md` for the requirements and `docs/` for the design.

## Quickstart

Commands are given for Linux (bash) and Windows (PowerShell). Run them all from the
project root, the directory that contains `pyproject.toml`.

### Requirements

- Python 3.12 or newer (3.12 is what we test with)
- A webcam
- About 1 GB of disk space for the dependencies

### 1. Download the face model

The program needs mediapipe's face landmarker model, which is not in the repository.

1. Open <https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker>.
2. In the "Models" section, download the FaceLandmarker model bundle (the "Latest"
   link in the "Versions" column). The file is named `face_landmarker.task`.
3. Put it at `data/face_landmarker.task` in the project directory.

### 2. Install

Choose one of uv, pip or conda.

The project must be installed in editable mode (`uv sync` does this; with pip, use
`pip install -e .`). The program finds `experiments/`, `stimuli/`, `data/` and
`results/` relative to the source tree, so a regular install will not work.

#### uv

```bash
uv sync
```

uv creates `.venv/` and installs everything there. Prefix commands with `uv run`
(for example `uv run nfmimiface`), and there is no need to activate the environment.

#### pip (venv)

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip uninstall -y opencv-contrib-python
pip install --force-reinstall --no-deps opencv-python-headless
```

Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
pip uninstall -y opencv-contrib-python
pip install --force-reinstall --no-deps opencv-python-headless
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

The last two `pip` lines are needed because mediapipe depends on `opencv-contrib-python`,
which installs the same `cv2` module as `opencv-python-headless` and, on Linux, bundles a
Qt that conflicts with PySide6. uv handles this automatically (see `[tool.uv]` in
`pyproject.toml`); pip cannot, so we remove that package by hand and reinstall the
headless build, whose files the removal deleted. Run these two lines again whenever
you re-run `pip install -e .`.

#### conda (Anaconda / Miniconda)

Use conda only to provide Python; install the packages with pip, because mediapipe and
PySide6 are taken from PyPI.

```bash
conda create -n nfmimiface python=3.12
conda activate nfmimiface
pip install -e .
pip uninstall -y opencv-contrib-python
pip install --force-reinstall --no-deps opencv-python-headless
```

On Windows, run these in the Anaconda Prompt. The last two lines are explained in the
pip section above.

### 3. Add the stimuli

Put the face pictures to imitate in `stimuli/` and list them in the experiment file
(next section). `stimuli/` is not version-controlled, so it is empty after cloning.

### 4. Run

```bash
nfmimiface                                         # uses experiments/default.toml
nfmimiface --experiment experiments/another.toml   # uses another experiment file
```

With uv, write `uv run nfmimiface` instead. With pip or conda, activate the
environment first. Results are saved under `results/`, one subdirectory per day.

Without a webcam, `--fake-camera IMAGE` shows a still image in place of the camera.

### Files you will edit

- `experiments/`: one TOML file per experiment set-up. `default.toml` is used when
  `--experiment` is not given. It sets the experiment name, the machine name (used in
  session IDs), the camera, the stimuli and their order, the scorer, the feedback
  messages by score, and the form shown on the closing page. The comments in
  `default.toml` describe each entry.
- `stimuli/`: the pictures of the faces to imitate. Each picture should show one clearly
  visible face; the program refuses to start if it finds no face in a stimulus. In the TOML, refer to them by paths relative to the project
  root, for example `path = "stimuli/face01.jpg"`.

### Uninstall

With uv and pip, everything is installed in `.venv/` inside the project directory, so
deleting the project directory removes the installation. Copy `results/` elsewhere
first if you need the results. The only things left outside are download caches,
which you can remove if you want the disk space back. These commands clear the caches
for all projects, not only this one:

- uv: `uv cache clean`. If uv downloaded Python for this project (shown by
  `uv python list --only-installed`), `uv python uninstall 3.12` removes it.
- pip: `pip cache purge` (run it before deleting `.venv/`, with the environment
  activated).

With conda, the environment lives outside the project directory. Remove it with

```bash
conda deactivate
conda env remove -n nfmimiface
```

and then delete the project directory. `conda clean --all` removes conda's download
cache.
