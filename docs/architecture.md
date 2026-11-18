# Architecture

See `project.md` for the requirements and terminology (experiment, session, trial, stimulus).

## Modules

All under `src/nfmimiface/`. Only `app.py` and `gui/` import Qt; everything else is
plain Python and can be tested without a camera or a window.

| Module | Contents |
|---|---|
| `constants.py` | Static paths: project root, model file, default experiment, results root |
| `config.py` | `load_config()` → `ExperimentConfig` (plain data read from the TOML) |
| `images.py` | `RGBImage` type; `load_rgb`, `save_png` (safe with non-ASCII paths on Windows) |
| `face.py` | `FaceAnalyzer` (mediapipe), `Blendshapes`, `Rect`, `crop_rect` |
| `scoring.py` | `Scorer` type, `@register`, `make_scorer`, the scorers themselves |
| `experiment.py` | `Experiment`, `Session`, steps (`Imitate`, `Feedback`, `Closing`), `TrialResult` |
| `show_blendshapes.py` | Command-line tool `nfmimiface-blendshapes` |
| `compare_scorers.py` | Command-line tool `nfmimiface-compare-scorers`: scorers on labelled photos ([scorers.md](scorers.md)) |
| `live_score.py` | `nfmimiface-live-score`: live scores of the webcam face against a target image |
| `app.py` | Entry point `nfmimiface`: loads everything (errors → dialog), opens the window |
| `gui/camera.py` | `FrameSource` threads emitting mirrored RGB frames: `Camera`, `StillImage` (`--fake-camera`) |
| `gui/widgets.py` | `ImageView` (aspect-preserving image), `to_qimage` |
| `gui/pages.py` | `StartPage`, `ImitatePage`, `FeedbackPage`, `ClosingPage`: display only |
| `gui/main_window.py` | `MainWindow`: holds the session, shows the page for `session.step`, handles buttons and Space/Enter |

## How the classes relate

```
ExperimentConfig ──(read by)──▶ Experiment ◀──(passed in)── FaceAnalyzer
   (from TOML)                   │  holds: config, analyzer,
                                 │         scorer   ← make_scorer(config.scorer)
                                 │         stimuli  : tuple[Stimulus]   (loaded once)
                                 │  build_steps() → list[Step]
                                 │  message_for(score) → str
                                 │  start_session(frame) → Session | None
                                 ▼
                              Session   (one participant)
                                 │  id, crop: Rect, dir (results/YYYYMMDD/)
                                 │  steps : list[Step]  ← experiment.build_steps()
                                 │  step  : current Step;  advance()
                                 │  capture(frame) → TrialResult | None
                                 │  trials: list[TrialResult]
                                 ▼
                              TrialResult  (one imitation: face crop, blendshapes, score, message)

Step = Imitate(stimulus) | Feedback(trial: index into Session.trials) | Closing()
```

- `Experiment` lives for the whole run. It is created once at start-up, after the config
  is loaded and the camera is checked.
- `Session` is created when 「スタート」 is pressed and discarded after the closing page.
- Steps are plain data. The GUI shows one page per step type, and the session tells it
  which step is current: `MainWindow._show_step()` matches on `session.step`.

## One session

1. **Start:** `experiment.start_session(frame)` finds the face and fixes a generous crop
   (`crop_scale` × face height) for the whole session. The webcam is wide-angle, and the
   participant is assumed to stay roughly within the crop. No face: show a retry message.
2. **`Imitate`:** the stimulus and the live, mirrored crop are shown side by side.
   On 「この表情を撮影」 (or a key), `session.capture(frame)` crops, analyses, scores and saves the
   frame. No face: show a retry message; nothing is recorded.
3. **`Feedback`:** shows `session.trials[step.trial]`: the face, the score and the message.
   The button reads 「次へ」, or 「終了」 when the next step is `Closing`
   (`session.next_is_closing`).
4. Steps 2–3 repeat for each stimulus, then **`Closing`**.

Frames are RGB and mirrored once, right after capture from the camera. The face crop is
therefore analysed and saved exactly as the participant saw it, so left/right
blendshapes match what the participant was imitating. Stimuli are never mirrored.

## Outputs

`results/YYYYMMDD/` contains, per session:

- `{session_id}.json`: written when the session starts and rewritten after each trial.
  It records the scorer, the crop, the experiment's stimuli (path + md5), and per trial
  the stimulus path, face image file name, unrounded score, message, and both sets of
  blendshapes.
- `{session_id}_t{n}.png`: the mirrored face crop of trial *n*.

Session IDs are `{machine_name}_{NNN}_{xxxxx}`. NNN is one more than the highest number
the machine has used in any `results/*/` directory. xxxxx is 5 random lowercase letters
and digits (without 0, 1, i, l, o), so that an ID typed wrongly (e.g. into a form) does
not silently match another session. The date and time are in the JSON (`started_at`).

## Extension points

- **New scorer:** see the docstring of `scoring.py`, then select it with `[scorer] name`
  in the TOML. The candidates and their behaviour: [scorers.md](scorers.md).
- **Different order (e.g. shuffling):** change `Experiment.build_steps()`.
- **New kind of step** (e.g. a neutral face before the task): add a step class to
  `experiment.py`, include it in `Step`, emit it from `build_steps()`, and add a GUI page.
