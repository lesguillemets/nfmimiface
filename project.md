# nfmimiface

This is a project for demonstrative purpose in an outreach event.

- In the GUI, one picture is displayed on one side, and on the other, (mirrored) image from the webcam is displayed.
- The participants will mimic the facial expression displayed, then click 'finish' to proceed
- The picture will be taken and compared according to the pre-defined scoring function, then the result is displayed
- The result is saved elsewhere
- We begin with one picture (trial) per session but make it configurable (below), so that we can multiple mimic tasks per session (participant)
- The GUI should be light-themed and should look attractive, rather than stiff and experimental


## Terminology

- Experiment: the set-up defined by the TOML file (stimuli, number of trials, scorer).
- Session: one participant's visit, from "start" to their last trial.
- Trial: one imitation of one stimulus: show → imitate → capture → score. A session has one or more trials; one JSON record is saved per Session.
- Stimulus == target face: a reference picture to imitate.

## States

- Init (load the config and the model, check for camera)
- Main loop:
  - The user will click "start" for initiating the task
  - The location of the face within webcam is used to clipping the input from the webcam
    - How much clipping should be done will be configured
    - The clipping here is for enlarge the face of participants appropriately; the webcam will be wide-angled
  - The current face is displayed along the stimulus
    - The current face is mirrored on display
  - Mimic, then click finish
  - The picture is saved
    - If no face is detected, show a retry message and don't record anything. Assume exactly one face; use the largest/first
  - The score is calculated and then displayed, along with messages based on the scores
  - Although not currently implemented, make it easy to add variants in experiments; as such, there should probably be a "Experiment" class that handles trials

## Scoring

- Scoring will be based on mediapipe's blendshape scores.
- During the development, it is essential to be able to change and swap the function for calculating the score. No change is intended among actual trials, so it can either be configured in the toml or within python codes. Make proper abstractions to declare the scoring functions.


## Configs

Mostly the configurations should be declared within code or within dedicated files, rather than selected from the GUI.

### Changeable

Use a toml file for configuration.

- Side of the stimuli
- The path(s) of the target picture(s) in their order
- Name of the experiment
- The messages and corresponding threshold scores
- The amount of clipping around participants' faces

### Static

The following configurations are to be stored as python constants, which can, if we decide to implement it, later be overridden by command line arguments.

- The directories where the results are saved (should include day as timestamp, like `{project_root}/results/20250101/`)

## Outputs

- The picture from webcam when the user clicked Proceed
- JSON file, one per one session, that records the id, timestamp, score(s), blendshape scores and the paths to the stimuli image & recorded face


## Development

- The project is developed in Fedora/KDE but will be run under Windows 11.
- The User interface will be in the Japanese language.
- Prefer well-typed codes with proper abstractions

### Dependencies
- Pyside6 for GUI
- mediapipe for face image processing
- opencv for image manipulation

### Coding

- Lint and format with pyright and ruff
- Add type annotations whenever appropriate
- Don't be too verbose and prefer readability over "handle every possible case" strategy
  - There are many edge cases where we can assume not happening; rather than handling everything with code, make a clear set of assumptions
- Future variants may add extra capture steps (e.g. a neutral face before the task); the flow should allow inserting steps without restructuring

