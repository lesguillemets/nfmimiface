# Scorers

How well does the participant's face imitate the stimulus? Every scorer compares two
vectors of MediaPipe's 52 blendshapes (participant, target) and returns 0–100.
This document describes the candidates in `src/nfmimiface/scoring.py`, why they are
built the way they are, and how they behave.

Select one in the experiment TOML:

```toml
[scorer]
name = "shape_strength"
params = { strength_weight = 0.5 }
```

Compare them with

    uv run nfmimiface-compare-scorers [DIR] [--edits] [--mirror] [--matrix]

(see the module docstring of `compare_scorers.py`; candidates are listed in its
`CANDIDATES`). To try them on yourself in front of the webcam:

    uv run nfmimiface-live-score --img-path TARGET_IMAGE_OR_DIR [--scorer NAME ...]

shows the target, the mirrored webcam image, and every candidate's score live
(← / → switch targets in a directory).

## What a good imitation score should do

1. **Right movements:** reward the facial movements the target shows, penalise
   movements it does not show.
2. **Strength:** a faint version of the target's expression is a partial imitation, not a
   perfect one.
3. **Doing nothing scores low.** A quiet face must not get a decent score against an
   expressive target.
4. **Ignore gaze.** Where the eyes look is not part of the expression, and the participant
   is looking at the screen anyway.
5. **Robust to measurement noise**: MediaPipe gives somewhat different values for the
   same face (see below).
6. Nice to have: **explainable** (which part of the face to fix), and a score range that
   feels fair when displayed.

## What MediaPipe's blendshapes look like

Observed on the photos in `data/expressions/` (casual picks: different people, not
imitation pairs, so only a sanity check):

- **Mostly near zero.** Of the 43 non-gaze blendshapes, a typical face has 8–19 above
  0.05; the rest sum to only about 0.2. A noise floor (dead zone) would barely matter.
- **Gaze is large and arbitrary.** `eyeLook*` values reach 0.8 and flip with the image.
  With plain cosine, gaze alone drags a face against its own mirror image down to 45.
- **Flipping the image swaps Left/Right names** (roughly). Participants imitate what
  they see on the screen, and the app analyses the mirrored camera image, so the sides
  of the participant's blendshapes match those of the stimulus. Mirroring is only ever
  done on images, never on blendshape values.
- **Measurement noise is substantial.** The flipped image is effectively a re-measurement
  of the same face, and some values move a lot (one surprised face: `browInnerUp` 0.13
  vs 0.47). Even after swapping the sides back, a face scores only ~75 (`overlap`)
  against its own flipped copy, so ~100 is out of reach in practice.
- **`cheekSquint` hardly moves.** It stays at 0.000 in two broad smiles; the cheek
  raise of a genuine smile shows up as `eyeSquint`. `browDown` can be high (0.8) in a
  smile. MediaPipe's blendshapes "loosely correspond" to FACS action units
  ([Py-Feat model card](https://huggingface.co/py-feat/mp_blendshapes)), and these
  quirks are where the correspondence breaks down.

## Candidates

Unless noted, `_neutral` and the gaze blendshapes (`scoring.NON_EXPRESSIVE`) are ignored.

### `cosine` (current default): baseline

    100 × p·t / (|p| |t|)

Compares the *pattern* only. With default params it includes gaze, which is its biggest
problem; `exclude = NON_EXPRESSIVE` fixes that. Even then it fails (2) and (3): any
strength of the right pattern scores 100, and a quiet face scores ~35 because
normalisation blows its small, random values up to full length.

### `pearson`: not recommended

Cosine after subtracting each face's mean. A quiet face now scores ~0 (its residual is
uncorrelated with the target), but strength is still ignored, and centring a sparse
vector makes the inactive blendshapes count as weakly "negative", which has no clear
meaning. Kept for comparison.

### ★ `overlap`: weighted Jaccard

    100 × Σ min(p, t) / Σ max(p, t)

The shared activity divided by the total activity. One line, no parameters, and it
meets (1)–(3) directly: extra movements enlarge the denominator, missing ones shrink
the numerator, a half-strength imitation scores exactly 50, and a quiet face ~5.

Weaknesses: it is strict. Exaggeration is penalised as much as under-doing (150%
strength → 70), measurement noise lowers every score, and real same-expression pairs
land around 30–45. The message thresholds must be set accordingly (see below).

### ★ `shape_strength`: pattern × strength

    100 × cosine(p, t) × (min(|p|, |t|) / max(|p|, |t|)) ** strength_weight

Splits the question into *which* movements (cosine of the direction) and *how strongly*
(ratio of the vector lengths). `strength_weight` sets how much strength matters: 0 is
plain cosine, 0.5 (default) gives a half-strength imitation 71, 1 gives it 50. A quiet
face scores ~8 because its vector is short.

It is the most forgiving candidate that still satisfies (2) and (3), and its scores sit
in a range people expect (same expression ~55, different ~25). The parameter has a plain
meaning that can be explained to participants ("how strongly you did it"). Being
cosine-based, it is less hurt by noise in small values than `overlap`.

### ★ `regions`: per facial region

The face is split into brows, eyes, cheeks & nose, mouth, and jaw (`scoring.REGIONS`).
Each region is compared with `overlap`; the region scores are averaged, weighted by each
region's activity capped at `full_activity` (0.5).

The mouth has 23 blendshapes and the brows 5, so in whole-face measures the mouth
dominates. Here every clearly moving region counts equally: "smile right, brows wrong"
loses as much as "brows right, smile wrong". Quiet regions barely count, so a quiet
participant still scores low (~6).

Its main promise is **feedback**: the per-region scores could drive messages like
「口元はばっちり！眉ももっと動かしてみよう」. That needs the scorer to return the
breakdown, which the current `Scorer` type (→ float) does not; this is an easy extension
if wanted.

### ★ `emotions`: basic-emotion profiles

Blendshapes are mapped to FACS action units (`scoring.ACTION_UNITS`, both sides
averaged), following the expert-consensus mapping of Turrisi et al. (2026, *Computers in
Human Behavior Reports* 22, 101125, "Blendshape features meet action units", Fig. 4).
Each basic emotion's evidence is the mean of its prototypical action units
(`scoring.EMOTIONS`, after Ekman & Friesen's EMFACS: happiness = AU6+12, sadness = 1+4+15,
surprise = 1+2+5+26, fear = 1+2+4+5+20+26, anger = 4+5+24, disgust = 9+15+16). The two
6-emotion profiles are compared with `overlap`.

This asks the semantic question: *does the participant show the same emotion?* Two
different-looking smiles count as the same. It is symmetric by construction (sides are
averaged) and robust to small irrelevant movements.

Limits:
- It only knows the six emotions. Puffed cheeks, pouting, sticking out the tongue,
  winking: anything outside the prototypes is invisible. Use it only if all stimuli are
  basic emotional expressions.
- One deliberate deviation from the paper: it maps `cheekSquint` to the cheek raiser
  AU6 and `eyeSquint` to the lid tightener AU7, judged from what the blendshapes are
  *meant* to capture (the authors note the mapping is not yet tested on data). In
  MediaPipe's actual output `cheekSquint` stays at 0 in broad smiles (see above), which
  would make happiness depend on the mouth alone. So `eyeSquint` stands for AU6 here, and
  AU7, which MediaPipe then cannot tell apart, is left out (of anger and fear). The
  lip tightener AU23 has no blendshape; the lip pressor AU24 stands in for it in anger.
  The paper maps no blendshape to AU10, so disgust uses the 9+15+16 variant.
- Because evidence is a *mean*, one action unit alone counts partly: the two sad faces
  get happiness evidence 0.22–0.28 from squinting alone, and one smiling face shows
  "anger" 0.39 (from `browDown` and `mouthPress`). See "soft-AND" below.
- Emotions overlap in their action units (surprise ⊂ fear), so related emotions score
  each other highly.

## Behaviour

### Controlled edits (`--edits`)

Each target is scored against an edited copy of itself; means over the 6 targets.
These are the most meaningful numbers here, because the ideal answer is known.

| edit (participant) | ideal | cosine | cos-nogaze | pearson | overlap | shape_strength | regions | emotions |
|---|---|---|---|---|---|---|---|---|
| half strength | partial | 100 | 100 | 100 | 50 | 71 | 50 | 50 |
| 1.5× strength | high | 99 | 100 | 100 | 70 | 84 | 69 | 71 |
| mouth & jaw only | partial | 63 | 73 | 67 | 53 | 64 | 41 | 38 |
| brows & eyes only | partial | 56 | 65 | 57 | 49 | 53 | 60 | 66 |
| gaze moved | 100 | 85 | 100 | 100 | 100 | 100 | 100 | 100 |
| quiet face | 0 | 37 | 35 | 1 | 5 | 8 | 6 | 5 |

### Photos (`--mirror`)

Same/different-expression pairs across the 6 photos (ordered pairs, 6 same / 24
different), and each face against its own flipped image. AUC = probability that a
same-expression pair outscores a different-expression pair. **Do not read much into
these**: six casually chosen photos of different people, and one "surprise" face is a
delighted one with raised lip corners.

| scorer | same mean | diff mean | AUC | own flipped image |
|---|---|---|---|---|
| cosine | 53 | 34 | 0.75 | 73 |
| cosine (no gaze) | 61 | 29 | 0.87 | 90 |
| pearson | 51 | 16 | 0.86 | 88 |
| overlap | 37 | 16 | 0.91 | 69 |
| shape_strength | 56 | 24 | 0.88 | 88 |
| regions | 38 | 19 | 0.89 | 68 |
| emotions | 51 | 33 | 0.95 | 81 |

Selected pairs (participant → target; "~m" = flipped image):

| pair | cosine | overlap | shape_strength | regions | emotions | note |
|---|---|---|---|---|---|---|
| smile1 → smile2 | 68 | 39 | 60 | 31 | 48 | same emotion, different smiles |
| surp1 → surp2 | 39 | 28 | 38 | 41 | 44 | surp1 has raised lip corners |
| smile2 → surp1 | 63 | 46 | 69 | 34 | 54 | the shared smile wins |
| smile2~m → smile2 | 52 | 54 | 78 | 52 | 80 | same face, flipped |

`emotions` separates the labels best, as expected since the labels *are* emotions;
`regions` is the only one that ranks the surprise pair above smile→surprise, because
it weighs the raised brows as much as the mouth.

## Which one when

- **Stimuli are basic emotions and "same feeling" is the point:** `emotions`, possibly
  with message thresholds around 30/50.
- **Any stimuli, forgiving, easy to explain:** `shape_strength`. The natural successor
  of the current cosine.
- **Any stimuli, strict, simplest formula:** `overlap`, with low message thresholds.
- **Want per-region feedback messages:** `regions`, after extending the scorer to return
  the breakdown.

All of them fix the two concrete flaws of the current default (gaze counted; a quiet
face scores ~35). Comparing them on the event's actual stimuli, with a few people
imitating them in front of the real webcam (`nfmimiface-live-score`), is the next
useful test.

## Display and thresholds

The scorers rank faces differently, but they also live on different scales: a good
imitation is ~55 for `shape_strength` but ~35 for `overlap`. Because measurement noise
alone costs 10–30 points, nobody will reach 100. Rather than rescaling the scorers, set
the `[[messages]]` thresholds per scorer. If displayed numbers should look higher, a
monotone curve (e.g. `100 × (s/100) ** 0.5`) changes the display without changing the
ranking.

## Ideas not implemented

- **Delta from the participant's own neutral face.** Capture a resting face at "start"
  and compare `participant − resting` against the target. This removes person-specific
  resting expressions (some faces look slightly frowning at rest), which none of the
  above can do. It needs a neutral capture for the stimulus too, or treats the stimulus
  as-is.
- **Target-weighted scoring:** weight each blendshape by how much the target uses it,
  so missing the target's key movement costs more than an extra small movement.
- **Soft-AND emotion evidence:** the geometric mean of an emotion's action units instead
  of the mean, so that one action unit alone (e.g. squinting) does not count as half a
  smile. More principled, but more sensitive to MediaPipe's mapping quirks.
