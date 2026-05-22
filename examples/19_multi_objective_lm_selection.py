# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Multi-objective HP Search: Picking the Best LM for Classification

Companion to `examples/18_hyperparameter_search_with_keras_tuner.py`. Here
the goal is **model selection**: given several candidate language models
plus a couple of prompting knobs, pick the configuration that maximizes
both `ExactMatch` reward (strict accuracy across all class fields) **and**
macro `BinaryF1Score` on a 6-class emotion classification task — the
classic trade-off when the label distribution is imbalanced.

We feed `synalinks.tuners.RandomSearch` a **list** of objectives. Internally
keras-tuner wraps it in a `MultiObjective`, aggregates the two metrics into
a single scalar per trial, and ranks accordingly. Each individual metric
remains recorded on the trial so you can inspect the Pareto trade-off.

## Why `BinaryF1Score` instead of `F1Score`

`synalinks.metrics.F1Score` is token-level (QA-oriented) — for a
single-token categorical label it collapses to accuracy and gives no
extra signal. `synalinks.metrics.BinaryF1Score` operates **per class
field**, computing per-class precision and recall and averaging across
classes. With `average="macro"`, a model that always predicts the
majority class scores high on accuracy but low on F1 — exactly the
trade-off we want to surface during search.

That means the output data model needs **one boolean field per class** —
the per-class layout `BinaryF1Score` expects. Each class is an
independent boolean, so this layout naturally supports multi-label
classification; the LM can in principle set zero, one, or several
fields to `True`. For *strict* single-label enforcement, use a single
`Literal` field with the `Categorical*` metrics instead.

## Dataset

[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) — 6
imbalanced classes (`sadness`, `joy`, `love`, `anger`, `fear`, `surprise`).
We use a tiny slice (24 train / 12 val / 12 test) so a full search fits
in a few minutes. Loaded via `synalinks.HuggingFaceDataset` — a Jinja2
output template maps the dataset's integer `label` (0-5) into a boolean
record (one `True`, five `False` — since this source is single-label)
that the `Emotion` schema and `BinaryF1Score` consume.

The dataset-helpers idiom we use here is worth a callout:

- `synalinks.datasets.load_split(...)` — one-call shortcut that builds
  a non-streaming `HuggingFaceDataset` and `.materialize()`s it into
  `(x, y)` numpy object arrays.
- `synalinks.datasets.split_train_test(x, y, validation_split=...)` —
  deterministic head/tail slicer returning
  `((x_train, y_train), (x_val, y_val))`. Works on any `(x, y)` pair,
  not just HF — handy when the source only ships a single labeled
  split (HumanEval, IFEval, BBH, ...).

## Search space

A single Choice: `model` — which language model to use. Since this is a
small discrete sweep, we use `synalinks.tuners.GridSearch` to enumerate
every candidate exactly once instead of `RandomSearch` (which would
re-sample with replacement).

## Installation

```bash
uv pip install keras-tuner datasets
```

## API References

- [synalinks.tuners.RandomSearch](https://synalinks.github.io/synalinks/Synalinks%20API/Tuners/)
- [synalinks.metrics.BinaryF1Score](https://synalinks.github.io/synalinks/Synalinks%20API/Metrics/FScore%20metrics/)
- [synalinks.rewards.ExactMatch](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/ExactMatch%20reward/)
- [synalinks.HuggingFaceDataset](https://synalinks.github.io/synalinks/Synalinks%20API/Datasets/HuggingFaceDataset/)
- [keras-tuner Multi-objective](https://keras.io/keras_tuner/api/keras_tuner/Objective/)
- [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# MUST run before any code path imports `keras_tuner`.
synalinks.disable_keras_backend()

# We load one labeled chunk from the `train` split and slice a
# validation tail off via `synalinks.datasets.split_train_test`.
NB_TRAINVAL_SAMPLES = 36  # → 24 train / 12 val after splitting
VALIDATION_SPLIT = 1.0 / 3.0
NB_TEST_SAMPLES = 12
EPOCHS = 1
BATCH_SIZE = 4

FOLDER = "examples"
PROJECT_NAME = "emotion_lm_selection"

# Output-schema toggle. False = per-class booleans (one independent
# `bool` per class; multi-label-friendly — the schema does not enforce
# "exactly one True"); True = per-class `synalinks.Score` confidences
# (same independence, but the LM can express graded uncertainty). The
# rest of the file branches on this single flag.
USE_SCORE_LABELS = False

# Candidate language models. The grid search visits each exactly once;
# add or remove entries to broaden / narrow the comparison.
CANDIDATE_MODELS = [
    "gemini/gemini-3.1-flash-lite-preview",
    "gemini/gemini-3.1-flash-preview",
    "gemini/gemini-3.1-pro-preview",
]

# Order matches `dair-ai/emotion`'s `ClassLabel` so `row["label"]` (int)
# maps to the right field on the output data model.
EMOTION_LABELS = ("sadness", "joy", "love", "anger", "fear", "surprise")


# =============================================================================
# Data models
# =============================================================================


class Tweet(synalinks.DataModel):
    """A short piece of text whose emotion we want to classify."""

    text: str = synalinks.Field(
        description="A short tweet-length text to classify",
    )


class Emotion(synalinks.DataModel):
    """Per-class boolean labels.

    One independent `bool` per class — the layout `BinaryF1Score`
    expects. The schema does not enforce "exactly one True":
    six independent booleans can take any of 2⁶ combinations, so
    this same layout naturally generalizes to multi-label tasks
    where more than one class can be positive at once. For
    *strict* single-label enforcement, use a single `Literal`
    field with the `Categorical*` metrics instead.

    On `dair-ai/emotion` each row happens to have exactly one
    `True` field in the ground truth, but the LM is technically
    free to predict any combination — the field descriptions are
    a soft hint, not a hard constraint.
    """

    sadness: bool = synalinks.Field(
        description="True if the dominant emotion is sadness",
    )
    joy: bool = synalinks.Field(
        description="True if the dominant emotion is joy",
    )
    love: bool = synalinks.Field(
        description="True if the dominant emotion is love",
    )
    anger: bool = synalinks.Field(
        description="True if the dominant emotion is anger",
    )
    fear: bool = synalinks.Field(
        description="True if the dominant emotion is fear",
    )
    surprise: bool = synalinks.Field(
        description="True if the dominant emotion is surprise",
    )


class EmotionScore(synalinks.DataModel):
    """Per-class confidence scores.

    One `synalinks.Score` per class — a discretized `[0, 1]` enum
    the LM is constrained to emit one of eleven values from (`0.0`,
    `0.1`, ..., `1.0`). Like the boolean layout, each class is an
    independent field; the schema does not enforce that exactly one
    field is high. `BinaryF1Score(threshold=0.5)` binarizes the
    confidences at evaluation time. Use this layout when you want
    the LM to express uncertainty, or for multi-label cases where
    several classes can be simultaneously positive.
    """

    sadness: synalinks.Score = synalinks.Field(
        description="Confidence that the dominant emotion is sadness",
    )
    joy: synalinks.Score = synalinks.Field(
        description="Confidence that the dominant emotion is joy",
    )
    love: synalinks.Score = synalinks.Field(
        description="Confidence that the dominant emotion is love",
    )
    anger: synalinks.Score = synalinks.Field(
        description="Confidence that the dominant emotion is anger",
    )
    fear: synalinks.Score = synalinks.Field(
        description="Confidence that the dominant emotion is fear",
    )
    surprise: synalinks.Score = synalinks.Field(
        description="Confidence that the dominant emotion is surprise",
    )


# Templates render each HF row into JSON that round-trips through Pydantic
# into `Tweet` / (`Emotion` or `EmotionScore`).
INPUT_TEMPLATE = '{"text": {{ text | tojson }}}'

# Boolean layout: emit `true`/`false` per class.
OUTPUT_TEMPLATE_BOOL = (
    "{"
    + ",".join(
        f'"{name}": {{{{ (label == {i}) | tojson }}}}'
        for i, name in enumerate(EMOTION_LABELS)
    )
    + "}"
)

# Score layout: emit `1.0` for the gold class, `0.0` for the rest. Both
# are valid `synalinks.Score` values, so Pydantic accepts them.
OUTPUT_TEMPLATE_SCORE = (
    "{"
    + ", ".join(
        f'"{name}": {{{{ 1.0 if label == {i} else 0.0 }}}}'
        for i, name in enumerate(EMOTION_LABELS)
    )
    + "}"
)

# Pick the active output schema and template from the toggle.
EmotionLabel = EmotionScore if USE_SCORE_LABELS else Emotion
OUTPUT_TEMPLATE = OUTPUT_TEMPLATE_SCORE if USE_SCORE_LABELS else OUTPUT_TEMPLATE_BOOL


# =============================================================================
# Dataset loader
# =============================================================================


def load_emotion_split(split: str, limit: int):
    """Materialize one HF split into `(x, y)` arrays of DataModels."""
    return synalinks.datasets.load_split(
        path="dair-ai/emotion",
        split=split,
        input_data_model=Tweet,
        input_template=INPUT_TEMPLATE,
        output_data_model=EmotionLabel,
        output_template=OUTPUT_TEMPLATE,
        limit=limit,
    )


# =============================================================================
# Hypermodel: build a fresh classifier for one trial
# =============================================================================


async def build_program(hp):
    """Sample HPs and return a compiled classifier.

    Following the keras-tuner convention, this is a top-level function the
    tuner calls once per trial via `hypermodel.build(hp)`. The only HP
    here is which language model to use.
    """
    model_name = hp.Choice("model", CANDIDATE_MODELS)

    synalinks.clear_session()
    language_model = synalinks.LanguageModel(model=model_name)

    inputs = synalinks.Input(data_model=Tweet)
    outputs = await synalinks.Generator(
        data_model=EmotionLabel,
        language_model=language_model,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="emotion_classifier",
    )
    # Macro-F1 over the six per-class fields. With the Score layout we
    # pass `threshold=0.5` so float confidences get binarized before F1;
    # with the bool layout the threshold is irrelevant (booleans round-
    # trip as 0/1) but still safe to pass.
    program.compile(
        # Reward = strict accuracy: 1.0 iff every class field matches.
        # NOTE: under the Score layout, `ExactMatch` is too strict
        # (0.9 != 1.0). Swap to `BinaryF1Score` as the reward in that
        # case, e.g.:
        #
        #     reward=synalinks.metrics.BinaryF1Score(threshold=0.5),
        #
        reward=synalinks.rewards.ExactMatch(),
        metrics=[
            synalinks.metrics.BinaryF1Score(average="macro", threshold=0.5),
        ],
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    return program


# =============================================================================
# Main
# =============================================================================


async def main():
    load_dotenv()

    print("Loading dair-ai/emotion via synalinks.HuggingFaceDataset...")
    # Demo the dataset helpers: load one labeled chunk from `train` and
    # slice a validation tail off with `split_train_test`. (We could
    # also pass `split="validation"` directly — this dataset ships a
    # native validation split — but the split helper is the general
    # recipe for sources that only ship a single labeled split.)
    x_trainval, y_trainval = load_emotion_split("train", NB_TRAINVAL_SAMPLES)
    (x_train, y_train), (x_val, y_val) = synalinks.datasets.split_train_test(
        x_trainval,
        y_trainval,
        validation_split=VALIDATION_SPLIT,
    )
    x_test, y_test = load_emotion_split("test", NB_TEST_SAMPLES)
    print(
        f"Per trial: fit on {len(x_train)} samples, "
        f"validate on {len(x_val)} samples. "
        f"Test split: {len(x_test)} samples."
    )

    # Multi-objective: pass a list of `Objective`s and kt aggregates them
    # into a `MultiObjective` automatically. Both metrics are maximized
    # here, but you could mix directions (e.g. add a min-latency proxy).
    #
    # `GridSearch` enumerates every value of every `Choice` — here that
    # means every candidate model exactly once. `max_trials` has to be at
    # least the number of combinations, hence `len(CANDIDATE_MODELS)`.
    tuner = synalinks.tuners.GridSearch(
        build_program,
        objective=[
            synalinks.tuners.Objective("val_reward", direction="max"),
            synalinks.tuners.Objective("val_binary_f1_score", direction="max"),
        ],
        max_trials=len(CANDIDATE_MODELS),
        directory=FOLDER,
        project_name=PROJECT_NAME,
        overwrite=True,
    )

    print(f"\nEvaluating {len(CANDIDATE_MODELS)} candidate models...")
    tuner.search(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    print("Done.\n")

    tuner.results_summary(num_trials=len(CANDIDATE_MODELS))

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]

    print("\n" + "=" * 70)
    print("Winning model (aggregated objective)")
    print("=" * 70)
    print(f"  model:               {best_hp.get('model')}")
    reward = best_trial.metrics.get_best_value("val_reward") or 0.0
    f1 = best_trial.metrics.get_best_value("val_binary_f1_score") or 0.0
    print(f"  val_reward:          {reward:.3f}")
    print(f"  val_binary_f1_score: {f1:.3f}")
    print(f"\nFull trial history persisted under {FOLDER}/{PROJECT_NAME}")

    # Per-model breakdown so you can see the trade-off, not just the
    # winner. For a real comparison you'd want multiple seeds per model
    # and plot the Pareto frontier.
    print("\n" + "=" * 70)
    print("Per-model results (sorted by aggregated score)")
    print("=" * 70)
    print(f"{'trial':<6} {'model':<42} {'reward':>8} {'f1':>8}")
    for trial in tuner.oracle.get_best_trials(num_trials=len(CANDIDATE_MODELS)):
        hps = trial.hyperparameters
        r = trial.metrics.get_best_value("val_reward") or 0.0
        f = trial.metrics.get_best_value("val_binary_f1_score") or 0.0
        print(f"{trial.trial_id:<6} {hps.get('model'):<42} {r:>8.3f} {f:>8.3f}")

    # Optional: rebuild the winner from the best HPs and evaluate on the
    # held-out test slice (same pattern as keras-tuner's `get_best_models`).
    print("\nRebuilding winner and evaluating on test split...")
    program = await build_program(best_hp)
    metrics = await program.evaluate(
        x=x_test,
        y=y_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    print(f"Test reward:          {metrics.get('reward', 0.0):.3f}")
    print(f"Test binary_f1_score: {metrics.get('binary_f1_score', 0.0):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
