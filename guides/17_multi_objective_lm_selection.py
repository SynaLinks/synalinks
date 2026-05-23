# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Picking the Best Language Model: Multi-Objective Search

[Guide 16](https://synalinks.github.io/synalinks/guides/Hyperparameter%20Search/) introduced hyperparameter search and tuned three knobs of
a single language model: chain-of-thought on or off, sampling
temperature, and reasoning effort. In this guide we change the
kind of question we ask. Instead of tuning *one* model, we
**compare several models** on the same task — and we score each
one on **two metrics** at once, not just a single reward.

This is a very practical question. Modern LM providers ship whole
families of models — a small "lite" version, a medium "flash"
version, a large "pro" version. Before you deploy, you usually
have to pick one. The cheap model is fast and inexpensive but may
miss edge cases; the expensive model is accurate but slow and
pricey. What you actually want is the **trade-off curve**, not
just the single winner. The winner depends on how you weight cost
against accuracy, and that weighting is a decision you should make
with the data in front of you.

Two ideas show up in this guide for the first time:

- **Multi-objective optimization.** Instead of a single
  `Objective("val_reward", "max")`, we hand the tuner a *list* of
  objectives. Keras-Tuner aggregates them into a single scalar per
  trial and ranks trials by that aggregate, while keeping each
  individual metric recorded so you can inspect the trade-off
  after the fact.
- **Grid search.** When the search space is small and discrete —
  for example, "try each of these three models exactly once" —
  `GridSearch` enumerates the combinations deterministically.
  `RandomSearch` would sample *with replacement* and could visit
  the same model twice while skipping another; `GridSearch` is
  guaranteed not to.

## The Task: Six-Way Emotion Classification

The dataset we use is
[`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion):
short tweets each labeled with one of six emotions — `sadness`,
`joy`, `love`, `anger`, `fear`, or `surprise`. What makes this
dataset interesting is that the label distribution is
**imbalanced**: `joy` and `sadness` together cover most of the
corpus, while `love` and `surprise` are rare.

That imbalance is the entire point. A lazy model that always
predicts the majority class can post a respectable accuracy and
still be useless — it gets the easy half right by accident, and
flunks the rare half completely. To surface that failure mode we
need a metric that scores **per class**, not in aggregate.

## Why Two Metrics, and Which Two

We compile each candidate with two metrics:

- **`ExactMatch`** — strict accuracy. The reward is `1.0` if and
  only if *every* class field on the predicted output matches the
  truth, else `0.0`. It captures "did we get the label exactly
  right?" but it does not distinguish between a model that gets
  every class right occasionally and one that gets only the
  majority class right but does so reliably.
- **`BinaryF1Score(average="macro")`** — averaged per-class F1.
  **F1** is the harmonic mean of precision and recall; intuitively,
  it punishes a classifier that wins on one of those at the
  expense of the other. **Macro averaging** computes the F1 of
  each class separately and then averages, so a class with only a
  handful of examples (like `love` or `surprise`) contributes the
  same weight as a frequent one (like `joy`). A model that
  collapses onto the majority class will score high on accuracy
  and low on macro-F1 — exactly the trade-off we are trying to
  see.

A footnote you will want before reading the source:
`synalinks.metrics.F1Score` (without `Binary`) is **token-level**
and oriented at open-text QA tasks. For a single categorical label
it collapses to accuracy and tells you nothing new. Whenever your
output has one boolean per class — the layout we are about to
introduce — the right metric is `BinaryF1Score`.

## The Output Schema Is Doing Real Work

Because `BinaryF1Score` operates per class, the output `DataModel`
needs **one boolean field per class**. The LM emits a 0/1 for
each:

```python
class Emotion(synalinks.DataModel):
    sadness:  bool = synalinks.Field(description="True if the dominant emotion is sadness")
    joy:      bool = synalinks.Field(description="True if the dominant emotion is joy")
    love:     bool = synalinks.Field(description="True if the dominant emotion is love")
    anger:    bool = synalinks.Field(description="True if the dominant emotion is anger")
    fear:     bool = synalinks.Field(description="True if the dominant emotion is fear")
    surprise: bool = synalinks.Field(description="True if the dominant emotion is surprise")
```

It is tempting to call this layout "one-hot" — and the **ground
truth** is one-hot in this dataset, because `dair-ai/emotion`
assigns exactly one class label per tweet. But the schema itself
does **not** enforce that. Six independent booleans can take any
of 2⁶ = 64 combinations; the LM is free to mark two as `True` if
two emotions seem present, or none at all if it cannot decide.
That is a *feature*, not a bug: the same layout doubles as
**multi-label classification** — many real-world tasks (genre
tagging, content moderation, symptom checklists) genuinely have
more than one positive class at once.

If you actually need to enforce "exactly one label," that is a
different schema: a **single field with a `Literal` of the class
names**, paired with the `Categorical*` F1 family (which compares
label *sets* instead of per-field booleans). Sketch:

```python
from typing import Literal

class EmotionCategorical(synalinks.DataModel):
    label: Literal["sadness", "joy", "love", "anger", "fear", "surprise"] = (
        synalinks.Field(description="The dominant emotion")
    )

# Pair with the Categorical F1 family in compile():
program.compile(
    reward=synalinks.rewards.ExactMatch(),
    metrics=[synalinks.metrics.CategoricalF1Score(average="macro")],
    optimizer=synalinks.optimizers.RandomFewShot(),
)
```

So there are three reasonable layouts for a classification task,
each pairing with a different metric family:

| Layout | Schema | Metric family | When to use |
|---|---|---|---|
| **Booleans** | one `bool` per class | `Binary*` | Multi-label; or multi-class where multiple labels are allowed in principle. |
| **Scores** | one `Score` per class | `Binary*(threshold=…)` | Same as booleans, plus you want the LM to express confidence. |
| **Categorical** | single `Literal` field | `Categorical*` | Strictly one label per example, enforced by the schema. |

The take-away worth internalizing: **in Synalinks, the schema you
declare and the metric you measure have to be designed
together.** This guide uses the boolean layout because it makes
the F1-vs-accuracy trade-off vivid even on a single-label dataset;
the variant section below shows the Score version of the same
story.

## A Variant: Confidence Scores Instead of Booleans

`BinaryF1Score` accepts not just `bool` fields but also **floats in
`[0, 1]`** — it just thresholds them at runtime to decide which
side of `0`/`1` each prediction lands on. That opens up a second
output layout: instead of independent booleans per class, ask the
LM for a **confidence per class**. `synalinks.Score` ([Guide 2](https://synalinks.github.io/synalinks/guides/Data%20Models/)) is the
natural type — a discretized `[0, 1]` enum that the LM is
constrained to emit one of eleven values from (`0.0`, `0.1`, ...,
`1.0`).

```python
class EmotionScore(synalinks.DataModel):
    sadness:  synalinks.Score = synalinks.Field(description="Confidence that the dominant emotion is sadness")
    joy:      synalinks.Score = synalinks.Field(description="Confidence that the dominant emotion is joy")
    love:     synalinks.Score = synalinks.Field(description="Confidence that the dominant emotion is love")
    anger:    synalinks.Score = synalinks.Field(description="Confidence that the dominant emotion is anger")
    fear:     synalinks.Score = synalinks.Field(description="Confidence that the dominant emotion is fear")
    surprise: synalinks.Score = synalinks.Field(description="Confidence that the dominant emotion is surprise")
```

The ground-truth template now emits the literal floats `1.0` /
`0.0` instead of the JSON booleans `true` / `false` — both are
valid `Score` values, so Pydantic accepts them:

```python
OUTPUT_TEMPLATE_SCORE = (
    "{"
    + ", ".join(
        f'"{name}": {{{{ 1.0 if label == {i} else 0.0 }}}}'
        for i, name in enumerate(EMOTION_LABELS)
    )
    + "}"
)
```

The metric line changes by exactly one keyword argument — add
`threshold=0.5` so `BinaryF1Score` knows where to cut the
confidence values:

```python
program.compile(
    reward=synalinks.rewards.ExactMatch(),    # see caveat below
    metrics=[synalinks.metrics.BinaryF1Score(average="macro", threshold=0.5)],
    optimizer=synalinks.optimizers.RandomFewShot(),
)
```

**When to prefer which layout.** A short decision guide:

- **Booleans** when the task really is multi-class with one
  winner. The LM has nothing to express beyond "this one." The
  reward `ExactMatch` is the natural strict accuracy.
- **Scores** when you want the LM to express **uncertainty**
  ("`joy: 0.7, love: 0.3`"), when more than one class can be
  simultaneously true (multi-label), or when you plan to use the
  confidence values downstream (e.g. as a ranking signal). The
  reward `ExactMatch` is too strict here because `0.9 ≠ 1.0`;
  `BinaryF1Score` itself (with a threshold) is the usual reward
  choice in this regime.

A small reward-side caveat worth flagging: if you switch to
Score-typed labels, `ExactMatch` will give you `0.0` whenever the
LM's confidence is anything other than the exact ground-truth
value — even `0.99` against `1.0`. Use `BinaryF1Score` as both the
**reward** and the **metric** in that case (or write your own
threshold-based reward via `RewardFunctionWrapper`, [Guide 12](https://synalinks.github.io/synalinks/guides/Rewards/)).

The runnable example below demonstrates both layouts behind a
`USE_SCORE_LABELS` toggle so you can see the wiring of each.

## Loading a Hugging Face Dataset Through Templates

The `dair-ai/emotion` dataset ships its labels as integers `0`
through `5`. Each row has exactly one label, so the rendered
ground-truth record happens to have exactly one `True` field —
the dataset *itself* is single-label, even though our schema
would tolerate multi-label. Before we can train on it, we have to
convert each row into the boolean record our `Emotion` data model
expects.
`synalinks.HuggingFaceDataset` handles this with two **Jinja2
templates** — one for the input side of each row, one for the
output side — that render each Hugging Face row into a JSON
snippet matching the target `DataModel`.

(**Jinja2** is the standard Python templating language; you can
think of it as "string formatting on steroids." The
double-curly-brace syntax `{{ ... }}` evaluates an expression and
substitutes its value into the surrounding text.)

```python
INPUT_TEMPLATE = '{"text": {{ text | tojson }}}'
OUTPUT_TEMPLATE = (
    "{"
    + ",".join(
        f'"{name}": {{{{ (label == {i}) | tojson }}}}'
        for i, name in enumerate(EMOTION_LABELS)
    )
    + "}"
)
```

The `tojson` filter is the safe way to embed a value into JSON —
it escapes quotes, backslashes, and control characters so they
cannot accidentally break the JSON output. Skipping it is the
templating equivalent of forgetting to parameterize a SQL query
([Guide 6](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/)); the bugs it prevents are unpleasant in exactly the
same way. For this dataset, the template emits `true` for the
matching class and `false` for every other one — which yields a
single-label record because the source dataset is single-label.
The `Emotion` schema would happily accept multiple `True` fields
if the data ever called for it.

## Dataset Helpers: `load_split`, `materialize`, `split_train_test`

Three helpers from `synalinks.datasets` (introduced in [Guide 10](https://synalinks.github.io/synalinks/guides/Datasets/))
do all of the heavy lifting:

- **`synalinks.datasets.load_split(...)`** — a one-call shortcut
  that builds a `HuggingFaceDataset` with `streaming=False`,
  iterates it to exhaustion, and hands you back numpy object
  arrays. The return shape is `(x, y)` when an output template is
  set (as here), or `(x,)` for inputs-only datasets.
- **`Dataset.materialize()`** — the underlying method on the
  `Dataset` base class that `load_split` calls into. Any
  `Dataset` subclass (HuggingFace, a custom CSV loader, your own
  generator) gets this method for free. Use it when you want to
  build the dataset object explicitly — for instance, to inspect
  it before materializing, or to swap streaming on and off.
- **`synalinks.datasets.split_train_test(x, y, validation_split=0.2)`** —
  a deterministic head/tail slicer. It returns
  `((x_train, y_train), (x_val, y_val))` after cutting at
  `int(n * (1 - validation_split))`. It is the standard way to
  carve a validation slice out of a single labeled split — useful
  when the source dataset does not ship its own validation split
  (HumanEval, IFEval, BBH, TruthfulQA, BBQ all fit that pattern).

For this guide we *do* have a native `validation` split on
`dair-ai/emotion`, but we use `split_train_test` against the
`train` split anyway — partly because it is the more general
recipe (it works on any single-split source), and partly so you
can see the helper in action:

```python
(x_trainval, y_trainval) = synalinks.datasets.load_split(
    path="dair-ai/emotion",
    split="train",
    input_data_model=Tweet,
    input_template=INPUT_TEMPLATE,
    output_data_model=EmotionLabel,
    output_template=OUTPUT_TEMPLATE,
    limit=NB_TRAINVAL_SAMPLES,
)
(x_train, y_train), (x_val, y_val) = synalinks.datasets.split_train_test(
    x_trainval, y_trainval, validation_split=VALIDATION_SPLIT,
)
(x_test, y_test) = synalinks.datasets.load_split(
    path="dair-ai/emotion", split="test", ..., limit=NB_TEST_SAMPLES,
)
```

The shuffle question is worth flagging: `split_train_test` is
deterministic and order-preserving — it slices the head off for
train and the tail off for val. If your source rows are not
already shuffled (HumanEval's prompts, for instance, are sorted
by task ID), shuffle *before* you split. The `datasets` library
lets you pass `shuffle=True` through `load_dataset` kwargs, and
those kwargs forward through `synalinks.datasets.load_split`.

## A Multi-Objective `GridSearch`

The tuner construction is almost the same as in [Guide 16](https://synalinks.github.io/synalinks/guides/Hyperparameter%20Search/), with
two differences:

- **`objective` is a *list*.** When you pass more than one
  objective, Keras-Tuner internally wraps the list in a
  `MultiObjective` object and aggregates the metrics into a
  single scalar (a weighted sum, by default with equal weights)
  for the oracle's ranking. The individual metrics stay recorded
  on each trial, so you can inspect the trade-off later — the
  aggregate is just there to let the oracle compare apples to
  apples.
- We use **`GridSearch`** instead of `RandomSearch`, because we
  want each model evaluated *exactly* once. (More on this below.)

```python
tuner = synalinks.tuners.GridSearch(
    build_program,
    objective=[
        synalinks.tuners.Objective("val_reward", direction="max"),
        synalinks.tuners.Objective("val_binary_f1_score", direction="max"),
    ],
    max_trials=len(CANDIDATE_MODELS),
    directory="examples",
    project_name="emotion_lm_selection",
    overwrite=True,
)
```

Both objectives are *maximized* in this example, but you can mix
directions freely. A common third axis is a `min` objective on
cost or latency — for instance, if you record
`val_tokens_per_request` as a metric, you can ask the tuner to
maximize accuracy *and* minimize tokens at the same time, and the
ranking will reflect both. That is exactly the "I want a model
that is accurate *and* cheap" question we set out to answer.

## Reading the Pareto Trade-off

When the search finishes, the *winning* trial is simply the one
with the highest *aggregated* score. But the genuinely interesting
output of a multi-objective search is the **per-model table** that
lists every model's individual scores side by side. That table is
your view of the **Pareto frontier** — a term worth pinning down.

The **Pareto frontier** is the set of configurations where you
cannot improve one metric without hurting another. A configuration
is on the frontier if no other configuration is strictly better
than it on *both* metrics at once. The frontier is not a single
point but a curve (or, in higher dimensions, a surface) of
"acceptable" trade-offs. Choosing one point on the frontier over
another is a *business* decision — how much accuracy am I willing
to give up for a fraction of the cost? — not a math problem.

```
trial  model                                       reward       f1
0001   gemini/gemini-3.1-flash-lite-preview        0.500    0.402
0002   gemini/gemini-3.1-flash-preview             0.667    0.531
0003   gemini/gemini-3.1-pro-preview               0.750    0.624
```

(Numbers shown here are illustrative; your run will differ.)

A model that scores high on `reward` but low on `f1` is
overfitting to the majority classes — it gets the easy half right
and quietly fails on the rare half. A model that scores about the
same on both metrics, even at a lower absolute level, is making
**balanced** mistakes — and that is often what you want in
practice, because the rare classes are usually the ones the user
notices.

For a really honest comparison you would run **multiple seeds per
model** and plot the Pareto frontier in two dimensions, so the
trade-off has uncertainty bars on it. The script below does a
single seed per model so that one search fits in a few minutes on
a laptop — but in production work, do not skip the multiple-seed
step.

## Take-Home Summary

- **Multi-objective search** passes a *list* of `Objective`s.
  Keras-Tuner aggregates them per trial and ranks accordingly,
  while keeping the individual metrics recorded so you can
  inspect the trade-off.
- **`BinaryF1Score` with `average="macro"`** is the standard
  metric for *imbalanced* multi-class classification. Plain
  accuracy rewards majority-class collapse; macro-F1 does not.
- The output `DataModel` **must have one boolean field per
  class** for `BinaryF1Score` to read it. The shape of the
  schema is dictated by the metric.
- **`GridSearch`** is the right tuner when the search space is a
  small discrete sweep — for example, "each model in this list,
  exactly once."
- A `HuggingFaceDataset` plus Jinja2 input/output templates is
  how you pipe a public dataset's raw rows into your `DataModel`
  schemas.
- Always **rebuild the winner from `best_hp`** and evaluate on a
  held-out test split; the validation scores were used by the
  oracle and overstate generalization.
- The point of multi-objective search is **the Pareto frontier**,
  not a single winner. The frontier is your engineering tool for
  the cost-vs-accuracy decision.

## API References

- [synalinks.tuners.GridSearch](https://synalinks.github.io/synalinks/Synalinks%20API/Tuners/)
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

# We pull one labeled chunk from the `train` split and then carve a
# validation tail off with `synalinks.datasets.split_train_test`. The
# constants below are stated in those terms: how many rows to load,
# and what fraction of them goes to validation.
NB_TRAINVAL_SAMPLES = 36  # → 24 train / 12 val after splitting
VALIDATION_SPLIT = 1.0 / 3.0
NB_TEST_SAMPLES = 12
EPOCHS = 1
BATCH_SIZE = 4

FOLDER = "guides"
PROJECT_NAME = "emotion_lm_selection"

# Output-schema toggle. False = per-class booleans (one independent
# `bool` per class; multi-label-friendly — the schema does not enforce
# "exactly one True"); True = per-class `synalinks.Score` confidences
# (same independence, but the LM can express graded uncertainty). The
# rest of the code branches on this single flag — see the "Variant"
# section above.
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
    *strict* single-label enforcement, use a `Literal` field with
    the `Categorical*` metrics instead (see the guide prose).

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
    """Materialize one HF split into `(x, y)` arrays of DataModels.

    `synalinks.datasets.load_split` is the one-call shortcut for
    "build a non-streaming HuggingFaceDataset and materialize it" —
    see [Guide 10](https://synalinks.github.io/synalinks/guides/Datasets/).
    """
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
    """Sample hyperparameters and return a compiled classifier.

    The only HP here is which language model to use.
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
    # Demo the dataset-base-class helpers: pull a single labeled chunk
    # from `train`, then slice off a validation tail with
    # `split_train_test`. We could also pass `split="validation"`
    # directly (this dataset ships a native validation split), but the
    # split helper is the general recipe — it works on any source that
    # only ships a single labeled split.
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

    # Rebuild the winner from the best HPs and evaluate on the held-out
    # test slice. Same pattern as keras-tuner's `tuner.get_best_models()`
    # followed by `model.evaluate(...)`.
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
