# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Metrics

[Guide 13](https://synalinks.github.io/synalinks/guides/Rewards/) covered **rewards** — the single number the *optimizer*
uses to decide whether one program is better than another. This
guide is about **metrics**, the other half of the scoring story.
A metric is what *you* watch. Some of them you would write down
in a paper; others you would put on a dashboard next to a billing
graph. Both reward and metric are scores, but they answer
different questions and they go to different consumers.

It is worth getting the distinction clear before we list the
options.

## Reward vs. Metric

- A **reward** drives optimization. There is at most one reward
  per `compile()`, and the optimizer treats it as the truth.
- A **metric** is *observed*, not optimized. You can attach as
  many as you like; the framework reports them in the progress
  bar, in `history.history`, and (in [Guide 17](https://synalinks.github.io/synalinks/guides/Hyperparameter%20Search/) / [Guide 18](https://synalinks.github.io/synalinks/guides/Multi-Objective%20LM%20Selection/)) to the tuner's
  oracle. The optimizer never reads them.

In other words: reward is the *steering wheel*, metrics are the
*dashboard*. You can stare at the dashboard all day, but if the
steering wheel points the wrong way, the car still goes off the
road.

A handy convention to internalize from [Guide 17](https://synalinks.github.io/synalinks/guides/Hyperparameter%20Search/): every training
metric `m` you declare also gets a mirrored **`val_m`** measured
on the validation split, *if* you pass `validation_split=` or
`validation_data=` to `fit()`. So adding one metric to `compile`
usually gives you two columns in the log.

## The Picture

```mermaid
flowchart LR
    P["program(x)"] --> Y["y_pred"]
    G["dataset"] --> T["y_true"]
    T --> R["Reward (drives optimizer)"]
    Y --> R
    R --> O["Optimizer"]

    Y -.-> M["Metrics (observed only)"]
    T -.-> M
    M --> L["progress bar, history, tuner"]
```

Solid arrows show the optimization loop you have known since
[Guide 15](https://synalinks.github.io/synalinks/guides/Training/). The dashed arrows are metrics — the same prediction and
ground truth flow into them, but nothing they say loops back
into the optimizer. They exist for you, not for the algorithm.

## Wiring Metrics Into a Program

The contract is identical to rewards. You build them once and
pass them through `compile()`:

```python
program.compile(
    reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
    optimizer=synalinks.optimizers.RandomFewShot(),
    metrics=[
        synalinks.metrics.Accuracy(),
        synalinks.metrics.F1Score(average="macro"),
        synalinks.metrics.Cost(),                  # operational
        synalinks.metrics.AvgCostPerCall(),        # operational
    ],
)
```

The `metrics=` argument is a **list**. The order matters only
for display — the framework will compute every metric on every
batch and reduce it over the epoch.

Every metric, like every reward, accepts the masking arguments
`in_mask`, `out_mask`, `in_mask_pattern`, and `out_mask_pattern`
(see [Guide 13](https://synalinks.github.io/synalinks/guides/Rewards/)). They behave identically. Use them to focus the
metric on the field(s) where it actually means something.

## The Three Families of Metric

Synalinks groups its built-in metrics into three loose families.
You will end up with metrics from at least two of them on any
non-trivial training run.

```mermaid
graph TD
    A["synalinks.metrics"] --> B["Quality metrics<br/>(accuracy, F1, precision/recall, similarity)"]
    A --> C["Reduction wrappers<br/>(Mean, Sum, MeanMetricWrapper)"]
    A --> D["Operational metrics<br/>(cost, tokens, throughput, cache)"]
```

### Family 1 — Quality Metrics

These tell you how well the program is performing the task.
The right one depends on what your output looks like.

**Free-text or QA-style answers.** Use word-level metrics —
they tokenize both `y_true` and `y_pred` and compare the token
sets.

- **`Accuracy`** — per-field token Jaccard index:
  `|truth ∩ pred| / |truth ∪ pred|`. Tolerant of extra or
  missing words; the closest token-level analogue of "how
  much of the right answer did we get."
- **`F1Score` / `FBetaScore`** — the harmonic mean of
  token-level precision and recall. The standard QA metric.
  `F1` weights precision and recall equally; `FBeta` lets you
  tilt toward one (`beta > 1` favors recall, `beta < 1`
  favors precision).
- **`Precision` / `Recall`** — if you want to inspect the
  components of F1 directly.

All four accept an `average=` argument (`None`, `"micro"`,
`"macro"`, `"weighted"`) that controls how multiple output
fields are combined into one number. We met `macro` in
[Guide 18](https://synalinks.github.io/synalinks/guides/Multi-Objective%20LM%20Selection/): it averages per-field scores with equal weight, so
rare classes count as much as common ones.

**Multi-class classification with one boolean per class.** Use
the **Binary*** variants. They treat each class field as an
independent 0/1 prediction.

- **`BinaryAccuracy`**, **`BinaryF1Score`**,
  **`BinaryFBetaScore`**, **`BinaryPrecision`**,
  **`BinaryRecall`** — same recipe as the QA-level ones, but
  scored field-by-field on booleans. This is the metric
  family [Guide 18](https://synalinks.github.io/synalinks/guides/Multi-Objective%20LM%20Selection/) used for emotion classification.

**Multi-class classification with one list of labels.** Use the
**Categorical*** variants — they aggregate over a single list-
valued field.

- **`CategoricalAccuracy`**, **`CategoricalF1Score`**
  (aliased as **`ListF1Score`**), **`CategoricalFBetaScore`**
  (aliased as **`ListFBetaScore`**), **`CategoricalPrecision`**,
  **`CategoricalRecall`**.

**Free-text where paraphrases should earn credit.** The
**regression** flavor:

- **`CosineSimilarity`** — same idea as the reward of the
  same name ([Guide 13](https://synalinks.github.io/synalinks/guides/Rewards/)), but used as an observed metric. Needs
  an `embedding_model`.

A quick rule of thumb to navigate the prefixes:

- No prefix → **word-level** (QA-style, free text).
- **`Binary`** → one boolean per class.
- **`Categorical`** → one list of labels.

### Family 2 — Reduction Wrappers

These are bookkeeping helpers that combine many per-step
values into one number.

- **`Mean`** — running average of the values fed in.
- **`Sum`** — running sum.
- **`MeanMetricWrapper(fn=..., name=...)`** — wrap any
  reward-shaped function `(y_true, y_pred) -> float` into a
  metric, tracking its running mean.

You met `MeanMetricWrapper` in [Guide 15](https://synalinks.github.io/synalinks/guides/Training/)'s compile block,
turning the same `ExactMatch` you used as the reward into an
observed mean metric called `mean_reward`. That is the
canonical use of this wrapper: "I want to see the reward as a
metric too, averaged across the epoch."

### Family 3 — Operational Metrics

These tell you how *expensive* the program was. They are the
metrics you would put on a billing dashboard, not in a paper.
Three sources of LM calls show up in a Synalinks run:

- the **program** itself (your `Generator`s, agents, …),
- the **optimizer** (some optimizers call LMs to mutate
  prompts),
- the **rewards** (e.g. `LMAsJudge` calls a judge LM).

Each operational metric exists in three variants, one per
source. The default name with no prefix sums all three; the
**`Optimizer`** and **`Reward`** prefixes restrict to one
source.

#### LM-call metrics

- **`Cost`**, **`OptimizerCost`**, **`RewardCost`** — dollars
  per epoch.
- **`AvgCostPerCall`**, **`AvgOptimizerCostPerCall`**,
  **`AvgRewardCostPerCall`** — dollars per LM call.
- **`InputTokens`**, **`OutputTokens`**, **`TotalTokens`** —
  cumulative token counts; same `Optimizer*` / `Reward*`
  variants.
- **`AvgInputTokensPerCall`**, **`AvgOutputTokensPerCall`** —
  per-call averages.
- **`ReasoningTokens`**, **`ReasoningTokenShare`** — for
  reasoning models, how many tokens were spent on hidden
  reasoning vs. visible output.
- **`CachedTokens`**, **`CacheCreationTokens`**,
  **`CacheHitRate`** — for providers that support prompt
  caching, how often the cache is hitting.
- **`Throughput`**, **`TokensPerSecond`** — speed of the
  pipeline.

#### Embedding-model metrics

If you use `CosineSimilarity` as a reward or anywhere else
that calls an embedding model, the same family exists for
embeddings:
**`EmbeddingCost`**, **`EmbeddingTokens`**,
**`EmbeddingVectors`**, **`EmbeddingCacheHitRate`**,
**`EmbeddingThroughput`**, and so on — with `Optimizer*` /
`Reward*` variants when applicable.

#### Program-level metrics

These describe the program's invocation pattern, independent
of which LMs it called:

- **`ProgramCalls`** — how many times the program was
  invoked.
- **`ProgramCallsPerSecond`** — call rate.
- **`ProgramCost`** — total cost across every nested LM.
- **`ProgramElapsedTime`** — wall-clock time spent.
- **`ProgramAvgCostPerInvocation`** — cost / call.

A practical starter set for any real training run:

```python
metrics=[
    synalinks.metrics.Cost(),
    synalinks.metrics.AvgCostPerCall(),
    synalinks.metrics.TotalTokens(),
    synalinks.metrics.ProgramElapsedTime(),
]
```

If you also use an `LMAsJudge` reward, add
`RewardCost()` and `AvgRewardCostPerCall()` — that is usually
where the surprise bills hide.

## Reading `history` and the Progress Bar

After `program.fit(...)` returns a `history` object, the dict
`history.history` is keyed by metric name. Every metric named
`m` produces a `m` column (training) and, if you supplied
validation data, a `val_m` column. With the example from
[Guide 15](https://synalinks.github.io/synalinks/guides/Training/):

```python
print(list(history.history.keys()))
# ['mean_reward', 'reward', 'val_mean_reward', 'val_reward']
```

`reward` is always present — it is the reward driving training.
The rest is whatever you put in `metrics=[...]`. Plot them
with any library you like; they are plain lists of floats.

## Custom Metrics

For anything Synalinks does not ship out of the box, subclass
`synalinks.Metric`. The contract is three methods:

- **`update_state(y_true, y_pred)`** — called once per
  sample; mutate the metric's internal variables.
- **`result()`** — called at the end of each batch / epoch;
  return the current scalar value.
- **`reset_state()`** — called at the start of each epoch.

For the common case where your metric is just a running mean
of some scalar function, do *not* subclass — use
`MeanMetricWrapper`:

```python
@synalinks.saving.register_synalinks_serializable()
async def starts_with_capital(y_true, y_pred):
    answer = y_pred.get("answer", "") or ""
    return 1.0 if (answer[:1].isupper()) else 0.0

metric = synalinks.metrics.MeanMetricWrapper(
    fn=starts_with_capital,
    name="capital_rate",
)
```

The wrapper handles `update_state` / `result` / `reset_state`
for you.

## Picking Metrics: a Short Recipe

A pragmatic default for a non-trivial training run:

1. **The reward, exposed as a metric** so you can watch its
   running mean across the epoch:
   `MeanMetricWrapper(fn=reward, name="mean_reward")`.
2. **A quality metric in the right family** — `F1Score`,
   `BinaryF1Score`, or `CategoricalF1Score` depending on the
   shape of your output. F1 is more informative than plain
   accuracy on most tasks.
3. **One or two operational metrics**, at least `Cost()` and
   `TotalTokens()`. They are cheap to record and they often
   reveal something surprising (the optimizer's calls were
   half your bill; the reward's calls were the other half).
4. **One thing you care about that no built-in covers**,
   wrapped via `MeanMetricWrapper`. Format compliance, output
   length, refusal rate — anything that would make you
   nervous if it changed without telling you.

## Take-Home Summary

- A **reward** drives optimization (one per `compile`). A
  **metric** is observed only (any number). Both share the
  same masking API ([Guide 13](https://synalinks.github.io/synalinks/guides/Rewards/)).
- Three families: **quality** (Accuracy, F1, FBeta,
  Precision/Recall in Binary / Categorical / word-level
  variants), **reductions** (`Mean`, `Sum`,
  `MeanMetricWrapper`), and **operational** (cost, tokens,
  throughput, cache, broken down by `Program` /
  `Optimizer` / `Reward` source).
- Naming convention: **no prefix** → word-level / QA;
  **`Binary`** → one boolean per class;
  **`Categorical`** → one list of labels.
- Always carry **at least one operational metric** when
  training a real LM program. Surprise bills come from where
  you are not looking.
- **`MeanMetricWrapper`** is the easiest way to add a custom
  metric: write an async `(y_true, y_pred) -> float`
  function and wrap it.

## API References

- [synalinks.Metric](https://synalinks.github.io/synalinks/Synalinks%20API/Metrics/Base%20Metric%20class/)
- [Accuracy / F1 / Precision / Recall](https://synalinks.github.io/synalinks/Synalinks%20API/Metrics/)
- [BinaryF1Score](https://synalinks.github.io/synalinks/Synalinks%20API/Metrics/FScore%20metrics/)
- [CategoricalF1Score (alias ListF1Score)](https://synalinks.github.io/synalinks/Synalinks%20API/Metrics/FScore%20metrics/)
- [MeanMetricWrapper](https://synalinks.github.io/synalinks/Synalinks%20API/Metrics/Metric%20wrappers%20and%20reduction%20metrics/)
- [Cost / AvgCostPerCall / TotalTokens / CacheHitRate](https://synalinks.github.io/synalinks/Synalinks%20API/Metrics/)
- [ProgramCost / ProgramElapsedTime](https://synalinks.github.io/synalinks/Synalinks%20API/Metrics/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks


# =============================================================================
# Data Models
# =============================================================================


class MathProblem(synalinks.DataModel):
    """A math problem to solve."""

    problem: str = synalinks.Field(description="The math problem to solve")


class MathAnswer(synalinks.DataModel):
    """An answer to a math problem."""

    thinking: str = synalinks.Field(description="Step-by-step reasoning")
    answer: str = synalinks.Field(description="The final numerical answer")


# =============================================================================
# A small custom metric: how often the answer starts with a digit.
# =============================================================================


@synalinks.saving.register_synalinks_serializable()
async def starts_with_digit(y_true, y_pred):
    """Score 1.0 if the answer's first character is a digit, else 0.0."""
    if y_pred is None:
        return 0.0
    answer = y_pred.get("answer", "") or ""
    return 1.0 if answer[:1].isdigit() else 0.0


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    lm = synalinks.LanguageModel(model="ollama/mistral:latest")

    inputs = synalinks.Input(data_model=MathProblem)
    outputs = await synalinks.Generator(
        data_model=MathAnswer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="math_program",
    )

    # The reward ([Guide 13](https://synalinks.github.io/synalinks/guides/Rewards/)) drives the optimizer.
    reward = synalinks.rewards.ExactMatch(in_mask=["answer"])

    # The metrics are observed only. We mix all three families:
    #   - quality:      F1Score on the answer field
    #   - reduction:    MeanMetricWrapper exposing the reward as a metric,
    #                   plus a custom format-compliance metric
    #   - operational:  Cost, AvgCostPerCall, TotalTokens, ProgramElapsedTime
    program.compile(
        reward=reward,
        optimizer=synalinks.optimizers.RandomFewShot(),
        metrics=[
            # Quality:
            synalinks.metrics.F1Score(in_mask=["answer"]),
            # Reduction wrappers:
            synalinks.metrics.MeanMetricWrapper(fn=reward, name="mean_reward"),
            synalinks.metrics.MeanMetricWrapper(
                fn=starts_with_digit, name="starts_with_digit"
            ),
            # Operational:
            synalinks.metrics.Cost(),
            synalinks.metrics.AvgCostPerCall(),
            synalinks.metrics.TotalTokens(),
            synalinks.metrics.ProgramElapsedTime(),
        ],
    )

    print("Compiled program with metrics from all three families:")
    print("  - F1Score(in_mask=['answer'])")
    print("  - MeanMetricWrapper(fn=ExactMatch)   # exposes the reward")
    print("  - MeanMetricWrapper(fn=starts_with_digit)   # custom")
    print("  - Cost / AvgCostPerCall / TotalTokens / ProgramElapsedTime")

    print(
        "\nNothing was trained in this guide; running program.fit(...)"
        " would now populate every one of these columns in history.history."
    )


if __name__ == "__main__":
    asyncio.run(main())
