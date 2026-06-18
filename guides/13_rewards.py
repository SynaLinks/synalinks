# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Rewards

Training, which we will meet in [Guide 15](https://synalinks.github.io/synalinks/guides/Training/), has one input you cannot
afford to choose carelessly: a **reward function** that tells the
optimizer how good each prediction was. Pick the wrong reward and
the optimizer cheerfully optimizes the wrong thing — happily, on
your dime. So before we get to the training loop itself, this
guide makes the reward concrete: what it is, what the built-ins
do, and how to write your own. Rewards are the steering wheel of
the entire training process; this guide is the chapter on
steering.

A **reward** in Synalinks is a function

    r : (y_true, y_pred) → [0, 1]

where `y_true` is the known correct answer (a `DataModel`
instance), `y_pred` is what the program produced (another
`DataModel` instance), and the output is a real number between
`0.0` and `1.0`. Higher is better; `1.0` means perfect; `0.0`
means worst. Rewards play the role that *negative loss* plays in
classical machine learning, but with one crucial difference:
**they do not need to be differentiable.** We never take their
derivative.

That non-differentiability is liberating. A reward can call a
regex, run a unit test, ask another LM to grade the answer, hit
a real database — anything you can express in async Python. The
optimizer treats the reward as a black box and only cares about
the scalar it returns.

## The Picture, In One Diagram

```mermaid
flowchart LR
    P["program(x)"] --> Y["y_pred"]
    G["ground truth"] --> T["y_true"]
    T --> R["reward(y_true, y_pred)"]
    Y --> R
    R --> S["scalar in [0, 1]"]
    S --> O["Optimizer"]
```

One forward pass through the program produces `y_pred`. The
training loop hands `y_true` (from your dataset) and `y_pred`
to the reward, gets back a scalar, and gives it to the optimizer
to score the configuration that produced it.

## The Anatomy of a Reward

Every reward inherits from `synalinks.Reward` and accepts the
same handful of constructor arguments. The most important ones:

- **`in_mask=["field_a", "field_b"]`** — *whitelist*. Only the
  named fields of `y_true` / `y_pred` participate in the score;
  everything else is ignored.
- **`out_mask=["thinking"]`** — *blacklist*. Drop the named
  fields before scoring; keep the rest.
- **`in_mask_pattern=r"^answer.*"`** — like `in_mask` but the
  field set is described by a regex (regular expression — a
  short language for matching text patterns). Useful when you
  have a lot of fields with a common prefix.
- **`out_mask_pattern=r".*_thinking$"`** — same idea, for the
  blacklist side.
- **`reduction="mean"`** — how the per-sample rewards in a
  batch are reduced to a single scalar in the progress logs.
  Valid values are `"mean"` (the default), `"sum"`, `"min"`,
  `"max"`, and `"none"`. `"min"` scores by the worst sample in
  the batch — pessimistic; `"max"` scores by the best —
  optimistic / best-of-N. The per-sample values are always
  preserved for the optimizer; reduction only affects the
  number you see in the log and the number used to compare
  candidates.

You will almost always use `in_mask` or `out_mask` to focus the
reward on the field that actually matters. Without it, an answer
that gets the final `answer` right but has a slightly different
`thinking` string would still score `0.0` under `ExactMatch` —
because *every* field has to match.

## The Four Built-In Rewards

Synalinks ships four reward types out of the box. They cover the
common cases; for anything else, you write a small async
function and wrap it (next section).

### 1. `ExactMatch` — strict equality

The simplest reward there is. Compare the JSON of `y_pred` with
the JSON of `y_true`; return `1.0` if they are equal, else
`0.0`.

```python
reward = synalinks.rewards.ExactMatch(in_mask=["answer"])
```

`ExactMatch` is **discrete** — its only possible values are `0`
or `1` — and that bluntness is both its strength and its weak
spot. The strength: when the right answer is clearly defined
(a number, a label, a name), exact equality is exactly the
right standard. The weak spot: an answer that is *almost*
right scores the same as one that is wildly wrong, which gives
the optimizer no gradient to climb. If the task allows it, a
smoother reward like cosine similarity is easier to learn from.

A second sharp edge: `ExactMatch` does literal string equality.
Trailing whitespace, units, capitalization, and Unicode
normalization all matter. `"42 "` does not equal `"42"`, and
`"Paris"` does not equal `"paris"`. The mitigation is to lock
down the output schema ([Guide 2](https://synalinks.github.io/synalinks/guides/Data%20Models/)) so the LM produces stable
formats, and to use `in_mask` to focus on the field where
strict equality is genuinely the right test.

### 2. `CosineSimilarity` — meaning, not letters

A reward that scores `y_true` and `y_pred` by **semantic**
similarity. It embeds both into vectors using an embedding
model, measures the angle between them, and returns a number
in `[0, 1]`.

```python
reward = synalinks.rewards.CosineSimilarity(
    embedding_model=embedding_model,
    in_mask=["answer"],
)
```

The exact formula is

    r = (cos(emb(y_true), emb(y_pred)) + 1) / 2

The `+1` and `/2` rescale the usual cosine similarity (which
lives in `[-1, 1]`) into `[0, 1]` so it composes cleanly with
the other rewards. `1.0` means "embeddings point in the same
direction"; `0.5` means orthogonal; `0.0` means opposite.

Use `CosineSimilarity` when:

- Paraphrases of the right answer should still earn credit
  ("The capital of France is Paris" vs "Paris").
- The output is free-form text and exact-match is too strict.
- You want a smooth signal the optimizer can climb gradually.

Two cautions:

- **Embeddings cost money.** Every reward call embeds two
  strings. On a long-running training loop this adds up
  quickly. Pick a cheap embedding model unless you have
  measured that you need a stronger one.
- **The scale is the cosine scale.** A score of `0.7` on a
  scaled cosine is "moderately similar," not "70% right."
  Calibrate your expectations to the metric, not to a
  classroom grading scheme.

### 3. `LMAsJudge` — ask a second model

When the task is too open-ended for exact-match and too
nuanced for cosine — for instance, "is this summary
helpful?" or "did the assistant follow the policy?" — let
*another* LM grade the output:

```python
reward = synalinks.rewards.LMAsJudge(
    language_model=judge_model,
    instructions="Score the answer on accuracy and clarity. "
                 "Return a single number in [0, 1].",
)
```

Under the hood `LMAsJudge` is just a `Program` (the
`LMAsJudgeProgram` class) wrapped as a reward — a wrapper
called `ProgramAsJudge`. The judge sees both `y_true` and
`y_pred` (or just `y_pred` if no ground truth is provided), and
returns a numeric score that the framework normalizes to
`[0, 1]`.

Three things to know about LM judges:

- **They are the most flexible reward.** You can grade things
  like helpfulness, tone, format compliance, or anything else
  a regex cannot capture.
- **They are the most expensive.** Every reward call costs
  another LM call. Pair them with a cheap, fast judge model.
- **They are the noisiest.** The judge can be wrong; biases
  in the judge become biases in the optimizer. Always spot-
  check a sample of the judge's verdicts against a human.

### 4. Custom rewards via `RewardFunctionWrapper`

For everything else, write a plain async function and wrap it:

```python
@synalinks.saving.register_synalinks_serializable()
async def length_under(y_true, y_pred, limit=200):
    \"\"\"Score 1.0 if the answer is under `limit` characters, else 0.0.\"\"\"
    answer = y_pred.get("answer", "")
    return 1.0 if len(answer) <= limit else 0.0

reward = synalinks.rewards.RewardFunctionWrapper(
    fn=length_under,
    limit=200,
    name="length_under_200",
)
```

The function signature is `fn(y_true, y_pred, **kwargs)`. Any
keyword arguments you pass to the wrapper are forwarded to the
function on every call. The function must be `async` because
the wrapper awaits it.

The decorator
`@synalinks.saving.register_synalinks_serializable()` is what
lets your custom reward survive `program.save(...)` /
`Program.load(...)` ([Guide 3](https://synalinks.github.io/synalinks/guides/Programs/)): without it, the loader will not
know how to reconstruct the function.

### Bonus: combining rewards

Want both exact-match *and* a length penalty? Wrap them in a
function:

```python
exact = synalinks.rewards.ExactMatch(in_mask=["answer"])

@synalinks.saving.register_synalinks_serializable()
async def exact_and_short(y_true, y_pred):
    em = await exact(y_true, y_pred)
    short = 1.0 if len(y_pred.get("answer", "")) < 80 else 0.0
    return 0.7 * em + 0.3 * short

reward = synalinks.rewards.RewardFunctionWrapper(fn=exact_and_short)
```

The optimizer sees a single number, just as before. You have
hidden a multi-objective reward inside a one-dimensional
scalar with weights you chose.

## Batched Rewards: when samples need each other

The four rewards above score each sample in isolation. Some
ideas — group-relative scores, batch normalization, paired
comparisons — need to see the whole batch at once. For those
cases there is `BatchReward`:

```python
class GroupRelativeReward(synalinks.BatchReward):
    async def call(self, y_true, y_pred):
        # y_true and y_pred are LISTS of length batch_size.
        # Must return a list[float] of the same length.
        raw = [await score_single(t, p) for t, p in zip(y_true, y_pred)]
        mean = sum(raw) / max(1, len(raw))
        return [r - mean for r in raw]   # centered around the batch mean
```

A `BatchReward` subclass receives the **entire batch** at once
and must return one reward per sample. Use it when the meaning
of "good" depends on what the *other* samples in the batch did
— for instance, "this answer is better than the median of
its peers." A `BatchRewardFunctionWrapper` exists for the
common case where you only want to wrap a stateless function.

For most tasks you will not need `BatchReward`. The default
sample-by-sample rewards are simpler to write, faster, and
easier to debug.

## Picking a Reward: a Short Decision Tree

When you do not know which reward to start with, walk this
ladder top to bottom and stop at the first match:

1. **The right answer is a fixed value or label?** Use
   `ExactMatch(in_mask=[field])`.
2. **The right answer is open-ended text where paraphrases
   should earn credit?** Use `CosineSimilarity` with a cheap
   embedding model.
3. **The "right answer" is a judgment call — helpfulness,
   tone, policy compliance?** Use `LMAsJudge` with a small
   judge model. Spot-check the judge.
4. **None of the above?** Write a custom function and wrap it
   with `RewardFunctionWrapper`. If the score depends on the
   whole batch at once, subclass `BatchReward` instead.

## Failure Modes Worth Watching For

- **The reward is constantly `0.0`.** Usually a schema
  mismatch: the field you put in `in_mask` does not exist on
  the output, or the type of the field does not match. Print
  one `(y_pred, y_true)` pair before training to confirm.
- **The reward saturates at `1.0` instantly.** The task is
  too easy for this model — there is no signal left for the
  optimizer to chase. Make the task harder, the reward
  stricter, or move on.
- **The reward rewards the wrong thing.** Classic
  *reward hacking*: the LM learns to game the metric without
  actually getting better at the task. Symptom: training
  reward keeps rising, but a human reading the outputs sees
  them getting worse. Mitigation: spot-check outputs by hand
  every few epochs; add a second, sanity-check reward and
  watch them together.
- **The reward depends on a non-deterministic resource.**
  Using `LMAsJudge` with a high-temperature judge, or a
  reward that hits a flaky web API, produces noisy scores
  that confuse the optimizer. Use a deterministic judge
  (`temperature=0.0`) where you can.

## Take-Home Summary

- A **reward** is a function `(y_true, y_pred) → [0, 1]`.
  Higher is better. Non-differentiable is fine — we never
  take its derivative.
- **`in_mask` / `out_mask`** focus the reward on the
  field(s) that actually matter. Regex variants
  (`in_mask_pattern`, `out_mask_pattern`) handle dynamic
  field sets.
- The **four built-ins** cover most needs:
  `ExactMatch` (strict), `CosineSimilarity` (semantic),
  `LMAsJudge` (judgment calls), and
  `RewardFunctionWrapper` (anything else).
- **`BatchReward`** is the escape hatch when the score needs
  to look at the whole batch at once (group-relative scores,
  paired comparisons). Most tasks do not need it.
- **Reward design *is* task design.** A blunt 0/1 reward
  gives the optimizer no gradient; a smooth reward is easier
  to climb. Spot-check outputs by hand to catch reward
  hacking before it metastasizes.

## API References

- [synalinks.Reward](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/)
- [synalinks.rewards.ExactMatch](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/ExactMatch%20reward/)
- [synalinks.rewards.CosineSimilarity](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/CosineSimilarity%20reward/)
- [synalinks.rewards.LMAsJudge](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/LMAsJudge%20reward/)
- [synalinks.rewards.RewardFunctionWrapper](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/Reward%20wrappers/)
- [synalinks.BatchReward](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/Batch%20reward%20wrappers/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks


# =============================================================================
# Data Models
# =============================================================================


class Question(synalinks.DataModel):
    """A question for the program to answer."""

    question: str = synalinks.Field(description="The question to answer")


class Answer(synalinks.DataModel):
    """An answer to a question, with reasoning."""

    thinking: str = synalinks.Field(description="Step-by-step reasoning")
    answer: str = synalinks.Field(description="The final answer")


# =============================================================================
# A small custom reward
# =============================================================================


@synalinks.saving.register_synalinks_serializable()
async def length_under(y_true, y_pred, limit=200):
    """Score 1.0 if the answer is at most `limit` characters, else 0.0."""
    if y_pred is None:
        return 0.0
    answer = y_pred.get("answer", "") or ""
    return 1.0 if len(answer) <= limit else 0.0


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()
    synalinks.enable_logging()

    # Build a tiny QA program. The reward types we demonstrate below all
    # operate on `(y_true, y_pred)` pairs that match the `Answer` schema.
    lm = synalinks.LanguageModel(model="ollama/mistral:latest")

    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="qa_program",
    )
    program.summary()

    y_pred = await program(Question(question="What is the capital of France?"))
    y_true = Answer(thinking="France has Paris as its capital.", answer="Paris")
    print(f"y_pred: {y_pred.get_json()}")
    print(f"y_true: {y_true.get_json()}")

    # -------------------------------------------------------------------------
    # 1. ExactMatch — strict equality on the `answer` field
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. ExactMatch(in_mask=['answer'])")
    print("=" * 60)
    # Rewards accept either a DataModel or a JsonDataModel for both
    # arguments — the framework converts internally — so we can pass
    # `y_true` straight in without `.to_json_data_model()`.
    em = synalinks.rewards.ExactMatch(in_mask=["answer"])
    score = await em(y_true, y_pred)
    print(f"  reward: {score:.3f}")

    # -------------------------------------------------------------------------
    # 2. Custom reward — wrap a plain async function
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. RewardFunctionWrapper(fn=length_under, limit=200)")
    print("=" * 60)
    short = synalinks.rewards.RewardFunctionWrapper(
        fn=length_under,
        limit=200,
        name="length_under_200",
    )
    score = await short(y_true, y_pred)
    print(f"  reward: {score:.3f}")

    # -------------------------------------------------------------------------
    # 3. Compose two rewards by hand into one scalar
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. Blended reward: 0.7 * ExactMatch + 0.3 * length_under_200")
    print("=" * 60)

    @synalinks.saving.register_synalinks_serializable()
    async def exact_and_short(y_true, y_pred):
        em_score = await em(y_true, y_pred)
        short_score = await length_under(y_true, y_pred, limit=200)
        return 0.7 * em_score + 0.3 * short_score

    blended = synalinks.rewards.RewardFunctionWrapper(fn=exact_and_short)
    score = await blended(y_true, y_pred)
    print(f"  reward: {score:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
