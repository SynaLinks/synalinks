"""
# Training

So far we have *used* language models. In this guide we **improve**
them — without ever touching their weights. The pieces have been
assembled in the last four guides: [Guide 11](https://synalinks.github.io/synalinks/guides/Datasets/) explained where the
training data comes from; [Guide 12](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/) explained what trainable state
lives on a `Module`; [Guide 13](https://synalinks.github.io/synalinks/guides/Rewards/) explained how a reward function
scores a prediction; [Guide 14](https://synalinks.github.io/synalinks/guides/Metrics/) explained the metrics you watch
while it all runs. This guide is where those four ingredients come
together.

Training, in Synalinks, means: take a set of examples where you
already know the right answer (or not if you use an LMasJudge),
run the program on each one, score how well it did with the reward,
and let an **optimizer** rewrite the program's trainable variables
to do better next time. The LM's internal weights stay frozen the
whole time; what changes is the JSON state we place in front of it.

We need to be precise with terminology in this guide, because words
like *loss*, *gradient*, *optimizer*, *epoch*, and *batch* are
borrowed from classical deep learning — and they mean something
subtly different here. Confusing the two mental models is the single
most common pitfall for new users, so we spend a section up front
contrasting them.

## What We Are *Not* Doing: a contrast with classical ML

In a typical "train a neural net" course, a model is a mathematical
function `f_θ(x)` with a big bag of numeric parameters `θ` ("theta")
— the **weights**. Training searches for values of `θ` that make the
model's outputs match the targets on a dataset. The classical recipe
looks like:

    L(θ) = (1/N) * Σ_i loss(f_θ(x_i), y_i)

In words: average the per-example error over the `N` training
examples. That average is called the **empirical risk**. Classical
training then computes the partial derivative of `L` with respect to
every weight (a procedure called **backpropagation**) and nudges each
weight a little in the direction that lowers `L` (this is **gradient
descent**). The thing being mutated is a giant array of floating-point
numbers inside the network.

Synalinks does **none of this**. A Synalinks `Program` is a DAG of
`Module`s. The parameters exposed to training are **JSON objects**
attached to the modules — each one obeying a fixed schema (a
subclass of `synalinks.Trainable`, the topic of [Guide 12](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/)). On a
`Generator`, the two trainable variables you will meet most often
are:

- the **instruction** variable — a JSON object whose primary
  field is a string of system-prompt text the LM reads on every
  call, and
- the **examples** variable — a JSON object whose primary field
  is a list of `(input, output)` pairs shown to the LM as
  few-shot demonstrations.

Both are *special cases*: a trainable variable can in general
hold any structured data its schema permits (you will see a
`Persona` variable with a custom field in [Guide 12](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/)). We never
look inside the LM, and we never compute gradients. The LM is
treated as a complete black box: we call it, we get text back,
we score the text. What we optimize is the **context** — the
trainable JSON objects — that drives the LM on each call.

```mermaid
graph LR
    subgraph SGD["Classical SGD"]
        A1["batch (x, y)"] --> B1["forward f_theta"]
        B1 --> C1["loss"]
        C1 --> D1["backprop grad L"]
        D1 --> E1["theta := theta - eta * grad"]
    end
    subgraph ICO["Synalinks in-context optimization"]
        A2["batch (x, y)"] --> B2["forward Program"]
        B2 --> C2["reward r in [0, 1]"]
        C2 --> D2["optimizer.optimize"]
        D2 --> E2["update instruction and examples variables"]
    end
```

Two consequences follow from this design:

1. **No differentiation needed.** Any LM you can call through an API
   works — open or closed, local or hosted. We never need access to
   the model's internal weights.
2. **Interpretable state.** After training, the learned state is
   a small collection of JSON objects — typically one with an
   instruction string inside, and one with a list of
   `(input, output)` pairs inside. You can open the saved JSON
   file in a text editor and literally **read** what was learned.
   Compare this to a 7-billion-parameter neural network, where
   the "learning" is an inscrutable tensor.

## The Cast of Characters

Before the precise definitions, here is the cast in plain English:

- A **Module** is a building block you can call like a function.
- A **Generator** is the specific kind of module that talks to the
  LM.
- A **Reward** is a score saying how good one output was.
- A **Metric** summarizes rewards over many examples.
- An **Optimizer** is the thing that rewrites the instruction and
  examples based on what it has seen.
- An **epoch** is one full sweep through your training data.

Now the precise versions, used consistently throughout this guide:

- **Module.** A unit of computation `m : (input, state) -> output`,
  where `state` may include zero or more trainable variables (values
  the optimizer is allowed to change).
- **Generator.** A `Module` that wraps a `LanguageModel` and emits
  a structured output. It exposes two trainable variables, each
  one a JSON object obeying a `Trainable` schema:
    - `instruction_variable` — a JSON object whose primary field
      is the system instruction shown to the LM (think: "You are
      an expert at solving math problems..."). The variable also
      carries optimizer bookkeeping fields; see [Guide 12](https://synalinks.github.io/synalinks/guides/Trainable%20Variables/).
    - `examples_variable` — a JSON object whose primary field is
      a list of `(input, target)` pairs inserted into the prompt
      as *few-shot demonstrations* (worked examples the LM can
      imitate).
- **Reward.** A function `r : (y_pred, y_true) -> [0, 1]`, where
  `y_pred` is the program's output and `y_true` is the known correct
  answer. Higher is better; `1.0` means perfect. It plays the role
  that *negative loss* plays in classical ML. It does not need to be
  differentiable, because we never take its derivative.
- **Metric.** A number tracked per step and aggregated over an epoch.
  The default `mean_reward` is just the running average of the reward
  across the epoch.
- **Optimizer.** An object with an
  `optimize(trainable_variables, ...)` method. Given the current
  variables and the `(y_pred, y_true, reward)` triples it has
  observed, it proposes new JSON values for those variables — a
  new instruction, a new list of examples, a new persona record,
  whatever the variables' schemas describe.
- **Epoch.** One full pass over the training set.
- **Batch / step.** A *batch* is a chunk of examples processed together;
  a *step* is one such chunk. With the default settings here, the data
  loader yields one example per step, so an epoch over `N_train`
  examples produces `N_train` training steps followed by a validation
  pass (a check on data the program has not trained on).

## What Actually Changes When You Call `fit`

```mermaid
graph TD
    P["Program"] --> G["Generator"]
    G --> IV["instruction_variable: str"]
    G --> EV["examples_variable: list of (x, y)"]
    O["Optimizer"] -->|"writes"| IV
    O -->|"writes"| EV
    R["Reward"] -->|"reads forward output"| O
```

For each training step `i`:

1. The program is run on input `x_i` to produce a prediction `y_pred_i`.
2. We compute `r_i = reward(y_pred_i, y_true_i)`: a single number
   saying how good that prediction was.
3. The optimizer records the tuple `(x_i, y_true_i, y_pred_i, r_i)` in
   a buffer (it remembers everything it has seen this epoch).
4. After the epoch finishes (or whenever it is configured to fire),
   the optimizer rewrites `instruction_variable` and/or
   `examples_variable` using what is in its buffer.

A crucial property: **between two steps inside the same epoch, the
trainable state does not change.** By default, updates happen only
at epoch boundaries. So if you watch the reward stay flat through an
epoch and worry, do not — that is the design, not a bug. Improvement
shows up *across* epochs, not within one.

## The Training Loop in Code

```python
import synalinks

program = synalinks.Program(inputs=inputs, outputs=outputs)

program.compile(
    optimizer=synalinks.optimizers.RandomFewShot(nb_max_examples=3),
    reward=synalinks.ExactMatch(in_mask=["answer"]),
    metrics=[
        synalinks.metrics.MeanMetricWrapper(
            fn=synalinks.ExactMatch(in_mask=["answer"]),
            name="mean_reward",
        ),
    ],
)

history = await program.fit(
    x=x_train,
    y=y_train,
    epochs=2,
    validation_split=0.2,
    verbose=1,
)

program.save("trained_program.json")
```

## Data Format

`fit` expects NumPy arrays where each element is a `DataModel`
instance (a Pydantic object). You **must** pass `dtype="object"` so
NumPy stores the objects as-is rather than trying to fit them into a
numeric dtype:

```python
import numpy as np

x_train = np.array([InputModel(field="..."), ...], dtype="object")
y_train = np.array([OutputModel(field="..."), ...], dtype="object")
```

The arrays are matched by position: the *i*-th element of `y_train`
is the target for the *i*-th element of `x_train`. To carve off a
validation set, use `validation_split=p` (which reserves the last
`p` fraction of the array; e.g. `0.2` holds out the last 20%), or
pass an explicit pair via `validation_data=(x_val, y_val)`.

A common trap: do **not** omit `dtype="object"`. Without it, NumPy
will try to fit the Pydantic objects into a structured numeric dtype
and silently produce garbage that does not crash but does not work
either.

## Optimizers

### `RandomFewShot`

The simplest **in-context optimizer** (an optimizer that improves
the program by editing the LM's prompt instead of its weights). After
each epoch, it picks `k = nb_max_examples` examples whose reward was
above a threshold and pastes them into `examples_variable` as worked
demonstrations for the LM to imitate. Cost per epoch is `O(N_train)`
LM calls — one per training example — plus negligible bookkeeping.

```python
optimizer = synalinks.optimizers.RandomFewShot(nb_max_examples=3)
```

`RandomFewShot` is the right baseline to start with. If it already
saturates the reward (pushes it to its ceiling), no more sophisticated
optimizer will improve on it.

### `OMEGA`

An **evolutionary** optimizer — that is, an optimizer modeled on
biological evolution. It keeps a **population** of candidate prompt
variants, scores each on a held-out slice of data, throws out the
worst ones (**selection**), and randomly tweaks the survivors
(**mutation**). Reach for `OMEGA` when `RandomFewShot` plateaus well
below `1.0` and you suspect the bottleneck is the instruction itself,
not the examples.

`OMEGA` needs both a `language_model` (to propose the mutated
variants) and an `embedding_model` (to measure how novel each
candidate is, so the population stays diverse):

```python
optimizer = synalinks.OMEGA(
    language_model=lm,
    embedding_model=embedding_model,
    population_size=10,
)
```

## Rewards

A reward is your "how good was this answer?" function. Synalinks
ships three useful ones out of the box.

### `ExactMatch`

Compare only the fields named in `in_mask`. Return `1` if every one
of those fields is exactly equal in `y_pred` and `y_true`, else `0`.
This is a **discrete** (zero-or-one) reward that is
**non-differentiable** — there is no notion of "almost correct."
Neither matters here, because we never take derivatives.

```python
reward = synalinks.ExactMatch(in_mask=["answer"])
```

A common failure mode: trailing whitespace, missing/extra units, or
different capitalization in the LM output drops the reward to `0`
even when the answer is semantically right ("42 " vs "42"). The
mitigation is to lock down the schema (e.g. `answer: str` with a
precise description) so the LM produces stable formats.

### `CosineSimilarity`

Formula: `r = max(0, cos(emb(y_pred), emb(y_true)))`. In words: turn
both strings into **embedding vectors** (numeric representations of
meaning, the same kind we used in [Guide 7](https://synalinks.github.io/synalinks/guides/Knowledge%20Base/)), measure the angle between
them, and use that as the score. Use this reward when paraphrases of
the right answer should still earn partial credit.

```python
reward = synalinks.CosineSimilarity(
    embedding_model=embedding_model,
    in_mask=["answer"],
)
```

### `LMAsJudge`

A **second** LM reads the output and scores it against a rubric you
provide. This is the most flexible reward (it can grade open-ended
answers a regex never could), the most expensive (every reward
computation costs an LM call), and the noisiest (the judge can be
wrong, and biases in the judge become biases in the optimizer).

```python
reward = synalinks.LMAsJudge(
    language_model=judge_model,
    instructions="accuracy, helpfulness, clarity",
)
```

## Metrics

A metric **reduces** (combines) per-step rewards into a single
tracked number.
`MeanMetricWrapper(fn=reward, name="mean_reward")` keeps the running
average of `fn` across the epoch. After `fit` returns,
`history.history` is a dictionary keyed by metric name. When you use
a validation split, every training metric `m` has a mirrored
`val_m` measured on the held-out data.

## A Concrete Example: tiny arithmetic task

The runnable section below trains on an 8-example arithmetic dataset
with `validation_split=0.2` (the last 20% is held out for validation)
and `epochs=2`, using a local `ollama/mistral:latest`. The dataset
is deliberately tiny so the guide finishes in a few minutes; do *not*
read the reported rewards as evidence of model quality at scale.

Expected output (numbers match a fresh run on the same model, up to the
usual nondeterminism of LM sampling — the same prompt can give slightly
different text on different runs):

```
============================================================
Step 1: Prepare Training Data
============================================================

Training examples: 8
Test examples: 2

============================================================
Step 2: Create and Compile Program
============================================================

Program compiled with:
  - Optimizer: RandomFewShot(nb_max_examples=3)
  - Reward: ExactMatch(in_mask=['answer'])

============================================================
Step 3: Train the Program
============================================================
Epoch 1/2
6/6 - 24s 4s/step - mean_reward: 1.0000 - reward: 1.0000 - val_mean_reward: 1.0000 - val_reward: 1.0000
Epoch 2/2
6/6 - 36s 6s/step - mean_reward: 1.0000 - reward: 1.0000 - val_mean_reward: 1.0000 - val_reward: 1.0000

Training complete!
History keys: ['mean_reward', 'reward', 'val_mean_reward', 'val_reward']

============================================================
Step 4: Test Trained Program
============================================================

Problem: 9 + 1
Answer: 10

Problem: 5 * 5
Answer: 25

Problem: 20 - 8
Answer: 12

============================================================
Step 5: Save and Load
============================================================

Saved trained program to trained_math_solver.json
Loaded program: math_solver

Loaded program test: 100 / 10 = 10
```

A reward of `1.0` on every example here means the task is too easy
for this model — the optimizer has nothing to fix. On a more
challenging task you should expect the training reward to start
below `1.0` and rise across epochs, with `val_reward` (the reward on
held-out data) lagging slightly behind `reward`. That small gap is
the standard signature of mild train/validation mismatch, and it is
normal.

## Diagnostics and Failure Modes

- **Reward is exactly `0.0` on every step.** Either the schema is
  wrong (e.g. the LM emits a number where you declared a string), or
  the keys you listed in `in_mask` do not exist on the output. Print
  one `(y_pred, y_true)` pair *before* training to confirm shapes
  match.
- **`mean_reward` rises, but `val_mean_reward` falls.** This is the
  classic shape of **overfitting** — memorizing the training data
  without learning anything that generalizes. The few-shot pool is
  filled with examples too similar to the training split. Shrink
  `nb_max_examples`, or enlarge the training set.
- **`mean_reward` is flat at a high value.** The task is saturated:
  either the program is already optimal, or the reward is too coarse
  (only `0` or `1`) to surface the remaining errors.

## Best Practices, Distilled

1. **Start with `RandomFewShot(nb_max_examples=3)` + `ExactMatch`.**
   Move to anything fancier only when this baseline plateaus below
   the accuracy you need.
2. **Always pass `validation_split` or `validation_data`.** A curve
   with only training scores cannot tell you whether you are
   overfitting.
3. **Save with `program.save("path.json")` after every successful
   run.** The file is human-readable; `diff` it against the previous
   run to see exactly what the optimizer learned.
4. **Treat reward design as task design.** A reward that is only `0`
   or `1`, with nothing in between, gives the optimizer no signal
   about *how close* a wrong answer was. Prefer a reward that varies
   smoothly with output quality whenever the task allows it.

## API References

- [Program.compile](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
- [Program.fit](https://synalinks.github.io/synalinks/Synalinks%20API/Programs%20API/The%20Program%20class/)
- [RandomFewShot](https://synalinks.github.io/synalinks/Synalinks%20API/Optimizers%20API/RandomFewShot/)
- [ExactMatch](https://synalinks.github.io/synalinks/Synalinks%20API/Rewards/ExactMatch%20reward/)
- [Metrics](https://synalinks.github.io/synalinks/Synalinks%20API/Metrics/)
"""

import asyncio
import os

from dotenv import load_dotenv

import synalinks

# =============================================================================
# Data Models
# =============================================================================


class MathProblem(synalinks.DataModel):
    """A math problem."""

    problem: str = synalinks.Field(description="The math problem to solve")


class MathAnswer(synalinks.DataModel):
    """A math answer."""

    thinking: str = synalinks.Field(description="Step by step calculation")
    answer: str = synalinks.Field(description="The numerical answer only")


# =============================================================================
# Main Demonstration
# =============================================================================


async def main():
    load_dotenv()
    synalinks.clear_session()

    # synalinks.enable_observability(
    #     tracking_uri="http://localhost:5000",
    #     experiment_name="guide_15_training",
    # )

    lm = synalinks.LanguageModel(model="ollama/mistral:latest")

    # -------------------------------------------------------------------------
    # Prepare Training Data (as NumPy arrays)
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Prepare Training Data")
    print("=" * 60)

    import numpy as np

    # Training data: separate arrays for inputs (x) and expected outputs (y).
    # We keep the dataset small (8 train + 2 test, 2 epochs) so that the guide
    # completes in a few minutes against a local ollama model.
    x_train = np.array(
        [
            MathProblem(problem="2 + 3"),
            MathProblem(problem="5 * 4"),
            MathProblem(problem="10 - 3"),
            MathProblem(problem="8 / 2"),
            MathProblem(problem="3 + 3 + 3"),
            MathProblem(problem="7 * 2"),
            MathProblem(problem="15 - 5"),
            MathProblem(problem="12 / 3"),
        ],
        dtype="object",
    )
    y_train = np.array(
        [
            MathAnswer(thinking="2 + 3 = 5", answer="5"),
            MathAnswer(thinking="5 * 4 = 20", answer="20"),
            MathAnswer(thinking="10 - 3 = 7", answer="7"),
            MathAnswer(thinking="8 / 2 = 4", answer="4"),
            MathAnswer(thinking="3 + 3 + 3 = 9", answer="9"),
            MathAnswer(thinking="7 * 2 = 14", answer="14"),
            MathAnswer(thinking="15 - 5 = 10", answer="10"),
            MathAnswer(thinking="12 / 3 = 4", answer="4"),
        ],
        dtype="object",
    )

    # Test data
    x_test = np.array(
        [
            MathProblem(problem="4 + 5"),
            MathProblem(problem="6 * 3"),
        ],
        dtype="object",
    )
    _y_test = np.array(  # noqa: F841
        [
            MathAnswer(thinking="4 + 5 = 9", answer="9"),
            MathAnswer(thinking="6 * 3 = 18", answer="18"),
        ],
        dtype="object",
    )

    print(f"\nTraining examples: {len(x_train)}")
    print(f"Test examples: {len(x_test)}")

    # -------------------------------------------------------------------------
    # Create and Compile Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Create and Compile Program")
    print("=" * 60)

    inputs = synalinks.Input(data_model=MathProblem)
    outputs = await synalinks.Generator(
        data_model=MathAnswer,
        language_model=lm,
    )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="math_solver",
    )

    reward = synalinks.ExactMatch(in_mask=["answer"])

    program.compile(
        optimizer=synalinks.optimizers.RandomFewShot(nb_max_examples=3),
        reward=reward,
        metrics=[
            synalinks.metrics.MeanMetricWrapper(fn=reward, name="mean_reward"),
        ],
    )

    print("\nProgram compiled with:")
    print("  - Optimizer: RandomFewShot(nb_max_examples=3)")
    print("  - Reward: ExactMatch(in_mask=['answer'])")

    # -------------------------------------------------------------------------
    # Train the Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Train the Program")
    print("=" * 60)

    history = await program.fit(
        x=x_train,
        y=y_train,
        epochs=2,
        validation_split=0.2,
        verbose=1,
        callbacks=[],
    )

    print("\nTraining complete!")
    print(f"History keys: {list(history.history.keys())}")

    # -------------------------------------------------------------------------
    # Test the Trained Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Test Trained Program")
    print("=" * 60)

    test_problems = [
        "9 + 1",
        "5 * 5",
        "20 - 8",
    ]

    for problem in test_problems:
        result = await program(MathProblem(problem=problem))
        print(f"\nProblem: {problem}")
        print(f"Answer: {result['answer']}")

    # -------------------------------------------------------------------------
    # Save and Load Trained Program
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Save and Load")
    print("=" * 60)

    program.save("trained_math_solver.json")
    print("\nSaved trained program to trained_math_solver.json")

    loaded = synalinks.Program.load("trained_math_solver.json")
    print(f"Loaded program: {loaded.name}")

    result = await loaded(MathProblem(problem="100 / 10"))
    print(f"\nLoaded program test: 100 / 10 = {result['answer']}")

    if os.path.exists("trained_math_solver.json"):
        os.remove("trained_math_solver.json")


if __name__ == "__main__":
    asyncio.run(main())
