# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Hyperparameter Search

So far you have been making design choices by hand. Should the
program use plain `Generator` or `ChainOfThought`? What sampling
temperature should the LM use? Should the optimizer keep three
few-shot examples or eight? In [Guide 14](Training.md) we picked sensible defaults
and moved on. That works for one or two knobs, but as your
programs get richer the number of choices grows quickly, and
"sensible default" stops being a defensible engineering decision.

This guide introduces the way out: instead of guessing, you tell
the framework which choices you would like to try, and a **search
procedure** runs many trials for you and reports back which
combination worked best.

The technical name for those handpicked design choices is
**hyperparameters**, and the procedure that searches over them is
called **hyperparameter search**. The distinction between a
parameter and a hyperparameter is worth getting straight up front,
because the words sound almost identical:

- A **parameter** is something the *training loop* updates as it
  runs. In a neural network it would be a floating-point weight;
  in Synalinks ([Guide 14](Training.md)) it is a JSON object obeying a
  `Trainable` schema — most often holding an instruction string
  or a list of few-shot examples, but in general any structured
  data the schema permits. Training changes these values
  automatically.
- A **hyperparameter** is something you fix *before* training even
  starts. The training loop never changes it. To try a different
  value you have to re-train from scratch.

Picking hyperparameters is like adjusting the dials on a stove
before you turn on the heat — once you start cooking, you do not
fiddle with the dials anymore. Search is the process of cooking
the same dish many times, each time with slightly different dial
settings, then keeping the best version.

If you happen to have used **Keras-Tuner** before, this guide will
feel familiar from sentence one: Synalinks ships drop-in wrappers
around the four standard Keras-Tuner tuners — `RandomSearch`,
`BayesianOptimization`, `Hyperband`, and `GridSearch` — under the
`synalinks.tuners` namespace, and the user-facing API is identical.
If you have *not* used Keras-Tuner before, that is fine; we explain
every piece below from scratch.

## The Search Loop in One Picture

```mermaid
flowchart LR
    HP["hp.Boolean / hp.Float / hp.Choice"] --> B["build_program(hp)"]
    B --> P["compiled Program"]
    P --> F["program.fit(...)"]
    F --> M["val_reward, val_loss, ..."]
    M --> O["Oracle (RandomSearch / Bayes / ...)"]
    O -->|"propose next HPs"| HP
```

One trip around this loop is called a **trial**. On each trial the
search procedure picks a fresh set of hyperparameter values, builds
a new program from those values, fits it on the training data,
measures it on the validation data, and reports the score back to
the **oracle**.

The "oracle" is a colorful name for a small, ordinary object — it
is just the part of the tuner that, given the trials run so far,
decides which hyperparameters to try next. Different tuners use
different oracles: `RandomSearch` rolls dice; `BayesianOptimization`
fits a statistical model; `Hyperband` uses a tournament. We come
back to all four in a moment.

After many trials the oracle ranks the configurations it tried and
tells you which set of dial settings won.

## The `build_program(hp)` Hypermodel

The only thing **you** have to write is one function:
`build_program(hp)`. The convention is straightforward — wherever
you would normally hard-code a value, you instead *sample* it from
the `hp` argument:

- `hp.Boolean("use_chain_of_thought", default=True)` — sample
  `True` or `False`.
- `hp.Float("temperature", 0.0, 1.0, default=0.0)` — sample a real
  number in `[0.0, 1.0]`.
- `hp.Choice("reasoning_effort", ["minimal", "low", "medium"])` —
  sample one item from a fixed list.

The function returns a *compiled* `Program` — a `Program` on which
you have already called `.compile(...)` exactly as in [Guide 14](Training.md).
This is the same shape as the `build_model(hp)` function you would
write in a Keras-Tuner script. The only Synalinks-specific detail
is that `build_program` is **`async`**, because module calls are
awaitable. The tuner awaits the coroutine for you, so the rest of
the code stays the way you remember it.

```python
async def build_program(hp):
    use_cot = hp.Boolean("use_chain_of_thought", default=True)
    temperature = hp.Float("temperature", 0.0, 1.0, default=0.0)
    reasoning_effort = hp.Choice(
        "reasoning_effort", ["minimal", "low", "medium"], default="low",
    )

    synalinks.clear_session()

    inputs = synalinks.Input(
        data_model=synalinks.datasets.gsm8k.get_input_data_model(),
    )
    if use_cot:
        outputs = await synalinks.ChainOfThought(
            data_model=synalinks.datasets.gsm8k.get_output_data_model(),
            language_model=language_model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )(inputs)
    else:
        outputs = await synalinks.Generator(
            data_model=synalinks.datasets.gsm8k.get_output_data_model(),
            language_model=language_model,
            temperature=temperature,
        )(inputs)

    program = synalinks.Program(inputs=inputs, outputs=outputs, name="trial")
    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    return program
```

A small detail that catches everyone the first time: notice the
`synalinks.clear_session()` call at the top of the function. You
first met `clear_session` in [Guide 1](Getting%20Started.md). Without it, every trial would
accumulate module-name suffixes (`generator_1`, `generator_2`,
`generator_3`, ...) and your trial logs would drift between runs,
making the search non-reproducible. Calling `clear_session()` at
the top of each trial is the search-time equivalent of "wipe the
blackboard before the next student attempts the problem."

## The Tuner

Once you have the hypermodel, constructing the tuner is a single
call. The simplest tuner is `RandomSearch` — it samples
hyperparameter values **uniformly at random** from the declared
ranges and runs trials until it hits `max_trials`. "Uniformly at
random" here means every value in the range is equally likely; the
search has no memory of which values it has tried.

```python
tuner = synalinks.tuners.RandomSearch(
    build_program,
    objective=synalinks.tuners.Objective("val_reward", direction="max"),
    max_trials=4,
    seed=1234,
    directory="examples",
    project_name="gsm8k_hp_search",
    overwrite=True,
)
```

Two ideas in that block are new:

- **`Objective`.** A pair of `(metric_name, direction)` telling
  the oracle what counts as "better." Here we want to *maximize*
  `val_reward` — the reward measured on the validation split.
  Whenever you care about something smaller-is-better (latency,
  token cost, dollars), use `direction="min"` instead. The
  `val_` prefix in the metric name is a convention: training
  metrics get reported as-is, and the same metric measured on the
  validation split gets prefixed with `val_`.
- **The project directory.** The tuner persists every trial — the
  hyperparameter values it tried, the metrics it observed — to a
  folder on disk. That means you can stop a search halfway
  through, come back later, and resume from the last completed
  trial. `overwrite=True` says "I am starting fresh, please
  discard any previous results in this folder."

To actually run the search you call `tuner.search(...)`. Anything
you pass to `search` gets forwarded into `program.fit(...)` under
the hood, so the arguments will look very familiar from [Guide 14](Training.md):

```python
tuner.search(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    epochs=1,
    batch_size=4,
)
```

`epochs=1` keeps each trial quick. For a serious run you would
turn up both `epochs` and `max_trials`. There is a trade-off here
worth recognizing: more **trials** means more *exploration* — the
search sees more configurations — while more **epochs per trial**
means more *depth*, so each individual trial is a more accurate
estimate of how that configuration really performs. Total cost is
the product, so you cannot just have everything.

## The Four Tuners and When to Use Them

The choice of tuner is really the choice of *how the oracle
proposes new hyperparameter sets*. All four Synalinks tuners share
the constructor shape above, so swapping between them is one line.

- **`RandomSearch`** — sample uniformly at random from the
  declared ranges. Embarrassingly simple, and embarrassingly
  hard to beat at small budgets. This is the right default.
- **`BayesianOptimization`** — fit a small statistical model of
  "given hyperparameters, predict the score," update it after
  every trial, and propose the point the model thinks is most
  likely to improve. Much more sample-efficient than random
  when each trial is expensive (which is true for LM programs,
  where one trial costs many API calls).
- **`Hyperband`** — start many trials cheaply, kill the worst
  ones early, hand the saved budget to the survivors so they
  can train longer. Picture a tournament that eliminates half
  the players each round. Useful when training is *cheap* and
  the scarce resource is total training steps.
- **`GridSearch`** — enumerate every combination of every
  `Choice` exactly once. Use this when your search space is
  small and discrete — for instance, "try each of these three
  models" — and you do not want any to be visited twice. We
  meet `GridSearch` head-on in [Guide 17](Multi-Objective%20LM%20Selection.md).

## One Bit of Plumbing: `disable_keras_backend`

Keras-Tuner does `import keras` at module load time. Synalinks
does *not* depend on Keras at runtime, so we ship a small shim
that installs a minimal stub of the Keras namespace **before**
Keras-Tuner has a chance to look for it. You have to call this
shim before any `keras_tuner` import path runs — that is, right
at the top of your script, immediately after `import synalinks`:

```python
import synalinks
synalinks.disable_keras_backend()   # MUST run before importing tuners
```

Forget this call and the error you get is unhelpful: a
`ModuleNotFoundError` about a missing tensor backend (TensorFlow,
JAX, PyTorch — none of which Synalinks needs). If you see that
error, your call to `disable_keras_backend()` is either missing
or in the wrong place.

## Looking at the Results

After `tuner.search()` returns, three useful entry points let you
poke at what happened:

- **`tuner.results_summary(num_trials=N)`** — prints the top
  `N` trials, their hyperparameters, and their scores. Good for
  a quick eyeball.
- **`tuner.get_best_hyperparameters(num_trials=1)[0]`** — returns
  the best `HyperParameters` object, which you can hand back to
  `build_program(hp)` to rebuild the winning program from
  scratch.
- **`tuner.oracle.get_best_trials(num_trials=1)[0]`** — returns
  the underlying `Trial` object, which carries every recorded
  metric and the duration of the trial. Use this when you want
  more than the aggregated score.

The standard last step is to **rebuild the winning program from
the best hyperparameters and evaluate it on a third split** that
was held out from the entire search:

```python
async def _final_eval():
    program = await build_program(best_hp)
    return await program.evaluate(x=x_test, y=y_test, batch_size=4, verbose=0)

metrics = run_maybe_nested(_final_eval())
print(f"Test reward: {metrics.get('reward'):.3f}")
```

Why a *third* split? Because the validation reward was used by
the oracle to **choose** hyperparameters. As soon as you select a
configuration on the basis of its val score, that val score stops
being an unbiased estimate of how the program generalizes — by
construction, you picked the configuration most flattered by the
val data. The held-out test split has never been seen by the
search, so its score is an honest measurement of what you will
get in production.

This is a slightly subtle point, and worth pausing on. With one
split, validation gives you an unbiased estimate. With two splits
*and* a search procedure that picks the best val score, validation
becomes biased *upward*. A separate, untouched test split restores
honesty.

## Searching at Scale: Practical Cautions

Hyperparameter search is one of the most expensive things you can
do with a Synalinks program. The total cost is roughly:

    total_LM_calls ≈ max_trials × epochs × N_train

where `N_train` is the number of training examples per epoch.
Before you press "go," it is worth knowing four levers:

1. **Start with a tiny dataset and a small `max_trials`.** Get
   the pipeline working end-to-end first. Scale up only once you
   have seen one full search complete without surprises.
2. **Pick the cheapest LM that still distinguishes
   configurations.** You only need the LM to be sensitive enough
   to the hyperparameters for the oracle to get a useful signal.
   Save the strong (expensive) model for the final evaluation on
   the test split.
3. **Always set `seed=...` on the tuner.** Without a seed, the
   search trajectory is non-reproducible — two runs over the same
   dataset can find different winners. With a seed, the
   trajectory is identical every time, which makes debugging far
   easier.
4. **`directory=` / `project_name=` are not optional in
   practice.** When a trial crashes — and at some point, one
   will — the tuner can resume from the last completed trial,
   but only if there is a folder on disk for it to read from.

## Take-Home Summary

- A **hyperparameter** is a choice fixed *before* training; a
  *parameter* is a value that training updates as it runs.
- The contract you write is **one `async` function**,
  `build_program(hp)`, that samples hyperparameters from `hp` and
  returns a *compiled* `Program` (as in [Guide 14](Training.md)).
- A **tuner** plus an **objective** (`("val_reward", "max")`)
  drives a loop of **trials**; each trial calls `build_program`
  with a new HP set and then runs `program.fit`.
- **`RandomSearch`** is the safe default;
  **`BayesianOptimization`** is the sample-efficient upgrade;
  **`Hyperband`** front-loads cheap trials; **`GridSearch`**
  enumerates a small discrete grid.
- **`synalinks.disable_keras_backend()` must run first**, before
  any `keras_tuner` import path executes. It installs the Keras
  shim that Keras-Tuner expects to find.
- Always **evaluate the winner on a held-out test split** at the
  very end. The validation split is contaminated by the search
  itself; the test split is your only honest measurement.

## API References

- [synalinks.tuners](https://synalinks.github.io/synalinks/Synalinks%20API/Tuners/)
- [disable_keras_backend](https://synalinks.github.io/synalinks/Synalinks%20API/Utils/disable_keras_backend/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [ChainOfThought](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Test%20Time%20Compute%20Modules/ChainOfThought%20module/)
- [GSM8K dataset](https://synalinks.github.io/synalinks/Synalinks%20API/Built-in%20Datasets/GSM8K/)
- [keras-tuner](https://keras.io/keras_tuner/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# MUST run before any code path imports `keras_tuner`.
synalinks.disable_keras_backend()

# GSM8K is large; these caps keep one search affordable on a laptop.
NB_TRAIN_SAMPLES = 12
NB_VAL_SAMPLES = 6
NB_TEST_SAMPLES = 6
EPOCHS = 1
BATCH_SIZE = 4
MAX_TRIALS = 4
SEED = 1234

FOLDER = "guides"
PROJECT_NAME = "gsm8k_hp_search"

# Module-level handle the hypermodel reads. Populated in `main()` before
# the tuner is constructed — the keras-tuner pattern of "build_program(hp)
# just reads the world" depends on the world being ready.
language_model: synalinks.LanguageModel | None = None


# =============================================================================
# Hypermodel: build a fresh program for one trial
# =============================================================================


async def build_program(hp):
    """Sample hyperparameters and return a compiled `synalinks.Program`."""
    use_cot = hp.Boolean("use_chain_of_thought", default=True)
    temperature = hp.Float("temperature", 0.0, 1.0, default=0.0)
    reasoning_effort = hp.Choice(
        "reasoning_effort",
        ["minimal", "low", "medium"],
        default="low",
    )

    # Reset module name counters so trials don't accumulate suffixes
    # ("generator_1", "generator_2", ...) across the search.
    synalinks.clear_session()

    inputs = synalinks.Input(
        data_model=synalinks.datasets.gsm8k.get_input_data_model(),
    )
    if use_cot:
        outputs = await synalinks.ChainOfThought(
            data_model=synalinks.datasets.gsm8k.get_output_data_model(),
            language_model=language_model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )(inputs)
    else:
        outputs = await synalinks.Generator(
            data_model=synalinks.datasets.gsm8k.get_output_data_model(),
            language_model=language_model,
            temperature=temperature,
        )(inputs)

    program = synalinks.Program(
        inputs=inputs,
        outputs=outputs,
        name="gsm8k_trial",
    )
    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    return program


# =============================================================================
# Main
# =============================================================================


async def main():
    load_dotenv()

    global language_model
    language_model = synalinks.LanguageModel(
        model="gemini/gemini-3.1-flash-lite-preview",
    )

    print("Loading GSM8K...")
    (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()
    x_train_s = x_train[:NB_TRAIN_SAMPLES]
    y_train_s = y_train[:NB_TRAIN_SAMPLES]
    x_val_s = x_train[NB_TRAIN_SAMPLES : NB_TRAIN_SAMPLES + NB_VAL_SAMPLES]
    y_val_s = y_train[NB_TRAIN_SAMPLES : NB_TRAIN_SAMPLES + NB_VAL_SAMPLES]
    x_test_s = x_test[:NB_TEST_SAMPLES]
    y_test_s = y_test[:NB_TEST_SAMPLES]
    print(
        f"Per trial: fit on {len(x_train_s)} samples, validate on {len(x_val_s)} samples."
    )

    tuner = synalinks.tuners.RandomSearch(
        build_program,
        objective=synalinks.tuners.Objective("val_reward", direction="max"),
        max_trials=MAX_TRIALS,
        seed=SEED,
        directory=FOLDER,
        project_name=PROJECT_NAME,
        overwrite=True,
    )

    print(f"\nStarting search ({MAX_TRIALS} trials)...")
    # Everything after `tuner.search()` is forwarded into `program.fit(...)`
    # by `synalinks.tuners.RandomSearch.run_trial`, exactly like keras-tuner
    # forwards them into `model.fit(...)`.
    tuner.search(
        x=x_train_s,
        y=y_train_s,
        validation_data=(x_val_s, y_val_s),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )
    print("Done.\n")

    tuner.results_summary(num_trials=MAX_TRIALS)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    print("\n" + "=" * 70)
    print("Best hyperparameters")
    print("=" * 70)
    print(f"  use_chain_of_thought: {best_hp.get('use_chain_of_thought')}")
    print(f"  temperature:          {best_hp.get('temperature'):.3f}")
    print(f"  reasoning_effort:     {best_hp.get('reasoning_effort')}")
    print(f"  val_reward:           {best_trial.score:.3f}")
    print(f"\nFull trial history persisted under {FOLDER}/{PROJECT_NAME}")

    # Rebuild the winner from the best HPs and evaluate on the held-out
    # test slice — the validation reward was used by the oracle to pick
    # hyperparameters, so it overstates the true generalization score.
    print("\nRebuilding winner and evaluating on test split...")
    program = await build_program(best_hp)
    metrics = await program.evaluate(
        x=x_test_s,
        y=y_test_s,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    print(f"Test reward: {metrics.get('reward'):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
