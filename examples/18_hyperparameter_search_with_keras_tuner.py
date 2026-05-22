# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""
# Hyperparameter Search with Keras Tuner

Synalinks ships drop-in wrappers around the four standard `keras_tuner`
tuners — `RandomSearch`, `BayesianOptimization`, `Hyperband`, and
`GridSearch` — under the `synalinks.tuners` namespace. The user-facing
API is the same as `keras_tuner`'s: write a `build_program(hp)` function
that samples hyperparameters and returns a compiled program, hand it to
a tuner, call `tuner.search(...)`.

Under the hood the wrapper's `run_trial` builds the program from your
hypermodel and drives it through `await program.fit(...)` instead of
`model.fit(...)`. The resulting `synalinks.callbacks.History` is reduced
to a metrics dict the oracle understands (best-per-direction for
objective metrics, last-epoch for the rest).

`disable_keras_backend()` is still required: `keras_tuner` itself does
`import keras` at module load time. The shim installs a minimal stub of
the Keras namespace so kt can load without TensorFlow, JAX, or PyTorch.

## Script structure (mirrors the Keras-tuner pattern)

```python
def build_program(hp):
    program = synalinks.Program(...)
    program.compile(...)
    return program

tuner = synalinks.tuners.RandomSearch(
    build_program,
    objective=synalinks.tuners.Objective("val_reward", direction="max"),
    max_trials=4,
)
tuner.search(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=1)
```

## Search space

For each trial we sample three hyperparameters of a GSM8K solver:

- `use_chain_of_thought` (Boolean) — `ChainOfThought` vs plain `Generator`
- `temperature` (Float) — LM sampling temperature
- `reasoning_effort` (Choice) — only used by `ChainOfThought`

GSM8K is large — we deliberately use a *tiny* subset here (12 train, 6 val,
6 test) so a full search fits in a few minutes. Bump the constants below
for a serious run.

## Installation

```bash
uv pip install keras-tuner
```

## API References

- [disable_keras_backend](https://synalinks.github.io/synalinks/Synalinks%20API/Utils/disable_keras_backend/)
- [synalinks.tuners](https://synalinks.github.io/synalinks/Synalinks%20API/Tuners/)
- [Generator](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Core%20Modules/Generator%20module/)
- [ChainOfThought](https://synalinks.github.io/synalinks/Synalinks%20API/Modules%20API/Test%20Time%20Compute%20Modules/ChainOfThought%20module/)
- [Built-in Datasets (GSM8K)](https://synalinks.github.io/synalinks/Synalinks%20API/Built-in%20Datasets/GSM8K/)
- [keras-tuner](https://keras.io/keras_tuner/)
"""

import asyncio

from dotenv import load_dotenv

import synalinks

# MUST run before any code path imports `keras_tuner`.
synalinks.disable_keras_backend()

# GSM8K is large; these caps keep one search affordable.
NB_TRAIN_SAMPLES = 12
NB_VAL_SAMPLES = 6
NB_TEST_SAMPLES = 6
EPOCHS = 1
BATCH_SIZE = 4
MAX_TRIALS = 4
SEED = 1234

FOLDER = "examples"
PROJECT_NAME = "gsm8k_hp_search"

# Module-level handle the hypermodel reads. Populated in `main()` before
# the tuner is constructed — the keras-tuner pattern of "build_model(hp)
# just reads the world" depends on the world being ready.
language_model: synalinks.LanguageModel | None = None


# =============================================================================
# Hypermodel: build a fresh program for one trial
# =============================================================================


async def build_program(hp):
    """Sample HPs and return a compiled `synalinks.Program`.

    This is the synalinks equivalent of `build_model(hp)` in a keras-tuner
    script. `async` because synalinks `Module.__call__` is awaitable —
    `synalinks.tuners.*` awaits the coroutine for you.
    """
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

    print("Loading GSM8k...")
    (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()
    x_train_s, y_train_s = x_train[:NB_TRAIN_SAMPLES], y_train[:NB_TRAIN_SAMPLES]
    x_val_s = x_train[NB_TRAIN_SAMPLES : NB_TRAIN_SAMPLES + NB_VAL_SAMPLES]
    y_val_s = y_train[NB_TRAIN_SAMPLES : NB_TRAIN_SAMPLES + NB_VAL_SAMPLES]
    x_test_s, y_test_s = x_test[:NB_TEST_SAMPLES], y_test[:NB_TEST_SAMPLES]
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

    # Optional: rebuild the winner from the best HPs and evaluate on the
    # held-out test slice. Same pattern as keras-tuner's
    # `tuner.get_best_models()` → `model.evaluate(...)`.
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
