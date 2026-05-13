"""
# Callbacks: Hooking Into the Training Loop

[Guide 14](Training.md) walked through `program.fit(...)`: the trainer pulls
batches from your dataset, scores predictions against the reward,
asks the optimizer for an updated set of trainable variables, and
loops. That description is true, but it leaves out one detail.
Between every step of the loop, the trainer pauses for a moment
and asks a list of objects called **callbacks** whether they would
like to do something.

A **callback** is a small object whose methods are called by the
trainer at well-defined points in `fit()` — when an epoch starts,
when a batch ends, when training finishes. Each method receives
the current metrics and gets to react. Inside a callback you can
**save the program to disk**, **stop training early**, **append a
row to a CSV file**, **upload metrics to MLflow**, or **anything
else** you can write Python for. The trainer itself stays simple;
all the "what should I do at this point in time?" logic is pushed
out to callbacks.

This is the standard Keras pattern, and Synalinks inherits it
directly. If you have used `keras.callbacks.EarlyStopping` before,
everything in this guide will feel familiar; the names match.

By the end of this guide you will be able to:

- attach one or more callbacks to `program.fit(...)`,
- use the built-ins for the four jobs you actually want done
  (early-stopping, checkpointing, fault-tolerance, CSV logging),
- read the lifecycle hooks well enough to write your own,
- recognize the `Monitor` callback as the bridge between training
  and the observability story from [Guide 9](Observability.md).

## What a Callback Looks Like

A callback is a Python object — usually a subclass of
`synalinks.callbacks.Callback` — that overrides one or more of
these **lifecycle hooks**:

```python
class MyCallback(synalinks.callbacks.Callback):
    def on_train_begin(self, logs=None): ...
    def on_train_end(self, logs=None): ...

    def on_epoch_begin(self, epoch, logs=None): ...
    def on_epoch_end(self, epoch, logs=None): ...

    def on_train_batch_begin(self, batch, logs=None): ...
    def on_train_batch_end(self, batch, logs=None): ...

    def on_test_begin(self, logs=None): ...
    def on_test_end(self, logs=None): ...

    def on_predict_begin(self, logs=None): ...
    def on_predict_end(self, logs=None): ...
```

The names should read like an event timeline. Every `_begin`
fires once at the start of its phase; every `_end` fires once at
the end. The `logs` argument is a Python `dict` that holds
the current metric values (`{"reward": 0.62, "val_reward": 0.51,
...}`). The `epoch` and `batch` arguments are integer counters.
Inside any hook you can also read `self.program`, the live
`Program` being trained.

You wire one or more callbacks into a training run by passing them
to `fit()`:

```python
history = await program.fit(
    x=x_train,
    y=y_train,
    epochs=10,
    validation_split=0.2,
    callbacks=[
        synalinks.callbacks.EarlyStopping(patience=2),
        synalinks.callbacks.ProgramCheckpoint(
            filepath="best.program.json", save_best_only=True
        ),
    ],
)
```

The list is ordered: hooks fire in the order the callbacks appear
in the list. For most callbacks the order does not matter; when it
does, it is usually because one callback writes a file and the
next one reads it.

## The Four Built-Ins You Will Actually Use

Synalinks ships five callbacks. One of them (`Monitor`) belongs in
the observability story and is covered in [Guide 9](Observability.md); the other four
solve concrete operational problems that come up on almost every
non-trivial run.

### `EarlyStopping` — Stop When Improvement Plateaus

**The problem.** You ask for `epochs=20`, but the reward stops
improving after epoch 7. You burn thirteen epochs of LM calls for
no gain — possibly even regressing as the optimizer overfits the
prompt to noise in the training set.

**The fix.** `EarlyStopping` watches a metric and halts training
when it has not improved for `patience` consecutive epochs.

```python
es = synalinks.callbacks.EarlyStopping(
    monitor="val_reward",   # what to watch
    mode="max",             # "max" → higher is better
    patience=2,             # tolerate 2 epochs without improvement
    min_delta=0.01,         # under 1% gain → counts as "no improvement"
    restore_best_variables=True,  # roll the program back to its peak
)
```

The arguments form a sentence: *watch `val_reward`, expect it to
go up; if two epochs in a row produce less than a 0.01 gain, give
up; and when you do, restore the program to whatever variables
were in place at the best epoch*. That last knob matters: without
`restore_best_variables=True`, training stops at the (possibly
worse) **last** epoch, not the best one.

Use it on every run that takes more than a few minutes. The cost
of a false positive (stopping a hair too early) is one epoch of
training; the cost of running thirteen useless epochs at $0.01 per
LM call adds up fast.

### `ProgramCheckpoint` — Save the Best Program

**The problem.** You finish a long training run and discover the
final program is *worse* than what you had at epoch 5. Without
checkpointing, the peak is gone.

**The fix.** `ProgramCheckpoint` writes the program to disk every
time the monitored metric improves.

```python
ckpt = synalinks.callbacks.ProgramCheckpoint(
    filepath="best.program.json",
    monitor="val_reward",
    mode="max",
    save_best_only=True,
)
```

If `save_best_only=True`, the file is overwritten only when a new
peak is observed; the final file is always the best program seen.
With `save_best_only=False`, every epoch produces a checkpoint —
useful when you want a full history but expensive in disk space.

Two extras worth knowing:

- **`save_variables_only=True`** writes only the trainable JSON
  variables (`*.variables.json`) instead of the full program.
  Faster, smaller; reload with `program.load_variables(...)`
  instead of `synalinks.programs.load_program(...)`.
- **`filepath` accepts Python format strings.** If you write
  `filepath="epoch_{epoch:02d}_{val_reward:.3f}.json"`, the
  values from `logs` are substituted at save time. This gives
  every checkpoint a unique, descriptive name.

`EarlyStopping(restore_best_variables=True)` and
`ProgramCheckpoint(save_best_only=True)` overlap in purpose. The
difference: early-stopping keeps the best variables **in memory**
for the current run; checkpointing writes them **to disk** so they
survive a crash or a fresh Python process. Real training runs
usually use both.

### `BackupAndRestore` — Survive Crashes Mid-Run

**The problem.** Your training run is at epoch 7 of 20 when the
process dies — out-of-memory, network blip, somebody hit Ctrl-C
in the wrong terminal. You restart the script and it begins from
epoch 0. Every epoch you ran was wasted.

**The fix.** `BackupAndRestore` writes a small **resume file** at
the end of every epoch. If the script crashes and is re-run with
the same callback pointing at the same directory, the trainer
loads the resume file and continues from the next epoch.

```python
backup = synalinks.callbacks.BackupAndRestore(
    backup_dir="/tmp/synalinks/my_run_backup",
)
```

The contract is strict: the program, the optimizer, the dataset,
and the `epochs=` count must all be the same across the
interrupted and resumed runs. If you change them, the backup is
invalid and the trainer will refuse to load it.

`BackupAndRestore` is for **fault tolerance**.
`ProgramCheckpoint` is for **preserving the best program**. They
do *different* jobs even though both write files; on a long, real
run you will want both.

### `CSVLogger` — Append Metrics to a CSV

**The problem.** The progress bar shows metrics as they go, but
when you want to *plot* them — to compare two training runs, or
to make a figure for a report — you need them in a file.

**The fix.** `CSVLogger` appends one row per epoch to a CSV file,
with one column per metric.

```python
log = synalinks.callbacks.CSVLogger(
    filepath="run_001.csv",
    append=True,   # don't blow away an existing log on re-run
)
```

The resulting file plays nicely with `pandas.read_csv`,
`matplotlib`, spreadsheets, anything. No structured logging
service required for the basics.

### `Monitor` — MLflow Integration ([Guide 9](Observability.md))

Synalinks also ships a `Monitor` callback that logs everything
`CSVLogger` does — plus the program plot and the saved program
itself — to an **MLflow** tracking server. You usually do not
instantiate it directly; the recommended path is

```python
synalinks.enable_observability(
    tracking_uri="http://localhost:5000",
    experiment_name="my_experiment",
)
```

which configures both the `Monitor` hook (per-call traces) **and**
the `Monitor` callback (per-epoch metrics) for you. [Guide 9](Observability.md)
covers the observability story end-to-end.

## Putting Them Together

In production you usually run all four operational callbacks at
once. The list is short and the order does not matter much (each
one does its own thing on `on_epoch_end`):

```python
history = await program.fit(
    x=x_train,
    y=y_train,
    epochs=20,
    validation_split=0.2,
    callbacks=[
        synalinks.callbacks.BackupAndRestore(backup_dir="./backup"),
        synalinks.callbacks.ProgramCheckpoint(
            filepath="best.program.json",
            monitor="val_reward",
            mode="max",
            save_best_only=True,
        ),
        synalinks.callbacks.EarlyStopping(
            monitor="val_reward",
            mode="max",
            patience=3,
            restore_best_variables=True,
        ),
        synalinks.callbacks.CSVLogger(filepath="run.csv"),
    ],
)
```

That four-line list captures the operational reality of training
something at LM-call cost: *back up so a crash does not lose work;
checkpoint so the peak is preserved; stop early so we do not burn
budget after improvement plateaus; log to CSV for the post-mortem*.

## Writing a Custom Callback

When none of the built-ins do what you want, subclass
`synalinks.callbacks.Callback` and override the hooks you care
about. A toy example — a callback that prints a one-line summary
at the end of each epoch:

```python
class TerseLogger(synalinks.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = " ".join(f"{k}={v:.3f}" for k, v in sorted(logs.items()))
        print(f"[epoch {epoch:02d}] {msg}")
```

Inside a hook you can also reach the live program via
`self.program`. That makes it possible to, for example, snapshot
the current value of a specific trainable variable to a structured
log:

```python
class InstructionLogger(synalinks.callbacks.Callback):
    def __init__(self, module_name, out_path):
        super().__init__()
        self.module_name = module_name
        self.out_path = out_path

    def on_epoch_end(self, epoch, logs=None):
        mod = self.program.get_module(self.module_name)
        # The current instruction text the LM is seeing:
        instruction = mod.instruction.get()
        with open(self.out_path, "a") as f:
            f.write(f"epoch={epoch}\\n{instruction}\\n\\n")
```

Two rules of thumb when writing callbacks:

1. **Read, do not write, mid-batch.** The `on_train_batch_*`
   hooks fire dozens of times per epoch. Anything heavier than a
   `print` belongs in `on_epoch_end` instead.
2. **Mutate the program with care.** A callback *can* call
   `self.program.set_variables(...)` and the change will take
   effect; but every other callback in the list runs against the
   mutated state too. Reserve direct mutation for callbacks
   designed to do so, like `EarlyStopping(restore_best_variables=True)`.

## The Lifecycle, Pictured

```mermaid
sequenceDiagram
    participant fit as program.fit()
    participant cbs as callbacks
    fit->>cbs: on_train_begin
    loop each epoch
        fit->>cbs: on_epoch_begin
        loop each batch
            fit->>cbs: on_train_batch_begin
            fit->>fit: train on batch
            fit->>cbs: on_train_batch_end
        end
        opt validation_data given
            fit->>cbs: on_test_begin
            fit->>fit: evaluate on val set
            fit->>cbs: on_test_end
        end
        fit->>cbs: on_epoch_end
    end
    fit->>cbs: on_train_end
```

Read top-to-bottom: `on_train_begin` fires once at the start, then
each epoch fires its own begin/end with a stream of batch hooks in
between. Validation, when present, slots in **before**
`on_epoch_end` — which is why metric keys like `val_reward`
already have values when `on_epoch_end` runs.

## Failure Modes Worth Watching For

- **Wrong `monitor` name.** Callbacks that take a `monitor=`
  argument (`EarlyStopping`, `ProgramCheckpoint`) silently no-op
  if the key is not in `logs`. If you compiled the program with
  `metrics=[BinaryF1Score(average="macro")]`, the key is
  `val_binary_f1_score` — not `val_f1`, not `val_macro_f1`.
  Print `history.history.keys()` after a short run to discover
  the exact names.
- **`mode="min"` on a reward.** Synalinks rewards are *maximized*
  by convention (higher = better); accidentally passing
  `mode="min"` will make the callback stop training as the reward
  *improves*. The default `mode="auto"` infers direction from the
  metric name and is usually correct.
- **Re-using `backup_dir` across runs.** `BackupAndRestore` is
  strict about reuse — its directory must not be shared with
  another callback or another training run. Use one per
  experiment.
- **CSV file growing across re-runs.** With `append=True`,
  re-running the script appends a fresh column header *and* a new
  block of rows. Set `append=False` (or delete the file) when you
  truly want a clean log.

## Take-Home Summary

- **Callbacks are pluggable side-effects** the trainer fires at
  specific points in `fit()` — epoch begin/end, batch begin/end,
  train begin/end.
- **Four built-ins cover the bread-and-butter operational
  needs**: `EarlyStopping` (stop when plateaued),
  `ProgramCheckpoint` (preserve the best), `BackupAndRestore`
  (resume after crashes), `CSVLogger` (log per-epoch metrics).
  Most real runs use all four.
- **`Monitor`** is the MLflow bridge; configure it via
  `synalinks.enable_observability(...)` rather than instantiating
  it directly ([Guide 9](Observability.md)).
- **`monitor=` is the most error-prone argument.** Verify the
  exact metric name by inspecting `history.history.keys()`.
- **Custom callbacks** subclass `synalinks.callbacks.Callback`
  and override the hooks you need. Keep batch-level hooks cheap;
  do real work in `on_epoch_end`.

## API References

- [synalinks.callbacks.Callback](https://synalinks.github.io/synalinks/Synalinks%20API/Callbacks%20API/Base%20Callback%20class/)
- [synalinks.callbacks.EarlyStopping](https://synalinks.github.io/synalinks/Synalinks%20API/Callbacks%20API/EarlyStopping/)
- [synalinks.callbacks.ProgramCheckpoint](https://synalinks.github.io/synalinks/Synalinks%20API/Callbacks%20API/ProgramCheckpoint/)
- [synalinks.callbacks.BackupAndRestore](https://synalinks.github.io/synalinks/Synalinks%20API/Callbacks%20API/BackUpAndRestore/)
- [synalinks.callbacks.CSVLogger](https://synalinks.github.io/synalinks/Synalinks%20API/Callbacks%20API/CSVLogger/)
- [synalinks.callbacks.Monitor](https://synalinks.github.io/synalinks/Synalinks%20API/Callbacks%20API/Monitor/)
"""

import asyncio
import os
import tempfile

import numpy as np

import synalinks


# =============================================================================
# Data Models
# =============================================================================


class MathProblem(synalinks.DataModel):
    """A short arithmetic word problem."""

    problem: str = synalinks.Field(
        description="An arithmetic expression to solve",
    )


class NumericAnswer(synalinks.DataModel):
    """A numeric answer to a math problem."""

    answer: float = synalinks.Field(
        description="The numeric answer to the problem",
    )


# =============================================================================
# A toy custom callback
# =============================================================================


class TerseLogger(synalinks.callbacks.Callback):
    """Prints one short line per epoch — illustrates the on_epoch_end hook."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = " ".join(
            f"{k}={v:.3f}" for k, v in sorted(logs.items()) if isinstance(v, float)
        )
        print(f"  [TerseLogger] epoch={epoch} {msg}")


# =============================================================================
# Main demonstration
# =============================================================================


async def main():
    # -------------------------------------------------------------------------
    # Build a tiny program. The exact program does not matter for the
    # callbacks demo — what matters is that fit() runs.
    # -------------------------------------------------------------------------
    language_model = synalinks.LanguageModel(model="ollama/mistral")

    inputs = synalinks.Input(data_model=MathProblem)
    outputs = await synalinks.Generator(
        data_model=NumericAnswer,
        language_model=language_model,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs, name="math_solver")

    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )

    # A toy training set. Real runs would use a Dataset ([Guide 10](Datasets.md)).
    x_train = np.array(
        [MathProblem(problem=p) for p in ("1+1", "2+3", "4+5", "7-2", "3*3", "8/2")],
        dtype="object",
    )
    y_train = np.array(
        [NumericAnswer(answer=a) for a in (2.0, 5.0, 9.0, 5.0, 9.0, 4.0)],
        dtype="object",
    )

    # -------------------------------------------------------------------------
    # Wire all four operational callbacks plus the custom TerseLogger,
    # using a temporary directory so the demo cleans up after itself.
    # -------------------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "best.program.json")
        csv_path = os.path.join(tmp, "run.csv")
        backup_dir = os.path.join(tmp, "backup")

        callbacks = [
            synalinks.callbacks.BackupAndRestore(backup_dir=backup_dir),
            synalinks.callbacks.ProgramCheckpoint(
                filepath=ckpt_path,
                monitor="val_reward",
                mode="max",
                save_best_only=True,
            ),
            synalinks.callbacks.EarlyStopping(
                monitor="val_reward",
                mode="max",
                patience=1,
                restore_best_variables=True,
            ),
            synalinks.callbacks.CSVLogger(filepath=csv_path),
            TerseLogger(),
        ]

        print("=" * 60)
        print("Training with five callbacks (4 built-ins + TerseLogger)")
        print("=" * 60)
        history = await program.fit(
            x=x_train,
            y=y_train,
            epochs=3,
            validation_split=0.3,
            callbacks=callbacks,
            verbose=1,
        )

        print("\nHistory keys:", list(history.history.keys()))
        print(f"Checkpoint written: {os.path.exists(ckpt_path)}")
        print(f"CSV log written:    {os.path.exists(csv_path)}")


if __name__ == "__main__":
    asyncio.run(main())
