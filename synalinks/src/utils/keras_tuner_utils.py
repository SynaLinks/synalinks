# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Synalinks-aware wrappers around the standard `keras_tuner` tuners.

`keras_tuner.RandomSearch` (and friends) are designed for Keras models:
their default `run_trial` builds a model via `hypermodel(hp)` and calls
`model.fit(...)`. Synalinks programs work the same way but their `fit`
is `async`, and `program.fit` returns a `synalinks.callbacks.History`
that `keras_tuner` does not recognize (its `isinstance` check is against
the stubbed `keras.callbacks.History`).

The wrappers here override `run_trial` to:

1. Build the program by calling `tuner.hypermodel.build(hp)` — the
   hypermodel may return either a `synalinks.Program` directly or a
   coroutine that resolves to one.
2. Run `await program.fit(*args, **kwargs)` with whatever the user passed
   to `tuner.search(...)`.
3. Reduce the resulting `History` into a flat metrics dict keyed by
   metric name and return it. The oracle's objective(s) get the
   best-across-epochs value (max for "max" direction, min for "min");
   non-objective metrics get their final-epoch value (informational).

Everything else (oracle plumbing, trial persistence, `get_best_hyperparameters`)
comes for free from `keras_tuner`.

Usage:

```python
import synalinks
synalinks.disable_keras_backend()   # must come before kt is imported

async def build_program(hp):
    inputs = synalinks.Input(data_model=Query)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=language_model,
        temperature=hp.Float("temperature", 0.0, 1.0),
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs)
    program.compile(reward=synalinks.rewards.ExactMatch())
    return program

tuner = synalinks.tuners.RandomSearch(
    hypermodel=build_program,
    objective=synalinks.tuners.Objective("val_reward", direction="max"),
    max_trials=5,
    directory="runs",
    project_name="my_search",
)
tuner.search(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    epochs=3,
    batch_size=4,
)
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
```

`keras_tuner` is an optional dependency. Importing this module does **not**
import `keras_tuner` — the kt subclasses are built lazily on the first
instantiation of `RandomSearch` / `BayesianOptimization` / `Hyperband` /
`GridSearch`. If `keras_tuner` (and either real Keras or the
`disable_keras_backend()` stub) cannot be loaded then, a clear error is
raised pointing the user at the fix.
"""

import inspect

from synalinks.src.api_export import synalinks_export
from synalinks.src.utils.async_utils import run_maybe_nested

_TUNER_CLASS_NAMES = (
    "RandomSearch",
    "BayesianOptimization",
    "Hyperband",
    "GridSearch",
)

# Lazily-built synalinks-aware subclasses of `kt.<Name>`. Populated on first
# instantiation so that simply importing this module never pulls in
# `keras_tuner`.
_REAL_SUBCLASSES = {}


def _resolve_kt_tuner(name):
    """Build the synalinks-aware subclass of `keras_tuner.<name>` on demand."""
    if name in _REAL_SUBCLASSES:
        return _REAL_SUBCLASSES[name]
    try:
        import keras_tuner as kt
    except ModuleNotFoundError as e:
        # `import keras_tuner` triggers `import keras` transitively. If
        # Keras isn't installed and the user hasn't called
        # `disable_keras_backend()`, the failure surfaces here with
        # `e.name == "keras"`.
        missing = e.name or ""
        if missing == "keras" or missing.startswith("keras."):
            raise RuntimeError(
                f"`synalinks.tuners.{name}` could not import `keras_tuner`: "
                f"no `{missing}` module is available. Either install Keras, "
                "or call `synalinks.disable_keras_backend()` *before* "
                f"instantiating `synalinks.tuners.{name}` so kt can load "
                "without a Keras backend."
            ) from e
        raise ImportError(
            f"`synalinks.tuners.{name}` requires `keras-tuner`. "
            "Install it with `pip install keras-tuner`."
        ) from e

    base = getattr(kt, name)

    class _SynalinksAwareTuner(base):
        def run_trial(self, trial, *fit_args, **fit_kwargs):
            # `run_maybe_nested` works whether or not an event loop is
            # already running (notebooks, IPython, async test harnesses),
            # unlike `asyncio.run` which crashes inside a live loop.
            return run_maybe_nested(
                _run_trial_async(self, trial, *fit_args, **fit_kwargs)
            )

    _SynalinksAwareTuner.__name__ = name
    _SynalinksAwareTuner.__qualname__ = name
    _SynalinksAwareTuner.__doc__ = (
        f"Synalinks-aware `keras_tuner.{name}`. "
        f"See `synalinks.utils.keras_tuner_utils` for usage."
    )
    _REAL_SUBCLASSES[name] = _SynalinksAwareTuner
    return _SynalinksAwareTuner


async def _run_trial_async(tuner, trial, *fit_args, **fit_kwargs):
    """Build the program, score it, and return a metrics dict for the oracle.

    Dispatches between `program.fit(...)` and `program.evaluate(...)`:

      - With an optimizer set on the compiled program, run the normal
        fit loop and reduce the History across epochs.
      - With no optimizer, `fit()` would just iterate the training data
        without applying any updates — wasted compute. Dispatch to
        `program.evaluate(...)` on `validation_data` (or on the
        positional `(x, y)` when no validation pair was passed) and
        report those metrics to the oracle. Each metric is reported
        under both its bare name (`reward`) and a `val_` prefix
        (`val_reward`) so objectives written for either fit-mode or
        evaluate-mode resolve.
    """
    program = tuner.hypermodel.build(trial.hyperparameters)
    if inspect.isawaitable(program):
        program = await program
    if program.optimizer is None:
        result = await _evaluate_only(program, *fit_args, **fit_kwargs)
    else:
        history = await program.fit(*fit_args, **fit_kwargs)
        result = _history_to_metrics_dict(history, tuner.oracle.objective)
    _attach_multi_objective_aggregate(result, tuner.oracle.objective)
    return result


async def _evaluate_only(program, *fit_args, **fit_kwargs):
    """Run a single `program.evaluate(...)` in place of `fit(...)`.

    Eval-set selection mirrors what `fit()` would have used:
      1. `validation_data=(x_val, y_val)` if the caller provided it —
         keeps objectives named `val_*` consistent across fit and
         evaluate dispatch.
      2. otherwise the positional `(x, y)` — the only data available.

    Only the kwargs `evaluate()` actually accepts are forwarded
    (`batch_size`, `verbose`, `steps`, `callbacks`); training-only
    kwargs like `epochs` / `validation_split` / `shuffle` are dropped.

    Every metric in the result is duplicated under a `val_` prefix so
    `objective="val_reward"` (the natural choice when the caller passed
    `validation_data`) resolves the same as `objective="reward"`.
    """
    val = fit_kwargs.get("validation_data")
    if val is not None:
        x, y = val
    else:
        x = fit_kwargs.get("x") or (fit_args[0] if len(fit_args) >= 1 else None)
        y = fit_kwargs.get("y") or (fit_args[1] if len(fit_args) >= 2 else None)

    eval_kwargs = {
        k: fit_kwargs[k]
        for k in ("batch_size", "verbose", "steps", "callbacks")
        if k in fit_kwargs
    }
    metrics = await program.evaluate(x=x, y=y, return_dict=True, **eval_kwargs)
    # Mirror each metric under a `val_` prefix so objectives match
    # regardless of which dispatch (fit / evaluate) the caller assumed.
    return {**metrics, **{f"val_{k}": v for k, v in metrics.items()}}


def _history_to_metrics_dict(history, objective):
    """Flatten a synalinks `History` into the dict keras-tuner expects.

    For each metric in `history.history`:
      - if the metric is part of the oracle's objective, take the best
        value across epochs (`max` for "max" direction, `min` for "min"),
      - otherwise, take the final-epoch value (kt records it but does not
        use it for ranking).
    """
    hist = getattr(history, "history", None) or {}
    directions = _objective_directions(objective)

    result = {}
    for metric_name, values in hist.items():
        if not values:
            continue
        direction = directions.get(metric_name)
        if direction == "max":
            result[metric_name] = max(values)
        elif direction == "min":
            result[metric_name] = min(values)
        else:
            result[metric_name] = values[-1]
    return result


def _attach_multi_objective_aggregate(result, objective):
    """Add the `MultiObjective` aggregate (if any) into the result dict.

    `kt.MultiObjective` looks up the trial's score by the objective's
    synthetic `.name` (e.g. `"multi_objective"`). Single-objective specs
    expose `.name` too but the metric is already keyed by that name in
    `result`, so the call is a no-op in that case.
    """
    name = getattr(objective, "name", None)
    has_value = getattr(objective, "has_value", None)
    get_value = getattr(objective, "get_value", None)
    if (
        name
        and name not in result
        and callable(has_value)
        and callable(get_value)
        and has_value(result)
    ):
        result[name] = get_value(result)


def _objective_directions(objective):
    """Return a `{metric_name: direction}` map from a kt objective spec.

    Handles `Objective`, `MultiObjective`, and the `DefaultObjective` that
    kt installs when none is passed.
    """
    # `MultiObjective` exposes `.name_to_direction` directly.
    name_to_direction = getattr(objective, "name_to_direction", None)
    if name_to_direction is not None:
        return dict(name_to_direction)
    if hasattr(objective, "name") and hasattr(objective, "direction"):
        return {objective.name: objective.direction}
    return {}


def _make_stub(name, kt_doc_hint):
    """Build a stub class that, when instantiated, returns a real kt tuner.

    The stub carries the `@synalinks_export` decorator so the api generator
    routes it to `synalinks.tuners.<name>`. It does not subclass anything
    from `keras_tuner` — `keras_tuner` is only imported inside `__new__`,
    which makes the real subclass on the first call.
    """

    @synalinks_export(
        [
            f"synalinks.tuners.{name}",
            f"synalinks.utils.tuners.{name}",
        ]
    )
    class _Stub:
        """Placeholder set by `_make_stub`. Replaced below."""

        def __new__(cls, *args, **kwargs):
            real_cls = _resolve_kt_tuner(name)
            return real_cls(*args, **kwargs)

    _Stub.__name__ = name
    _Stub.__qualname__ = name
    _Stub.__doc__ = (
        f"Synalinks-aware wrapper around `keras_tuner.{name}`.\n\n"
        f"Same constructor as `keras_tuner.{name}`; `run_trial` is "
        "overridden to build a `synalinks.Program` from "
        "`hypermodel(hp)` and run `await program.fit(*args, **kwargs)` "
        "with whatever was passed to `tuner.search(...)`.\n\n"
        f"{kt_doc_hint}"
    )
    return _Stub


RandomSearch = _make_stub(
    "RandomSearch",
    "Random sampling from the hyperparameter space. Cheap and the right "
    "default when you have no prior on the search shape.",
)
BayesianOptimization = _make_stub(
    "BayesianOptimization",
    "Gaussian-process based sequential search. Spends more compute per "
    "trial choosing the next point — worth it when each trial is expensive.",
)
Hyperband = _make_stub(
    "Hyperband",
    "Successive-halving search: many cheap trials, few expensive ones. "
    "Needs an `objective` that improves smoothly with `epochs`.",
)
GridSearch = _make_stub(
    "GridSearch",
    "Exhaustive enumeration of `Choice` / `Boolean` / discretized `Int` "
    "spaces. Use only when the grid is small.",
)


@synalinks_export(
    [
        "synalinks.tuners.Objective",
        "synalinks.utils.tuners.Objective",
    ]
)
def Objective(name, direction):
    """Re-export of `keras_tuner.Objective` for ergonomic access.

    Loads `keras_tuner` lazily on first call so importing
    `synalinks.tuners` never pulls kt into the process by itself.
    """
    try:
        import keras_tuner as kt
    except ModuleNotFoundError as e:
        missing = e.name or ""
        if missing == "keras" or missing.startswith("keras."):
            raise RuntimeError(
                "`synalinks.tuners.Objective` could not import "
                "`keras_tuner`: call `synalinks.disable_keras_backend()` "
                "first, or install Keras."
            ) from e
        raise ImportError(
            "`synalinks.tuners.Objective` requires `keras-tuner`. "
            "Install with `pip install keras-tuner`."
        ) from e
    return kt.Objective(name=name, direction=direction)
