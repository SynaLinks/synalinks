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

import asyncio
import functools
import inspect

from synalinks.src.api_export import synalinks_export
from synalinks.src.utils.async_utils import _LoopCoroRunner
from synalinks.src.utils.async_utils import _close_loop
from synalinks.src.utils.async_utils import run_maybe_nested


def _close_litellm_async_clients(loop):
    """Best-effort: close litellm's cached async HTTP client(s) on ``loop``.

    litellm caches a module-level async httpx client bound to the first event
    loop it touches. Closing it ON the search loop, before that loop is torn
    down, prevents an httpx pool being left bound to a dead loop — which GC would
    otherwise try to close on a closed loop, emitting a final "Event loop is
    closed" and occasionally wedging interpreter exit. litellm is a hard
    dependency, but its teardown API varies by version, so every step is guarded
    and failure is non-fatal (the loop still closes cleanly without it).
    """
    try:
        import litellm

        closer = getattr(litellm, "close_litellm_async_clients", None)
        if closer is None:
            return
        result = closer()
        if inspect.isawaitable(result):
            loop.run_until_complete(result)
    except Exception:
        pass

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

_kt_inference_patched = False


def _synalinks_name_direction_map():
    """Build a ``{default_metric_name: "max"|"min"}`` map from the metrics
    registry.

    Each shipped metric class declares its ``direction`` (``"up"``/``"down"``)
    at class scope (see ``synalinks.Metric``). The default ``name=`` kwarg in
    each ``__init__`` signature is what shows up in ``history.history`` and
    is what kt's ``infer_metric_direction`` is asked about.

    ``"reward"`` is added as a special case: it's the conventional name for a
    ``Mean``/``MeanMetricWrapper`` instance wrapping a synalinks reward —
    not a class name — so no class introspection would catch it.
    """
    from synalinks.src import metrics as _metrics_pkg

    name_to_direction = {"reward": "max"}
    _UP_DOWN = {"up": "max", "down": "min"}
    for cls in _metrics_pkg.ALL_OBJECTS:
        cls_direction = getattr(cls, "direction", None)
        kt_direction = _UP_DOWN.get(cls_direction)
        if kt_direction is None:
            continue
        try:
            sig = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue
        name_param = sig.parameters.get("name")
        if name_param is None:
            continue
        default_name = name_param.default
        if isinstance(default_name, str):
            name_to_direction.setdefault(default_name, kt_direction)
    return name_to_direction


def _patch_kt_inference():
    """Patch kt's ``infer_metric_direction`` to know synalinks metric names.

    Idempotent. Called from ``_resolve_kt_tuner`` (so string objectives like
    ``objective="val_reward"`` resolve when a tuner is built) and from
    ``Objective(...)`` (so omitting ``direction=`` resolves via inference).

    Keras Tuner's ``infer_metric_direction`` resolves keras metric names via
    ``keras.metrics.deserialize`` then matches the class name against a
    hardcoded ``_MAX_METRICS`` whitelist. Synalinks metrics don't live in
    ``keras.metrics``, and ``keras_backend.py`` ships deserialize stubs that
    always raise — so without help, inference returns None for every
    synalinks metric name. The replacement here consults the synalinks
    metrics registry first (via `_synalinks_name_direction_map`) and
    falls through to kt's original for keras compat (``"loss"`` etc.).

    Both kt call sites — ``metrics_tracking.register`` and
    ``objective.create_objective`` — look up the function via attribute
    access on the ``metrics_tracking`` module, so a single replacement on
    that module covers both.
    """
    global _kt_inference_patched
    if _kt_inference_patched:
        return
    from keras_tuner.src.engine import metrics_tracking

    _kt_original_infer = metrics_tracking.infer_metric_direction
    name_to_direction = _synalinks_name_direction_map()

    def _synalinks_infer(metric):
        if isinstance(metric, str):
            name = metric
            if name.startswith("val_"):
                name = name[len("val_") :]
            inferred = name_to_direction.get(name)
            if inferred is not None:
                return inferred
        return _kt_original_infer(metric)

    metrics_tracking.infer_metric_direction = _synalinks_infer
    _kt_inference_patched = True


def _sync_driving_hypermodel(hypermodel):
    """Wrap a (possibly async) hypermodel so `build(hp)` returns a concrete
    program, never an un-awaited coroutine.

    keras_tuner explores the hyperparameter space **synchronously** during
    tuner construction: ``BaseTuner.__init__`` → ``_populate_initial_space``
    → ``_activate_all_conditions`` calls ``self.hypermodel.build(hp)`` (and
    relies on the ``hp.Float(...)`` / ``hp.Choice(...)`` calls inside it to
    register the search space, including any conditional scopes). A synalinks
    ``build_program`` is typically ``async def``, so calling it synchronously
    just returns a coroutine that is never awaited — its body never runs.

    The visible symptom is ``RuntimeWarning: coroutine 'build_program' was
    never awaited`` (raised from ``base_tuner.py`` at construction); the
    latent bug is that the initial search space is left empty and conditional
    hyperparameters are never discovered. (Random/Grid search happen to
    survive an empty initial space by populating it lazily on the first
    trial, but that masks the conditional-HP loss.)

    Driving the coroutine to completion here fixes both. ``run_maybe_nested``
    is safe whether or not an event loop is already running, so this also
    works inside notebooks and async test harnesses. The same wrapper is used
    for the per-trial build in ``_run_trial_async`` (via ``tuner.hypermodel``),
    so building never returns an awaitable there either.
    """
    import keras_tuner as kt

    if isinstance(hypermodel, kt.HyperModel):
        inner_build = hypermodel.build

        @functools.wraps(inner_build)
        def sync_build(hp, *args, **kwargs):
            result = inner_build(hp, *args, **kwargs)
            if inspect.isawaitable(result):
                result = run_maybe_nested(result)
            return result

        hypermodel.build = sync_build
        return hypermodel

    if callable(hypermodel):

        @functools.wraps(hypermodel)
        def sync_build(hp, *args, **kwargs):
            result = hypermodel(hp, *args, **kwargs)
            if inspect.isawaitable(result):
                result = run_maybe_nested(result)
            return result

        return sync_build

    # `None` (subclass defines the space in run_trial) or anything else: leave
    # it untouched and let keras_tuner validate it.
    return hypermodel


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

    _patch_kt_inference()
    base = getattr(kt, name)

    class _SynalinksAwareTuner(base):
        def __init__(self, *args, **kwargs):
            # `hypermodel` is the first positional argument of every public
            # kt tuner (RandomSearch/BayesianOptimization/Hyperband/GridSearch).
            # Wrap it before `super().__init__` runs the synchronous space
            # exploration that would otherwise drop an un-awaited coroutine.
            if "hypermodel" in kwargs:
                kwargs["hypermodel"] = _sync_driving_hypermodel(kwargs["hypermodel"])
            elif args:
                args = (_sync_driving_hypermodel(args[0]),) + args[1:]
            super().__init__(*args, **kwargs)

        def search(self, *fit_args, **fit_kwargs):
            # Run EVERY trial on ONE event loop — created here, closed once at
            # the end — instead of letting `run_maybe_nested` spin up and tear
            # down a fresh loop per trial (see `run_trial`).
            #
            # Why: litellm caches a *module-level* async httpx client bound to
            # the first event loop it touches. With a per-trial loop that client
            # is stranded on a closed loop every trial, which (a) floods stderr
            # with "Event loop is closed" / "Task exception was never retrieved"
            # and (b) leaves unclosed clients/connections that can wedge process
            # exit. A single persistent loop keeps the client valid for the whole
            # search and tears everything down cleanly once.
            #
            # If a loop is already running (notebook / IPython / async harness)
            # we cannot install our own, so leave `_synalinks_search_loop` unset
            # and let `run_trial` fall back to the per-trial `run_maybe_nested`.
            # NB: keep `super().search()` OUT of this try. It raises
            # RuntimeError of its own (e.g. keras_tuner's consecutive-failure
            # abort); catching that here would swallow the real error and
            # wrongly fall through to spin up our own loop on top of the
            # already-running one — which then dies with "Cannot run the event
            # loop while another loop is running".
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                pass
            else:
                return super().search(*fit_args, **fit_kwargs)
            loop = asyncio.new_event_loop()
            self._synalinks_search_loop = loop
            try:
                return super().search(*fit_args, **fit_kwargs)
            finally:
                self._synalinks_search_loop = None
                # Close litellm's async client ON this loop before tearing it
                # down, so nothing httpx-related is left bound to a dead loop.
                _close_litellm_async_clients(loop)
                _close_loop(loop)

        def run_trial(self, trial, *fit_args, **fit_kwargs):
            # Reuse the search-wide loop set up by `search` when present;
            # otherwise `run_maybe_nested` makes a per-call loop (correct, but it
            # is exactly the per-trial churn that strands litellm's client).
            # Both paths work whether or not an outer loop is already running
            # (notebooks, IPython, async test harnesses), unlike `asyncio.run`.
            coro = _run_trial_async(self, trial, *fit_args, **fit_kwargs)
            loop = getattr(self, "_synalinks_search_loop", None)
            if loop is None:
                return run_maybe_nested(coro)
            # _LoopCoroRunner(loop) runs ON the given loop without closing it (it
            # only closes loops it created) and applies the same nested-run +
            # contextvar propagation that `run_maybe_nested` does.
            with _LoopCoroRunner(loop) as runner:
                return runner.run(coro)

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
def Objective(name, direction=None):
    """Re-export of `keras_tuner.Objective` for ergonomic access.

    When ``direction`` is omitted, it's inferred from the synalinks metrics
    registry: shipped metrics declare ``direction`` at class scope (e.g.
    ``Accuracy.direction == "up"``, ``ProgramCost.direction == "down"``), and
    ``"reward"`` is special-cased as ``"max"``. Unknown names fall through
    to keras-tuner's own inference (which handles ``"loss"`` etc.). Pass
    ``direction=`` explicitly if neither table covers your metric.

    Loads ``keras_tuner`` lazily on first call so importing
    ``synalinks.tuners`` never pulls kt into the process by itself.
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
    _patch_kt_inference()
    if direction is None:
        from keras_tuner.src.engine import metrics_tracking

        direction = metrics_tracking.infer_metric_direction(name)
        if direction is None:
            raise ValueError(
                f"Could not infer optimization direction for objective {name!r}. "
                "Pass it explicitly: "
                f"synalinks.tuners.Objective({name!r}, direction='max')."
            )
    return kt.Objective(name=name, direction=direction)
