# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Disable the Keras backend that `keras-tuner` would otherwise load.

`keras-tuner` does `import keras` (and `from keras... import *`) at module
load time. Real Keras 3 then needs a backend (TensorFlow, JAX, PyTorch, or
numpy) — none of which `synalinks` itself requires. This module installs
minimal stubs in `sys.modules` so `import keras_tuner` succeeds without
pulling a full Keras runtime into the process.

The intended workflow is:

```python
import synalinks
synalinks.disable_keras_backend()  # MUST come before `import keras_tuner`
import keras_tuner as kt
```

Caveats — read before adopting:

- The stub mutates `sys.modules["keras"]` and a handful of submodules. Real
  Keras becomes unreachable in the same Python process. If you mix
  `synalinks` with real Keras in one script, do not call this — you don't
  need it!
- The stub fakes a small surface (`keras.callbacks.Callback`,
  `keras.utils.serialize_keras_object`, etc.) that is enough for
  `keras-tuner`'s search loop when `BaseTuner.run_trial` is overridden.
  Anything that reaches deeper into the Keras runtime will fail.
- Any dependency that imports `keras` *after* this call will see the stub
  and likely break — including TensorFlow ≥ 2.16, AutoKeras, KerasCV, and
  KerasNLP. Only call this in scripts that don't pull those in.
- `keras-tuner` workers spawned via `multiprocessing` start with a fresh
  `sys.modules`. If you use kt's parallel search, call
  `disable_keras_backend()` in the worker init too, not just the parent.
- The stub references some `keras-tuner` internal paths (e.g.
  `keras_tuner.applications`). A `keras-tuner` release that changes those
  paths could require an update here.
"""

import sys
import types
import warnings

from synalinks.src.api_export import synalinks_export
from synalinks.src.saving.object_registration import get_registered_object


def _build_keras_stubs() -> dict[str, types.ModuleType]:
    keras = types.ModuleType("keras")
    keras.__version__ = "3.0.0"
    keras.version = lambda: "3.0.0"

    config = types.ModuleType("keras.config")
    config.backend = lambda: "numpy"
    keras.config = config

    src = types.ModuleType("keras.src")
    ops = types.ModuleType("keras.src.ops")
    src.ops = ops

    random_mod = types.ModuleType("keras.random")
    keras.random = random_mod

    callbacks = types.ModuleType("keras.callbacks")

    # `Callback` is referenced as the base class for `TunerCallback` /
    # `SaveBestEpoch` at `import keras_tuner` time. `History` is used in
    # `isinstance(results, History)` checks inside `tuner_utils` from
    # `BaseTuner.search`. Both must be real classes; the body is unused
    # because user-overridden `run_trial` returns a float/dict, not a
    # History, and the `set_model` lifecycle hook is only invoked by
    # `Model.fit` (which kt never reaches when `run_trial` is overridden).
    class Callback:
        pass

    class History(Callback):
        pass

    callbacks.Callback = Callback
    callbacks.History = History
    keras.callbacks = callbacks

    # `keras_tuner.src.engine.metrics_tracking.infer_metric_direction` walks
    # `keras.metrics.deserialize` / `keras.losses.deserialize` to figure out
    # whether an unknown metric name should be maximized or minimized. For
    # synalinks users that's never a hit — synalinks metrics live in their
    # own namespace — so we wire up stubs that always raise. kt catches the
    # exception and returns `None` (direction unknown). The `Metric` /
    # `Loss` base classes are needed for kt's `isinstance(metric, ...)`
    # branches; keeping them as empty classes is enough because the
    # `infer_metric_direction` flow never produces an instance of them via
    # the stub.
    def _stub_metric_deserialize(name, *args, **kwargs):
        raise ValueError(f"keras_stub: unknown metric {name!r}")

    def _stub_loss_deserialize(name, *args, **kwargs):
        raise ValueError(f"keras_stub: unknown loss {name!r}")

    class _StubMetric:
        pass

    class _StubLoss:
        pass

    metrics = types.ModuleType("keras.metrics")
    metrics.deserialize = _stub_metric_deserialize
    metrics.Metric = _StubMetric
    keras.metrics = metrics

    losses = types.ModuleType("keras.losses")
    losses.deserialize = _stub_loss_deserialize
    losses.Loss = _StubLoss
    keras.losses = losses

    utils = types.ModuleType("keras.utils")

    def serialize_keras_object(obj):
        return {
            "class_name": type(obj).__name__,
            "config": obj.get_config() if hasattr(obj, "get_config") else {},
        }

    def deserialize_keras_object(config, custom_objects=None, module_objects=None):
        if not isinstance(config, dict) or "class_name" not in config:
            return config
        class_name = config["class_name"]
        cfg = config.get("config", {})
        # Reuse synalinks's resolver: same `(custom_objects, module_objects)`
        # contract, plus honors any active `CustomObjectScope`.
        cls = get_registered_object(
            class_name,
            custom_objects=custom_objects,
            module_objects=module_objects,
        )
        if cls is None:
            raise ValueError(f"keras_stub: unknown class {class_name!r}")
        if hasattr(cls, "from_config"):
            return cls.from_config(cfg)
        return cls(**cfg)

    utils.serialize_keras_object = serialize_keras_object
    utils.deserialize_keras_object = deserialize_keras_object
    keras.utils = utils

    # keras-tuner's `applications` submodule imports `keras.layers` and other
    # runtime-backed symbols this stub does not provide. Replace it with an
    # empty module so `import keras_tuner` doesn't crash; the HyperResNet /
    # HyperEfficientNet image-model helpers in there are not relevant to
    # synalinks workflows.
    kt_apps = types.ModuleType("keras_tuner.applications")
    kt_src_apps = types.ModuleType("keras_tuner.src.applications")

    return {
        "keras": keras,
        "keras.config": config,
        "keras.src": src,
        "keras.src.ops": ops,
        "keras.random": random_mod,
        "keras.callbacks": callbacks,
        "keras.metrics": metrics,
        "keras.losses": losses,
        "keras.utils": utils,
        "keras_tuner.applications": kt_apps,
        "keras_tuner.src.applications": kt_src_apps,
    }


@synalinks_export(
    [
        "synalinks.disable_keras_backend",
        "synalinks.utils.disable_keras_backend",
    ]
)
def disable_keras_backend() -> None:
    """Stub the Keras import surface so `keras-tuner` can be used without Keras.

    Installs minimal fakes in `sys.modules` for the `keras` namespace (and a
    couple of `keras-tuner` internal submodules) so that `import keras_tuner`
    succeeds without pulling in TensorFlow / JAX / PyTorch.

    Call this **once, before importing `keras_tuner`**.

    Import-order rules
    ------------------
    - If `keras` is already in `sys.modules` (real or stubbed), this function
      is a **no-op**. So if you need real Keras alongside `synalinks`+`kt`,
      `import keras` *before* calling `disable_keras_backend()` — the stub
      will detect Keras and stay out of the way. kt will use real Keras.
    - If `keras_tuner` is already imported when this is called, kt has
      already bound to whatever keras it found at import time, so the stub
      cannot help anymore. A `RuntimeWarning` fires and this function is a
      no-op. Move the call before `import keras_tuner`.
    - You cannot use real Keras and the stub simultaneously in the same
      process: `keras_tuner` caches its backend at module-load. If you want
      separate `kt.search` runs against Keras *and* against synalinks, run
      them in separate Python processes.

    Idempotent: calling more than once is a no-op once `sys.modules["keras"]`
    is set.

    Example:
        ```python
        import synalinks
        synalinks.disable_keras_backend()
        import keras_tuner as kt
        ```
    """
    if "keras" in sys.modules:
        return
    if "keras_tuner" in sys.modules:
        warnings.warn(
            "`disable_keras_backend()` was called after `keras_tuner` was "
            "imported; keras-tuner has already bound to whatever keras it "
            "found at import time, so installing the stub now has no effect. "
            "Move this call before `import keras_tuner`.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    for name, module in _build_keras_stubs().items():
        sys.modules[name] = module
