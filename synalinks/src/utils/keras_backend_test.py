# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
import types
import warnings

from synalinks.src import testing
from synalinks.src.utils.keras_backend import disable_keras_backend


def _is_managed_module(name: str) -> bool:
    """True for any `keras*` or `keras_tuner*` module name.

    Once `keras_tuner` is imported it caches references to the stub (e.g.
    `TunerCallback` baked in `keras.callbacks.Callback`). Clearing the
    whole subtree between tests prevents a stale kt module from leaking
    into the next test's stub install.
    """
    return (
        name == "keras"
        or name.startswith("keras.")
        or name == "keras_tuner"
        or name.startswith("keras_tuner.")
    )


def _make_fake_real_keras() -> types.ModuleType:
    """A `keras` module that *looks* real (has `.layers`, non-stub version)."""
    fake = types.ModuleType("keras")
    fake.__version__ = "3.12.0"
    fake.layers = types.ModuleType("keras.layers")  # only real Keras has this
    fake.version = lambda: "3.12.0"
    return fake


class DisableKerasBackendTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self._removed = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if _is_managed_module(name)
        }

    def tearDown(self):
        for name in list(sys.modules):
            if _is_managed_module(name):
                sys.modules.pop(name, None)
        for name, module in self._removed.items():
            sys.modules[name] = module
        super().tearDown()

    # ------------------------------------------------------------------
    # Core stub behavior
    # ------------------------------------------------------------------

    def test_installs_minimal_keras_surface(self):
        self.assertNotIn("keras", sys.modules)
        disable_keras_backend()
        import keras
        import keras.callbacks
        import keras.utils

        self.assertTrue(callable(keras.utils.serialize_keras_object))
        self.assertTrue(callable(keras.utils.deserialize_keras_object))
        # kt declares `class TunerCallback(keras.callbacks.Callback)` at
        # import time, so the stub class must be subclassable.
        self.assertTrue(isinstance(keras.callbacks.Callback(), keras.callbacks.Callback))
        self.assertTrue(issubclass(keras.callbacks.History, keras.callbacks.Callback))

    def test_is_idempotent(self):
        disable_keras_backend()
        first = sys.modules["keras"]
        disable_keras_backend()
        self.assertIs(sys.modules["keras"], first)

    def test_serialize_round_trip(self):
        disable_keras_backend()
        from keras.utils import deserialize_keras_object
        from keras.utils import serialize_keras_object

        class Widget:
            def __init__(self, kind: str = "alpha"):
                self.kind = kind

            def get_config(self):
                return {"kind": self.kind}

            @classmethod
            def from_config(cls, cfg):
                return cls(**cfg)

        cfg = serialize_keras_object(Widget(kind="beta"))
        self.assertEqual(cfg["class_name"], "Widget")
        self.assertEqual(cfg["config"], {"kind": "beta"})

        restored = deserialize_keras_object(cfg, custom_objects={"Widget": Widget})
        self.assertEqual(restored.kind, "beta")

    def test_deserialize_unknown_class_raises(self):
        disable_keras_backend()
        from keras.utils import deserialize_keras_object

        with self.assertRaisesRegex(ValueError, "unknown class"):
            deserialize_keras_object({"class_name": "Nope", "config": {}})

    def test_deserialize_passthrough_for_non_dict(self):
        disable_keras_backend()
        from keras.utils import deserialize_keras_object

        # kt occasionally serializes primitives (ints, strings) in HP configs;
        # those must round-trip unchanged.
        self.assertEqual(deserialize_keras_object(42), 42)
        self.assertEqual(deserialize_keras_object("hello"), "hello")
        self.assertEqual(deserialize_keras_object(None), None)

    # ------------------------------------------------------------------
    # Coexistence with real Keras
    # ------------------------------------------------------------------

    def test_real_keras_already_loaded_is_not_shadowed(self):
        """Scenario A: `import keras` first → `disable_keras_backend()` is a no-op.

        This is the documented path for users who want to use real Keras
        alongside synalinks in the same process.
        """
        real = _make_fake_real_keras()
        sys.modules["keras"] = real

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # would raise if a warning fires
            disable_keras_backend()

        self.assertIs(sys.modules["keras"], real)
        # Stub-specific submodules must NOT have been installed.
        for stub_only in ("keras.config", "keras.src.ops", "keras.callbacks"):
            self.assertNotIn(stub_only, sys.modules)

    def test_happy_path_emits_no_warnings(self):
        """Fresh state: stub installs silently.

        Whether or not real Keras is on disk no longer affects this — the
        only signal we use is `sys.modules`. So the typical
        `synalinks + keras-tuner` user (kt pulls keras transitively but
        never imports it) gets a quiet install.
        """
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            disable_keras_backend()
        relevant = [
            w for w in captured if issubclass(w.category, (UserWarning, RuntimeWarning))
        ]

        self.assertEqual(
            relevant,
            [],
            f"Expected no warnings, got: {[str(w.message) for w in relevant]}",
        )
        self.assertIn("keras", sys.modules)

    def test_warns_and_noops_when_keras_tuner_already_imported(self):
        """Scenario: user called `disable_keras_backend()` too late.

        `keras_tuner` was imported first, so it has already bound to whatever
        keras it found at import time. Installing the stub now would not
        rewire kt and could mask the real failure mode — warn and bail.
        """
        sys.modules["keras_tuner"] = types.ModuleType("keras_tuner")

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            disable_keras_backend()
        runtime_warnings = [w for w in captured if issubclass(w.category, RuntimeWarning)]

        self.assertEqual(len(runtime_warnings), 1, f"Got: {runtime_warnings!r}")
        self.assertIn("keras_tuner", str(runtime_warnings[0].message))
        # No-op: no stub installed.
        self.assertNotIn("keras", sys.modules)
        self.assertNotIn("keras.callbacks", sys.modules)


class KerasTunerIntegrationTest(testing.TestCase):
    """End-to-end: verify a synalinks user can actually drive `keras_tuner`.

    These exercise the canonical pattern — subclass `kt.Tuner`, override
    `run_trial` to return a float or dict — against the stub. Skipped
    automatically when `keras_tuner` isn't installed.

    Each test gets a fresh kt import: once kt loads it caches references
    to `keras.callbacks.Callback` and friends, so we must clear the whole
    `keras*` / `keras_tuner*` subtree from `sys.modules` between tests.
    """

    def setUp(self):
        super().setUp()
        if importlib.util.find_spec("keras_tuner") is None:
            self.skipTest("keras_tuner is not installed")
        self._removed = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if _is_managed_module(name)
        }
        self._tmpdirs = []

    def tearDown(self):
        for d in self._tmpdirs:
            shutil.rmtree(d, ignore_errors=True)
        for name in list(sys.modules):
            if _is_managed_module(name):
                sys.modules.pop(name, None)
        for name, module in self._removed.items():
            sys.modules[name] = module
        super().tearDown()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tmpdir(self):
        d = tempfile.mkdtemp(prefix="kt_test_")
        self._tmpdirs.append(d)
        return d

    def _kt(self):
        """Install the stub, import kt fresh, return the module."""
        disable_keras_backend()
        import keras_tuner

        return keras_tuner

    @staticmethod
    @contextlib.contextmanager
    def _quiet():
        """Suppress kt's progress prints during search()."""
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                yield

    # ------------------------------------------------------------------
    # Import-time wiring
    # ------------------------------------------------------------------

    def test_kt_imports_and_exposes_core_api(self):
        """The stub lets `import keras_tuner` succeed and surfaces the
        symbols a synalinks user reaches for."""
        kt = self._kt()
        for name in (
            "Tuner",
            "Oracle",
            "Objective",
            "HyperParameters",
            "RandomSearch",
            "GridSearch",
            "oracles",
        ):
            self.assertTrue(hasattr(kt, name), f"kt is missing `{name}`")
        self.assertTrue(hasattr(kt.oracles, "RandomSearchOracle"))
        self.assertTrue(hasattr(kt.oracles, "GridSearchOracle"))

    def test_tuner_callback_subclasses_stub(self):
        """kt's internal `TunerCallback` must successfully subclass the
        stub's `keras.callbacks.Callback` at import time."""
        kt = self._kt()  # noqa: F841 — imports trigger kt module loads
        import keras
        from keras_tuner.src.engine.tuner_utils import TunerCallback

        self.assertTrue(issubclass(TunerCallback, keras.callbacks.Callback))

    # ------------------------------------------------------------------
    # `run_trial` return-value handling
    # ------------------------------------------------------------------

    def test_run_trial_returning_dict_is_recorded(self):
        """The recommended return form: `{objective_name: value}`."""
        kt = self._kt()

        class T(kt.Tuner):
            def run_trial(self, trial, *_, **__):
                x = trial.hyperparameters.Float("x", 0.0, 1.0)
                return {"reward": 1.0 - abs(x - 0.3)}

        tuner = T(
            oracle=kt.oracles.RandomSearchOracle(
                objective=kt.Objective("reward", direction="max"),
                max_trials=4,
                seed=1,
            ),
            directory=self._tmpdir(),
            project_name="dict_return",
        )
        with self._quiet():
            tuner.search()

        self.assertEqual(len(tuner.oracle.trials), 4)
        for trial in tuner.oracle.trials.values():
            self.assertEqual(trial.status, "COMPLETED")
            self.assertIsNotNone(trial.score)

    def test_run_trial_returning_bare_float_is_recorded(self):
        """kt also accepts a bare float (uses it as the objective value)."""
        kt = self._kt()

        class T(kt.Tuner):
            def run_trial(self, trial, *_, **__):
                x = trial.hyperparameters.Float("x", 0.0, 1.0)
                return 1.0 - abs(x - 0.5)

        tuner = T(
            oracle=kt.oracles.RandomSearchOracle(
                objective=kt.Objective("score", direction="max"),
                max_trials=3,
                seed=2,
            ),
            directory=self._tmpdir(),
            project_name="float_return",
        )
        with self._quiet():
            tuner.search()

        scores = [t.score for t in tuner.oracle.trials.values()]
        self.assertEqual(len(scores), 3)
        for s in scores:
            self.assertIsNotNone(s)

    # ------------------------------------------------------------------
    # Hyperparameter surface
    # ------------------------------------------------------------------

    def test_all_hyperparameter_types(self):
        """Float / Int / Choice / Boolean all sample valid values."""
        kt = self._kt()
        captured = {}

        class T(kt.Tuner):
            def run_trial(self, trial, *_, **__):
                hp = trial.hyperparameters
                captured["f"] = hp.Float("f", 0.0, 1.0, default=0.5)
                captured["i"] = hp.Int("i", 1, 10, default=5)
                captured["c"] = hp.Choice("c", ["a", "b", "c"], default="a")
                captured["b"] = hp.Boolean("b", default=True)
                return 0.5

        tuner = T(
            oracle=kt.oracles.RandomSearchOracle(
                objective=kt.Objective("score", direction="max"),
                max_trials=1,
                seed=3,
            ),
            directory=self._tmpdir(),
            project_name="hp_types",
        )
        with self._quiet():
            tuner.search()

        self.assertIsInstance(captured["f"], float)
        self.assertGreaterEqual(captured["f"], 0.0)
        self.assertLessEqual(captured["f"], 1.0)
        self.assertIsInstance(captured["i"], int)
        self.assertGreaterEqual(captured["i"], 1)
        self.assertLessEqual(captured["i"], 10)
        self.assertIn(captured["c"], ["a", "b", "c"])
        self.assertIsInstance(captured["b"], bool)

    def test_hyperparameters_serialize_round_trip_via_stub(self):
        """kt persists HPs through `keras.utils.serialize_keras_object`
        (the stub's serializer). Verify a round-trip preserves values."""
        kt = self._kt()
        from keras.utils import deserialize_keras_object
        from keras.utils import serialize_keras_object

        hp = kt.HyperParameters()
        hp.Float("lr", 1e-4, 1e-1, default=1e-3)
        hp.Choice("opt", ["adam", "sgd"], default="adam")
        hp.Int("batch", 8, 128, default=32)

        cfg = serialize_keras_object(hp)
        restored = deserialize_keras_object(
            cfg, custom_objects={"HyperParameters": kt.HyperParameters}
        )
        self.assertAlmostEqual(restored.get("lr"), hp.get("lr"))
        self.assertEqual(restored.get("opt"), hp.get("opt"))
        self.assertEqual(restored.get("batch"), hp.get("batch"))

    # ------------------------------------------------------------------
    # Search behavior
    # ------------------------------------------------------------------

    def test_get_best_hyperparameters_returns_top_trial(self):
        """The trial closest to the reward peak should win."""
        kt = self._kt()

        class T(kt.Tuner):
            def run_trial(self, trial, *_, **__):
                x = trial.hyperparameters.Float("x", 0.0, 1.0)
                return {"reward": 1.0 - abs(x - 0.3)}

        tuner = T(
            oracle=kt.oracles.RandomSearchOracle(
                objective=kt.Objective("reward", direction="max"),
                max_trials=20,
                seed=4,
            ),
            directory=self._tmpdir(),
            project_name="best_hp",
        )
        with self._quiet():
            tuner.search()

        best = tuner.get_best_hyperparameters(num_trials=1)
        self.assertEqual(len(best), 1)
        # 20 random samples in [0,1] — best should land within 0.3 of the peak.
        self.assertLess(abs(best[0].get("x") - 0.3), 0.3)

    def test_search_respects_min_direction(self):
        """`direction='min'` should pick the lowest-loss trial."""
        kt = self._kt()

        class T(kt.Tuner):
            def run_trial(self, trial, *_, **__):
                x = trial.hyperparameters.Float("x", 0.0, 1.0)
                return {"loss": abs(x - 0.7)}

        tuner = T(
            oracle=kt.oracles.RandomSearchOracle(
                objective=kt.Objective("loss", direction="min"),
                max_trials=20,
                seed=5,
            ),
            directory=self._tmpdir(),
            project_name="min_dir",
        )
        with self._quiet():
            tuner.search()

        best = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.assertLess(abs(best.get("x") - 0.7), 0.3)

    def test_grid_search_oracle_enumerates_choices(self):
        """A different oracle (GridSearch) exercises a non-trivial path."""
        kt = self._kt()
        seen_choices = set()

        class T(kt.Tuner):
            def run_trial(self, trial, *_, **__):
                c = trial.hyperparameters.Choice("c", ["a", "b", "c"])
                seen_choices.add(c)
                return {"reward": 0.5}

        tuner = T(
            oracle=kt.oracles.GridSearchOracle(
                objective=kt.Objective("reward", direction="max"),
                max_trials=10,
            ),
            directory=self._tmpdir(),
            project_name="grid",
        )
        with self._quiet():
            tuner.search()

        # Grid search must have covered every choice value.
        self.assertEqual(seen_choices, {"a", "b", "c"})

    def test_trials_persist_to_disk(self):
        """kt writes each trial under the project directory — verify our
        stub doesn't interfere with that persistence."""
        kt = self._kt()
        import os

        class T(kt.Tuner):
            def run_trial(self, trial, *_, **__):
                x = trial.hyperparameters.Float("x", 0.0, 1.0)
                return {"reward": x}

        directory = self._tmpdir()
        tuner = T(
            oracle=kt.oracles.RandomSearchOracle(
                objective=kt.Objective("reward", direction="max"),
                max_trials=3,
                seed=7,
            ),
            directory=directory,
            project_name="persist",
        )
        with self._quiet():
            tuner.search()

        project_dir = os.path.join(directory, "persist")
        self.assertTrue(os.path.isdir(project_dir))
        # At least one trial subdirectory should exist.
        trial_dirs = [d for d in os.listdir(project_dir) if d.startswith("trial_")]
        self.assertEqual(len(trial_dirs), 3)

    def test_user_can_pass_synalinks_callbacks_to_program_fit(self):
        """Sanity check: synalinks's `Callback` lives in a different
        namespace from the stub's `keras.callbacks.Callback`. A user who
        wires synalinks callbacks into `program.fit(callbacks=[...])`
        inside `run_trial` is unaffected by the stub."""
        self._kt()  # install stub + import kt
        import keras

        from synalinks.src.callbacks.callback import Callback as SynalinksCallback

        # The two Callback classes are unrelated.
        self.assertIsNot(SynalinksCallback, keras.callbacks.Callback)
        self.assertFalse(issubclass(SynalinksCallback, keras.callbacks.Callback))
        self.assertFalse(issubclass(keras.callbacks.Callback, SynalinksCallback))
