# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import importlib.util
import shutil
import sys
import tempfile
import types

from synalinks.src import testing
from synalinks.src.callbacks.history import History as SynalinksHistory
from synalinks.src.utils import keras_tuner_utils
from synalinks.src.utils.keras_backend import disable_keras_backend
from synalinks.src.utils.keras_tuner_utils import BayesianOptimization
from synalinks.src.utils.keras_tuner_utils import GridSearch
from synalinks.src.utils.keras_tuner_utils import Hyperband
from synalinks.src.utils.keras_tuner_utils import Objective
from synalinks.src.utils.keras_tuner_utils import RandomSearch
from synalinks.src.utils.keras_tuner_utils import _history_to_metrics_dict
from synalinks.src.utils.keras_tuner_utils import _objective_directions


def _is_managed_module(name: str) -> bool:
    return (
        name == "keras"
        or name.startswith("keras.")
        or name == "keras_tuner"
        or name.startswith("keras_tuner.")
    )


class StubExportTest(testing.TestCase):
    """The stub classes that get decorated with `@synalinks_export` and
    discovered by the API generator. They must:

      - exist as module-level symbols (so namex finds them),
      - carry the `_api_export_path` attribute set by the decorator,
      - resolve to the right path, and
      - NOT pull in `keras_tuner` simply by being imported.
    """

    def test_stub_classes_are_module_level(self):
        for sym in (RandomSearch, BayesianOptimization, Hyperband, GridSearch):
            self.assertTrue(callable(sym))
            self.assertEqual(sym.__module__, keras_tuner_utils.__name__)

    def test_stub_class_names_round_trip(self):
        self.assertEqual(RandomSearch.__name__, "RandomSearch")
        self.assertEqual(BayesianOptimization.__name__, "BayesianOptimization")
        self.assertEqual(Hyperband.__name__, "Hyperband")
        self.assertEqual(GridSearch.__name__, "GridSearch")

    def test_importing_module_does_not_load_keras_tuner(self):
        """Critical: `keras_tuner` must remain lazy.

        Otherwise the api auto-import path (`synalinks/api/tuners/__init__.py`
        is loaded eagerly from `synalinks/__init__.py`) would force every
        synalinks user to have keras-tuner installed.

        We can't reliably *unload* kt for this test, so instead we re-import
        a fresh copy of `keras_tuner_utils` after stripping kt from
        `sys.modules`, and confirm the stub access alone does not reload it.
        """
        saved = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if _is_managed_module(name)
        }
        try:
            sys.modules.pop("synalinks.src.utils.keras_tuner_utils", None)
            spec = importlib.util.find_spec("synalinks.src.utils.keras_tuner_utils")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            self.assertNotIn(
                "keras_tuner",
                sys.modules,
                "Importing `keras_tuner_utils` must not import keras_tuner.",
            )
            # Touching the symbol on the module is still lazy — only
            # instantiation triggers the kt import.
            _ = mod.RandomSearch
            self.assertNotIn("keras_tuner", sys.modules)
        finally:
            for name in list(sys.modules):
                if _is_managed_module(name):
                    sys.modules.pop(name, None)
            for name, m in saved.items():
                sys.modules[name] = m

    def test_export_decorator_registered_paths(self):
        """`@synalinks_export` should have routed each stub to the tuners ns."""
        from synalinks.src.api_export import REGISTERED_NAMES_TO_OBJS

        for name in ("RandomSearch", "BayesianOptimization", "Hyperband", "GridSearch"):
            path = f"synalinks.tuners.{name}"
            self.assertIn(path, REGISTERED_NAMES_TO_OBJS, f"Missing export: {path}")


class HistoryReductionTest(testing.TestCase):
    """`_history_to_metrics_dict` is the bit we own — keras-tuner doesn't
    know how to unwrap a synalinks History on its own."""

    def _make_history(self, history_dict):
        h = SynalinksHistory()
        h.history = history_dict
        return h

    def test_single_objective_max_picks_max_across_epochs(self):
        history = self._make_history(
            {
                "reward": [0.1, 0.7, 0.4],
                "val_reward": [0.2, 0.5, 0.9],
            }
        )
        obj = types.SimpleNamespace(name="val_reward", direction="max")
        result = _history_to_metrics_dict(history, obj)
        self.assertEqual(result["val_reward"], 0.9)
        # Non-objective metrics use the *last* value, not the best.
        self.assertEqual(result["reward"], 0.4)

    def test_single_objective_min_picks_min_across_epochs(self):
        history = self._make_history({"loss": [1.0, 0.3, 0.6]})
        obj = types.SimpleNamespace(name="loss", direction="min")
        result = _history_to_metrics_dict(history, obj)
        self.assertEqual(result["loss"], 0.3)

    def test_multi_objective_reduces_each_metric_in_its_own_direction(self):
        history = self._make_history(
            {
                "loss": [1.0, 0.4, 0.5],
                "reward": [0.2, 0.9, 0.7],
            }
        )
        # MultiObjective exposes `name_to_direction` directly.
        multi = types.SimpleNamespace(
            name_to_direction={"loss": "min", "reward": "max"},
        )
        result = _history_to_metrics_dict(history, multi)
        self.assertEqual(result["loss"], 0.4)
        self.assertEqual(result["reward"], 0.9)

    def test_empty_metric_lists_are_skipped(self):
        history = self._make_history({"reward": [], "val_reward": [0.5]})
        obj = types.SimpleNamespace(name="val_reward", direction="max")
        result = _history_to_metrics_dict(history, obj)
        self.assertNotIn("reward", result)
        self.assertEqual(result["val_reward"], 0.5)

    def test_objective_directions_handles_default_objective(self):
        # kt's `DefaultObjective` has name="default_objective", direction="min".
        default = types.SimpleNamespace(name="default_objective", direction="min")
        directions = _objective_directions(default)
        self.assertEqual(directions, {"default_objective": "min"})


class ResolveErrorTest(testing.TestCase):
    """When `keras_tuner` can't be imported, the message should tell the
    user exactly what to fix (call `disable_keras_backend()` or install
    keras-tuner)."""

    def setUp(self):
        super().setUp()
        self._saved = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if _is_managed_module(name)
        }
        # Clear the resolved-class cache so each test re-runs the resolver.
        self._cache_backup = dict(keras_tuner_utils._REAL_SUBCLASSES)
        keras_tuner_utils._REAL_SUBCLASSES.clear()

    def tearDown(self):
        keras_tuner_utils._REAL_SUBCLASSES.clear()
        keras_tuner_utils._REAL_SUBCLASSES.update(self._cache_backup)
        for name in list(sys.modules):
            if _is_managed_module(name):
                sys.modules.pop(name, None)
        for name, m in self._saved.items():
            sys.modules[name] = m
        super().tearDown()

    def test_missing_keras_backend_message_points_at_disable_keras_backend(self):
        """Without Keras and without the stub, kt fails to import. The
        wrapped error must mention `disable_keras_backend` so the user
        knows the remediation. Either `RuntimeError` (keras/keras_tuner
        missing after a clean import attempt) or `ImportError` (kt itself
        not installed) is acceptable here — both bodies point the user at
        the same fix path."""
        with self.assertRaises((RuntimeError, ImportError)) as ctx:
            RandomSearch()  # triggers _resolve_kt_tuner
        msg = str(ctx.exception)
        self.assertTrue(
            "disable_keras_backend" in msg or "keras-tuner" in msg,
            f"Unhelpful error message: {msg!r}",
        )

    def test_resolved_subclass_is_cached(self):
        """Repeated instantiation must not re-build the kt subclass."""
        disable_keras_backend()
        try:
            import keras_tuner  # noqa: F401
        except Exception:
            self.skipTest("keras_tuner is not importable in this env")
        cls_1 = keras_tuner_utils._resolve_kt_tuner("RandomSearch")
        cls_2 = keras_tuner_utils._resolve_kt_tuner("RandomSearch")
        self.assertIs(cls_1, cls_2)


class TunerEndToEndTest(testing.TestCase):
    """Drive a real `kt.RandomSearch` through `tuner.search()` with a fake
    synalinks Program. We mock `program.fit` to return a synthetic History
    so the test runs offline (no LM)."""

    def setUp(self):
        super().setUp()
        # `find_spec` interacts oddly with pytest's collection (it returns
        # `None` here even when `keras_tuner` is importable in a fresh
        # interpreter). Probe by actually attempting an import instead.
        from synalinks.src.utils.keras_backend import disable_keras_backend

        disable_keras_backend()
        try:
            import keras_tuner  # noqa: F401
        except Exception as e:
            self.skipTest(f"keras_tuner is not importable: {type(e).__name__}: {e}")
        self._saved = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if _is_managed_module(name)
        }
        keras_tuner_utils._REAL_SUBCLASSES.clear()
        self._tmpdir = tempfile.mkdtemp(prefix="synalinks_kt_test_")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)
        keras_tuner_utils._REAL_SUBCLASSES.clear()
        for name in list(sys.modules):
            if _is_managed_module(name):
                sys.modules.pop(name, None)
        for name, m in self._saved.items():
            sys.modules[name] = m
        super().tearDown()

    def _build_fake_program_factory(self):
        """Return a hypermodel callable + the list of HP values it saw."""
        seen = []

        async def build(hp):
            x = hp.Float("x", 0.0, 1.0)
            seen.append(x)

            class _FakeProgram:
                optimizer = object()  # truthy → dispatch to fit() branch

                async def fit(self_inner, *args, **kwargs):
                    # Reward peaks at x=0.3 — RandomSearch with enough trials
                    # should converge near that.
                    score = 1.0 - abs(x - 0.3)
                    h = SynalinksHistory()
                    h.history = {"val_reward": [score]}
                    return h

            return _FakeProgram()

        return build, seen

    def test_random_search_drives_fake_synalinks_program_via_fit(self):
        disable_keras_backend()
        build, seen = self._build_fake_program_factory()

        tuner = RandomSearch(
            hypermodel=build,
            objective=Objective("val_reward", direction="max"),
            max_trials=10,
            directory=self._tmpdir,
            project_name="fake_program_fit",
            overwrite=True,
            seed=42,
        )
        # The space is explored once at construction (`_populate_initial_space`)
        # and then once per trial, so the hypermodel runs `1 + max_trials` times.
        self.assertEqual(
            len(seen), 1, "hypermodel should run once to explore the space at init"
        )

        # Args go through to `program.fit(...)`. Our fake ignores them.
        tuner.search(epochs=1)

        self.assertEqual(
            len(seen), 11, "hypermodel should run once per trial after the init build"
        )
        best = tuner.get_best_hyperparameters(num_trials=1)[0]
        # 10 random samples in [0, 1] — best x should land near 0.3.
        self.assertLess(abs(best.get("x") - 0.3), 0.3)

    def test_async_hypermodel_populates_space_at_construction(self):
        """An `async def build` must have its body run during the synchronous
        space exploration in `BaseTuner.__init__`.

        Regression test for the `RuntimeWarning: coroutine 'build_program' was
        never awaited` raised when kt called the async hypermodel synchronously:
        the coroutine body never ran, so `hp.Float(...)` never registered and
        the initial search space was left empty. We assert both that no such
        warning fires at construction and that the space is populated before
        `search()` is ever called.
        """
        import warnings

        disable_keras_backend()

        async def build(hp):
            hp.Float("x", 0.0, 1.0)

            class _FakeProgram:
                optimizer = object()

                async def fit(self_inner, *args, **kwargs):
                    h = SynalinksHistory()
                    h.history = {"val_reward": [1.0]}
                    return h

            return _FakeProgram()

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            tuner = RandomSearch(
                hypermodel=build,
                objective=Objective("val_reward", direction="max"),
                max_trials=2,
                directory=self._tmpdir,
                project_name="async_space_population",
                overwrite=True,
                seed=1,
            )

        # The hyperparameter registered inside the async build must be visible
        # in the oracle's space *before* any trial runs.
        space_names = {hp.name for hp in tuner.oracle.get_space().space}
        self.assertIn("x", space_names)

    def test_sync_hypermodel_is_also_accepted(self):
        """Hypermodels that return a Program directly (no `async def`) must
        work — `inspect.isawaitable` falls through to the program as-is."""
        disable_keras_backend()

        def build(hp):
            x = hp.Float("x", 0.0, 1.0)

            class _FakeProgram:
                optimizer = object()  # truthy → dispatch to fit() branch

                async def fit(self_inner, *args, **kwargs):
                    h = SynalinksHistory()
                    h.history = {"val_reward": [1.0 - abs(x - 0.5)]}
                    return h

            return _FakeProgram()

        tuner = RandomSearch(
            hypermodel=build,
            objective=Objective("val_reward", direction="max"),
            max_trials=4,
            directory=self._tmpdir,
            project_name="sync_hypermodel",
            overwrite=True,
            seed=7,
        )
        tuner.search()
        self.assertEqual(len(tuner.oracle.trials), 4)
        for t in tuner.oracle.trials.values():
            self.assertEqual(t.status, "COMPLETED")

    def test_multi_objective_drives_search_end_to_end(self):
        """Multi-objective: pass a list of `Objective`s to the tuner. kt
        wraps them in a `MultiObjective`, and we reduce each metric in
        its own direction inside `_history_to_metrics_dict`."""
        disable_keras_backend()
        seen = []

        async def build(hp):
            x = hp.Float("x", 0.0, 1.0)
            seen.append(x)

            class _FakeProgram:
                optimizer = object()  # truthy → dispatch to fit() branch

                async def fit(self_inner, *args, **kwargs):
                    h = SynalinksHistory()
                    # Two competing metrics in different directions.
                    h.history = {
                        "val_reward": [1.0 - abs(x - 0.3)],  # maximize
                        "val_loss": [abs(x - 0.3)],  # minimize
                    }
                    return h

            return _FakeProgram()

        tuner = RandomSearch(
            hypermodel=build,
            objective=[
                Objective("val_reward", direction="max"),
                Objective("val_loss", direction="min"),
            ],
            max_trials=10,
            directory=self._tmpdir,
            project_name="multi_objective",
            overwrite=True,
            seed=13,
        )
        tuner.search()

        self.assertEqual(len(tuner.oracle.trials), 10)
        # kt's `MultiObjective` aggregates as (-max metrics + min metrics);
        # both peaks coincide at x=0.3 in this fake, so the best trial
        # should land near it.
        best = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.assertLess(abs(best.get("x") - 0.3), 0.3)

        # Each completed trial must carry both metric values (not just the
        # objective name) — confirms we don't drop non-aggregated metrics
        # when the objective is a MultiObjective.
        for trial in tuner.oracle.trials.values():
            self.assertEqual(trial.status, "COMPLETED")
            self.assertIn("val_reward", trial.metrics.metrics)
            self.assertIn("val_loss", trial.metrics.metrics)

    def test_min_direction_picks_lowest_value(self):
        disable_keras_backend()

        async def build(hp):
            x = hp.Float("x", 0.0, 1.0)

            class _FakeProgram:
                optimizer = object()  # truthy → dispatch to fit() branch

                async def fit(self_inner, *args, **kwargs):
                    h = SynalinksHistory()
                    h.history = {"val_loss": [abs(x - 0.8)]}
                    return h

            return _FakeProgram()

        tuner = RandomSearch(
            hypermodel=build,
            objective=Objective("val_loss", direction="min"),
            max_trials=10,
            directory=self._tmpdir,
            project_name="min_direction",
            overwrite=True,
            seed=11,
        )
        tuner.search()
        best = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.assertLess(abs(best.get("x") - 0.8), 0.3)


class ObjectiveDirectionInferenceTest(testing.TestCase):
    """When `direction` is omitted on `synalinks.tuners.Objective` or when a
    bare string objective like `objective="val_reward"` is passed to a tuner,
    direction should resolve via the synalinks metrics registry (each metric
    class carries a `direction` class attribute) and fall through to
    keras-tuner's original inference for keras-compatible names like
    `"loss"`. Reset the patch state per test so the patch is re-applied to
    a freshly-imported `keras_tuner` module."""

    def setUp(self):
        super().setUp()
        from synalinks.src.utils.keras_backend import disable_keras_backend

        disable_keras_backend()
        try:
            import keras_tuner  # noqa: F401
        except Exception as e:
            self.skipTest(f"keras_tuner is not importable: {type(e).__name__}: {e}")
        self._saved = {
            name: sys.modules.pop(name)
            for name in list(sys.modules)
            if _is_managed_module(name)
        }
        # Popping cleared the stubbed `keras` from `sys.modules`; re-install it
        # so the test body's fresh `import keras_tuner` binds to the stub rather
        # than pulling real Keras (which needs a backend like TensorFlow).
        disable_keras_backend()
        keras_tuner_utils._kt_inference_patched = False

    def tearDown(self):
        keras_tuner_utils._kt_inference_patched = False
        for name in list(sys.modules):
            if _is_managed_module(name):
                sys.modules.pop(name, None)
        for name, m in self._saved.items():
            sys.modules[name] = m
        super().tearDown()

    def test_reward_infers_as_max(self):
        """`"reward"` is the conventional name for a `Mean`/wrapper around a
        synalinks `Reward` and is not a class name in the registry — it's
        special-cased to `"max"` in `_synalinks_name_direction_map`."""
        obj = Objective("val_reward")
        self.assertEqual(obj.direction, "max")

    def test_accuracy_infers_as_max(self):
        """`Accuracy.direction == "up"` (class attr) → kt `"max"`."""
        obj = Objective("accuracy")
        self.assertEqual(obj.direction, "max")

    def test_f1_score_infers_as_max(self):
        """Inherited via `FBetaScore.direction = "up"` on the parent."""
        obj = Objective("val_f1_score")
        self.assertEqual(obj.direction, "max")

    def test_program_cost_infers_as_min(self):
        """`ProgramOperationalMetric.direction == "down"` → kt `"min"`."""
        obj = Objective("val_program_cost")
        self.assertEqual(obj.direction, "min")

    def test_loss_falls_through_to_kt_inference(self):
        """kt's original `infer_metric_direction` handles `"loss"` as a
        special case. The synalinks patch falls through for names not in
        its table — verifying compatibility is preserved."""
        obj = Objective("val_loss")
        self.assertEqual(obj.direction, "min")

    def test_unknown_name_raises_with_remediation(self):
        with self.assertRaises(ValueError) as ctx:
            Objective("val_totally_made_up")
        self.assertIn("Could not infer", str(ctx.exception))
        self.assertIn("direction=", str(ctx.exception))

    def test_explicit_direction_bypasses_inference(self):
        """A user-supplied direction overrides registry lookup — useful for
        custom metrics with no `direction` set."""
        obj = Objective("any_unknown_metric", direction="min")
        self.assertEqual(obj.direction, "min")
