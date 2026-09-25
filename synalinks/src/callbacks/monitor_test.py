# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.callbacks import monitor as monitor_module
from synalinks.src.callbacks.monitor import Monitor
from synalinks.src.callbacks.monitor import SynalinksProgramModel
from synalinks.src.version import __version__


class _FakeRunInfo:
    def __init__(self, run_id="run-123", artifact_uri=""):
        self.run_id = run_id
        self.artifact_uri = artifact_uri


class _FakeRun:
    def __init__(self, run_id="run-123", artifact_uri=""):
        self.info = _FakeRunInfo(run_id=run_id, artifact_uri=artifact_uri)


class _FakeProgram:
    def __init__(self, name="prog", description="desc", built=True):
        self.name = name
        self.description = description
        self.built = built
        self.trainable_variables = []

    def get_state_tree(self):
        return {"trainable_variables": {}}


def _run_coro(coro):
    """Drive an async helper directly in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class MonitorInitTest(testing.TestCase):
    def test_missing_mlflow_raises(self):
        with patch.object(monitor_module, "MLFLOW_AVAILABLE", False):
            with self.assertRaisesRegex(ImportError, "mlflow is required"):
                Monitor()

    def test_default_attributes(self):
        cb = Monitor()
        self.assertIsNone(cb.experiment_name)
        self.assertIsNone(cb.run_name)
        self.assertIsNone(cb.tracking_uri)
        self.assertFalse(cb.log_batch_metrics)
        self.assertTrue(cb.log_epoch_metrics)
        self.assertTrue(cb.log_program_plot)
        self.assertTrue(cb.log_program_model)
        self.assertEqual(cb.tags, {})
        self.assertIsNone(cb._run)
        self.assertEqual(cb._steps, {"train": 0, "val": 0, "test": 0})
        self.assertEqual(cb._epoch, 0)
        self.assertFalse(cb._in_training)


class MonitorSetupAndRunTest(testing.TestCase):
    def test_setup_uses_tracking_uri_and_program_name(self):
        cb = Monitor(tracking_uri="http://example", experiment_name=None)
        cb.set_program(_FakeProgram(name="my_prog"))
        with patch.object(monitor_module, "mlflow") as mlf:
            cb._setup_mlflow()
        mlf.set_tracking_uri.assert_called_once_with("http://example")
        mlf.set_experiment.assert_called_once_with("my_prog")

    def test_setup_falls_back_to_default_experiment_name(self):
        cb = Monitor()
        cb.set_program(_FakeProgram(name=""))
        with patch.object(monitor_module, "mlflow") as mlf:
            cb._setup_mlflow()
        mlf.set_experiment.assert_called_once_with("synalinks_experiment")

    def test_start_run_tags_with_program_metadata(self):
        cb = Monitor(run_name="base", tags={"extra": "v"})
        cb.set_program(_FakeProgram(name="p1", description="d1"))
        fake_run = _FakeRun()
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = fake_run
            cb._start_run(run_name_suffix="train")
        mlf.start_run.assert_called_once_with(run_name="base_train")
        tags_arg = mlf.set_tags.call_args[0][0]
        self.assertEqual(tags_arg["extra"], "v")
        self.assertEqual(tags_arg["program_name"], "p1")
        self.assertEqual(tags_arg["program_description"], "d1")
        self.assertIs(cb._run, fake_run)

    def test_start_run_without_program(self):
        cb = Monitor()
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = _FakeRun()
            cb._start_run(run_name_suffix="test")
        mlf.start_run.assert_called_once_with(run_name="test")

    def test_end_run_clears_state(self):
        cb = Monitor()
        cb._run = _FakeRun()
        with patch.object(monitor_module, "mlflow") as mlf:
            cb._end_run()
        mlf.end_run.assert_called_once()
        self.assertIsNone(cb._run)

    def test_end_run_noop_without_active_run(self):
        cb = Monitor()
        with patch.object(monitor_module, "mlflow") as mlf:
            cb._end_run()
        mlf.end_run.assert_not_called()


class MonitorLogMetricsTest(testing.TestCase):
    async def test_log_metrics_filters_non_scalars(self):
        cb = Monitor()
        cb._run = _FakeRun()
        with patch.object(monitor_module, "mlflow") as mlf:
            await cb._log_metrics({"a": 1.0, "b": 2, "c": "str", "d": None}, step=4)
        mlf.log_metrics.assert_called_once_with(
            {"a": 1.0, "b": 2}, step=4, run_id="run-123"
        )

    async def test_log_metrics_skips_when_no_run(self):
        cb = Monitor()
        with patch.object(monitor_module, "mlflow") as mlf:
            await cb._log_metrics({"a": 1.0})
        mlf.log_metrics.assert_not_called()

    async def test_log_metrics_skips_when_no_logs(self):
        cb = Monitor()
        cb._run = _FakeRun()
        with patch.object(monitor_module, "mlflow") as mlf:
            await cb._log_metrics(None)
        mlf.log_metrics.assert_not_called()

    async def test_log_metrics_with_no_scalars_skips_call(self):
        cb = Monitor()
        cb._run = _FakeRun()
        with patch.object(monitor_module, "mlflow") as mlf:
            await cb._log_metrics({"a": "str"}, step=1)
        mlf.log_metrics.assert_not_called()

    async def test_log_params_filters_non_scalars(self):
        cb = Monitor()
        cb._run = _FakeRun()
        cb.set_params({"epochs": 10, "lr": 0.1, "tag": "x", "obj": object()})
        with patch.object(monitor_module, "mlflow") as mlf:
            await cb._log_params()
        logged = mlf.log_params.call_args[0][0]
        self.assertEqual(
            {k: logged[k] for k in ("epochs", "lr", "tag")},
            {"epochs": 10, "lr": 0.1, "tag": "x"},
        )
        self.assertNotIn("obj", logged)

    async def test_log_params_swallows_errors(self):
        cb = Monitor()
        cb._run = _FakeRun()
        cb.set_params({"epochs": 10, "batch_size": 2, "lr": 0.1})
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.log_params.side_effect = RuntimeError("boom")
            client = mlf.MlflowClient.return_value
            client.log_param.side_effect = [None, RuntimeError("conflict"), None]
            # Should not raise; Monitor falls back to per-key logging.
            await cb._log_params()
        self.assertEqual(client.log_param.call_count, 3)
        self.assertEqual(
            client.log_param.call_args_list[0].args[:2], ("run-123", "epochs")
        )

    async def test_log_params_skips_when_no_run(self):
        cb = Monitor()
        cb.set_params({"epochs": 10})
        with patch.object(monitor_module, "mlflow") as mlf:
            await cb._log_params()
        mlf.log_params.assert_not_called()


class MonitorLifecycleTest(testing.TestCase):
    def _patched_monitor(self, **kwargs):
        cb = Monitor(**kwargs)
        cb.set_program(_FakeProgram())
        cb.set_params({"epochs": 3})
        return cb

    def test_train_lifecycle_drives_setup_and_logging(self):
        cb = self._patched_monitor(log_program_plot=False, log_program_model=False)
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = _FakeRun()
            cb.on_train_begin()
            cb.on_epoch_begin(0)
            cb.on_epoch_end(0, logs={"loss": 0.5})
            cb.on_train_end(logs={"loss": 0.4})
        # set_experiment from _setup_mlflow + start_run for training
        mlf.set_experiment.assert_called_once()
        mlf.start_run.assert_called_once()
        # epoch end + fit duration at train end = 2 metrics calls
        self.assertEqual(mlf.log_metrics.call_count, 2)
        epoch_metrics = mlf.log_metrics.call_args_list[0].args[0]
        self.assertEqual(epoch_metrics["loss"], 0.5)
        self.assertIn("epoch_duration_s", epoch_metrics)
        self.assertEqual(epoch_metrics["total_cost"], 0)
        self.assertEqual(mlf.log_metrics.call_args_list[0].kwargs["step"], 0)
        self.assertIn("fit_duration_s", mlf.log_metrics.call_args_list[1].args[0])
        mlf.end_run.assert_called_once()
        self.assertFalse(cb._in_training)

    def test_epoch_end_skipped_when_log_epoch_metrics_false(self):
        cb = self._patched_monitor(
            log_epoch_metrics=False,
            log_program_plot=False,
            log_program_model=False,
        )
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = _FakeRun()
            cb.on_train_begin()
            mlf.log_metrics.reset_mock()
            cb.on_epoch_end(0, logs={"loss": 0.1})
        mlf.log_metrics.assert_not_called()

    def test_batch_metrics_logged_when_enabled(self):
        cb = self._patched_monitor(
            log_batch_metrics=True,
            log_program_plot=False,
            log_program_model=False,
        )
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = _FakeRun()
            cb.on_train_begin()
            mlf.log_metrics.reset_mock()
            cb.on_train_batch_end(0, logs={"loss": 0.5})
            cb.on_train_batch_end(1, logs={"loss": 0.4})
            cb.on_test_batch_end(0, logs={"loss": 0.3})
        self.assertEqual(mlf.log_metrics.call_count, 3)
        self.assertEqual(cb._steps, {"train": 2, "val": 1, "test": 0})
        logged = [(c.args[0], c.kwargs["step"]) for c in mlf.log_metrics.call_args_list]
        self.assertEqual(
            logged,
            [
                ({"train_batch_loss": 0.5}, 0),
                ({"train_batch_loss": 0.4}, 1),
                ({"val_batch_loss": 0.3}, 0),
            ],
        )

    def test_batch_metrics_disabled_by_default(self):
        cb = self._patched_monitor(log_program_plot=False, log_program_model=False)
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = _FakeRun()
            cb.on_train_begin()
            mlf.log_metrics.reset_mock()
            cb.on_train_batch_end(0, logs={"loss": 0.5})
            cb.on_test_batch_end(0, logs={"loss": 0.5})
        mlf.log_metrics.assert_not_called()

    def test_test_lifecycle_outside_training_starts_and_ends_run(self):
        cb = self._patched_monitor()
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = _FakeRun()
            cb.on_test_begin()
            cb.on_test_end(logs={"loss": 0.1})
        mlf.start_run.assert_called_once()
        mlf.end_run.assert_called_once()

    def test_test_inside_training_does_not_end_run(self):
        cb = self._patched_monitor(log_program_plot=False, log_program_model=False)
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = _FakeRun()
            cb.on_train_begin()
            mlf.start_run.reset_mock()
            cb.on_test_begin()
            cb.on_test_end(logs={"loss": 0.1})
        mlf.start_run.assert_not_called()
        mlf.end_run.assert_not_called()

    def test_predict_callbacks_are_noops(self):
        cb = Monitor()
        cb.on_predict_begin()
        cb.on_predict_end()
        cb.on_predict_batch_begin(0)
        cb.on_predict_batch_end(0)


class _FakeMetric:
    def __init__(self, step):
        self.step = step


class MonitorResumeTest(testing.TestCase):
    def test_run_id_resumes_run_and_continues_step(self):
        cb = Monitor(run_id="run-abc")
        cb.set_program(_FakeProgram())
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = _FakeRun(run_id="run-abc")
            client = mlf.MlflowClient.return_value
            client.get_run.return_value.data.metrics = {"reward": 0.5, "acc": 0.1}
            client.get_metric_history.side_effect = lambda run_id, key: {
                "reward": [_FakeMetric(0), _FakeMetric(4)],
                "acc": [_FakeMetric(2)],
            }[key]
            cb._start_run(run_name_suffix="test")
        mlf.start_run.assert_called_once_with(run_id="run-abc")
        self.assertEqual(cb._steps["test"], 5)
        self.assertEqual(cb.run_id, "run-abc")

    def test_resume_looks_up_run_by_name(self):
        cb = Monitor(run_name="nightly", resume=True)
        cb.set_program(_FakeProgram())
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.set_experiment.return_value.experiment_id = "exp-1"
            cb._setup_mlflow()
            client = mlf.MlflowClient.return_value
            client.search_runs.return_value = [_FakeRun(run_id="found-1")]
            client.get_run.return_value.data.metrics = {}
            mlf.start_run.return_value = _FakeRun(run_id="found-1")
            cb._start_run(run_name_suffix="test")
        search_kwargs = client.search_runs.call_args.kwargs
        self.assertEqual(search_kwargs["experiment_ids"], ["exp-1"])
        self.assertIn("'nightly_test'", search_kwargs["filter_string"])
        mlf.start_run.assert_called_once_with(run_id="found-1")
        self.assertEqual(cb.run_id, "found-1")

    def test_resume_creates_run_when_none_found(self):
        cb = Monitor(run_name="nightly", resume=True)
        cb.set_program(_FakeProgram())
        with patch.object(monitor_module, "mlflow") as mlf:
            cb._setup_mlflow()
            mlf.MlflowClient.return_value.search_runs.return_value = []
            mlf.start_run.return_value = _FakeRun(run_id="new-1")
            cb._start_run(run_name_suffix="test")
        mlf.start_run.assert_called_once_with(run_name="nightly_test")
        self.assertEqual(cb.run_id, "new-1")
        self.assertEqual(cb._steps["test"], 0)

    def test_standalone_evaluate_tags_run_and_advances_step(self):
        cb = Monitor(run_name="nightly", resume=True)
        cb.set_program(_FakeProgram())
        cb.set_params({})
        with patch.object(monitor_module, "mlflow") as mlf:
            client = mlf.MlflowClient.return_value
            client.search_runs.return_value = []
            client.get_run.return_value.data.metrics = {}
            mlf.start_run.return_value = _FakeRun(run_id="r1")
            cb.on_test_begin()
            cb.on_test_end(logs={"reward": 0.3})
            # second evaluate() in the same process resumes the run found by name
            client.search_runs.return_value = [_FakeRun(run_id="r1")]
            client.get_metric_history.return_value = [_FakeMetric(0)]
            client.get_run.return_value.data.metrics = {"reward": 0.3}
            cb.on_test_begin()
            cb.on_test_end(logs={"reward": 0.6})
        mlf.set_tag.assert_called_with("mlflow.runType", "genai_evaluate")
        steps = [c.kwargs["step"] for c in mlf.log_metrics.call_args_list]
        self.assertEqual(steps, [0, 1])
        self.assertEqual(mlf.end_run.call_count, 2)

    def test_experiment_defaults_to_observability_experiment(self):
        cb = Monitor()
        cb.set_program(_FakeProgram(name="my_prog"))
        with (
            patch.object(monitor_module, "mlflow") as mlf,
            patch.object(monitor_module, "is_observability_enabled", return_value=True),
            patch.object(monitor_module, "mlflow_experiment_name", return_value="obs"),
        ):
            cb._setup_mlflow()
        mlf.set_experiment.assert_called_once_with("obs")


class MonitorThreadSafetyTest(testing.TestCase):
    """`log_metrics` / `log_params` run in a worker thread where MLflow's
    thread-local active run is not visible: without an explicit `run_id`
    MLflow silently starts a stray run and logs there."""

    async def test_log_metrics_passes_run_id(self):
        cb = Monitor()
        cb._run = _FakeRun(run_id="run-xyz")
        with patch.object(monitor_module, "mlflow") as mlf:
            await cb._log_metrics({"reward": 0.5}, step=3)
        mlf.log_metrics.assert_called_once_with({"reward": 0.5}, step=3, run_id="run-xyz")

    async def test_log_params_passes_run_id(self):
        cb = Monitor()
        cb._run = _FakeRun(run_id="run-xyz")
        cb.set_params({"epochs": 2})
        with patch.object(monitor_module, "mlflow") as mlf:
            await cb._log_params()
        self.assertEqual(mlf.log_params.call_args.kwargs["run_id"], "run-xyz")


class MonitorAssessmentsTest(testing.TestCase):
    def _monitor_with_run(self, **kwargs):
        cb = Monitor(**kwargs)
        program = _FakeProgram()
        program._per_sample_rewards = [1.0, 0.0]
        cb.set_program(program)
        cb._run = _FakeRun(run_id="run-1")
        return cb

    def test_batch_rewards_logged_as_feedback_on_traces(self):
        cb = self._monitor_with_run()
        with (
            patch.object(monitor_module, "mlflow") as mlf,
            patch.object(monitor_module.monitor_hook, "root_trace_mark", return_value=7),
            patch.object(
                monitor_module.monitor_hook,
                "root_trace_ids_since",
                return_value=["tr-a", "tr-b"],
            ) as since,
        ):
            cb.on_test_batch_begin(0)
            cb.on_test_batch_end(0, logs={"reward": 0.5})
        since.assert_called_once_with(7)
        calls = sorted(
            (c.kwargs["trace_id"], c.kwargs["value"], c.kwargs["name"])
            for c in mlf.log_feedback.call_args_list
        )
        self.assertEqual(calls, [("tr-a", 1.0, "reward"), ("tr-b", 0.0, "reward")])
        metadata = mlf.log_feedback.call_args.kwargs["metadata"]
        self.assertEqual(metadata["mlflow.assessment.sourceRunId"], "run-1")

    def test_count_mismatch_skips_assessments(self):
        cb = self._monitor_with_run()
        with (
            patch.object(monitor_module, "mlflow") as mlf,
            patch.object(
                monitor_module.monitor_hook, "root_trace_ids_since", return_value=["tr-a"]
            ),
        ):
            cb.on_test_batch_begin(0)
            cb.on_test_batch_end(0, logs={"reward": 0.5})
        mlf.log_feedback.assert_not_called()

    def test_log_assessments_false_disables(self):
        cb = self._monitor_with_run(log_assessments=False)
        with (
            patch.object(monitor_module, "mlflow") as mlf,
            patch.object(
                monitor_module.monitor_hook,
                "root_trace_ids_since",
                return_value=["tr-a", "tr-b"],
            ),
        ):
            cb.on_test_batch_begin(0)
            cb.on_test_batch_end(0, logs={"reward": 0.5})
        mlf.log_feedback.assert_not_called()


class _FakeMonitoredProgram(_FakeProgram):
    """A program exposing language models so spend counters can be summed."""

    def __init__(self, models=(), **kwargs):
        super().__init__(**kwargs)
        self._models = list(models)

    def _flatten_modules(self, include_self=True, recursive=True):
        out = [self] if include_self else []
        out.extend(self._models)
        return out


def _lm(name, cost=0.0, tokens=0):
    from synalinks.src.modules.language_models import LanguageModel

    lm = LanguageModel(model="ollama/mistral", name=name, temperature=0.2)
    lm.cumulated_cost = cost
    lm.cumulated_tokens = tokens
    lm.inference_cumulated_cost = cost
    return lm


class MonitorValidationMetricsTest(testing.TestCase):
    def _monitor(self, program=None, **kwargs):
        cb = Monitor(log_program_plot=False, log_program_model=False, **kwargs)
        cb.set_program(program or _FakeProgram())
        cb.set_params({"epochs": 2})
        return cb

    def test_validation_inside_fit_is_not_logged_by_on_test_end(self):
        cb = self._monitor()
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = _FakeRun()
            cb.on_train_begin()
            mlf.log_metrics.reset_mock()
            cb.on_epoch_begin(0)
            cb.on_test_begin()
            cb.on_test_end(logs={"reward": 0.9})
            cb.on_epoch_end(0, logs={"reward": 0.5, "val_reward": 0.9})
        self.assertEqual(mlf.log_metrics.call_count, 1)
        metrics, kwargs = (
            mlf.log_metrics.call_args.args[0],
            mlf.log_metrics.call_args.kwargs,
        )
        self.assertEqual(metrics["reward"], 0.5)
        self.assertEqual(metrics["val_reward"], 0.9)
        self.assertEqual(kwargs["step"], 0)
        self.assertEqual(cb._steps["test"], 0)

    def test_standalone_evaluate_logs_spend_and_duration(self):
        lm = _lm("lm")
        cb = self._monitor(program=_FakeMonitoredProgram([lm]))
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = _FakeRun()
            cb.on_test_begin()
            lm.cumulated_cost += 0.25
            lm.cumulated_tokens += 300
            cb.on_test_end(logs={"reward": 0.3})
        metrics = mlf.log_metrics.call_args.args[0]
        self.assertEqual(metrics["reward"], 0.3)
        self.assertAlmostEqual(metrics["eval_cost"], 0.25)
        self.assertEqual(metrics["eval_tokens"], 300)
        self.assertIn("eval_duration_s", metrics)
        self.assertEqual(mlf.log_metrics.call_args.kwargs["step"], 0)
        self.assertEqual(cb._steps["test"], 1)

    def test_epoch_spend_deltas(self):
        lm = _lm("lm", cost=10.0, tokens=1000)
        cb = self._monitor(program=_FakeMonitoredProgram([lm]))
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = _FakeRun()
            cb.on_train_begin()
            mlf.log_metrics.reset_mock()
            cb.on_epoch_begin(0)
            lm.cumulated_cost = 15.0
            lm.cumulated_tokens = 1500
            lm.inference_cumulated_cost = 15.0
            cb.on_epoch_end(0, logs={"reward": 0.5})
            cb.on_epoch_begin(1)
            lm.cumulated_cost = 25.0
            lm.cumulated_tokens = 2000
            cb.on_epoch_end(1, logs={"reward": 0.6})
        first, second = (c.args[0] for c in mlf.log_metrics.call_args_list)
        self.assertAlmostEqual(first["epoch_cost"], 5.0)
        self.assertEqual(first["epoch_tokens"], 500)
        self.assertAlmostEqual(first["epoch_cost_inference"], 5.0)
        self.assertAlmostEqual(first["fit_cost"], 5.0)
        self.assertAlmostEqual(first["total_cost"], 15.0)
        self.assertAlmostEqual(second["epoch_cost"], 10.0)
        self.assertAlmostEqual(second["fit_cost"], 15.0)
        self.assertEqual(second["fit_tokens"], 1000)
        self.assertAlmostEqual(second["total_cost"], 25.0)
        self.assertEqual(second["total_tokens"], 2000)


class _FakeReward:
    name = "exact_match"

    def get_config(self):
        return {"name": "exact_match", "reduction": "mean"}


class _FakeOptimizer:
    def get_config(self):
        return {"name": "random_few_shot", "population_size": 4}


class _FakeCompileMetrics:
    def __init__(self):
        self.metrics = [_FakeReward()]


class MonitorParamsAndDatasetTest(testing.TestCase):
    def test_collect_params_covers_compile_program_and_models(self):
        program = _FakeMonitoredProgram([_lm("judge")], name="qa")
        program.reward = _FakeReward()
        program.optimizer = _FakeOptimizer()
        program._compile_metrics = _FakeCompileMetrics()
        cb = Monitor()
        cb.set_program(program)
        cb.set_params({"epochs": 3, "batch_size": 2, "steps": None})
        params = cb._collect_params()
        self.assertEqual(params["epochs"], 3)
        self.assertNotIn("steps", params)
        self.assertEqual(params["program_name"], "qa")
        self.assertEqual(params["num_modules"], 1)
        self.assertEqual(params["reward"], "exact_match")
        self.assertIn('"reduction": "mean"', params["reward_config"])
        self.assertIn('"population_size": 4', params["optimizer_config"])
        self.assertEqual(params["metrics"], "exact_match")
        self.assertEqual(params["lm.judge.model"], "ollama_chat/mistral")
        self.assertEqual(params["lm.judge.temperature"], 0.2)
        self.assertIn("lm.judge.api_base", params)

    def test_collect_params_truncates_long_values(self):
        program = _FakeProgram()
        program.reward = _FakeReward()
        program.reward.get_config = lambda: {"blob": "x" * 10000}
        cb = Monitor()
        cb.set_program(program)
        params = cb._collect_params()
        self.assertEqual(len(params["reward_config"]), 6000)

    def test_train_begin_logs_train_and_validation_datasets(self):
        class _Item:
            def __init__(self, v):
                self.v = v

            def get_json(self):
                return {"v": self.v}

        program = _FakeProgram(name="qa")
        program._fit_inputs = {
            "x": [_Item(1), _Item(2)],
            "y": [_Item("a"), _Item("b")],
            "val_x": [_Item(3)],
            "val_y": None,
        }
        cb = Monitor(log_program_plot=False, log_program_model=False)
        cb.set_program(program)
        cb.set_params({})
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = _FakeRun(run_id="r-ds")
            cb.on_train_begin()
        from_pandas_calls = mlf.data.from_pandas.call_args_list
        self.assertEqual(len(from_pandas_calls), 2)
        train_df = from_pandas_calls[0].args[0]
        self.assertEqual(list(train_df.columns), ["inputs", "expectations"])
        self.assertEqual(train_df["inputs"].tolist(), ['{"v": 1}', '{"v": 2}'])
        self.assertEqual(from_pandas_calls[0].kwargs["name"], "qa_training")
        self.assertEqual(from_pandas_calls[0].kwargs["targets"], "expectations")
        self.assertIsNone(from_pandas_calls[1].kwargs["targets"])
        client = mlf.MlflowClient.return_value
        self.assertEqual(client.log_inputs.call_count, 2)
        self.assertEqual(client.log_inputs.call_args.args[0], "r-ds")

    def test_dataset_logging_failures_are_swallowed(self):
        program = _FakeProgram()
        program._fit_inputs = {"x": [], "y": None, "val_x": None, "val_y": None}
        cb = Monitor(log_program_plot=False, log_program_model=False)
        cb.set_program(program)
        cb.set_params({})
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.start_run.return_value = _FakeRun()
            mlf.data.from_pandas.side_effect = RuntimeError("no pandas")
            cb.on_train_begin()


class _SavingProgram(_FakeProgram):
    """Fake program that can be saved like a real one and exposes schemas."""

    input_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "hint": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["question"],
    }
    output_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.saved_paths = []

    def save(self, filepath, overwrite=True):
        self.saved_paths.append(filepath)
        with open(filepath, "w") as f:
            f.write('{"class_name": "Program"}')


class _Item:
    def __init__(self, **json):
        self._json = json

    def get_json(self):
        return dict(self._json)


class MonitorModelLoggingTest(testing.TestCase):
    def _monitor(self, program=None, **kwargs):
        kwargs.setdefault("log_program_plot", False)
        cb = Monitor(**kwargs)
        program = program or _SavingProgram(name="qa")
        program._fit_inputs = {"x": [_Item(question="q")], "y": None}
        cb.set_program(program)
        cb.set_params({"epochs": 2})
        return cb, program

    def _mlf(self, mlf, model_id="m-1"):
        mlf.start_run.return_value = _FakeRun(run_id="run-train")
        mlf.set_experiment.return_value.experiment_id = "exp-1"
        mlf.initialize_logged_model.return_value.model_id = model_id
        return mlf

    def test_model_logged_on_each_reward_improvement(self):
        cb, program = self._monitor()
        with patch.object(monitor_module, "mlflow") as mlf:
            self._mlf(mlf)
            cb.on_train_begin()
            cb.on_epoch_begin(0)
            cb.on_epoch_end(0, logs={"reward": 0.4, "val_reward": 0.5})
            self.assertEqual(mlf.models.Model.log.call_count, 1)
            cb.on_epoch_begin(1)
            cb.on_epoch_end(1, logs={"reward": 0.6, "val_reward": 0.7})
            self.assertEqual(mlf.models.Model.log.call_count, 2)
            cb.on_train_end(logs={"reward": 0.6, "val_reward": 0.7})
        # No extra model at train end: the last improvement is the final one.
        self.assertEqual(mlf.models.Model.log.call_count, 2)
        self.assertEqual(mlf.initialize_logged_model.call_count, 2)
        init_kwargs = mlf.initialize_logged_model.call_args.kwargs
        self.assertEqual(init_kwargs["source_run_id"], "run-train")
        self.assertEqual(init_kwargs["model_type"], "synalinks_program")
        log_kwargs = mlf.models.Model.log.call_args.kwargs
        self.assertEqual(log_kwargs["run_id"], "run-train")
        self.assertEqual(log_kwargs["model_id"], "m-1")
        self.assertEqual(log_kwargs["step"], 1)
        self.assertEqual(log_kwargs["model_type"], "synalinks_program")
        self.assertEqual(log_kwargs["registered_model_name"], "qa")
        self.assertEqual(log_kwargs["pip_requirements"], [f"synalinks=={__version__}"])
        self.assertEqual(log_kwargs["artifacts"], {"program": program.saved_paths[-1]})
        self.assertIsInstance(log_kwargs["python_model"], SynalinksProgramModel)
        self.assertEqual(log_kwargs["input_example"], {"question": "q"})
        self.assertIsNotNone(log_kwargs["signature"])
        self.assertIsNone(log_kwargs["prompts"])
        mlf.finalize_logged_model.assert_called_with("m-1", "READY")
        self.assertEqual(program._mlflow_model_id, "m-1")
        self.assertEqual(program._mlflow_run_id, "run-train")
        self.assertEqual(program._mlflow_experiment_id, "exp-1")
        mlf.set_active_model.assert_called_once_with(model_id="m-1")
        # The active model is restored when the run ends.
        mlf.set_active_model.return_value.__exit__.assert_called_once()
        mlf.MlflowClient.return_value.set_logged_model_tags.assert_called_once_with(
            "m-1", {"synalinks.best": "true"}
        )

    def test_model_not_logged_when_reward_does_not_improve(self):
        cb, program = self._monitor()
        with patch.object(monitor_module, "mlflow") as mlf:
            self._mlf(mlf)
            mlf.initialize_logged_model.return_value.model_id = "m-a"
            cb.on_train_begin()
            cb.on_epoch_end(0, logs={"reward": 0.4, "val_reward": 0.5})
            cb.on_epoch_end(1, logs={"reward": 0.6, "val_reward": 0.4})
            mlf.initialize_logged_model.return_value.model_id = "m-b"
            cb.on_epoch_end(2, logs={"reward": 0.6, "val_reward": 0.8})
            cb.on_train_end(logs={"reward": 0.6, "val_reward": 0.8})
        steps = [c.kwargs["step"] for c in mlf.models.Model.log.call_args_list]
        self.assertEqual(steps, [0, 2])
        self.assertEqual(
            mlf.MlflowClient.return_value.set_logged_model_tags.call_args.args[0], "m-b"
        )

    def test_model_logging_disabled(self):
        cb, _ = self._monitor(log_program_model=False)
        with patch.object(monitor_module, "mlflow") as mlf:
            self._mlf(mlf)
            cb.on_train_begin()
            cb.on_train_end(logs={"reward": 0.6})
        mlf.initialize_logged_model.assert_not_called()
        mlf.models.Model.log.assert_not_called()

    def test_standalone_evaluate_nests_under_training_run_and_links_model(self):
        cb, program = self._monitor()
        program._mlflow_model_id = "m-9"
        program._mlflow_run_id = "run-train"
        program._mlflow_experiment_id = "exp-1"
        with patch.object(monitor_module, "mlflow") as mlf:
            self._mlf(mlf)
            mlf.start_run.return_value = _FakeRun(run_id="run-eval")
            cb.on_test_begin()
            cb.on_test_end(logs={"reward": 0.3})
        mlf.start_run.assert_called_once_with(
            run_name="qa_test", tags={"mlflow.parentRunId": "run-train"}
        )
        mlf.set_active_model.assert_called_once_with(model_id="m-9")
        self.assertEqual(mlf.log_metrics.call_args.kwargs["model_id"], "m-9")
        mlf.set_active_model.return_value.__exit__.assert_called_once()

    def test_standalone_evaluate_in_other_experiment_is_not_nested(self):
        cb, program = self._monitor()
        program._mlflow_run_id = "run-train"
        program._mlflow_experiment_id = "exp-other"
        with patch.object(monitor_module, "mlflow") as mlf:
            self._mlf(mlf)
            cb.on_test_begin()
            cb.on_test_end(logs={"reward": 0.3})
        mlf.start_run.assert_called_once_with(run_name="qa_test")

    def test_expectations_logged_next_to_rewards(self):
        cb = Monitor()
        program = _FakeProgram()
        program._per_sample_rewards = [1.0]
        program._per_sample_targets = [{"answer": "42"}]
        cb.set_program(program)
        cb._run = _FakeRun(run_id="run-1")
        with (
            patch.object(monitor_module, "mlflow") as mlf,
            patch.object(
                monitor_module.monitor_hook, "root_trace_ids_since", return_value=["tr-a"]
            ),
        ):
            cb.on_test_batch_begin(0)
            cb.on_test_batch_end(0, logs={"reward": 1.0})
        kwargs = mlf.log_expectation.call_args.kwargs
        self.assertEqual(kwargs["trace_id"], "tr-a")
        self.assertEqual(kwargs["name"], "expected_output")
        self.assertEqual(kwargs["value"], {"answer": "42"})
        self.assertEqual(kwargs["metadata"]["mlflow.assessment.sourceRunId"], "run-1")


class PyfuncModelTest(testing.TestCase):
    def test_build_signature_from_json_schemas(self):
        from mlflow.types import DataType
        from mlflow.types.schema import AnyType
        from mlflow.types.schema import Array
        from mlflow.types.schema import Object

        input_schema = {
            "$defs": {
                "Tag": {"type": "object", "properties": {"n": {"type": "integer"}}}
            },
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "hint": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "tags": {"type": "array", "items": {"$ref": "#/$defs/Tag"}},
                "level": {"enum": ["a", "b"]},
                "score": {"type": "number"},
                "ok": {"type": "boolean"},
            },
            "required": ["question", "tags"],
        }
        output_schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        signature = monitor_module.build_signature(input_schema, output_schema)
        cols = {c.name: c for c in signature.inputs.inputs}
        self.assertEqual(cols["question"].type, DataType.string)
        self.assertTrue(cols["question"].required)
        self.assertEqual(cols["hint"].type, DataType.string)
        self.assertFalse(cols["hint"].required)
        self.assertIsInstance(cols["tags"].type, Array)
        self.assertIsInstance(cols["tags"].type.dtype, Object)
        self.assertIsInstance(cols["level"].type, AnyType)
        self.assertEqual(cols["score"].type, DataType.double)
        self.assertEqual(cols["ok"].type, DataType.boolean)
        self.assertEqual(signature.outputs.inputs[0].name, "answer")

    def test_build_signature_falls_back_to_example_then_none(self):
        signature = monitor_module.build_signature(
            {"type": "object"}, None, input_example={"question": "q"}
        )
        self.assertIsNotNone(signature)
        self.assertEqual(signature.inputs.inputs[0].name, "question")
        self.assertIsNone(monitor_module.build_signature(None, None))

    def test_to_records_accepts_dict_list_and_dataframe(self):
        import pandas as pd

        self.assertEqual(monitor_module._to_records({"a": 1}), [{"a": 1}])
        self.assertEqual(
            monitor_module._to_records([{"a": 1}, {"a": 2}]), [{"a": 1}, {"a": 2}]
        )
        df = pd.DataFrame({"a": [1, 2]})
        self.assertEqual(monitor_module._to_records(df), [{"a": 1}, {"a": 2}])

    async def test_predict_runs_program_from_inside_a_running_loop(self):
        class _Prog:
            input_schema = {"type": "object", "properties": {"q": {"type": "string"}}}

            async def predict(self, x, verbose=0):
                return [_Item(answer=item.get_json()["q"].upper()) for item in x] + [None]

        model = SynalinksProgramModel()
        model.program = _Prog()
        model._loop = monitor_module._BackgroundLoop()
        outputs = model.predict(None, [{"q": "hi"}])
        self.assertEqual(outputs, [{"answer": "HI"}, None])

    def test_load_context_loads_program(self):
        model = SynalinksProgramModel()
        context = type("Ctx", (), {"artifacts": {"program": "/tmp/p.json"}})()
        with (
            patch.object(monitor_module, "_BackgroundLoop"),
            patch("synalinks.src.programs.Program.load", return_value="loaded") as load,
        ):
            model.load_context(context)
        load.assert_called_once_with("/tmp/p.json")
        self.assertEqual(model.program, "loaded")


class _FakeSystemMessage:
    def __init__(self, content):
        self.content = content


class _FakeGenerator:
    """Mirrors the slice of `Generator` the prompt registry relies on."""

    def __init__(self, name, instructions, use_inputs_schema=False):
        self.name = name
        self.state = {"instructions": instructions, "examples": []}
        self.use_inputs_schema = use_inputs_schema
        self._build_schemas_dict = None
        self.schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        self.language_model = _lm("lm")
        self.temperature = None
        self.max_tokens = 256
        self.top_p = None
        self.top_k = None

    def _render_system_message(self, inputs):
        if self.use_inputs_schema:
            assert inputs is not None and inputs.get_schema()
        return _FakeSystemMessage(
            f"<instructions>{self.state['instructions']}</instructions>"
        )


class MonitorPromptRegistryTest(testing.TestCase):
    def _monitor(self, module, **kwargs):
        program = _FakeMonitoredProgram([module], name="qa prog")
        cb = Monitor(log_program_plot=False, log_program_model=False, **kwargs)
        cb.set_program(program)
        cb.set_params({"epochs": 3, "optimizer": "OMEGA"})
        return cb, program

    def _mlf(self, mlf):
        mlf.start_run.return_value = _FakeRun(run_id="run-train")
        mlf.MlflowClient.return_value.load_prompt.return_value = None
        versions = iter(range(1, 10))

        def register(**kwargs):
            pv = type("PV", (), {})()
            pv.name = kwargs["name"]
            pv.version = next(versions)
            pv.uri = f"prompts:/{pv.name}/{pv.version}"
            pv.template = kwargs["template"]
            return pv

        mlf.genai.register_prompt.side_effect = register
        return mlf

    def test_registers_a_version_per_changed_epoch_with_val_reward(self):
        module = _FakeGenerator("generator", "Answer briefly.")
        cb, program = self._monitor(module)
        with patch.object(monitor_module, "mlflow") as mlf:
            self._mlf(mlf)
            cb.on_train_begin()
            cb.on_epoch_begin(0)
            cb.on_test_begin()
            cb.on_epoch_end(0, logs={"reward": 0.4, "val_reward": 0.5})
            # unchanged prompt: no new version
            cb.on_epoch_begin(1)
            cb.on_test_begin()
            cb.on_epoch_end(1, logs={"reward": 0.5, "val_reward": 0.9})
            # the optimizer rewrote the instructions: new version, lower reward
            module.state["instructions"] = "Answer in detail."
            cb.on_epoch_begin(2)
            cb.on_test_begin()
            cb.on_epoch_end(2, logs={"reward": 0.6, "val_reward": 0.7})
            cb.on_train_end(logs={"reward": 0.6, "val_reward": 0.7})
        calls = mlf.genai.register_prompt.call_args_list
        self.assertEqual(len(calls), 2)
        first = calls[0].kwargs
        self.assertEqual(first["name"], "qa_prog.generator")
        self.assertEqual(first["template"][0]["role"], "system")
        self.assertIn("Answer briefly.", first["template"][0]["content"])
        self.assertEqual(first["template"][1]["content"], "{{ inputs }}")
        self.assertEqual(first["commit_message"], "epoch 0: val_reward=0.5000")
        self.assertEqual(first["tags"]["val_reward"], "0.5")
        self.assertEqual(first["tags"]["optimizer"], "OMEGA")
        self.assertEqual(first["tags"]["run_id"], "run-train")
        self.assertEqual(first["response_format"], module.schema)
        self.assertEqual(first["model_config"]["model_name"], "ollama_chat/mistral")
        self.assertEqual(first["model_config"]["provider"], "ollama_chat")
        self.assertEqual(first["model_config"]["temperature"], 0.2)
        self.assertEqual(first["model_config"]["max_tokens"], 256)
        self.assertIn("Answer in detail.", calls[1].kwargs["template"][0]["content"])
        self.assertEqual(calls[1].kwargs["commit_message"], "epoch 2: val_reward=0.7000")
        client = mlf.MlflowClient.return_value
        self.assertEqual(client.link_prompt_version_to_run.call_count, 2)
        self.assertEqual(client.link_prompt_version_to_run.call_args.args[0], "run-train")
        # best alias: version 1 (val_reward 0.5) vs version 2 (0.7) -> 2
        mlf.genai.set_prompt_alias.assert_called_once_with("qa_prog.generator", "best", 2)
        self.assertEqual(module._mlflow_prompt_version.version, 2)

    def test_reuses_identical_existing_prompt_version(self):
        module = _FakeGenerator("generator", "Answer briefly.")
        cb, program = self._monitor(module)
        with patch.object(monitor_module, "mlflow") as mlf:
            self._mlf(mlf)
            existing = type("PV", (), {})()
            existing.name = "qa_prog.generator"
            existing.version = 7
            existing.uri = "prompts:/qa_prog.generator/7"
            existing.template = monitor_module._render_prompt_messages(module)
            mlf.MlflowClient.return_value.load_prompt.return_value = existing
            cb.on_train_begin()
            cb.on_epoch_end(0, logs={"reward": 0.4, "val_reward": 0.5})
        mlf.genai.register_prompt.assert_not_called()
        mlf.MlflowClient.return_value.link_prompt_version_to_run.assert_called_once_with(
            "run-train", existing
        )
        self.assertIs(module._mlflow_prompt_version, existing)

    def test_prompt_versions_linked_to_logged_model(self):
        module = _FakeGenerator("generator", "Answer briefly.")
        program = _FakeMonitoredProgram([module], name="qa")
        program.save = lambda path, overwrite=True: open(path, "w").write("{}")
        cb = Monitor(log_program_plot=False)
        cb.set_program(program)
        cb.set_params({"epochs": 1})
        with patch.object(monitor_module, "mlflow") as mlf:
            self._mlf(mlf)
            mlf.initialize_logged_model.return_value.model_id = "m-1"
            cb.on_train_begin()
            cb.on_epoch_end(0, logs={"reward": 0.4, "val_reward": 0.5})
            cb.on_train_end(logs={"reward": 0.4, "val_reward": 0.5})
        self.assertEqual(
            mlf.models.Model.log.call_args.kwargs["prompts"], ["prompts:/qa.generator/1"]
        )
        # Recorded on the program before it was saved, so `Program.load()`
        # re-attaches the version to the module.
        self.assertEqual(
            program._mlflow_prompts,
            {
                "generator": {
                    "name": "qa.generator",
                    "version": 1,
                    "uri": "prompts:/qa.generator/1",
                }
            },
        )

    def test_inputs_schema_module_without_build_schema_is_skipped(self):
        module = _FakeGenerator("generator", "Answer.", use_inputs_schema=True)
        self.assertIsNone(monitor_module._render_prompt_messages(module))
        module._build_schemas_dict = {"inputs": {"type": "object", "properties": {}}}
        messages = monitor_module._render_prompt_messages(module)
        self.assertEqual(messages[0]["role"], "system")

    def test_prompt_name_sanitized(self):
        self.assertEqual(monitor_module._prompt_name("my prog", "gen/1"), "my_prog.gen_1")
        self.assertEqual(monitor_module._prompt_name(None, "gen"), "program.gen")
