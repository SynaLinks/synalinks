# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
from unittest.mock import MagicMock
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.callbacks import monitor as monitor_module
from synalinks.src.callbacks.monitor import Monitor


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
        self.assertEqual(cb._step, 0)
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
            await cb._log_metrics(
                {"a": 1.0, "b": 2, "c": "str", "d": None}, step=4
            )
        mlf.log_metrics.assert_called_once_with({"a": 1.0, "b": 2}, step=4)

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
        self.assertEqual(logged, {"epochs": 10, "lr": 0.1, "tag": "x"})

    async def test_log_params_swallows_errors(self):
        cb = Monitor()
        cb._run = _FakeRun()
        cb.set_params({"epochs": 10})
        with patch.object(monitor_module, "mlflow") as mlf:
            mlf.log_params.side_effect = RuntimeError("boom")
            # Should not raise — Monitor logs and continues.
            await cb._log_params()

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
        # epoch end + train end → 2 metrics calls
        self.assertEqual(mlf.log_metrics.call_count, 2)
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
        self.assertEqual(cb._step, 3)

    def test_batch_metrics_disabled_by_default(self):
        cb = self._patched_monitor(
            log_program_plot=False, log_program_model=False
        )
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
        cb = self._patched_monitor(
            log_program_plot=False, log_program_model=False
        )
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


class MonitorArtifactUploadTest(testing.TestCase):
    async def test_upload_requires_tracking_uri(self):
        cb = Monitor()
        with self.assertRaisesRegex(ValueError, "tracking_uri"):
            await cb._upload_artifact_via_http("/tmp/x", "art", "run-1")

    async def test_upload_parses_mlflow_artifacts_uri(self):
        cb = Monitor(tracking_uri="http://server:5000")
        fake_run = _FakeRun(
            run_id="r1",
            artifact_uri="mlflow-artifacts:/0/r1/artifacts",
        )

        tmp = self.get_temp_dir()
        path = f"{tmp}/plot.png"
        with open(path, "wb") as f:
            f.write(b"png")

        async def fake_get_run(run_id):
            return fake_run

        called = {}

        async def fake_put(url, data=None, headers=None):
            called["url"] = url
            called["headers"] = headers
            called["data"] = data
            resp = MagicMock()
            resp.status_code = 200
            return resp

        with patch.object(monitor_module, "mlflow") as mlf:
            client = MagicMock()
            client.get_run.return_value = fake_run
            mlf.MlflowClient.return_value = client
            with patch("requests.put") as mock_put:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_put.return_value = mock_resp
                await cb._upload_artifact_via_http(
                    path, artifact_path="plots", run_id="r1"
                )
        called_url = mock_put.call_args[0][0]
        # The uri path 0/r1/artifacts must be embedded; tracking_uri prefix used.
        self.assertIn("http://server:5000", called_url)
        self.assertIn("0/r1/artifacts/plots/plot.png", called_url)

    async def test_upload_handles_server_local_path(self):
        cb = Monitor(tracking_uri="http://server")
        fake_run = _FakeRun(
            run_id="r1",
            artifact_uri="/mlflow/artifacts/0/r1/artifacts",
        )
        tmp = self.get_temp_dir()
        path = f"{tmp}/state.json"
        with open(path, "wb") as f:
            f.write(b"{}")

        with patch.object(monitor_module, "mlflow") as mlf:
            client = MagicMock()
            client.get_run.return_value = fake_run
            mlf.MlflowClient.return_value = client
            with patch("requests.put") as mock_put:
                mock_resp = MagicMock()
                mock_resp.status_code = 201
                mock_put.return_value = mock_resp
                await cb._upload_artifact_via_http(
                    path, artifact_path=None, run_id="r1"
                )
        url = mock_put.call_args[0][0]
        self.assertIn("0/r1/artifacts/state.json", url)
        # No artifact subpath since artifact_path is None.
        self.assertNotIn("None/", url)

    async def test_upload_raises_on_non_success(self):
        cb = Monitor(tracking_uri="http://server")
        fake_run = _FakeRun(
            run_id="r1", artifact_uri="mlflow-artifacts:/0/r1/artifacts"
        )
        tmp = self.get_temp_dir()
        path = f"{tmp}/x.txt"
        with open(path, "wb") as f:
            f.write(b"x")
        with patch.object(monitor_module, "mlflow") as mlf:
            client = MagicMock()
            client.get_run.return_value = fake_run
            mlf.MlflowClient.return_value = client
            with patch("requests.put") as mock_put:
                mock_resp = MagicMock()
                mock_resp.status_code = 500
                mock_resp.text = "internal error"
                mock_put.return_value = mock_resp
                with self.assertRaisesRegex(Exception, "Failed to upload"):
                    await cb._upload_artifact_via_http(
                        path, artifact_path="x", run_id="r1"
                    )
