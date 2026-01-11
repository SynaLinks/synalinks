# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import logging
import os
import tempfile

from synalinks.src.api_export import synalinks_export
from synalinks.src.callbacks.callback import Callback
from synalinks.src.utils.async_utils import run_maybe_nested

try:
    import mlflow
    import mlflow.pyfunc

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


@synalinks_export("synalinks.callbacks.Monitor")
class Monitor(Callback):
    """Monitor callback for logging training metrics to MLflow.

    This callback logs training progress and evaluation metrics to MLflow
    for experiment tracking and visualization.

    Args:
        experiment_name (str): Name of the MLflow experiment. If None, uses
            the program name.
        run_name (str): Name of the MLflow run. If None, auto-generated.
        tracking_uri (str): MLflow tracking server URI. If None, uses the
            default (local ./mlruns directory or MLFLOW_TRACKING_URI env var).
        log_batch_metrics (bool): Whether to log metrics at batch level
            (default: False).
        log_epoch_metrics (bool): Whether to log metrics at epoch level
            (default: True).
        log_program_plot (bool): Whether to log the program plot as an artifact
            at the beginning of training (default: True).
        log_program_model (bool): Whether to log the program as an MLflow model
            at the end of training (default: True).
        tags (dict): Optional tags to add to the MLflow run.

    Example:

    ```python
    import synalinks

    # Basic usage - uses local MLflow storage
    monitor = synalinks.callbacks.Monitor(experiment_name="my_experiment")

    # With custom MLflow tracking server
    monitor = synalinks.callbacks.Monitor(
        tracking_uri="http://localhost:5000",
        experiment_name="my_experiment",
        run_name="training_run_1",
        log_program_plot=True,
        log_program_model=True,
        tags={"model_type": "chain_of_thought"}
    )

    # Use in training
    program.fit(
        x=train_data,
        y=train_labels,
        epochs=10,
        callbacks=[monitor]
    )
    ```

    Note:
        For tracing module calls along with training metrics, use
        `synalinks.enable_observability()` at the beggining of your script
        which configures the Monitor hook & callback:

        ```python
        synalinks.enable_observability(
            tracking_uri="http://localhost:5000",
            experiment_name="my_traces"
        )
        ```
    """

    def __init__(
        self,
        experiment_name=None,
        run_name=None,
        tracking_uri=None,
        log_batch_metrics=False,
        log_epoch_metrics=True,
        log_program_plot=True,
        log_program_model=True,
        tags=None,
    ):
        super().__init__()
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "mlflow is required for the Monitor callback. "
                "Install it with: pip install mlflow"
            )

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.log_batch_metrics = log_batch_metrics
        self.log_epoch_metrics = log_epoch_metrics
        self.log_program_plot = log_program_plot
        self.log_program_model = log_program_model
        self.tags = tags or {}
        self.logger = logging.getLogger(__name__)

        self._run = None
        self._step = 0
        self._epoch = 0
        # Track if we're inside fit() to avoid ending run during validation
        self._in_training = False

    def _setup_mlflow(self):
        """Configure MLflow tracking."""
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        experiment_name = self.experiment_name
        if experiment_name is None and self.program is not None:
            experiment_name = self.program.name or "synalinks_experiment"

        mlflow.set_experiment(experiment_name)

    def _start_run(self, run_name_suffix=""):
        """Start a new MLflow run."""
        run_name = self.run_name
        if run_name and run_name_suffix:
            run_name = f"{run_name}_{run_name_suffix}"
        elif run_name_suffix:
            run_name = run_name_suffix

        self._run = mlflow.start_run(run_name=run_name)

        tags = dict(self.tags)
        if self.program is not None:
            if self.program.name:
                tags["program_name"] = self.program.name
            if self.program.description:
                tags["program_description"] = self.program.description

        if tags:
            mlflow.set_tags(tags)

        self._step = 0
        self._epoch = 0

    def _end_run(self):
        """End the current MLflow run."""
        if self._run is not None:
            mlflow.end_run()
            self._run = None

    async def _log_metrics(self, logs, step=None):
        """Log metrics to MLflow asynchronously."""
        if logs is None or self._run is None:
            return

        metrics = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                metrics[key] = value

        if metrics:
            await asyncio.to_thread(mlflow.log_metrics, metrics, step=step)

    async def _upload_artifact_via_http(self, local_path, artifact_path, run_id):
        """Upload artifact via HTTP to MLflow server asynchronously.

        This method uses the MLflow REST API to upload artifacts directly,
        bypassing local filesystem artifact repo issues. Requires the MLflow
        server to be started with --serve-artifacts flag.
        """
        import requests

        if not self.tracking_uri:
            raise ValueError("tracking_uri is required for HTTP artifact upload")

        # Get the run's artifact URI to determine the correct upload path
        client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        run = await asyncio.to_thread(client.get_run, run_id)
        artifact_uri = run.info.artifact_uri

        filename = os.path.basename(local_path)
        if artifact_path:
            full_artifact_path = f"{artifact_path}/{filename}"
        else:
            full_artifact_path = filename

        # Parse the artifact URI to construct the correct upload URL
        # artifact_uri can be:
        #   - mlflow-artifacts:/<experiment_id>/<run_id>/artifacts
        #   - mlflow-artifacts://host:port/<experiment_id>/<run_id>/artifacts
        #   - /mlflow/artifacts/<experiment_id>/<run_id>/artifacts (server local path)
        if artifact_uri.startswith("mlflow-artifacts:"):
            # Extract the path part after the scheme
            uri_path = artifact_uri.replace("mlflow-artifacts://", "").replace(
                "mlflow-artifacts:/", ""
            )
            # Remove host:port if present (will use tracking_uri instead)
            if "/" in uri_path and not uri_path.startswith("/"):
                parts = uri_path.split("/", 1)
                if ":" in parts[0] or "." in parts[0]:
                    # First part looks like host:port, skip it
                    uri_path = parts[1] if len(parts) > 1 else ""
        elif artifact_uri.startswith("/"):
            # Server-side local path like /mlflow/artifacts/<exp_id>/<run_id>/artifacts
            # Extract the relative path: <exp_id>/<run_id>/artifacts
            # Find the pattern after the base artifacts directory
            parts = artifact_uri.split("/")
            # Look for 'artifacts' in the path and take everything after the first one
            try:
                artifacts_idx = parts.index("artifacts")
                uri_path = "/".join(parts[artifacts_idx + 1 :])
            except ValueError:
                # Fallback: use experiment_id/run_id/artifacts pattern
                uri_path = f"0/{run_id}/artifacts"
        else:
            # Fallback for other URI schemes
            uri_path = f"0/{run_id}/artifacts"

        # Construct the full upload URL
        base = f"{self.tracking_uri}/api/2.0/mlflow-artifacts/artifacts"
        url = f"{base}/{uri_path}/{full_artifact_path}"

        with open(local_path, "rb") as f:
            content = f.read()

        # Determine content type based on file extension
        content_type = "application/octet-stream"
        if local_path.endswith(".png"):
            content_type = "image/png"
        elif local_path.endswith(".json"):
            content_type = "application/json"

        headers = {"Content-Type": content_type}
        response = await asyncio.to_thread(
            requests.put, url, data=content, headers=headers
        )

        if response.status_code not in (200, 201, 204):
            raise Exception(
                f"Failed to upload artifact: {response.status_code} {response.text}"
            )

    async def _log_program_plot_artifact(self):
        """Log the program plot as an MLflow artifact asynchronously."""
        if self._run is None:
            self.logger.warning("No MLflow run active, skipping plot logging")
            return

        if self.program is None:
            self.logger.warning("No program set, skipping plot logging")
            return

        if not self.program.built:
            self.logger.warning("Program not built, skipping plot logging")
            return

        try:
            from synalinks.src.utils.program_visualization import check_graphviz
            from synalinks.src.utils.program_visualization import check_pydot
            from synalinks.src.utils.program_visualization import plot_program

            if not check_pydot() or not check_graphviz():
                self.logger.warning(
                    "pydot or graphviz not available, skipping program plot"
                )
                return

            run_id = self._run.info.run_id

            with tempfile.TemporaryDirectory() as tmpdir:
                plot_filename = f"{self.program.name or 'program'}.png"
                plot_path = os.path.join(tmpdir, plot_filename)

                # Run plot generation in thread pool
                await asyncio.to_thread(
                    plot_program,
                    self.program,
                    to_file=plot_filename,
                    to_folder=tmpdir,
                    show_schemas=True,
                    show_module_names=True,
                    show_trainable=True,
                    dpi=96,  # Lower DPI for smaller file size
                )

                if os.path.exists(plot_path):
                    # Use HTTP upload if tracking_uri is set (remote server),
                    # otherwise fall back to direct artifact logging (local)
                    if self.tracking_uri:
                        await self._upload_artifact_via_http(
                            plot_path, artifact_path="program_plots", run_id=run_id
                        )
                    else:
                        await asyncio.to_thread(
                            mlflow.log_artifact,
                            plot_path,
                            artifact_path="program_plots",
                            run_id=run_id,
                        )
                    self.logger.info(f"Logged program plot: {plot_filename}")
                else:
                    self.logger.warning(f"Plot file not created: {plot_path}")

        except Exception as e:
            self.logger.warning(f"Failed to log program plot: {e}")

    async def _log_params(self):
        """Log training hyperparameters to MLflow asynchronously."""
        if self._run is None or self.params is None:
            return

        try:
            params_to_log = {}
            for key, value in self.params.items():
                if isinstance(value, (str, int, float, bool)):
                    params_to_log[key] = value

            if params_to_log:
                await asyncio.to_thread(mlflow.log_params, params_to_log)
                self.logger.debug(f"Logged params: {params_to_log}")
        except Exception as e:
            self.logger.warning(f"Failed to log params: {e}")

    async def _log_program_model(self):
        """Log the program trainable state as an MLflow artifact asynchronously.

        This saves only the trainable variables (state), not the full
        program architecture. This is useful for checkpointing the learned
        parameters like few-shot examples, optimized prompts, etc.
        """
        if self._run is None or self.program is None:
            self.logger.warning("No run or program, skipping model logging")
            return

        try:
            import orjson

            # Get the state tree (trainable, non-trainable, optimizer variables)
            state_tree = self.program.get_state_tree()

            # Create model info
            model_info = {
                "program_name": self.program.name or "program",
                "program_description": self.program.description or "",
                "framework": "synalinks",
                "num_trainable_variables": len(self.program.trainable_variables),
            }

            run_id = self._run.info.run_id

            # Write to temp files and log as artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save state tree
                state_path = os.path.join(tmpdir, "state_tree.json")
                with open(state_path, "wb") as f:
                    f.write(orjson.dumps(state_tree, option=orjson.OPT_INDENT_2))

                # Save model info
                info_path = os.path.join(tmpdir, "model_info.json")
                with open(info_path, "wb") as f:
                    f.write(orjson.dumps(model_info, option=orjson.OPT_INDENT_2))

                # Upload artifacts
                if self.tracking_uri:
                    await self._upload_artifact_via_http(
                        state_path, artifact_path="model", run_id=run_id
                    )
                    await self._upload_artifact_via_http(
                        info_path, artifact_path="model", run_id=run_id
                    )
                else:
                    await asyncio.to_thread(
                        mlflow.log_artifact,
                        state_path,
                        artifact_path="model",
                        run_id=run_id,
                    )
                    await asyncio.to_thread(
                        mlflow.log_artifact,
                        info_path,
                        artifact_path="model",
                        run_id=run_id,
                    )

            self.logger.info(
                f"Logged program state: {self.program.name} "
                f"({len(self.program.trainable_variables)} trainable variables)"
            )

        except Exception as e:
            self.logger.warning(f"Failed to log program model: {e}")

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self._in_training = True
        self._setup_mlflow()
        self._start_run(run_name_suffix="train")
        self.logger.debug("MLflow run started for training")

        # Log hyperparameters
        run_maybe_nested(self._log_params())

        # Log program plot
        if self.log_program_plot:
            run_maybe_nested(self._log_program_plot_artifact())

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        run_maybe_nested(self._log_metrics(logs, step=self._step))

        # Log program as model at end of training
        if self.log_program_model:
            run_maybe_nested(self._log_program_model())

        self._end_run()
        self._in_training = False
        self.logger.debug("MLflow run ended for training")

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch."""
        self._epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        if not self.log_epoch_metrics:
            return

        self._epoch = epoch
        run_maybe_nested(self._log_metrics(logs, step=epoch))
        self.logger.debug(f"Logged metrics for epoch {epoch}")

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch."""
        pass

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch."""
        if not self.log_batch_metrics:
            return

        self._step += 1
        run_maybe_nested(self._log_metrics(logs, step=self._step))

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation."""
        # Only start a new run if we're not already in a training run
        if self._run is None and not self._in_training:
            self._setup_mlflow()
            self._start_run(run_name_suffix="test")
            self.logger.debug("MLflow run started for testing")

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation."""
        run_maybe_nested(self._log_metrics(logs, step=self._step))
        # Only end the run if we're not in training (standalone evaluate() call)
        if self._run is not None and not self._in_training:
            self._end_run()
            self.logger.debug("MLflow run ended for testing")

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a test batch."""
        pass

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a test batch."""
        if not self.log_batch_metrics:
            return

        self._step += 1
        run_maybe_nested(self._log_metrics(logs, step=self._step))

    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction."""
        pass

    def on_predict_end(self, logs=None):
        """Called at the end of prediction."""
        pass

    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a prediction batch."""
        pass

    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a prediction batch."""
        pass

    def __del__(self):
        """Cleanup any open MLflow run."""
        if hasattr(self, "_run") and self._run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass
