# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time

import numpy as np

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend.config import is_observability_enabled
from synalinks.src.backend.config import mlflow_experiment_name
from synalinks.src.backend.config import mlflow_tracking_uri
from synalinks.src.callbacks.callback import Callback
from synalinks.src.hooks import monitor as monitor_hook
from synalinks.src.metrics.em_metrics import _collect_embedding_models
from synalinks.src.metrics.lm_metrics import _collect_language_models
from synalinks.src.utils.async_utils import run_maybe_nested
from synalinks.src.version import __version__

try:
    import mlflow
    import mlflow.genai
    import mlflow.models
    import mlflow.pyfunc

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

_COUNTER_SUFFIXES = (
    "cost",
    "tokens",
    "prompt_tokens",
    "completion_tokens",
    "cached_tokens",
    "calls",
)
_PHASES = ("inference", "reward", "optimizer")
_MAX_PARAM_LENGTH = 6000
_MODEL_TYPE = "synalinks_program"
_JSON_PRIMITIVES = {"string": "string", "integer": "long", "number": "double"}
_PROMPT_USER_TURN = "{{ inputs }}"


def _prompt_name(program_name, module_name):
    """Registry name of a module's prompt: `<program>.<module>`, sanitized."""
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{program_name or 'program'}.{module_name}")


def _render_prompt_messages(module):
    """Chat template of a Generator-like module's current trainable state.

    The system turn is the module's rendered prompt (instructions and
    few-shot examples from its `state` variable); the user turn carries the
    `{{ inputs }}` template variable. `None` when the module renders its
    inputs schema and no build schema is known.
    """
    inputs = None
    if getattr(module, "use_inputs_schema", False):
        schemas = getattr(module, "_build_schemas_dict", None) or {}
        schema = next(iter(schemas.values()), None)
        if schema is None:
            return None
        inputs = SymbolicDataModel(schema=schema)
    system_message = module._render_system_message(inputs)
    return [
        {"role": "system", "content": system_message.content},
        {"role": "user", "content": _PROMPT_USER_TURN},
    ]


def _prompt_model_config(module):
    """`PromptModelConfig` fields of the module's language model settings."""
    language_model = getattr(module, "language_model", None)
    model = getattr(language_model, "model", None)
    if not model:
        return None
    defaults = getattr(language_model, "default_kwargs", None) or {}
    config = {"provider": model.split("/")[0], "model_name": model}
    for key in ("temperature", "max_tokens", "top_p", "top_k"):
        value = getattr(module, key, None)
        if value is None:
            value = defaults.get(key)
        if value is not None:
            config[key] = value
    return config


class _BackgroundLoop:
    """One daemon thread running its own event loop.

    `mlflow.pyfunc.PythonModel.predict` is synchronous and may be called
    with or without a running event loop (a notebook, `mlflow models serve`,
    a Spark UDF). Running the async program on a dedicated loop works in
    both cases and keeps litellm's per-loop HTTP clients alive between
    calls, which `asyncio.run()` per call would not.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=self._loop.run_forever, name="synalinks-pyfunc", daemon=True
        )
        thread.start()

    def run(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


@synalinks_export("synalinks.callbacks.SynalinksProgramModel")
class SynalinksProgramModel(mlflow.pyfunc.PythonModel if MLFLOW_AVAILABLE else object):
    """MLflow pyfunc wrapper around a saved Synalinks program.

    Logged by `synalinks.callbacks.Monitor` at the end of training, loadable
    with `mlflow.pyfunc.load_model(...)` and servable with `mlflow models
    serve`. The `program` artifact is the `program.json` written by
    `Program.save()`; inputs are the program's input JSON (one dict, a list
    of dicts or a DataFrame with one column per field) and outputs are the
    program's output JSON dicts (`None` for a failed sample).

    Custom `DataModel`, `Module` or `Program` subclasses are resolved by
    `Program.load()` from their import path, so the code declaring them only
    needs to be importable where the model is loaded.
    """

    # The input and output are validated against the model signature that
    # `Monitor` logs, derived from the program's own JSON schemas: a program's
    # inputs are arbitrary JSON, which no static type hint on `predict` can
    # describe (and a hint would reject the dict and DataFrame inputs). This
    # tells MLflow not to expect type-hint based validation.
    _skip_type_hint_validation = True

    def __init__(self):
        self.program = None
        self._loop = None

    def load_context(self, context):
        from synalinks.src.programs import Program

        self.program = Program.load(context.artifacts["program"])
        self._loop = _BackgroundLoop()

    def predict(self, context, model_input, params=None):
        records = _to_records(model_input)
        input_schema = getattr(self.program, "input_schema", None)
        x = np.array(
            [JsonDataModel(json=record, schema=input_schema) for record in records],
            dtype="object",
        )
        outputs = self._loop.run(self.program.predict(x, verbose=0))
        return [o.get_json() if o is not None else None for o in outputs]


def _to_records(model_input):
    """Normalize a pyfunc input into a list of JSON dicts."""
    if hasattr(model_input, "to_dict"):
        return model_input.to_dict(orient="records")
    if isinstance(model_input, dict):
        return [model_input]
    return list(model_input)


def _json_schema_to_mlflow_type(schema, defs):
    """Map a JSON schema node to an MLflow type, `AnyType` when unsure."""
    from mlflow.types import DataType
    from mlflow.types.schema import AnyType
    from mlflow.types.schema import Array
    from mlflow.types.schema import Map
    from mlflow.types.schema import Object
    from mlflow.types.schema import Property

    if "$ref" in schema:
        schema = defs.get(schema["$ref"].rsplit("/", 1)[-1], {})
    options = schema.get("anyOf") or schema.get("oneOf")
    if options:
        options = [o for o in options if o.get("type") != "null"]
        if len(options) != 1:
            return AnyType()
        return _json_schema_to_mlflow_type(options[0], defs)
    json_type = schema.get("type")
    if isinstance(json_type, list):
        json_type = [t for t in json_type if t != "null"]
        json_type = json_type[0] if len(json_type) == 1 else None
    if json_type == "boolean":
        return DataType.boolean
    if json_type in _JSON_PRIMITIVES:
        return getattr(DataType, _JSON_PRIMITIVES[json_type])
    if json_type == "array":
        return Array(_json_schema_to_mlflow_type(schema.get("items") or {}, defs))
    if json_type == "object":
        properties = schema.get("properties") or {}
        if properties:
            required = set(schema.get("required") or [])
            return Object(
                [
                    Property(
                        name,
                        _json_schema_to_mlflow_type(prop, defs),
                        required=name in required,
                    )
                    for name, prop in properties.items()
                ]
            )
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            return Map(_json_schema_to_mlflow_type(additional, defs))
    return AnyType()


def _json_schema_to_mlflow_schema(schema):
    """One `ColSpec` per top-level property of a JSON object schema."""
    from mlflow.types import ColSpec
    from mlflow.types import Schema

    defs = schema.get("$defs") or {}
    required = set(schema.get("required") or [])
    properties = schema.get("properties") or {}
    if not properties:
        return None
    return Schema(
        [
            ColSpec(
                _json_schema_to_mlflow_type(prop, defs),
                name=name,
                required=name in required,
            )
            for name, prop in properties.items()
        ]
    )


def build_signature(input_schema, output_schema, input_example=None, output_example=None):
    """Build the MLflow model signature of a program.

    Derived from the program's input and output JSON schemas; when that
    fails, inferred from the examples; `None` when neither is possible.
    """
    from mlflow.models import ModelSignature
    from mlflow.models import infer_signature

    try:
        inputs = _json_schema_to_mlflow_schema(input_schema or {})
        outputs = _json_schema_to_mlflow_schema(output_schema or {})
        if inputs is not None:
            return ModelSignature(inputs=inputs, outputs=outputs)
    except Exception:
        pass
    try:
        if input_example is not None:
            return infer_signature(input_example, output_example)
    except Exception:
        pass
    return None


@synalinks_export("synalinks.callbacks.Monitor")
class Monitor(Callback):
    """Monitor callback for logging training metrics to MLflow.

    This callback logs training progress and evaluation metrics to MLflow
    for experiment tracking and visualization.

    What gets logged on a `fit()` run:

    - **Metrics** per epoch (`step=epoch`): every training metric, the
      `val_*` validation metrics, the LM/EM spend of the epoch (`epoch_cost`,
      `epoch_tokens`, per phase `epoch_cost_inference|reward|optimizer`), the
      running spend of the fit (`fit_cost`, `fit_tokens`), the all-time
      in-process spend (`total_cost`, `total_tokens`) and `epoch_duration_s`;
      `fit_duration_s` at the end. With `log_batch_metrics=True`, batch
      metrics are logged too, prefixed `train_batch_` / `val_batch_` with
      their own step counters.
    - **Params**: the fit arguments (epochs, batch/minibatch size,
      validation split/freq, dataset sizes), the compile configuration
      (reward, metrics, optimizer and its config), the program (name, class,
      number of modules and trainable variables) and every language or
      embedding model reachable from the program (`lm.<name>.model`, api
      base, sampling settings).
    - **Datasets**: the train and validation sets as MLflow dataset inputs,
      one row per sample with JSON `inputs` and `expectations` columns.
    - **Model**: the program (architecture and trained variables, the JSON
      of `Program.save()`) as an MLflow pyfunc model wrapped in
      `SynalinksProgramModel`, so it can be loaded with
      `mlflow.pyfunc.load_model()` and served with `mlflow models serve`. A
      new version is logged every time `val_reward` improves (`reward` when
      there is no validation data, the end of training when neither is
      available), registered in the Model Registry under the program's name;
      the best one is tagged `synalinks.best`. Each is an MLflow LoggedModel
      linked to the run; its id is written into the saved program
      (`program._mlflow_model_id`) so later `evaluate()` runs of the same
      program link their traces and metrics to that version and nest under
      the training run.
    - **Prompts**: each trainable `Generator`-like module's prompt (the
      system turn rendered from its optimized instructions and examples, plus
      a `{{ inputs }}` user turn, the module's output schema as
      `response_format`, and its LM settings as `model_config`) is registered
      in the MLflow Prompt Registry under
      `<program>.<module>`. A new version is created at every epoch where the
      rendered prompt changed, its commit message and `val_reward` tag
      recording that epoch's validation reward; the best version gets the
      `best` alias at the end. Versions are linked to the run, to the logged
      model, and to the traces of the module's later calls.
    - **Artifacts**: the program plot at the beginning of training.

    A standalone `evaluate()` opens its own run (type `genai_evaluate`), logs
    the evaluation metrics, `eval_cost`, `eval_tokens`, `eval_duration_s`, the
    evaluated dataset, and every sample's reward as a `reward` feedback
    assessment on its trace.

    Args:
        experiment_name (str): Name of the MLflow experiment. If None, uses
            the experiment of `synalinks.enable_observability()` when it was
            called (so runs land next to their traces), else the program name.
        run_name (str): Name of the MLflow run. If None, the program's name,
            suffixed `_train` or `_test`.
        tracking_uri (str): MLflow tracking server URI. If None, uses the
            value from `synalinks.enable_observability()` or the default
            (local ./mlruns directory or MLFLOW_TRACKING_URI env var).
        log_batch_metrics (bool): Whether to log metrics at batch level
            (default: False).
        log_epoch_metrics (bool): Whether to log metrics at epoch level
            (default: True).
        log_program_plot (bool): Whether to log the program plot as an artifact
            at the beginning of training (default: True).
        log_program_model (bool): Whether to log the program as an MLflow model
            (default: True).
        tags (dict): Optional tags to add to the MLflow run.
        run_id (str): Optional. The id of an existing MLflow run to resume
            instead of starting a new one. Metrics keep being appended to it,
            the step counters continuing after the last step already logged.
        resume (bool): Whether to look up an existing run named `run_name`
            in the experiment and resume it (default: False). Creates the
            run the first time. This is how repeated `evaluate()` calls
            draw a chart over time: each evaluation adds one point to the
            metrics of the same run.
        log_assessments (bool): Whether to log each evaluated sample's reward
            as a `reward` feedback assessment on the sample's trace during
            `evaluate()` (default: True). Requires the traces of
            `synalinks.enable_observability()`; the assessments show up on
            the traces of the evaluation run in the MLflow UI.

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

    # Track evaluation results over time: every evaluate() call (in this
    # process or a later one) appends one point to the same run's charts
    monitor = synalinks.callbacks.Monitor(
        experiment_name="my_experiment",
        run_name="nightly_eval",
        resume=True,
    )
    program.evaluate(x=test_data, y=test_labels, callbacks=[monitor])
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
        run_id=None,
        resume=False,
        log_assessments=True,
    ):
        super().__init__()
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "mlflow is required for the Monitor callback. "
                "Install it with: pip install mlflow"
            )

        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tracking_uri = tracking_uri or mlflow_tracking_uri()
        self.log_batch_metrics = log_batch_metrics
        self.log_epoch_metrics = log_epoch_metrics
        self.log_program_plot = log_program_plot
        self.log_program_model = log_program_model
        self.tags = tags or {}
        self.run_id = run_id
        self.resume = resume
        self.log_assessments = log_assessments
        self.logger = logging.getLogger(__name__)

        self._run = None
        self._trace_mark = None
        self._steps = {"train": 0, "val": 0, "test": 0}
        self._epoch = 0
        # Track if we're inside fit() to avoid ending run during validation
        self._in_training = False
        self._fit_counters = None
        self._epoch_counters = None
        self._fit_t0 = None
        self._epoch_t0 = None
        self._model_best = -np.inf
        self._logged_models = []
        self._active_model = None
        self._prompt_versions = {}
        self._prompt_hashes = {}
        self._prompt_history = {}
        self._prompt_snapshot = {}

    def _batch_phase(self):
        return "val" if self._in_training else "test"

    def _setup_mlflow(self):
        """Configure MLflow tracking."""
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        experiment_name = self.experiment_name
        if experiment_name is None and is_observability_enabled():
            experiment_name = mlflow_experiment_name()
        if experiment_name is None and self.program is not None:
            experiment_name = self.program.name or "synalinks_experiment"

        self._experiment_id = mlflow.set_experiment(experiment_name).experiment_id

    def _start_run(self, run_name_suffix="", tags=None):
        """Start a new MLflow run, or resume one (`run_id` / `resume`)."""
        run_name = self.run_name
        if run_name is None and self.program is not None and self.program.name:
            run_name = self.program.name
        if run_name and run_name_suffix:
            run_name = f"{run_name}_{run_name_suffix}"
        elif run_name_suffix:
            run_name = run_name_suffix

        run_id = self.run_id
        if run_id is None and self.resume:
            client = mlflow.MlflowClient()
            found = client.search_runs(
                experiment_ids=[self._experiment_id],
                filter_string=f"tags.mlflow.runName = '{run_name}'",
                order_by=["attributes.start_time DESC"],
                max_results=1,
            )
            if found:
                run_id = found[0].info.run_id

        if run_id is not None:
            self._run = mlflow.start_run(run_id=run_id)
            self.run_id = run_id
        elif tags:
            self._run = mlflow.start_run(run_name=run_name, tags=tags)
            self.run_id = self._run.info.run_id
        else:
            self._run = mlflow.start_run(run_name=run_name)
            self.run_id = self._run.info.run_id

        tags = dict(self.tags)
        if self.program is not None:
            if self.program.name:
                tags["program_name"] = self.program.name
            if self.program.description:
                tags["program_description"] = self.program.description

        if tags:
            mlflow.set_tags(tags)

        self._steps = {"train": 0, "val": 0, "test": 0}
        self._epoch = 0
        if run_id is not None:
            client = mlflow.MlflowClient()
            last_step = 0
            for key in client.get_run(run_id).data.metrics:
                history = client.get_metric_history(run_id, key)
                last_step = max(last_step, *(m.step + 1 for m in history))
            self._steps = {phase: last_step for phase in self._steps}

    def _end_run(self):
        """End the current MLflow run."""
        if self._active_model is not None:
            try:
                self._active_model.__exit__(None, None, None)
            except Exception as e:
                self.logger.debug(f"Failed to restore the active model: {e}")
            self._active_model = None
        if self._run is not None:
            mlflow.end_run()
            self._run = None

    def _activate_model(self, model_id):
        """Make `model_id` the active LoggedModel so traces link to it."""
        if model_id is None or self._active_model is not None:
            return
        try:
            self._active_model = mlflow.set_active_model(model_id=model_id)
        except Exception as e:
            self.logger.warning(f"Failed to set the active model {model_id}: {e}")

    async def _log_metrics(self, logs, step=None, prefix="", model_id=None):
        """Log metrics to MLflow asynchronously."""
        if logs is None or self._run is None:
            return

        metrics = {}
        for key, value in logs.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                metrics[prefix + key] = value

        if metrics:
            kwargs = {"step": step, "run_id": self._run.info.run_id}
            if model_id is not None:
                kwargs["model_id"] = model_id
            # Explicit run_id: MLflow's active run is thread-local, not seen by the worker
            await asyncio.to_thread(mlflow.log_metrics, metrics, **kwargs)

    def _models(self):
        if self.program is None:
            return []
        return _collect_language_models(self.program) + _collect_embedding_models(
            self.program
        )

    def _snapshot_counters(self):
        """Sum the all-time and per-phase spend counters of every LM and EM."""
        models = self._models()
        counters = {}
        for suffix in _COUNTER_SUFFIXES:
            counters[suffix] = sum(getattr(m, f"cumulated_{suffix}", 0) for m in models)
            for phase in _PHASES:
                counters[f"{phase}_{suffix}"] = sum(
                    getattr(m, f"{phase}_cumulated_{suffix}", 0) for m in models
                )
        return counters

    def _spend_metrics(self, prefix, baseline, current=None):
        """Cost/token deltas since `baseline`, named `<prefix>_cost` etc."""
        if baseline is None:
            return {}
        current = current or self._snapshot_counters()
        metrics = {
            f"{prefix}_cost": current["cost"] - baseline["cost"],
            f"{prefix}_tokens": current["tokens"] - baseline["tokens"],
            f"{prefix}_calls": current["calls"] - baseline["calls"],
        }
        for phase in _PHASES:
            delta = current[f"{phase}_cost"] - baseline[f"{phase}_cost"]
            if delta:
                metrics[f"{prefix}_cost_{phase}"] = delta
        return metrics

    def _collect_params(self):
        """Fit arguments, compile configuration, program and model settings."""
        params = {}
        for key, value in (self.params or {}).items():
            if isinstance(value, (str, int, float, bool)):
                params[key] = value
        program = self.program
        if program is None:
            return params
        params["program_name"] = program.name or ""
        params["program_class"] = program.__class__.__name__
        trainable_variables = getattr(program, "trainable_variables", None)
        if trainable_variables is not None:
            params["num_trainable_variables"] = len(trainable_variables)
        if hasattr(program, "_flatten_modules"):
            params["num_modules"] = len(
                program._flatten_modules(include_self=False, recursive=True)
            )
        reward = getattr(program, "reward", None)
        if reward is not None:
            params["reward"] = getattr(reward, "name", reward.__class__.__name__)
            if hasattr(reward, "get_config"):
                params["reward_config"] = json.dumps(reward.get_config(), default=str)
        optimizer = getattr(program, "optimizer", None)
        if optimizer is not None and hasattr(optimizer, "get_config"):
            params["optimizer_config"] = json.dumps(optimizer.get_config(), default=str)
        compile_metrics = getattr(program, "_compile_metrics", None)
        metrics = getattr(compile_metrics, "metrics", None)
        if metrics:
            params["metrics"] = ",".join(m.name for m in metrics)
        for model in self._models():
            kind = "em" if "Embedding" in model.__class__.__name__ else "lm"
            key = f"{kind}.{model.name}"
            params[f"{key}.model"] = getattr(model, "model", "")
            api_base = getattr(model, "api_base", None)
            if api_base:
                params[f"{key}.api_base"] = api_base
            for name, value in (getattr(model, "default_kwargs", None) or {}).items():
                if isinstance(value, (str, int, float, bool)):
                    params[f"{key}.{name}"] = value
        for key, value in list(params.items()):
            if isinstance(value, str) and len(value) > _MAX_PARAM_LENGTH:
                params[key] = value[:_MAX_PARAM_LENGTH]
        return params

    async def _log_params(self):
        """Log the run parameters to MLflow asynchronously.

        Params are immutable in MLflow: resuming a run with a changed value
        raises, so on failure fall back to logging key by key and skip the
        conflicting ones.
        """
        if self._run is None:
            return
        run_id = self._run.info.run_id
        params = self._collect_params()
        if not params:
            return
        try:
            await asyncio.to_thread(mlflow.log_params, params, run_id=run_id)
        except Exception as e:
            self.logger.debug(f"Batch log_params failed ({e}), logging key by key")
            client = mlflow.MlflowClient()
            for key, value in params.items():
                try:
                    await asyncio.to_thread(client.log_param, run_id, key, value)
                except Exception as err:
                    self.logger.warning(f"Failed to log param {key}: {err}")

    async def _log_dataset(self, x, y, context):
        """Log `x`/`y` as an MLflow dataset input of the run.

        Rows hold JSON strings (not dicts) so MLflow's pandas digest covers
        the data. Only the schema, digest and size are stored on the run.
        """
        if self._run is None or x is None:
            return
        try:
            import pandas as pd
            from mlflow.entities import DatasetInput
            from mlflow.entities import InputTag

            records = {"inputs": [json.dumps(item.get_json()) for item in x]}
            if y is not None:
                records["expectations"] = [json.dumps(item.get_json()) for item in y]
            df = pd.DataFrame(records)
            name = f"{self.program.name or 'program'}_{context}"
            dataset = mlflow.data.from_pandas(
                df, name=name, targets="expectations" if y is not None else None
            )
            client = mlflow.MlflowClient()
            await asyncio.to_thread(
                client.log_inputs,
                self._run.info.run_id,
                datasets=[
                    DatasetInput(
                        dataset._to_mlflow_entity(),
                        tags=[InputTag("mlflow.data.context", context)],
                    )
                ],
            )
        except Exception as e:
            self.logger.warning(f"Failed to log {context} dataset: {e}")

    async def _log_artifact(self, local_path, artifact_path):
        await asyncio.to_thread(
            mlflow.log_artifact,
            local_path,
            artifact_path=artifact_path,
            run_id=self._run.info.run_id,
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

            with tempfile.TemporaryDirectory() as tmpdir:
                plot_filename = f"{self.program.name or 'program'}.png"
                plot_path = os.path.join(tmpdir, plot_filename)

                await asyncio.to_thread(
                    plot_program,
                    self.program,
                    to_file=plot_filename,
                    to_folder=tmpdir,
                    show_schemas=True,
                    show_module_names=True,
                    show_trainable=True,
                    dpi=96,
                )

                if os.path.exists(plot_path):
                    await self._log_artifact(plot_path, "program_plots")
                    self.logger.info(f"Logged program plot: {plot_filename}")
                else:
                    self.logger.warning(f"Plot file not created: {plot_path}")

        except Exception as e:
            self.logger.warning(f"Failed to log program plot: {e}")

    def _prompt_modules(self):
        if self.program is None or not hasattr(self.program, "_flatten_modules"):
            return []
        return [
            m
            for m in self.program._flatten_modules(include_self=False, recursive=True)
            if hasattr(m, "_render_system_message") and hasattr(m, "state")
        ]

    def _snapshot_prompts(self):
        """Render every trainable prompt: `{name: (hash, messages, module)}`."""
        snapshot = {}
        for module in self._prompt_modules():
            try:
                messages = _render_prompt_messages(module)
            except Exception as e:
                self.logger.warning(f"Failed to render the prompt of {module.name}: {e}")
                continue
            if messages is None:
                continue
            digest = hashlib.sha256(
                json.dumps(messages, sort_keys=True).encode()
            ).hexdigest()
            name = _prompt_name(self.program.name, module.name)
            snapshot[name] = (digest, messages, module)
        return snapshot

    async def _register_prompts(self, epoch, logs):
        """Register a new version of every prompt that changed this epoch."""
        if self._run is None or not self._prompt_snapshot:
            return
        value = self._monitored_value(logs)
        run_id = self._run.info.run_id
        model_id = getattr(self.program, "_mlflow_model_id", None)
        optimizer = (self.params or {}).get("optimizer")
        client = mlflow.MlflowClient()
        for name, (digest, messages, module) in self._prompt_snapshot.items():
            if self._prompt_hashes.get(name) == digest:
                continue
            try:
                prompt_version = None
                if name not in self._prompt_hashes:
                    existing = await asyncio.to_thread(
                        client.load_prompt, name, allow_missing=True
                    )
                    if existing is not None and existing.template == messages:
                        prompt_version = existing
                if prompt_version is None:
                    tags = {
                        "program": self.program.name or "",
                        "module": module.name or "",
                        "epoch": str(epoch),
                        "run_id": run_id,
                    }
                    if value is not None:
                        tags["val_reward"] = str(value)
                    if optimizer:
                        tags["optimizer"] = str(optimizer)
                    commit_message = f"epoch {epoch}"
                    if value is not None:
                        commit_message += f": val_reward={value:.4f}"
                    prompt_version = await asyncio.to_thread(
                        mlflow.genai.register_prompt,
                        name=name,
                        template=messages,
                        commit_message=commit_message,
                        tags=tags,
                        response_format=getattr(module, "schema", None),
                        model_config=_prompt_model_config(module),
                    )
                await asyncio.to_thread(
                    client.link_prompt_version_to_run, run_id, prompt_version
                )
                if model_id:
                    await asyncio.to_thread(
                        client.link_prompt_version_to_model,
                        name,
                        str(prompt_version.version),
                        model_id,
                    )
                self._prompt_hashes[name] = digest
                self._prompt_versions[name] = prompt_version
                self._prompt_history.setdefault(name, []).append(
                    (prompt_version.version, value)
                )
                module._mlflow_prompt_version = prompt_version
            except Exception as e:
                self.logger.warning(f"Failed to register prompt {name}: {e}")

    def _set_prompt_aliases(self):
        """Point the `best` alias of each prompt at its best-scoring version."""
        for name, history in self._prompt_history.items():
            version = history[-1][0]
            scored = [h for h in history if h[1] is not None]
            if scored:
                version = max(scored, key=lambda h: h[1])[0]
            try:
                mlflow.genai.set_prompt_alias(name, "best", version)
            except Exception as e:
                self.logger.debug(f"Failed to set the best alias of {name}: {e}")

    def _monitored_value(self, logs):
        """The epoch's `val_reward`, else `reward`, else `None`."""
        logs = logs or {}
        value = logs.get("val_reward")
        if value is None:
            value = logs.get("reward")
        return value

    def _maybe_log_model(self, epoch, logs):
        """Log a new model version when the reward improved this epoch."""
        if not self.log_program_model:
            return
        value = self._monitored_value(logs)
        if value is None or value <= self._model_best:
            return
        self._model_best = value
        self._log_model(step=epoch, value=value)

    def _input_example(self):
        inputs = getattr(self.program, "_fit_inputs", None) or {}
        x = inputs.get("x")
        if x is None or len(x) == 0:
            return None
        return x[0].get_json()

    def _log_model(self, step, value=None):
        """Log the program as a pyfunc model and a LoggedModel version.

        Two-phase so the saved `program.json` carries its own model id: the
        LoggedModel is created first, its id written on the program, then the
        model files are logged against it. Runs on the callback thread since
        MLflow resolves the active run thread-locally.
        """
        if self._run is None or self.program is None:
            return None

        run_id = self._run.info.run_id
        name = self.program.name or "program"
        tags = {"synalinks.program": name, "synalinks.epoch": str(step)}
        if value is not None:
            tags["synalinks.reward"] = str(value)
        try:
            logged_model = mlflow.initialize_logged_model(
                name=name, source_run_id=run_id, model_type=_MODEL_TYPE, tags=tags
            )
            model_id = logged_model.model_id
            self.program._mlflow_model_id = model_id
            self.program._mlflow_run_id = run_id
            self.program._mlflow_experiment_id = getattr(self, "_experiment_id", None)
            self.program._mlflow_prompts = {
                module.name: {
                    "name": pv.name,
                    "version": pv.version,
                    "uri": getattr(pv, "uri", None),
                }
                for module in self._prompt_modules()
                for pv in [getattr(module, "_mlflow_prompt_version", None)]
                if pv is not None
            }
            input_example = self._input_example()
            signature = build_signature(
                getattr(self.program, "input_schema", None),
                getattr(self.program, "output_schema", None),
                input_example=input_example,
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "program.json")
                self.program.save(path)
                info = mlflow.models.Model.log(
                    artifact_path=None,
                    flavor=mlflow.pyfunc,
                    name=name,
                    run_id=run_id,
                    model_id=model_id,
                    model_type=_MODEL_TYPE,
                    step=step,
                    tags=tags,
                    prompts=[pv.uri for pv in self._prompt_versions.values()] or None,
                    registered_model_name=name,
                    python_model=SynalinksProgramModel(),
                    artifacts={"program": path},
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=[f"synalinks=={__version__}"],
                )
            mlflow.finalize_logged_model(model_id, "READY")
            self._logged_models.append((step, model_id, value))
            self._activate_model(model_id)
            self.logger.info(f"Logged program model {name} (model_id={model_id})")
            return info
        except Exception as e:
            self.logger.warning(f"Failed to log program model: {e}")
            return None

    def _tag_best_model(self):
        if not self._logged_models:
            return
        best = self._logged_models[-1]
        scored = [m for m in self._logged_models if m[2] is not None]
        if scored:
            best = max(scored, key=lambda m: m[2])
        try:
            mlflow.MlflowClient().set_logged_model_tags(
                best[1], {"synalinks.best": "true"}
            )
        except Exception as e:
            self.logger.debug(f"Failed to tag the best model: {e}")

    def on_train_begin(self, logs=None):
        """Called at the beginning of training."""
        self._in_training = True
        self._setup_mlflow()
        self._start_run(run_name_suffix="train")
        self.logger.debug("MLflow run started for training")
        self._fit_t0 = time.perf_counter()
        self._fit_counters = self._snapshot_counters()
        self._logged_models = []
        self._model_best = -np.inf
        self._prompt_versions = {}
        self._prompt_hashes = {}
        self._prompt_history = {}
        self._prompt_snapshot = {}

        run_maybe_nested(self._log_params())

        inputs = getattr(self.program, "_fit_inputs", None) or {}
        run_maybe_nested(self._log_dataset(inputs.get("x"), inputs.get("y"), "training"))
        run_maybe_nested(
            self._log_dataset(inputs.get("val_x"), inputs.get("val_y"), "validation")
        )

        if self.log_program_plot:
            run_maybe_nested(self._log_program_plot_artifact())

    def on_train_end(self, logs=None):
        """Called at the end of training."""
        if self._fit_t0 is not None:
            run_maybe_nested(
                self._log_metrics(
                    {"fit_duration_s": time.perf_counter() - self._fit_t0},
                    step=self._epoch,
                )
            )

        if self.log_program_model and not self._logged_models:
            self._log_model(step=self._epoch, value=self._monitored_value(logs))
        self._tag_best_model()
        self._set_prompt_aliases()

        self._end_run()
        self._in_training = False
        self.logger.debug("MLflow run ended for training")

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch."""
        self._epoch = epoch
        self._epoch_t0 = time.perf_counter()
        self._epoch_counters = self._snapshot_counters()

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch."""
        self._epoch = epoch
        if not self.log_epoch_metrics:
            return

        metrics = dict(logs or {})
        current = self._snapshot_counters()
        metrics.update(self._spend_metrics("epoch", self._epoch_counters, current))
        metrics.update(self._spend_metrics("fit", self._fit_counters, current))
        metrics["total_cost"] = current["cost"]
        metrics["total_tokens"] = current["tokens"]
        if self._epoch_t0 is not None:
            metrics["epoch_duration_s"] = time.perf_counter() - self._epoch_t0
        run_maybe_nested(self._log_metrics(metrics, step=epoch))
        self.logger.debug(f"Logged metrics for epoch {epoch}")
        # Snapshot taken when validation began; without validation the
        # state at epoch end is the one to register.
        if not self._prompt_snapshot:
            self._prompt_snapshot = self._snapshot_prompts()
        run_maybe_nested(self._register_prompts(epoch, logs))
        self._prompt_snapshot = {}
        self._maybe_log_model(epoch, logs)

    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch."""
        pass

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch."""
        if not self.log_batch_metrics:
            return

        step = self._steps["train"]
        self._steps["train"] += 1
        run_maybe_nested(self._log_metrics(logs, step=step, prefix="train_batch_"))

    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation."""
        if self._in_training:
            # The state being validated is the one whose reward we record.
            self._prompt_snapshot = self._snapshot_prompts()
        # Only start a new run if we're not already in a training run
        if self._run is None and not self._in_training:
            self._setup_mlflow()
            tags = None
            parent_run_id = getattr(self.program, "_mlflow_run_id", None)
            same_experiment = (
                getattr(self.program, "_mlflow_experiment_id", None)
                == self._experiment_id
            )
            if parent_run_id and same_experiment and not (self.run_id or self.resume):
                tags = {"mlflow.parentRunId": parent_run_id}
            self._start_run(run_name_suffix="test", tags=tags)
            self._activate_model(getattr(self.program, "_mlflow_model_id", None))
            # Same run type as `mlflow.genai.evaluate()` runs
            mlflow.set_tag("mlflow.runType", "genai_evaluate")
            self.logger.debug("MLflow run started for testing")
            self._eval_t0 = time.perf_counter()
            self._eval_counters = self._snapshot_counters()
            run_maybe_nested(self._log_params())
            inputs = getattr(self.program, "_eval_inputs", None) or {}
            run_maybe_nested(
                self._log_dataset(inputs.get("x"), inputs.get("y"), "evaluation")
            )

    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.

        Inside `fit()` the trainer merges the validation metrics, prefixed
        `val_`, into the epoch logs, so nothing is logged here: logging the
        unprefixed values would overwrite the training metrics.
        """
        if self._in_training or self._run is None:
            return
        metrics = dict(logs or {})
        metrics.update(self._spend_metrics("eval", self._eval_counters))
        if self._eval_t0 is not None:
            metrics["eval_duration_s"] = time.perf_counter() - self._eval_t0
        run_maybe_nested(
            self._log_metrics(
                metrics,
                step=self._steps["test"],
                model_id=getattr(self.program, "_mlflow_model_id", None),
            )
        )
        self._steps["test"] += 1
        self._end_run()
        self.logger.debug("MLflow run ended for testing")

    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a test batch."""
        self._trace_mark = monitor_hook.root_trace_mark()

    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a test batch."""
        if self.log_assessments:
            run_maybe_nested(self._log_batch_assessments())

        if not self.log_batch_metrics:
            return

        phase = self._batch_phase()
        step = self._steps[phase]
        self._steps[phase] += 1
        run_maybe_nested(self._log_metrics(logs, step=step, prefix=f"{phase}_batch_"))

    async def _log_batch_assessments(self):
        """Log the per-sample rewards of the batch just evaluated as `reward`
        feedback assessments on the samples' traces.

        Sample i is matched with the i-th root trace started since
        `on_test_batch_begin`; when the counts differ (tracing disabled, or
        the batch's predictions came from the auto-build pass that ran before
        the run started) nothing is logged.
        """
        if self._run is None or self.program is None:
            return
        rewards = getattr(self.program, "_per_sample_rewards", None)
        trace_ids = monitor_hook.root_trace_ids_since(self._trace_mark)
        if not rewards or len(rewards) != len(trace_ids):
            self.logger.debug(
                "Skipping assessments: %s rewards for %s traces",
                None if rewards is None else len(rewards),
                len(trace_ids),
            )
            return

        reward_fn = getattr(self.program, "_compile_reward", None)
        reward_fn = getattr(reward_fn, "_user_reward", reward_fn)
        source_id = getattr(reward_fn, "name", None) or "reward"
        source = mlflow.entities.AssessmentSource(
            source_type=mlflow.entities.AssessmentSourceType.CODE,
            source_id=source_id,
        )
        run_id = self._run.info.run_id
        targets = getattr(self.program, "_per_sample_targets", None)
        if not targets or len(targets) != len(trace_ids):
            targets = [None] * len(trace_ids)
        expectation_source = mlflow.entities.AssessmentSource(
            source_type=mlflow.entities.AssessmentSourceType.HUMAN,
            source_id="dataset",
        )

        def log_one(trace_id, value, target):
            try:
                mlflow.log_feedback(
                    trace_id=trace_id,
                    name="reward",
                    value=float(value),
                    source=source,
                    metadata={"mlflow.assessment.sourceRunId": run_id},
                )
                if target is not None:
                    mlflow.log_expectation(
                        trace_id=trace_id,
                        name="expected_output",
                        value=target,
                        source=expectation_source,
                        metadata={"mlflow.assessment.sourceRunId": run_id},
                    )
            except Exception as e:
                self.logger.warning(f"Failed to log assessment on {trace_id}: {e}")

        await asyncio.gather(
            *(
                asyncio.to_thread(log_one, trace_id, value, target)
                for trace_id, value, target in zip(trace_ids, rewards, targets)
            )
        )

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
        """End our MLflow run if it was left open and is still the active one.

        Guarded by a run-id check: ``mlflow.end_run()`` always ends whatever run
        is *globally* active, so a finalizer firing at GC time must not end an
        unrelated run (this also keeps a leaked finalizer from polluting other
        code's, or another test's, active run).
        """
        run = getattr(self, "_run", None)
        if run is None:
            return
        try:
            active = mlflow.active_run()
            if active is not None and active.info.run_id == run.info.run_id:
                mlflow.end_run()
        except Exception:
            pass
