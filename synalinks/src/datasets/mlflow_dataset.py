# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.config import mlflow_tracking_uri
from synalinks.src.datasets.dataset import Dataset

try:
    import mlflow
    from mlflow.genai.datasets import create_dataset
    from mlflow.genai.datasets import get_dataset

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def _set_tracking_uri(tracking_uri):
    """Point MLflow at the given store, or at `synalinks.enable_observability()`'s."""
    if not MLFLOW_AVAILABLE:
        raise ImportError(
            "mlflow is required for `MLflowDataset`. Install it with: pip install mlflow"
        )
    tracking_uri = tracking_uri or mlflow_tracking_uri()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


@synalinks_export(
    [
        "synalinks.MLflowDataset",
        "synalinks.datasets.MLflowDataset",
    ]
)
class MLflowDataset(Dataset):
    """Dataset backed by an MLflow evaluation dataset.

    MLflow evaluation datasets (``mlflow.genai.datasets``) are stored in the
    tracking server and curated from the MLflow UI, from traces, or from
    code. Every record carries an ``inputs`` dict and, when labeled, an
    ``expectations`` dict. This loader renders each record through the
    Jinja2 ``input_template`` / ``output_template`` to JSON, validates it
    against the corresponding ``DataModel``, and yields batches of size
    ``batch_size``, the same contract as `HuggingFaceDataset` and the other
    loaders. Records are exposed to the templates as ``inputs``,
    ``expectations``, ``outputs``, ``tags`` and ``record_id``.

    Evaluation datasets need a SQL-backed tracking store (``sqlite:///...``
    or a tracking server), not the default ``./mlruns`` file store.

    Example:

    ```python
    ds = synalinks.MLflowDataset(
        name="qa_golden_set",
        tracking_uri="http://localhost:5000",
        input_data_model=Question,
        input_template='{"question": {{ inputs.question | tojson }}}',
        output_data_model=Answer,
        output_template='{"answer": {{ expectations.answer | tojson }}}',
        batch_size=8,
    )
    program.evaluate(x=ds())

    # Push in-memory arrays to MLflow; each example becomes one record with
    # its input JSON under `inputs` and its target JSON under `expectations`.
    synalinks.MLflowDataset.create(
        name="qa_golden_set", x=x_train, y=y_train, experiment_id="1"
    )
    ```

    Args:
        name (str): The evaluation dataset name. Either ``name`` or
            ``dataset_id`` is required.
        dataset_id (str): Optional. The evaluation dataset id, when a name
            is ambiguous or unknown.
        tracking_uri (str): Optional. MLflow tracking URI. Defaults to the
            value from `synalinks.enable_observability()` or MLflow's own
            default (``MLFLOW_TRACKING_URI`` env var).
        input_data_model (DataModel): See ``Dataset``.
        input_schema (dict | str): See ``Dataset``.
        input_template (str): See ``Dataset``.
        output_data_model (DataModel): See ``Dataset``.
        output_schema (dict | str): See ``Dataset``.
        output_template (str): See ``Dataset``. When given, every record
            must carry expectations.
        batch_size (int): Examples per yielded batch. Defaults to ``1``.
        limit (int): Optional. See ``Dataset``.
        repeat (int): See ``Dataset``.
    """

    def __init__(
        self,
        name=None,
        *,
        dataset_id=None,
        tracking_uri=None,
        input_data_model=None,
        input_schema=None,
        input_template=None,
        output_data_model=None,
        output_schema=None,
        output_template=None,
        batch_size=1,
        limit=None,
        repeat=1,
    ):
        if name is None and dataset_id is None:
            raise ValueError("Either `name` or `dataset_id` is required.")
        super().__init__(
            input_data_model=input_data_model,
            input_schema=input_schema,
            input_template=input_template,
            output_data_model=output_data_model,
            output_schema=output_schema,
            output_template=output_template,
            batch_size=batch_size,
            limit=limit,
            repeat=repeat,
        )
        _set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        self._dataset = get_dataset(name=name, dataset_id=dataset_id)
        self.name = self._dataset.name
        self.dataset_id = self._dataset.dataset_id
        self._records = self._dataset.to_dict().get("records") or []

    def _iter_rows(self):
        labeled = self._output_tmpl is not None
        for record in self._records:
            expectations = record.get("expectations") or {}
            if labeled and not expectations:
                raise ValueError(
                    f"Record {record.get('dataset_record_id')!r} of MLflow dataset "
                    f"{self.name!r} has no expectations; omit `output_template` "
                    "to load it as an inputs-only dataset."
                )
            yield {
                "inputs": record.get("inputs") or {},
                "expectations": expectations,
                "outputs": record.get("outputs") or {},
                "tags": record.get("tags") or {},
                "record_id": record.get("dataset_record_id"),
            }

    def __len__(self):
        num_rows = len(self._records)
        if self.limit is not None:
            num_rows = min(num_rows, self.limit)
        return self._total_batches(num_rows)

    @classmethod
    def create(
        cls,
        name,
        x,
        y=None,
        *,
        experiment_id=None,
        tags=None,
        tracking_uri=None,
    ):
        """Create an MLflow evaluation dataset from in-memory arrays.

        Each input's JSON becomes a record's ``inputs`` and, when ``y`` is
        given, the matching target's JSON its ``expectations``.

        Args:
            name (str): The evaluation dataset name.
            x (np.ndarray | list): The inputs, ``DataModel`` instances.
            y (np.ndarray | list): Optional. The targets, ``DataModel``
                instances aligned with ``x``.
            experiment_id (str | list): Optional. Experiment id(s) the
                dataset is associated with.
            tags (dict): Optional. Tags set on the dataset.
            tracking_uri (str): Optional. MLflow tracking URI
                (see `MLflowDataset`).

        Returns:
            (EvaluationDataset): The MLflow evaluation dataset.
        """
        if len(x) == 0:
            raise ValueError("`x` must contain at least one example.")
        if y is not None and len(y) != len(x):
            raise ValueError(
                f"`x` and `y` must have the same length; got {len(x)} and {len(y)}."
            )
        _set_tracking_uri(tracking_uri)
        dataset = create_dataset(name=name, experiment_id=experiment_id, tags=tags)
        records = []
        for i, item in enumerate(x):
            record = {"inputs": item.get_json()}
            if y is not None:
                record["expectations"] = y[i].get_json()
            records.append(record)
        dataset.merge_records(records)
        return dataset
