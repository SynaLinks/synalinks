# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from datasets import load_dataset

from synalinks.src.api_export import synalinks_export
from synalinks.src.datasets.dataset import Dataset


@synalinks_export(
    [
        "synalinks.HuggingFaceDataset",
        "synalinks.datasets.HuggingFaceDataset",
    ]
)
class HuggingFaceDataset(Dataset):
    """Streaming dataset backed by a HuggingFace ``datasets`` source.

    Each row produced by ``datasets.load_dataset`` is rendered through the
    Jinja2 ``input_template`` / ``output_template`` to JSON, validated
    against the corresponding ``DataModel`` (or ``synalinks.ChatMessages``
    when ``None``), and accumulated into batches of size ``batch_size``.
    Each batch is yielded as ``(x, y)`` — numpy object arrays of
    ``DataModel`` instances — matching the format synalinks'
    ``GeneratorDataAdapter`` expects.

    Templates should render to JSON. Use Jinja's ``tojson`` filter for
    safe string escaping.

    Example:

    ```python
    ds = synalinks.HuggingFaceDataset(
        path="gsm8k",
        name="main",
        split="train",
        input_data_model=MathQuestion,
        input_template='{"question": {{ question | tojson }}}',
        output_data_model=NumericalAnswer,
        output_template='{"answer": {{ answer.split("####")[-1].strip() | tojson }}}',
        batch_size=8,
    )
    program.fit(x=ds())
    ```

    Args:
        path (str): The HuggingFace dataset repo / builder name (first
            positional argument of ``datasets.load_dataset``).
        name (str): Optional. The dataset configuration name.
        split (str): Optional. The split to load (e.g. ``"train"``,
            ``"test"``). When ``None``, the entire ``DatasetDict`` is
            iterated in split order.
        revision (str): Optional. The dataset revision (commit hash,
            branch, tag).
        streaming (bool): If ``True`` (default), use HF's
            ``IterableDataset`` so rows are downloaded on demand —
            required for datasets that don't fit on disk. The generator
            naturally terminates when the source is exhausted, so the
            trainer ends the epoch on its own; pass ``steps_per_epoch``
            only if you also want shorter epochs.
        input_data_model (DataModel): See ``Dataset``.
        input_schema (dict | str): See ``Dataset``.
        input_template (str): See ``Dataset``.
        output_data_model (DataModel): See ``Dataset``.
        output_schema (dict | str): See ``Dataset``.
        output_template (str): See ``Dataset``.
        batch_size (int): Examples per yielded batch. Defaults to ``1``.
        limit (int): Optional. See ``Dataset``. Caps how many rows are
            consumed (across all splits). Also makes ``__len__``
            available for streaming datasets.
        repeat (int): See ``Dataset``.
        **kwargs: Forwarded to ``datasets.load_dataset`` (e.g.
            ``data_files``, ``token``, ``trust_remote_code``, ...).
    """

    def __init__(
        self,
        path,
        *,
        name=None,
        split=None,
        revision=None,
        streaming=True,
        input_data_model=None,
        input_schema=None,
        input_template=None,
        output_data_model=None,
        output_schema=None,
        output_template=None,
        batch_size=1,
        limit=None,
        repeat=1,
        **kwargs,
    ):
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
        self.path = path
        self.name = name
        self.split = split
        self.revision = revision
        self.streaming = streaming
        self.load_kwargs = kwargs

        self._dataset = load_dataset(
            path,
            name=name,
            split=split,
            revision=revision,
            streaming=streaming,
            **kwargs,
        )

    def _iter_rows(self):
        if hasattr(self._dataset, "keys") and not self.split:
            for split_name in self._dataset.keys():
                yield from self._dataset[split_name]
        else:
            yield from self._dataset

    def __len__(self):
        if self.streaming and self.limit is None:
            raise NotImplementedError("Streaming HF datasets have unknown length.")
        if self.limit is not None:
            num_rows = self.limit
        elif hasattr(self._dataset, "keys") and not self.split:
            num_rows = sum(len(self._dataset[s]) for s in self._dataset.keys())
        else:
            num_rows = len(self._dataset)
        return self._total_batches(num_rows)


@synalinks_export(["synalinks.datasets.load_split"])
def load_split(
    path,
    *,
    name=None,
    split,
    input_data_model,
    input_template,
    output_data_model=None,
    output_template=None,
    limit=None,
    **load_kwargs,
):
    """Materialize a single HF split into one ``(x, y)`` (or ``(x,)``) pair.

    A thin convenience wrapper around
    ``HuggingFaceDataset(streaming=False).materialize()`` that takes
    the same arguments as the ``HuggingFaceDataset`` constructor and
    returns numpy object arrays directly.

    Use this when you want a whole HF split as in-memory NumPy
    arrays — for evaluation, head/tail train/test splits via
    ``split_train_test``, or quick experiments. For streaming use
    cases, construct ``HuggingFaceDataset`` directly.
    """
    ds = HuggingFaceDataset(
        path=path,
        name=name,
        split=split,
        streaming=False,
        input_data_model=input_data_model,
        input_template=input_template,
        output_data_model=output_data_model,
        output_template=output_template,
        batch_size=None,
        limit=limit,
        **load_kwargs,
    )
    return ds.materialize()
