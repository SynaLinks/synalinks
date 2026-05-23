# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
from typing import List
from typing import Optional
from typing import Union

import pyarrow.parquet as pa_parquet

from synalinks.src.api_export import synalinks_export
from synalinks.src.datasets.dataset import Dataset


@synalinks_export(
    [
        "synalinks.ParquetDataset",
        "synalinks.datasets.ParquetDataset",
    ]
)
class ParquetDataset(Dataset):
    """Streaming dataset backed by one or more Parquet files.

    Rows are read with pyarrow's ``ParquetFile.iter_batches`` so memory
    usage stays bounded. Each row is rendered through the Jinja2
    ``input_template`` / ``output_template`` to JSON, validated against
    the corresponding ``DataModel`` (or ``synalinks.ChatMessages`` when
    ``None``), and accumulated into batches of size ``batch_size`` —
    the same contract as `HuggingFaceDataset`.

    Templates receive each Parquet row as a dict keyed by column name.
    The dataset reports an exact ``__len__`` derived from the files'
    metadata without reading any row data, unless ``limit`` is set in
    which case the limit wins.

    Example:

    ```python
    ds = synalinks.ParquetDataset(
        path="qa.parquet",
        columns=["question", "answer"],
        input_data_model=Question,
        input_template='{"question": {{ question | tojson }}}',
        output_data_model=Answer,
        output_template='{"answer": {{ answer | tojson }}}',
        batch_size=8,
    )
    program.fit(x=ds())
    ```

    Args:
        path (str | list): Path to a Parquet file, or a list of paths.
            Multiple files are read in order and concatenated; they
            must share a compatible schema.
        columns (list): Optional subset of columns to materialize.
            Parquet's columnar layout means restricting this list
            avoids reading the unused columns from disk entirely.
        batch_size_rows (int): Per-iteration row-group batch size
            handed to ``ParquetFile.iter_batches``. This controls how
            many rows the reader buffers at a time and is separate
            from the dataset's ``batch_size`` (which controls how
            many examples each yielded ``(x, y)`` tuple holds).
            Defaults to ``1024``.
        input_data_model (DataModel): See ``Dataset``.
        input_schema (dict | str): See ``Dataset``.
        input_template (str): See ``Dataset``.
        output_data_model (DataModel): See ``Dataset``.
        output_schema (dict | str): See ``Dataset``.
        output_template (str): See ``Dataset``.
        batch_size (int): Examples per yielded batch. Defaults to ``1``.
        limit (int): Optional. Caps how many rows are consumed across
            all input files. When set, ``__len__`` reflects the cap;
            otherwise it reflects the files' actual row counts.
        repeat (int): See ``Dataset``.
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        *,
        columns: Optional[List[str]] = None,
        batch_size_rows: int = 1024,
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

        if isinstance(path, (str, os.PathLike)):
            paths = [os.fspath(path)]
        else:
            paths = [os.fspath(p) for p in path]
        if not paths:
            raise ValueError("`path` must name at least one Parquet file.")
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Parquet file not found: {p}")
        self.paths = paths

        self.columns = list(columns) if columns is not None else None
        self.batch_size_rows = batch_size_rows

        # Open each file once at construction time. ``ParquetFile`` only
        # reads the footer metadata at this point — no row data — so
        # this is cheap even for many large files.
        self._files = [pa_parquet.ParquetFile(p) for p in self.paths]

    def _iter_rows(self):
        for pf in self._files:
            for batch in pf.iter_batches(
                batch_size=self.batch_size_rows,
                columns=self.columns,
            ):
                for row in batch.to_pylist():
                    yield row

    def __len__(self):
        if self.limit is not None:
            num_rows = self.limit
        else:
            # Pull row counts from footer metadata — no row data is read.
            num_rows = sum(pf.metadata.num_rows for pf in self._files)
        return self._total_batches(num_rows)
