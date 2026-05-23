# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import csv as _csv
import os
from typing import List
from typing import Optional
from typing import Union

from synalinks.src.api_export import synalinks_export
from synalinks.src.datasets.dataset import Dataset


@synalinks_export(
    [
        "synalinks.CSVDataset",
        "synalinks.datasets.CSVDataset",
    ]
)
class CSVDataset(Dataset):
    """Streaming dataset backed by one or more CSV files.

    Rows are read with Python's ``csv.DictReader`` so memory usage stays
    bounded — one row is materialized at a time. Every field comes back
    as a string (no type inference); the Jinja2 templates can then
    coerce as needed via filters like ``int``, ``float``, or ``tojson``.
    This matches how most LM-training CSV pipelines treat their data
    and avoids the surprise of a number-looking column being silently
    cast to int.

    Each row is rendered through the Jinja2 ``input_template`` /
    ``output_template`` to JSON, validated against the corresponding
    ``DataModel`` (or ``synalinks.ChatMessages`` when ``None``), and
    accumulated into batches of size ``batch_size`` — the same
    contract as `HuggingFaceDataset`.

    Templates receive each CSV row as a dict keyed by column name, so
    column names must be valid Python identifiers (or you can alias
    them via the ``column_names`` argument when the source CSV has
    inconvenient headers — or no header at all).

    Example:

    ```python
    ds = synalinks.CSVDataset(
        path="qa.csv",
        input_data_model=Question,
        input_template='{"question": {{ question | tojson }}}',
        output_data_model=Answer,
        output_template='{"answer": {{ answer | tojson }}}',
        batch_size=8,
    )
    program.fit(x=ds())
    ```

    Args:
        path (str | list): Path to a CSV file, or a list of paths.
            Multiple files are read in order and concatenated; they
            must share the same column layout.
        delimiter (str): Field delimiter. Defaults to ``","``.
        quotechar (str): Field-quote character. Defaults to ``'"'``.
        encoding (str): Source encoding. Defaults to ``"utf-8"``.
        column_names (list): Optional explicit column names. When
            given, the first row of each file is treated as data
            (not a header) — useful for headerless CSVs and to alias
            non-identifier-shaped headers into identifier-shaped names.
            When ``None`` (default), the first row of each file is
            used as the header.
        columns (list): Optional subset of columns to forward to the
            template. Filters each row dict to these keys; columns
            absent from a file are silently skipped on that row.
        input_data_model (DataModel): See ``Dataset``.
        input_schema (dict | str): See ``Dataset``.
        input_template (str): See ``Dataset``.
        output_data_model (DataModel): See ``Dataset``.
        output_schema (dict | str): See ``Dataset``.
        output_template (str): See ``Dataset``.
        batch_size (int): Examples per yielded batch. Defaults to ``1``.
        limit (int): Optional. Caps how many rows are consumed across
            all input files. Also makes ``__len__`` available — without
            a limit, streaming CSVs have unknown length.
        repeat (int): See ``Dataset``.
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        *,
        delimiter: str = ",",
        quotechar: str = '"',
        encoding: str = "utf-8",
        column_names: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
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
            raise ValueError("`path` must name at least one CSV file.")
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"CSV file not found: {p}")
        self.paths = paths

        self.delimiter = delimiter
        self.quotechar = quotechar
        self.encoding = encoding
        self.column_names = list(column_names) if column_names is not None else None
        self.columns = list(columns) if columns is not None else None

    def _iter_rows(self):
        # ``newline=""`` lets ``csv`` handle the line discipline itself —
        # mandatory per the stdlib docs to avoid splitting embedded newlines
        # inside quoted fields.
        columns = self.columns
        for path in self.paths:
            with open(path, "r", encoding=self.encoding, newline="") as f:
                reader = _csv.DictReader(
                    f,
                    fieldnames=self.column_names,
                    delimiter=self.delimiter,
                    quotechar=self.quotechar,
                )
                for row in reader:
                    # When a row has more fields than the header,
                    # DictReader puts the surplus values under the
                    # ``None`` key. That key would crash Jinja's
                    # ``**row`` expansion ("keywords must be strings"),
                    # and we'd rather just ignore the surplus —
                    # consistent with "the loader doesn't fail on
                    # ragged rows; the user's template decides what's
                    # required". Drop the None key before yielding.
                    row.pop(None, None)
                    if columns is not None:
                        # Filter to the requested subset; absent keys are
                        # silently dropped rather than raising, matching
                        # the columns-pushdown semantics of pyarrow's
                        # parquet reader.
                        row = {k: row[k] for k in columns if k in row}
                    yield row

    def __len__(self):
        if self.limit is None:
            raise NotImplementedError(
                "Streaming CSV datasets have unknown length. "
                "Pass `limit=...` if you need __len__."
            )
        return self._total_batches(self.limit)
