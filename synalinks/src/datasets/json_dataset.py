# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
from typing import List
from typing import Optional
from typing import Union

import orjson

from synalinks.src.api_export import synalinks_export
from synalinks.src.datasets.dataset import Dataset


def _coerce_path(path):
    if isinstance(path, (str, os.PathLike)):
        return [os.fspath(path)]
    return [os.fspath(p) for p in path]


def _filter_columns(row, columns):
    """Project a row dict to `columns`, silently dropping absent keys.

    Matches the column-pushdown semantics used by the Parquet loader —
    a missing column on one row doesn't fail the load, the template
    decides what's required."""
    return {k: row[k] for k in columns if k in row}


@synalinks_export(
    [
        "synalinks.JSONDataset",
        "synalinks.datasets.JSONDataset",
    ]
)
class JSONDataset(Dataset):
    """Streaming dataset backed by one or more JSON files.

    Each file must contain a JSON array of objects at its top level
    (``[{"id": ..., "text": ...}, ...]``). The whole array is parsed
    into memory at iteration time per file — JSON arrays aren't
    natively line-streamable, so use :class:`JSONLDataset` for huge
    sources you can't fit in RAM.

    Each row is rendered through the Jinja2 ``input_template`` /
    ``output_template`` to JSON, validated against the corresponding
    ``DataModel`` (or ``synalinks.ChatMessages`` when ``None``), and
    accumulated into batches of size ``batch_size`` — the same
    contract as :class:`HuggingFaceDataset` and the other loaders.

    Example:

    ```python
    ds = synalinks.JSONDataset(
        path="qa.json",
        input_data_model=Question,
        input_template='{"question": {{ question | tojson }}}',
        batch_size=8,
    )
    ```

    Args:
        path (str | list): Path to a JSON file, or a list of paths.
            Multiple files are concatenated in order; each must be a
            top-level JSON array of objects.
        encoding (str): File encoding. Defaults to ``"utf-8"``.
        columns (list): Optional subset of keys to forward to the
            template. Keys absent from a given row are silently
            dropped (matches the parquet/CSV column-pushdown
            behaviour).
        input_data_model (DataModel): See ``Dataset``.
        input_schema (dict | str): See ``Dataset``.
        input_template (str): See ``Dataset``.
        output_data_model (DataModel): See ``Dataset``.
        output_schema (dict | str): See ``Dataset``.
        output_template (str): See ``Dataset``.
        batch_size (int): Examples per yielded batch. Defaults to ``1``.
        limit (int): Optional. Caps how many rows are consumed across
            all input files. With a limit set, ``__len__`` is also
            available.
        repeat (int): See ``Dataset``.
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        *,
        encoding: str = "utf-8",
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

        paths = _coerce_path(path)
        if not paths:
            raise ValueError("`path` must name at least one JSON file.")
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"JSON file not found: {p}")
        self.paths = paths

        self.encoding = encoding
        self.columns = list(columns) if columns is not None else None

    def _iter_rows(self):
        columns = self.columns
        for path in self.paths:
            with open(path, "r", encoding=self.encoding) as f:
                data = orjson.loads(f.read())
            if not isinstance(data, list):
                raise ValueError(
                    f"{path}: expected a top-level JSON array of objects, "
                    f"got {type(data).__name__}. Use JSONLDataset for one "
                    f"object per line."
                )
            for row in data:
                if not isinstance(row, dict):
                    raise ValueError(
                        f"{path}: every element of the array must be an "
                        f"object, got {type(row).__name__}."
                    )
                if columns is not None:
                    row = _filter_columns(row, columns)
                yield row

    def __len__(self):
        if self.limit is None:
            raise NotImplementedError(
                "JSON datasets have unknown length without a limit. "
                "Pass `limit=...` if you need __len__."
            )
        return self._total_batches(self.limit)


@synalinks_export(
    [
        "synalinks.JSONLDataset",
        "synalinks.datasets.JSONLDataset",
    ]
)
class JSONLDataset(Dataset):
    """Streaming dataset backed by one or more JSON Lines (NDJSON) files.

    Each line in each file is a standalone JSON object. The reader
    streams line-by-line — memory usage stays bounded for arbitrarily
    large files, which is the main reason JSONL exists in the first
    place and the main reason to prefer it over :class:`JSONDataset`
    for sources you can't fit in RAM.

    Blank lines are skipped silently (matches CSV's blank-line
    behaviour); a non-blank line that isn't valid JSON raises
    ``json.JSONDecodeError`` with the underlying message so the
    caller can find the offending row.

    Args:
        path (str | list): Path to a JSONL file, or a list of paths.
            Multiple files are concatenated in order.
        encoding (str): File encoding. Defaults to ``"utf-8"``.
        columns (list): Optional subset of keys to forward to the
            template. Keys absent from a given row are silently
            dropped.
        input_data_model (DataModel): See ``Dataset``.
        input_schema (dict | str): See ``Dataset``.
        input_template (str): See ``Dataset``.
        output_data_model (DataModel): See ``Dataset``.
        output_schema (dict | str): See ``Dataset``.
        output_template (str): See ``Dataset``.
        batch_size (int): Examples per yielded batch. Defaults to ``1``.
        limit (int): Optional. Caps how many rows are consumed across
            all input files. With a limit set, ``__len__`` is also
            available.
        repeat (int): See ``Dataset``.
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        *,
        encoding: str = "utf-8",
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

        paths = _coerce_path(path)
        if not paths:
            raise ValueError("`path` must name at least one JSONL file.")
        for p in paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"JSONL file not found: {p}")
        self.paths = paths

        self.encoding = encoding
        self.columns = list(columns) if columns is not None else None

    def _iter_rows(self):
        columns = self.columns
        for path in self.paths:
            with open(path, "r", encoding=self.encoding) as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        # Blank lines are tolerated, same as CSV. Real
                        # JSONL pipelines occasionally leave trailing
                        # newlines or split-then-rejoined chunks.
                        continue
                    row = orjson.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"{path}: every line must decode to a JSON "
                            f"object, got {type(row).__name__}."
                        )
                    if columns is not None:
                        row = _filter_columns(row, columns)
                    yield row

    def __len__(self):
        if self.limit is None:
            raise NotImplementedError(
                "Streaming JSONL datasets have unknown length. "
                "Pass `limit=...` if you need __len__."
            )
        return self._total_batches(self.limit)
