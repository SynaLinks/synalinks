# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
from typing import Iterator
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.dataset import Dataset


@synalinks_export(
    [
        "synalinks.TextDocument",
        "synalinks.datasets.TextDocument",
    ]
)
class TextDocument(DataModel):
    """A plain-text document — one row per `.txt` file.

    ``filepath`` is the first declared field so it becomes the primary
    key when the row is inserted into a ``KnowledgeBase`` (see the
    "Primary Key Convention" section of the ``KnowledgeBase``
    docstring). Re-running the loader against the same corpus upserts
    deterministically on filepath.
    """

    filepath: str = Field(
        description="Path of the file, relative to the corpus root.",
    )
    text: str = Field(
        description="Full file contents, decoded with the loader's encoding.",
    )


# Default Jinja2 template that renders a raw row dict (``{"filepath": ...,
# "text": ...}``) into JSON matching ``TextDocument``. Module-level so it's
# compiled once.
_DEFAULT_INPUT_TEMPLATE = (
    '{"filepath": {{ filepath | tojson }}, "text": {{ text | tojson }}}'
)


@synalinks_export(
    [
        "synalinks.TextDataset",
        "synalinks.datasets.TextDataset",
    ]
)
class TextDataset(Dataset):
    """Streaming dataset over a directory of plain-text files.

    Walks ``root`` (optionally recursively), reads every file whose
    name ends in ``glob_pattern`` (default ``".txt"``, case-insensitive),
    and yields one row per file. Each row is rendered through the
    Jinja2 ``input_template`` to JSON, validated against
    ``input_data_model`` (defaults to `TextDocument`), and
    accumulated into batches of size ``batch_size`` — the same contract
    as `CSVDataset` and the other loaders.

    The yielded shape is inputs-only (no ``output_template``), so the
    dataset can be handed straight to `KnowledgeBase.update`:

    ```python
    import synalinks

    knowledge_base = synalinks.KnowledgeBase(
        uri="duckdb://./docs.db",
        data_models=[synalinks.TextDocument],
    )
    ds = synalinks.TextDataset(root="./corpus", batch_size=16)
    await knowledge_base.update(ds)  # streams batch-by-batch
    ```

    For non-default row shapes (e.g. an extra ``source`` column), pass
    ``input_data_model=YourModel`` and an ``input_template`` that
    renders the extra fields. The ``_iter_rows`` output always carries
    just ``filepath`` + ``text``; subclass and override
    `_iter_rows` if you need to inject more per-file metadata.

    Args:
        root (str): Directory to walk. Must exist.
        encoding (str): Source encoding. Defaults to ``"utf-8"``.
        recursive (bool): When True (default), descend into
            subdirectories. When False, only direct children of
            ``root`` are read.
        glob_pattern (str): Filename suffix to match
            (case-insensitive). Defaults to ``".txt"``.
        input_data_model (DataModel): See `Dataset`. Defaults to
            `TextDocument`.
        input_schema (dict | str): See `Dataset`.
        input_template (str): See `Dataset`. Defaults to a
            template producing ``TextDocument``-shaped JSON.
        batch_size (int): Examples per yielded batch. Defaults to 8.
        limit (int): Optional cap on the number of files consumed.
            With a limit set, ``__len__`` is also available.
        repeat (int): See `Dataset`.
    """

    def __init__(
        self,
        root: str,
        *,
        encoding: str = "utf-8",
        recursive: bool = True,
        glob_pattern: str = ".txt",
        input_data_model=None,
        input_schema=None,
        input_template: Optional[str] = None,
        batch_size: int = 8,
        limit: Optional[int] = None,
        repeat: int = 1,
    ):
        if input_data_model is None and input_schema is None:
            input_data_model = TextDocument
        if input_template is None:
            input_template = _DEFAULT_INPUT_TEMPLATE
        super().__init__(
            input_data_model=input_data_model,
            input_schema=input_schema,
            input_template=input_template,
            batch_size=batch_size,
            limit=limit,
            repeat=repeat,
        )

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Corpus root not found: {root}")
        self.root = root
        self.encoding = encoding
        self.recursive = recursive
        self.glob_pattern = glob_pattern.lower()

    def _iter_files(self) -> Iterator[str]:
        if self.recursive:
            # os.walk's order is filesystem-dependent — sort filenames
            # within each directory so the dataset is deterministic
            # across reruns on the same corpus.
            for dirpath, _, filenames in os.walk(self.root):
                for name in sorted(filenames):
                    if name.lower().endswith(self.glob_pattern):
                        yield os.path.join(dirpath, name)
        else:
            for name in sorted(os.listdir(self.root)):
                full = os.path.join(self.root, name)
                if os.path.isfile(full) and name.lower().endswith(self.glob_pattern):
                    yield full

    def _iter_rows(self):
        for path in self._iter_files():
            with open(path, "r", encoding=self.encoding) as f:
                text = f.read()
            yield {
                "filepath": os.path.relpath(path, self.root),
                "text": text,
            }

    def __len__(self):
        if self.limit is None:
            raise NotImplementedError(
                "TextDataset has unknown length without `limit=...`. "
                "Pass a limit if you need __len__ (e.g. for the progress "
                "bar shown by KnowledgeBase.update)."
            )
        return self._total_batches(self.limit)
