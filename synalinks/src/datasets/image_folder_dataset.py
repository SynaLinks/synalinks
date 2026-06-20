# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
from typing import Iterator
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.datasets.dataset import Dataset

# Image file extensions recognized by default (lower-cased, case-insensitive
# match). A superset of what every provider accepts — the resolver inlines
# whatever bytes it finds and lets the model decide.
_DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")

# Default Jinja2 template: a single user turn pairing a text prompt with the
# image. The image is referenced by its `file://` URI — a lightweight pointer,
# NOT the bytes — so the dataset stays cheap to iterate over. The actual file
# is read and inlined as base64 only when the batch is sent to the model (see
# `synalinks.backend.resolve_content_media`).
_DEFAULT_INPUT_TEMPLATE = (
    '{"messages":[{"role":"user","content":['
    '{"type":"text","text":{{ prompt | tojson }}},'
    '{"type":"image_url","image_url":{"url":{{ file_uri | tojson }}}}'
    "]}]}"
)


@synalinks_export(
    [
        "synalinks.ImageFolderDataset",
        "synalinks.datasets.ImageFolderDataset",
    ]
)
class ImageFolderDataset(Dataset):
    """Streaming dataset over a directory of image files.

    Walks ``root`` (optionally recursively), matches every file whose
    extension is in ``extensions`` (default common image types,
    case-insensitive), and yields one row per image — shaped, by default,
    as a `ChatMessages` with a text ``prompt`` next to the image.

    Crucially, an image is carried as a **reference** (a ``file://`` URI),
    never as bytes: iterating the dataset — even materializing it — does
    not load a single pixel into memory. The file is read and inlined as a
    base64 ``data:`` URI only when its batch is actually sent to the model,
    one batch at a time. A folder of a million photos therefore costs a list
    of paths to iterate, not a million decoded images.

    ```python
    import synalinks

    ds = synalinks.ImageFolderDataset(
        root="./photos",
        prompt="Describe this image in one short sentence.",
        batch_size=8,
    )
    # program.predict(x=ds())  # each batch's files are read on demand
    ```

    Each raw row exposes four template variables, so a custom
    ``input_template`` / ``output_template`` can reshape freely:

    - ``file_uri``: the image as ``file:///abs/path`` (drop into an
      ``image_url`` part — the resolver reads it per batch).
    - ``image_path``: the path relative to ``root``.
    - ``name``: the filename without extension.
    - ``label``: the immediate parent directory name (``""`` for images
      directly under ``root``) — the torchvision ``ImageFolder`` convention,
      handy as a classification target.

    For a supervised ``(image, label)`` dataset, pass an ``output_template``:

    ```python
    ds = synalinks.ImageFolderDataset(
        root="./photos",            # e.g. photos/cat/*.jpg, photos/dog/*.jpg
        output_template='{"label": {{ label | tojson }}}',
        output_data_model=MyLabel,
        batch_size=8,
    )
    ```

    Args:
        root (str): Directory to walk. Must exist.
        prompt (str): Text paired with each image by the default
            ``input_template``. Ignored if you pass your own
            ``input_template``. Defaults to ``"Describe this image."``.
        recursive (bool): When True (default), descend into
            subdirectories. When False, only direct children of ``root``.
        extensions (tuple): Image extensions to match (case-insensitive).
            Defaults to ``(".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")``.
        input_data_model (DataModel): See `Dataset`. Defaults to
            `synalinks.ChatMessages`.
        input_schema (dict | str): See `Dataset`.
        input_template (str): See `Dataset`. Defaults to a single user turn
            with the ``prompt`` and the image.
        output_data_model (DataModel): See `Dataset`. Optional target shape.
        output_schema (dict | str): See `Dataset`.
        output_template (str): See `Dataset`. Omit for an inputs-only
            dataset (e.g. captioning at inference); provide it to build a
            supervised ``(image, target)`` dataset for training.
        batch_size (int): Examples per yielded batch. Defaults to 8.
        limit (int): Optional cap on the number of images consumed. With a
            limit set, ``__len__`` is also available.
        repeat (int): See `Dataset`.
    """

    def __init__(
        self,
        root: str,
        *,
        prompt: str = "Describe this image.",
        recursive: bool = True,
        extensions=_DEFAULT_EXTENSIONS,
        input_data_model=None,
        input_schema=None,
        input_template: Optional[str] = None,
        output_data_model=None,
        output_schema=None,
        output_template: Optional[str] = None,
        batch_size: int = 8,
        limit: Optional[int] = None,
        repeat: int = 1,
    ):
        if input_template is None:
            input_template = _DEFAULT_INPUT_TEMPLATE
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

        if not os.path.isdir(root):
            raise FileNotFoundError(f"Image folder not found: {root}")
        self.root = root
        self.prompt = prompt
        self.recursive = recursive
        self.extensions = tuple(ext.lower() for ext in extensions)

    def _iter_files(self) -> Iterator[str]:
        if self.recursive:
            # os.walk's order is filesystem-dependent — sort within each
            # directory so the dataset is deterministic across reruns.
            for dirpath, _, filenames in os.walk(self.root):
                for name in sorted(filenames):
                    if name.lower().endswith(self.extensions):
                        yield os.path.join(dirpath, name)
        else:
            for name in sorted(os.listdir(self.root)):
                full = os.path.join(self.root, name)
                if os.path.isfile(full) and name.lower().endswith(self.extensions):
                    yield full

    def _iter_rows(self):
        for path in self._iter_files():
            abspath = os.path.abspath(path)
            parent = os.path.basename(os.path.dirname(abspath))
            root_name = os.path.basename(os.path.abspath(self.root))
            yield {
                "file_uri": f"file://{abspath}",
                "image_path": os.path.relpath(path, self.root),
                "name": os.path.splitext(os.path.basename(path))[0],
                "label": "" if parent == root_name else parent,
                "prompt": self.prompt,
            }

    def __len__(self):
        if self.limit is None:
            raise NotImplementedError(
                "ImageFolderDataset has unknown length without `limit=...`. "
                "Pass a limit if you need __len__ (e.g. for a progress bar)."
            )
        return self._total_batches(self.limit)
