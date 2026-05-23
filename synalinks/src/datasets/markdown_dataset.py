# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import re
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional

from markdown_it import MarkdownIt

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.dataset import Dataset


@synalinks_export(
    [
        "synalinks.MarkdownSection",
        "synalinks.datasets.MarkdownSection",
    ]
)
class MarkdownSection(DataModel):
    """A single heading-delimited region of a Markdown document.

    ``section_id`` is the first declared field so it becomes the
    primary key when the row is inserted into a ``KnowledgeBase``
    (see the "Primary Key Convention" section of the ``KnowledgeBase``
    docstring). The id is built as ``"<filepath>#<path>"`` so it's
    stable across reruns and unique within a corpus (assuming heading
    paths are unique within each file).
    """

    section_id: str = Field(
        description=(
            "Stable identifier of the form '<filepath>#<path>'. Used as "
            "the primary key for upserts."
        ),
    )
    filepath: str = Field(
        description="Source `.md` file, relative to the corpus root.",
    )
    path: str = Field(
        description=(
            "Breadcrumb of ancestor headings joined with ' / ', ending "
            "with this section's own heading. Empty string for content "
            "appearing before the first heading."
        ),
    )
    section_name: str = Field(
        description="This section's heading text (no leading '#').",
    )
    level: int = Field(
        description=(
            "Heading depth, 1-6. 0 is reserved for the pre-heading preamble of a file."
        ),
    )
    text: str = Field(
        description="Body of this section, between this heading and the next.",
    )


@synalinks_export(
    [
        "synalinks.MarkdownDocument",
        "synalinks.datasets.MarkdownDocument",
    ]
)
class MarkdownDocument(DataModel):
    """The nested view of a Markdown file — one object per source file.

    ``filepath`` is the first declared field so it serves as the PK
    when stored directly. In practice you usually want to store the
    flattened `MarkdownSection` rows instead (so retrieval
    chunks at heading granularity), and use ``MarkdownDocument`` only
    as the dataset's yield shape.
    """

    filepath: str = Field(
        description="Path of the file, relative to the corpus root.",
    )
    title: str = Field(
        description=(
            "Document title: the first h1's text if any, else the file "
            "basename without extension."
        ),
    )
    sections: List[MarkdownSection] = Field(
        description="Sections in document order.",
    )


# YAML front matter strip — ``markdown-it-py`` itself doesn't strip it
# (a plugin is required, ``mdit-py-plugins``). Rather than pull a second
# dep, we strip the leading ``---\n...\n---\n`` block ourselves before
# parsing. Front matter is metadata, not document content, so dropping
# it before parsing is what every downstream tool effectively does.
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*(?:\n|$)", re.DOTALL)


# Instantiate the parser once, module-level. ``commonmark`` is the strict
# CommonMark spec preset (no GFM tables, no autolinks) — sufficient for
# splitting on headings, which is all we use this for. The ``html``
# option doesn't change heading detection either way; we leave it on the
# default.
_MD = MarkdownIt("commonmark")


@synalinks_export(
    [
        "synalinks.parse_markdown_sections",
        "synalinks.datasets.parse_markdown_sections",
    ]
)
def parse_markdown_sections(text: str) -> List[Dict[str, Any]]:
    """Split a Markdown document into heading-delimited sections.

    Uses `markdown_it` under the hood, so all CommonMark heading
    forms are honored — ATX (``# ... ######``), setext (``===`` /
    ``---``), and (importantly) ``#`` characters inside fenced code
    blocks are NOT treated as headings. A leading YAML front-matter
    block (``---\\n...\\n---``) is stripped before parsing.

    Returns a list of dicts in document order, each with:

    - ``section_name`` — heading text, ``""`` for the preamble.
    - ``level`` — heading depth (1–6), or ``0`` for the preamble.
    - ``path`` — breadcrumb of ancestor heading names joined with
      ``" / "``, ending with the section's own name. ``""`` for the
      preamble.
    - ``text`` — body of the section, between this heading and the
      next (or EOF).

    A "preamble" entry is emitted only when there's non-empty content
    before the first heading. A file with no headings at all yields a
    single preamble-shaped section containing the whole document.
    """
    text = _FRONT_MATTER_RE.sub("", text, count=1)
    lines = text.splitlines()
    tokens = _MD.parse(text)

    # Walk tokens, collect (level, name, line range) for every heading.
    # ``markdown-it-py`` annotates block-level tokens with ``map =
    # [start, end]`` — 0-indexed source line numbers, end-exclusive.
    # For ``heading_open`` that range covers BOTH the heading text and
    # any setext underline, so ``end`` is the right place to start the
    # body of the section.
    headings: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            level = int(tok.tag[1])  # 'h3' -> 3
            inline = tokens[i + 1]  # the inline-content token
            name = (inline.content or "").strip()
            start_line, end_line = tok.map
            headings.append(
                {
                    "level": level,
                    "name": name,
                    "start": start_line,
                    "end": end_line,
                }
            )
            # heading_open / inline / heading_close — skip all three.
            i += 3
            continue
        i += 1

    # Body extraction. Each heading owns everything from its ``end``
    # line up to (but not including) the next heading's ``start``.
    sections: List[Dict[str, Any]] = []

    if headings:
        preamble = "\n".join(lines[: headings[0]["start"]]).strip("\n")
        if preamble.strip():
            sections.append(
                {"section_name": "", "level": 0, "path": "", "text": preamble}
            )
    else:
        # No headings — treat the whole file as a single preamble-shaped
        # section. Keeps the caller's loop uniform (always at least one
        # section per non-empty file).
        whole = "\n".join(lines).strip("\n")
        if whole.strip():
            sections.append({"section_name": "", "level": 0, "path": "", "text": whole})

    stack: List[str] = []  # names of currently-open ancestor headings
    for idx, h in enumerate(headings):
        body_start = h["end"]
        body_end = headings[idx + 1]["start"] if idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[body_start:body_end]).strip("\n")

        # Pop ancestors at the same or deeper level — a new h2 closes
        # any open h2 / h3 / h4 / ... sibling, but leaves the parent h1.
        while len(stack) >= h["level"]:
            stack.pop()
        stack.append(h["name"])
        path = " / ".join(stack)

        sections.append(
            {
                "section_name": h["name"],
                "level": h["level"],
                "path": path,
                "text": body,
            }
        )

    return sections


def _derive_title(filepath: str, sections: List[Dict[str, Any]]) -> str:
    """First h1's text, else the file basename without extension."""
    for s in sections:
        if s["level"] == 1:
            return s["section_name"]
    stem, _ = os.path.splitext(os.path.basename(filepath))
    return stem


_DEFAULT_INPUT_TEMPLATE = (
    '{"filepath": {{ filepath | tojson }},'
    ' "title": {{ title | tojson }},'
    ' "sections": {{ sections | tojson }}}'
)


@synalinks_export(
    [
        "synalinks.MarkdownDataset",
        "synalinks.datasets.MarkdownDataset",
    ]
)
class MarkdownDataset(Dataset):
    """Streaming dataset over a directory of Markdown files.

    Each file is parsed via `parse_markdown_sections` into a list
    of heading-delimited `MarkdownSection` objects, wrapped in a
    `MarkdownDocument` and yielded one per source file. Rows
    accumulate into batches of size ``batch_size`` — the same contract
    as `CSVDataset` and the other loaders.

    The yielded shape is inputs-only (no ``output_template``), so the
    dataset can be handed straight to `KnowledgeBase.update`.
    For section-level retrieval, iterate the dataset and store the
    flattened ``MarkdownSection`` rows in a dedicated table:

    ```python
    import synalinks

    knowledge_base = synalinks.KnowledgeBase(
        uri="duckdb://./docs.db",
        data_models=[synalinks.MarkdownSection],
    )
    ds = synalinks.MarkdownDataset(root="./docs")
    for batch in ds:
        (docs,) = batch
        for doc in docs:
            await knowledge_base.update(list(doc.sections))
    ```

    Args:
        root (str): Directory to walk. Must exist.
        encoding (str): Source encoding. Defaults to ``"utf-8"``.
        recursive (bool): When True (default), descend into
            subdirectories.
        glob_pattern (str): Filename suffix to match
            (case-insensitive). Defaults to ``".md"``.
        input_data_model (DataModel): See `Dataset`. Defaults
            to `MarkdownDocument`.
        input_schema (dict | str): See `Dataset`.
        input_template (str): See `Dataset`. Defaults to a
            template producing ``MarkdownDocument``-shaped JSON.
        batch_size (int): Examples per yielded batch. Defaults to 8.
        limit (int): Optional cap on the number of files consumed.
        repeat (int): See `Dataset`.
    """

    def __init__(
        self,
        root: str,
        *,
        encoding: str = "utf-8",
        recursive: bool = True,
        glob_pattern: str = ".md",
        input_data_model=None,
        input_schema=None,
        input_template: Optional[str] = None,
        batch_size: int = 8,
        limit: Optional[int] = None,
        repeat: int = 1,
    ):
        if input_data_model is None and input_schema is None:
            input_data_model = MarkdownDocument
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
            relpath = os.path.relpath(path, self.root)
            raw_sections = parse_markdown_sections(text)
            sections = [
                {
                    "section_id": f"{relpath}#{s['path']}",
                    "filepath": relpath,
                    "path": s["path"],
                    "section_name": s["section_name"],
                    "level": s["level"],
                    "text": s["text"],
                }
                for s in raw_sections
            ]
            yield {
                "filepath": relpath,
                "title": _derive_title(relpath, raw_sections),
                "sections": sections,
            }

    def __len__(self):
        if self.limit is None:
            raise NotImplementedError(
                "MarkdownDataset has unknown length without "
                "`limit=...`. Pass a limit if you need __len__."
            )
        return self._total_batches(self.limit)
