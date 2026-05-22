# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Shared helpers for sandbox-backed code agents (RLM, Deep Agent).

The agents only ever show the LM a *metadata summary* of large inputs — field
previews and sizes — never the full values. The full payload is made reachable
to the agent's code another way: bound as the ``inputs`` REPL variable (RLM), or
written as a JSON file in the sandbox's copy-on-write overlay (Deep Agent). The
summary's ``inputs_file`` field names that file when the file route is used.
"""

from typing import Optional

from synalinks.src.backend import DataModel
from synalinks.src.backend import Field


class InputsSummary(DataModel):
    """Metadata-only view of the user input shown to the LM.

    Only per-field previews and sizes are surfaced here to keep the prompt small
    when the input contains long documents or large collections. The **full**
    untruncated values are reachable from the agent's code: read them via
    ``inputs[field_name]`` (REPL agents) or by reading the JSON file named in
    ``inputs_file`` (filesystem agents) — never retype them from the preview.
    """

    fields: list[dict] = Field(
        default=[],
        description=(
            "One entry per top-level input field, each with `name`, `type`, "
            "`size` (len of string/list/dict, else null), `preview`, and "
            "`truncated` (true when preview omits part of the value). Read the "
            "complete value from the `inputs` variable or the `inputs_file`."
        ),
    )
    inputs_file: Optional[str] = Field(
        default=None,
        description=(
            "When set, the full untruncated inputs are stored as this JSON file "
            "in the sandbox filesystem — read it instead of retyping previews: "
            "the `read_file` tool, or in a `run_python_code` snippet "
            "`json.loads(pathlib.Path(inputs_file).read_text())` (the sandbox "
            "has no `open()`, and `json` provides only `loads` / `dumps`). "
            "When null, the full inputs are bound as the `inputs` variable."
        ),
    )


def summarize_inputs(
    inputs_json,
    inputs_file: Optional[str] = None,
    preview_chars: int = 200,
    preview_items: int = 5,
) -> InputsSummary:
    """Build a compact ``InputsSummary`` from a raw input JSON dict.

    Small values are previewed in full. Long strings, lists and dicts show only
    a head; the full value stays reachable (via the ``inputs`` variable, or the
    JSON file named by ``inputs_file`` when given).
    """
    import orjson

    fields = []
    for name, value in inputs_json.items():
        type_name = type(value).__name__
        size = None
        preview = None
        truncated = False

        if isinstance(value, str):
            size = len(value)
            if size > preview_chars:
                preview = value[:preview_chars]
                truncated = True
            else:
                preview = value
        elif isinstance(value, list):
            size = len(value)
            head = value[:preview_items]
            truncated = size > preview_items
            preview = orjson.dumps(head).decode()
            if len(preview) > preview_chars:
                preview = preview[:preview_chars] + "…"
                truncated = True
        elif isinstance(value, dict):
            size = len(value)
            head = dict(list(value.items())[:preview_items])
            truncated = size > preview_items
            preview = orjson.dumps(head).decode()
            if len(preview) > preview_chars:
                preview = preview[:preview_chars] + "…"
                truncated = True
        else:
            preview = orjson.dumps(value).decode()

        fields.append(
            {
                "name": name,
                "type": type_name,
                "size": size,
                "preview": preview,
                "truncated": truncated,
            }
        )

    return InputsSummary(fields=fields, inputs_file=inputs_file)


async def unique_inputs_path(sandbox, base: str = "inputs", ext: str = ".json") -> str:
    """Pick a sandbox path that doesn't collide with anything already there.

    ``list_files`` searches the merged view (workdir + overlay), so this never
    shadows a file the caller mounted via ``workdir``. Call it once while the
    overlay is still empty (e.g. on first use) and reuse the result, so the
    agent overwrites the same file each call instead of accumulating copies.

    Returns ``"<base><ext>"`` when free, else ``"<base>_1<ext>"``, ``..._2`` …
    """
    n = 0
    while True:
        candidate = f"{base}{ext}" if n == 0 else f"{base}_{n}{ext}"
        listing = await sandbox.list_files(candidate)
        if not listing.get("files"):
            return candidate
        n += 1
