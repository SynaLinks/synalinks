# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Shared helpers for the code agents (FunctionCalling, RLM, Deep Agent).

Covers workdir resolution, AGENTS.md (the agents.md standard) discovery +
injection, tool merging, and the metadata-only input summary the agents show the
LM — field previews and sizes, never the full values. The full payload is made
reachable to the agent's code another way: bound as the ``inputs`` REPL variable
(RLM), or written as a JSON file in the sandbox's copy-on-write overlay (Deep
Agent); the summary's ``inputs_file`` field names that file when the file route
is used. Agent Skills helpers live in ``skills_utils``.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from typing import Optional

from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.core.tool import Tool


def resolve_workdir(workdir: Optional[str]) -> Optional[str]:
    """Validate and resolve a ``workdir`` to an absolute path.

    Args:
        workdir: A path to an existing directory, or ``None``.

    Returns:
        The resolved absolute path as a string, or ``None`` when ``workdir``
        is falsy.

    Raises:
        ValueError: If ``workdir`` is set but does not exist or is not a
            directory.
    """
    if not workdir:
        return None
    resolved = Path(workdir).resolve()
    if not resolved.exists():
        raise ValueError(f"workdir does not exist: {workdir}")
    if not resolved.is_dir():
        raise ValueError(f"workdir is not a directory: {workdir}")
    return str(resolved)


def prepend_context_message(agent_messages: List, context_message) -> bool:
    """Insert a context message (AGENTS.md, available skills, …) at the front.

    Idempotent: when a message with the same ``role`` and ``content`` is already
    present, nothing is inserted. This matters in interactive mode and whenever a
    returned trajectory is fed back into the agent — otherwise a fresh copy would
    stack at the front of the messages on every turn.

    Args:
        agent_messages: The trajectory's list of message dicts, mutated in place.
        context_message: A ``ChatMessage`` (e.g. from ``read_agents_md`` /
            ``read_skills``) or ``None``.

    Returns:
        ``True`` if a message was inserted, ``False`` otherwise.
    """
    if context_message is None:
        return False
    msg_json = context_message.get_json()
    for existing in agent_messages:
        if existing.get("role") == msg_json.get("role") and existing.get(
            "content"
        ) == msg_json.get("content"):
            return False
    agent_messages.insert(0, msg_json)
    return True


# -- AGENTS.md --------------------------------------------------------------
#
# Support for the open AGENTS.md standard (https://agents.md): a project's
# ``AGENTS.md`` is plain Markdown (no frontmatter) holding the conventions an
# agent should follow — "a README for agents". Per the spec a monorepo may carry
# nested ``AGENTS.md`` files and the **nearest one wins** for a given path. These
# helpers — structured like the Agent Skills ones in ``skills_utils`` — discover
# the root + nested files; the prompt renderer emits the root file's content
# verbatim. The spec does not prescribe any prompt wording, so no framing is
# added.

_AGENTS_MD_NAME = "AGENTS.md"
# Vendored / build directories not descended into when discovering AGENTS.md.
_AGENTS_MD_SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".idea",
        ".vscode",
    }
)


@dataclass
class AgentsMd:
    """One discovered ``AGENTS.md`` and the subtree it governs.

    ``directory`` is the file's directory relative to the workdir (``""`` for the
    root file); ``content`` is the stripped Markdown body; ``path`` the file.
    """

    path: str
    content: str
    directory: str = ""


def find_agents_md(directory) -> Optional[Path]:
    """Return the ``AGENTS.md`` directly in ``directory``, or ``None``."""
    candidate = Path(directory) / _AGENTS_MD_NAME
    return candidate if candidate.is_file() else None


def discover_agents_md(workdir) -> List[AgentsMd]:
    """Discover every ``AGENTS.md`` under ``workdir`` (root first, then nested).

    Follows the agents.md monorepo model: subprojects may carry their own
    ``AGENTS.md`` alongside the root, and the **nearest** file to a path takes
    precedence. Returns the root file first (the primary conventions) followed by
    nested files ordered by depth then path, so a caller can present the root
    inline and surface nested ones as read-on-demand pointers. Vendored / build
    directories are skipped. Returns ``[]`` when ``workdir`` is missing/empty.
    """
    if not workdir:
        return []
    base = Path(workdir)
    if not base.is_dir():
        return []
    found: List[AgentsMd] = []
    for dirpath, dirnames, filenames in os.walk(base):
        # Prune vendored/hidden dirs in place so os.walk doesn't descend them.
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d not in _AGENTS_MD_SKIP_DIRS
        )
        if _AGENTS_MD_NAME not in filenames:
            continue
        file_path = Path(dirpath) / _AGENTS_MD_NAME
        try:
            content = file_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not content:
            continue
        rel = os.path.relpath(dirpath, base)
        found.append(
            AgentsMd(
                path=str(file_path.resolve()),
                content=content,
                directory="" if rel == "." else rel,
            )
        )
    # Root (directory == "") first, then nested by depth then path.
    found.sort(
        key=lambda a: (a.directory.count(os.sep) if a.directory else -1, a.directory)
    )
    return found


def agents_md_prompt(items: List[AgentsMd]) -> str:
    """Return the working directory's root ``AGENTS.md`` content for the prompt.

    The agents.md spec does not prescribe any wording for how an agent surfaces
    ``AGENTS.md`` to the model, so the root file's body is returned verbatim with
    no added framing. Nested (monorepo) files discovered by `discover_agents_md`
    are not surfaced here. Returns ``""`` when there is no root file.
    """
    top = next((item for item in items if not item.directory), None)
    return top.content if top is not None else ""


def merge_tools(builtin_tools: List, extra_tools: Optional[List], *, kind: str) -> List:
    """Merge built-in tools with user-supplied tools, rejecting name collisions.

    Args:
        builtin_tools: The agent's own ``Tool`` instances.
        extra_tools: User-supplied tools (``Tool`` instances or callables), or
            ``None``.
        kind: A short noun naming the built-in tool family, embedded in the
            collision error message (e.g. ``"SQL"``, ``"retrieval"``).

    Returns:
        A new list with the built-in tools followed by the (coerced) extra
        tools.

    Raises:
        ValueError: If an extra tool's name collides with a built-in tool.
    """
    builtin_names = {t.name for t in builtin_tools}
    merged = list(builtin_tools)
    for extra in extra_tools or []:
        # Already-built tools (either `Tool` flavor, or any tool-like object
        # exposing `.name`) pass through; only raw callables are wrapped.
        extra_tool = extra if hasattr(extra, "name") else Tool(extra)
        if extra_tool.name in builtin_names:
            raise ValueError(
                f"Tool name {extra_tool.name!r} collides with a built-in "
                f"{kind} tool. Rename the additional tool."
            )
        merged.append(extra_tool)
    return merged


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
