# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Pure helpers for the file tools: pagination, globbing and patch rendering."""

import difflib
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


def paginate(items: List[Any], offset: int, limit: int):
    """Slice ``items`` to a page starting at a **1-based** ``offset``.

    Returns ``(page, truncated)`` where ``truncated`` is True when more
    items follow the page. ``offset`` 1 is the first item (grep
    convention); ``limit <= 0`` means "all remaining".
    """
    start = max(offset - 1, 0)
    page = items[start:] if limit <= 0 else items[start : start + limit]
    truncated = start + len(page) < len(items)
    return page, truncated


def render_patch(
    base: Dict[str, str],
    current: Dict[str, str],
    paths: Optional[set] = None,
) -> str:
    """Render a git-style unified diff between two ``{vpath: text}`` snapshots.

    Mirrors ``git diff``: a ``diff --git a/p b/p`` header per changed file,
    ``new file``/``deleted file`` modes with ``/dev/null`` sides, real ``@@``
    hunks, and ``Binary files ... differ`` for NUL-containing content. The
    result is meant to be consumable by ``git apply`` / ``patch -p1``. ``paths``
    restricts output to a subset of virtual paths.
    """
    out: List[str] = []
    for path in sorted(set(base) | set(current)):
        if paths is not None and path not in paths:
            continue
        old = base.get(path)
        new = current.get(path)
        if old == new:
            continue
        rel = path.lstrip("/")
        a, b = f"a/{rel}", f"b/{rel}"
        out.append(f"diff --git {a} {b}\n")
        if (old is not None and "\x00" in old) or (new is not None and "\x00" in new):
            if old is None:
                out.append("new file mode 100644\n")
            elif new is None:
                out.append("deleted file mode 100644\n")
            old_side = a if old is not None else "/dev/null"
            new_side = b if new is not None else "/dev/null"
            out.append(f"Binary files {old_side} and {new_side} differ\n")
            continue
        if old is None:
            out.append("new file mode 100644\n")
            from_file, old_lines = "/dev/null", []
        else:
            from_file, old_lines = a, old.splitlines(keepends=True)
        if new is None:
            out.append("deleted file mode 100644\n")
            to_file, new_lines = "/dev/null", []
        else:
            to_file, new_lines = b, new.splitlines(keepends=True)
        # git's ``\ No newline at end of file`` marker follows any hunk line
        # whose source lacked a trailing newline, so the patch applies
        # cleanly. Inside a hunk, lines are never re-classified by prefix: a
        # deleted ``--foo`` becomes ``---foo`` and is not a header.
        in_hunk = False
        for line in difflib.unified_diff(
            old_lines, new_lines, fromfile=from_file, tofile=to_file, n=3
        ):
            if not in_hunk or line.startswith("@@") or line.endswith("\n"):
                out.append(line)
                in_hunk = in_hunk or line.startswith("@@")
            else:
                out.append(line + "\n\\ No newline at end of file\n")
    return "".join(out)


def glob_to_regex(pattern: str) -> "re.Pattern":
    """Translate a glob (``*``, ``?``, ``[seq]``, ``**``) to a regex.

    ``*`` matches within a single path segment, ``**`` spans segments, and
    ``**/`` additionally matches zero leading directories so that
    ``**/x.py`` finds ``x.py`` at the root too.
    """
    i, n = 0, len(pattern)
    out: List[str] = []
    while i < n:
        if pattern[i : i + 3] == "**/":
            out.append("(?:.*/)?")
            i += 3
        elif pattern[i : i + 2] == "**":
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[":
            j = i + 1
            if j < n and pattern[j] in "!^":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j >= n:  # unterminated class -> literal '['
                out.append("\\[")
                i += 1
            else:
                seq = pattern[i + 1 : j]
                seq = ("^" + seq[1:]) if seq[:1] in "!^" else seq
                out.append("[" + seq + "]")
                i = j + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")
