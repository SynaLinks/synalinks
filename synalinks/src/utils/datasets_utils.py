# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""File and row helpers shared by the file-backed datasets."""

import os


def iter_files(root, suffixes, recursive=True):
    """Yield the files under `root` whose name ends with `suffixes`, sorted.

    The match ignores case (`suffixes` is lowercase: a string or a tuple of
    strings). Directories and files are walked in sorted order, since
    `os.walk`'s order depends on the filesystem.
    """
    if not recursive:
        for name in sorted(os.listdir(root)):
            full = os.path.join(root, name)
            if os.path.isfile(full) and name.lower().endswith(suffixes):
                yield full
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if name.lower().endswith(suffixes):
                yield os.path.join(dirpath, name)


def resolve_paths(path, kind):
    """Return `path` (one path or several) as a list of existing paths.

    Raises `ValueError` when no path is given and `FileNotFoundError` naming
    the first missing one; `kind` names the file type in both messages.
    """
    if isinstance(path, (str, os.PathLike)):
        paths = [os.fspath(path)]
    else:
        paths = [os.fspath(p) for p in path]
    if not paths:
        raise ValueError(f"`path` must name at least one {kind} file.")
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{kind} file not found: {p}")
    return paths


def select_columns(row, columns):
    """Project a row dict to `columns`, silently dropping absent keys.

    `None` keeps the whole row. A column missing from one row does not fail
    the load (as with the Parquet reader's column pushdown): the template
    decides what is required.
    """
    if columns is None:
        return row
    return {k: row[k] for k in columns if k in row}
