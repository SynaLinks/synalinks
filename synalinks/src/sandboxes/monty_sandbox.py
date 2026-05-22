# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import base64
import re
import time
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import pydantic_monty

from synalinks.src.api_export import synalinks_export
from synalinks.src.sandboxes.sandbox import ExecutionResult
from synalinks.src.sandboxes.sandbox import Sandbox
from synalinks.src.saving.object_registration import register_synalinks_serializable

_REGULAR_FILE_MODE = 0o100644
_DIRECTORY_MODE = 0o040755


def _split_streams(collector: "pydantic_monty.CollectStreams"):
    stdout_parts = []
    stderr_parts = []
    for stream, text in collector.output:
        if stream == "stderr":
            stderr_parts.append(text)
        else:
            stdout_parts.append(text)
    return "".join(stdout_parts), "".join(stderr_parts)


def _paginate(items: List[Any], offset: int, limit: int):
    """Slice ``items`` to a page starting at a **1-based** ``offset``.

    Returns ``(page, truncated)`` where ``truncated`` is True when more
    items follow the page. ``offset`` 1 is the first item (grep
    convention); ``limit <= 0`` means "all remaining".
    """
    start = max(offset - 1, 0)
    page = items[start:] if limit <= 0 else items[start : start + limit]
    truncated = start + len(page) < len(items)
    return page, truncated


@register_synalinks_serializable()
@synalinks_export(
    [
        "synalinks.sandboxes.MontySandbox",
        "synalinks.MontySandbox",
    ]
)
class MontySandbox(Sandbox, pydantic_monty.AbstractOS):
    """A restricted Python sandbox backed by ``pydantic_monty``.

    !!! warning "Experimental"
        The sandbox API is experimental and may change in a future
        release.

    Wraps a `Monty <https://github.com/pydantic/monty>`_ REPL: a
    restricted Python interpreter with a small stdlib subset, no network
    access, and no ``class`` or ``match`` statements. Safe for executing
    LM-authored Python snippets.

    State (variables, imports, user-defined functions) accumulates across
    ``run`` calls. ``dump()`` serializes the full REPL namespace to bytes;
    ``load()`` restores it. Full state round-trips through
    ``get_config()`` / ``from_config()`` as well, so a sandbox can be
    persisted alongside a conversation trajectory.

    Example:

    ```python
    import synalinks

    sandbox = synalinks.MontySandbox(timeout=5)
    await sandbox.run("x = 42")
    result = await sandbox.run("print(x)")
    print(result.stdout)                    # -> "42\\n"

    # Snapshot + restore
    blob = sandbox.dump()
    restored = synalinks.MontySandbox.load(blob)
    ```

    ## Filesystem

    The sandbox **is** a copy-on-write virtual filesystem: it implements
    ``pydantic_monty.AbstractOS`` directly, so sandboxed code reaches it
    through ``pathlib.Path``. Reads fall through to a real host directory
    (``workdir``) when one is set; writes, edits and deletes are captured
    in an in-memory overlay and **never modify the host**. With no
    ``workdir`` it is a pure in-memory scratch filesystem.

    Inspect or persist what code did via the methods on the sandbox
    itself — ``changes()``, ``journal()``, ``glob()`` / ``rglob()``,
    ``read_overlay()`` and ``get_state()`` / ``set_state()``. Monty's
    ``os`` module has no filesystem functions, so the overlay (reached
    only via ``pathlib``) cannot be bypassed, and Monty's ``Path`` has no
    ``glob`` / ``rglob`` — so the sandbox instead exposes ``glob`` /
    ``rglob`` as async globals inside the snippet.

    Containment: virtual paths are normalized (``..`` / ``.`` flattened)
    and cannot escape the root; base reads are checked to resolve inside
    ``workdir``, so a symlink in the base pointing outside it is refused.
    ``os.getenv`` / ``os.environ`` see only the ``environ`` mapping passed
    in (empty by default), never the host environment.

    ## Bound functions

    Host callables can be exposed inside the sandbox as global async
    functions. Pass them per call (``run(..., external_functions=...)``)
    or bind them once — via the ``external_functions`` constructor arg or
    :meth:`bind_functions` — so they are available on every subsequent
    ``run`` without re-passing them. A per-call ``external_functions`` is
    merged on top of the bound set and wins on name clashes. Bound
    functions persist across :meth:`reset` (they configure the sandbox,
    they are not run-produced state) but are **not** serialized by
    ``get_config`` / ``dump`` — re-bind them after ``load`` /
    ``from_config``.

    Args:
        timeout (float): Per-snippet execution budget in seconds (Default 5).
            Each ``run`` call is guaranteed a fresh ``timeout`` of actual
            in-sandbox execution time; the sandbox internally resets
            Monty's cumulative clock between snippets via dump/load, so
            long idle gaps and prior snippets do not eat into the budget
            of the current one.
        name (str): Optional. Human-readable name for the sandbox.
        workdir (str): Optional. Host directory used as the read-through
            base. ``pathlib.Path`` reads see these files and writes land in
            the in-memory overlay only. ``None`` for a pure in-memory
            filesystem.
        environ (dict): Optional. Environment exposed to ``os.getenv`` /
            ``os.environ`` inside the sandbox (isolated from the host).
        external_functions (dict): Optional. Mapping of name → callable
            (sync or async) bound persistently and exposed as global
            async functions on every ``run``. Add more later with
            :meth:`bind_functions`.
    """

    description: str = (
        "Code runs inside a Monty sandbox: a restricted Python "
        "interpreter. Only this stdlib subset is importable: sys, os, "
        "typing, asyncio, re, datetime, json, math, pathlib. No "
        "third-party libraries, no `class` or `match` statements. A "
        "copy-on-write virtual filesystem is reachable via `pathlib.Path` "
        "(`read_text`, `write_text`, `exists`, `iterdir`, `unlink`, ...); "
        "reads fall through to the working directory when one is set, and "
        "writes and deletes are captured in an in-memory overlay that "
        "never touches the host. `Path.glob` / `Path.rglob` do not exist; "
        "instead two async globals `glob(pattern, root='/')` and "
        "`rglob(pattern, root='/')` are provided and return a list of "
        "matching path strings (e.g. `await glob('**/*.py')`). The `os` "
        "module has no filesystem functions (`open`, `os.listdir`, "
        "`os.system`, `os.path` do not exist); only `os.getenv` / "
        "`os.environ` work and see an isolated environment. `asyncio` is a "
        "stub: only `asyncio.run` and `asyncio.gather` exist (no "
        "`asyncio.sleep`, `wait_for`, `Future`, `create_task` or "
        "`TaskGroup`). `json` exposes only the string forms `json.loads` "
        "and `json.dumps`; the file-object variants `json.load` / "
        "`json.dump` do not exist (there is no `open` / file I/O). Any "
        "tools bound to the module are exposed as global "
        "async callables; call them inside an `async def main(): ...` "
        "using `await tool_name(...)` and run the coroutine with "
        "`asyncio.run(main())`. A tool call returns its value unchanged: a "
        "tool returning an `int` yields that `int`, one returning a `dict` "
        "yields that `dict`, one returning a `list` yields that `list`."
    )

    def __init__(
        self,
        timeout: float = 5.0,
        name: Optional[str] = None,
        *,
        workdir: Optional[str] = None,
        environ: Optional[Dict[str, str]] = None,
        external_functions: Optional[Dict[str, Callable]] = None,
    ):
        Sandbox.__init__(
            self, timeout=timeout, name=name, external_functions=external_functions
        )
        pydantic_monty.AbstractOS.__init__(self)
        self._repl = self._new_repl()
        self._init_overlay(workdir, environ)

    def _init_overlay(
        self, workdir: Optional[str], environ: Optional[Dict[str, str]]
    ) -> None:
        """(Re)initialize the empty copy-on-write overlay over ``workdir``."""
        self._workdir = Path(workdir).resolve() if workdir else None
        self._environ: Dict[str, str] = dict(environ or {})
        self._overlay: Dict[str, bytes] = {}  # relkey -> content
        self._tombstones: set = set()  # relkeys removed
        self._dirs: set = set()  # relkeys explicitly created with mkdir
        self._journal: List[Dict[str, Any]] = []  # ordered mutation log
        # In-memory snapshot of the read-through base, populated by
        # set_state() so a serialized filesystem restores without needing
        # the original host ``workdir`` on disk. Empty in normal operation
        # (reads then fall through to the live workdir).
        self._base_files: Dict[str, bytes] = {}

    def _new_repl(self) -> pydantic_monty.MontyRepl:
        return pydantic_monty.MontyRepl(
            limits=pydantic_monty.ResourceLimits(
                max_duration_secs=self.timeout,
            )
        )

    # -- Sandbox API ----------------------------------------------------

    async def run(
        self,
        code: str,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        external_functions: Optional[Dict[str, Callable]] = None,
        os: Optional[pydantic_monty.AbstractOS] = None,
    ) -> ExecutionResult:
        # Monty's `max_duration_secs` is a **cumulative** budget across all
        # `feed_run_async` calls on a single REPL — once exceeded, every
        # subsequent call fails, even trivial ones. Users expect ``timeout``
        # to be per-snippet, so before each run we refresh the internal REPL
        # via a dump/load round-trip. The namespace (variables, imports,
        # function definitions) is preserved; only the elapsed-time counter
        # is reset. Dump+load is sub-millisecond even with meaningful state.
        blob = self._repl.dump()
        self._repl = pydantic_monty.MontyRepl.load(blob)

        collector = pydantic_monty.CollectStreams()
        kwargs: Dict[str, Any] = {"print_callback": collector}
        if inputs is not None:
            kwargs["inputs"] = inputs
        # Layered, lowest precedence first: overlay glob/rglob helpers, then
        # persistently bound functions, then this call's `external_functions`.
        # A later layer overrides an earlier one on name clashes.
        merged_functions = {
            **self._filesystem_helpers(),
            **self._functions,
            **(external_functions or {}),
        }
        if merged_functions:
            kwargs["external_functions"] = merged_functions
        # The sandbox is its own filesystem; an explicit `os=` overrides it
        # for this call.
        kwargs["os"] = os if os is not None else self

        result = None
        error = None
        try:
            result = await self._repl.feed_run_async(code, **kwargs)
        except pydantic_monty.MontyError as e:
            error = f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001 — sandboxed code can raise anything
            error = f"{type(e).__name__}: {e}"

        stdout, stderr = _split_streams(collector)
        # `_record_run` (base class) logs the snippet + outcome into history.
        return self._record_run(
            code,
            ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                result=result,
                error=error,
            ),
        )

    def reset(self) -> None:
        self._repl = self._new_repl()
        self.clear_history()
        # Drop accumulated overlay writes but keep the workdir mount point.
        self._init_overlay(self.workdir, self._environ)

    def dump(self) -> bytes:
        return self._repl.dump()

    @classmethod
    def load(
        cls,
        data: bytes,
        *,
        timeout: float = 5.0,
        name: Optional[str] = None,
        workdir: Optional[str] = None,
        environ: Optional[Dict[str, str]] = None,
        external_functions: Optional[Dict[str, Callable]] = None,
    ) -> "MontySandbox":
        """Restore a sandbox from bytes produced by ``dump()``.

        Monty does not persist the original resource limits, the overlay
        filesystem or bound functions, so the caller re-supplies
        ``timeout`` and any ``workdir`` / ``environ`` /
        ``external_functions`` for the restored sandbox. The REPL
        namespace (variables, imports, functions) is restored from
        ``data``; overlay contents are not (use ``get_config`` /
        ``from_config``, or ``set_state``, to restore those).
        """
        instance = cls.__new__(cls)
        Sandbox.__init__(
            instance, timeout=timeout, name=name, external_functions=external_functions
        )
        pydantic_monty.AbstractOS.__init__(instance)
        instance._repl = pydantic_monty.MontyRepl.load(data)
        instance._init_overlay(workdir, environ)
        return instance

    def _obj_type(self):
        return "MontySandbox"

    def get_config(self):
        """Serialize the sandbox (config + REPL state + filesystem) to a dict.

        The REPL state is captured via ``dump()`` and base64-encoded so
        the resulting config is JSON-safe. The ``workdir`` / ``environ``
        and a **full filesystem snapshot** (base files + overlay
        writes/deletes) are serialized too, so the sandbox round-trips
        fully and restores even if the original ``workdir`` is no longer
        on disk.
        """
        return {
            "timeout": self.timeout,
            "name": self.name,
            "state": base64.b64encode(self.dump()).decode("ascii"),
            "history": [dict(entry) for entry in self._history],
            "workdir": self.workdir,
            "environ": self.environ,
            "filesystem_state": self.get_state(snapshot_base=True),
        }

    @classmethod
    def from_config(cls, config):
        state_b64 = config.pop("state", None)
        timeout = config.pop("timeout", 5.0)
        name = config.pop("name", None)
        workdir = config.pop("workdir", None)
        environ = config.pop("environ", None)
        filesystem_state = config.pop("filesystem_state", None)
        history = config.pop("history", None)
        if state_b64:
            instance = cls.load(
                base64.b64decode(state_b64),
                timeout=timeout,
                name=name,
                workdir=workdir,
                environ=environ,
            )
        else:
            instance = cls(timeout=timeout, name=name, workdir=workdir, environ=environ)
        if filesystem_state is not None:
            instance.set_state(filesystem_state)
        if history is not None:
            instance._history = [dict(entry) for entry in history]
        return instance

    def _filesystem_helpers(self) -> Dict[str, Callable]:
        """``glob`` / ``rglob`` async globals over the overlay.

        Monty's ``pathlib.Path`` lacks ``glob`` / ``rglob`` (there is no
        ``AbstractOS`` hook to add them), so the sandbox surfaces its own
        matching as two async globals returning lists of path strings. A
        bound or per-call function of the same name overrides them.
        """

        async def glob(pattern, root="/"):
            return self.glob(pattern, root=root)

        async def rglob(pattern, root="/"):
            return self.rglob(pattern, root=root)

        return {"glob": glob, "rglob": rglob}

    # -- AbstractOS / overlay filesystem --------------------------------

    @staticmethod
    def _key(path) -> str:
        """Normalize a virtual path to a root-relative POSIX key.

        Flattens ``.`` and ``..`` and strips the leading ``/`` so that
        relative and absolute spellings of the same location collapse to
        one key. ``..`` can never climb above the root.
        """
        parts: List[str] = []
        for part in PurePosixPath(str(path)).parts:
            if part in ("/", "", "."):
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(part)
        return "/".join(parts)

    @staticmethod
    def _join(key: str, name: str) -> str:
        return f"{key}/{name}" if key else name

    def _host_path(self, key: str) -> Optional[Path]:
        """Resolve a key to a real host path, or ``None`` if it escapes."""
        if self._workdir is None:
            return None
        candidate = (self._workdir / key) if key else self._workdir
        try:
            resolved = candidate.resolve()
            resolved.relative_to(self._workdir)
        except (OSError, ValueError):
            return None
        return resolved

    def _content(self, key: str) -> Optional[bytes]:
        """Effective bytes for a key, or ``None`` if missing/deleted.

        Precedence: tombstone, then overlay write, then the in-memory base
        snapshot, then the live host ``workdir``.
        """
        if key in self._tombstones:
            return None
        if key in self._overlay:
            return self._overlay[key]
        if key in self._base_files:
            return self._base_files[key]
        host = self._host_path(key)
        if host is not None and host.is_file():
            try:
                return host.read_bytes()
            except OSError:
                return None
        return None

    def _is_file(self, key: str) -> bool:
        return self._content(key) is not None

    def _is_dir(self, key: str) -> bool:
        if key == "":
            return True  # root always exists
        if key in self._tombstones:
            return False
        if key in self._dirs:
            return True
        prefix = key + "/"
        if any(k.startswith(prefix) for k in self._overlay):
            return True
        if any(k.startswith(prefix) for k in self._base_files):
            return True
        host = self._host_path(key)
        return host is not None and host.is_dir()

    def _write(self, key: str, content: bytes) -> int:
        self._overlay[key] = content
        self._tombstones.discard(key)
        return len(content)

    def _log(self, action: str, key: str, **detail) -> None:
        """Append one mutation to the ordered change journal."""
        entry: Dict[str, Any] = {"action": action, "path": "/" + key}
        entry.update(detail)
        self._journal.append(entry)

    def path_read_text(self, path):
        content = self._content(self._key(path))
        if content is None:
            raise FileNotFoundError(str(path))
        return content.decode("utf-8", errors="replace")

    def path_read_bytes(self, path):
        content = self._content(self._key(path))
        if content is None:
            raise FileNotFoundError(str(path))
        return content

    def path_write_text(self, path, data):
        key = self._key(path)
        kind = "modify" if self._is_file(key) else "create"
        size = self._write(key, data.encode("utf-8"))
        self._log("write", key, kind=kind, size=size)
        return size

    def path_write_bytes(self, path, data):
        key = self._key(path)
        kind = "modify" if self._is_file(key) else "create"
        size = self._write(key, bytes(data))
        self._log("write", key, kind=kind, size=size)
        return size

    def path_exists(self, path):
        key = self._key(path)
        return self._is_file(key) or self._is_dir(key)

    def path_is_file(self, path):
        return self._is_file(self._key(path))

    def path_is_dir(self, path):
        return self._is_dir(self._key(path))

    def path_is_symlink(self, path):
        return False  # the virtual filesystem has no symlinks

    def path_iterdir(self, path):
        key = self._key(path)
        names: set = set()
        host = self._host_path(key)
        if host is not None and host.is_dir():
            try:
                for child in host.iterdir():
                    names.add(child.name)
            except OSError:
                pass
        prefix = f"{key}/" if key else ""
        for k in list(self._overlay) + list(self._dirs) + list(self._base_files):
            if k == key or (prefix and not k.startswith(prefix)):
                continue
            rest = k[len(prefix) :]
            if rest:
                names.add(rest.split("/", 1)[0])
        result = []
        for name in sorted(names):
            childkey = self._join(key, name)
            if childkey in self._tombstones:
                continue
            result.append(PurePosixPath("/" + childkey))
        return result

    def path_mkdir(self, path, *args, **kwargs):
        key = self._key(path)
        self._dirs.add(key)
        self._tombstones.discard(key)
        self._log("mkdir", key)
        return None

    def path_rmdir(self, path):
        key = self._key(path)
        self._dirs.discard(key)
        self._tombstones.add(key)
        self._log("rmdir", key)
        return None

    def path_unlink(self, path):
        key = self._key(path)
        if self._content(key) is None:
            raise FileNotFoundError(str(path))
        self._overlay.pop(key, None)
        self._tombstones.add(key)
        self._log("delete", key)
        return None

    def path_rename(self, src, dst):
        srckey = self._key(src)
        content = self._content(srckey)
        if content is None:
            raise FileNotFoundError(str(src))
        dstkey = self._key(dst)
        self._write(dstkey, content)
        self._overlay.pop(srckey, None)
        self._tombstones.add(srckey)
        self._log("rename", dstkey, src="/" + srckey, size=len(content))
        return None

    def path_stat(self, path):
        key = self._key(path)
        content = self._content(key)
        now = time.time()
        if content is not None:
            mode, size = _REGULAR_FILE_MODE, len(content)
        elif self._is_dir(key):
            mode, size = _DIRECTORY_MODE, 0
        else:
            raise FileNotFoundError(str(path))
        return pydantic_monty.StatResult(
            st_mode=mode,
            st_ino=0,
            st_dev=0,
            st_nlink=1,
            st_uid=0,
            st_gid=0,
            st_size=size,
            st_atime=now,
            st_mtime=now,
            st_ctime=now,
        )

    def path_resolve(self, path):
        return "/" + self._key(path)

    def path_absolute(self, path):
        return "/" + self._key(path)

    def get_environ(self):
        return dict(self._environ)

    def getenv(self, key, default=None):
        return self._environ.get(key, default)

    @property
    def workdir(self) -> Optional[str]:
        """The read-through base directory, or ``None`` (in-memory)."""
        return str(self._workdir) if self._workdir is not None else None

    @property
    def environ(self) -> Dict[str, str]:
        return dict(self._environ)

    # -- host-side filesystem inspection & serialization ----------------

    def changes(self) -> Dict[str, List[str]]:
        """Summary of overlay mutations relative to the base.

        Returns ``{"written": [...], "deleted": [...]}`` with virtual
        paths created/modified and removed in the overlay. Useful to
        review (or diff) what sandboxed code did before optionally
        persisting any of it to the host.
        """
        return {
            "written": sorted("/" + k for k in self._overlay),
            "deleted": sorted("/" + k for k in self._tombstones),
        }

    def journal(self) -> List[Dict[str, Any]]:
        """Ordered log of every mutation performed on the filesystem.

        Where :meth:`changes` is a deduplicated summary of the *final*
        state, this returns one entry per action in the order it
        happened — so repeated writes, renames, and create-then-delete
        sequences are all visible. Each entry has an ``action``
        (``"write"``, ``"delete"``, ``"mkdir"``, ``"rmdir"`` or
        ``"rename"``) and a ``path``; writes additionally carry ``kind``
        (``"create"`` / ``"modify"``) and ``size``, and renames carry the
        origin ``src`` and ``size``. The list is JSON-safe and survives
        :meth:`get_state` / :meth:`set_state`.
        """
        return [dict(entry) for entry in self._journal]

    def read_overlay(self, path) -> Optional[bytes]:
        """Host-side read of the effective content for ``path``."""
        return self._content(self._key(path))

    @staticmethod
    def _glob_to_regex(pattern: str) -> "re.Pattern":
        """Translate a glob (``*``, ``?``, ``[seq]``, ``**``) to a regex.

        ``*`` matches within a single path segment, ``**`` spans segments,
        and ``**/`` additionally matches zero leading directories so that
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

    def _walk(self, base_key: str):
        """Yield every descendant key of ``base_key`` (merged view)."""
        for child in self.path_iterdir("/" + base_key if base_key else "/"):
            key = self._key(str(child))
            yield key
            if self._is_dir(key):
                yield from self._walk(key)

    def glob(self, pattern: str, root: str = "/") -> List[str]:
        """Merged paths under ``root`` matching a glob ``pattern``.

        Searches the same base+overlay view code sees, skipping tombstoned
        paths, and returns sorted absolute virtual paths (files and
        directories). ``*`` stays within a segment; use ``**`` to cross
        directories (or :meth:`rglob` for a recursive search).
        """
        root_key = self._key(root)
        regex = self._glob_to_regex(pattern)
        prefix = root_key + "/" if root_key else ""
        out = []
        for key in self._walk(root_key):
            rel = key[len(prefix) :] if prefix else key
            if regex.match(rel):
                out.append("/" + key)
        return sorted(out)

    def rglob(self, pattern: str, root: str = "/") -> List[str]:
        """Recursive :meth:`glob`: ``rglob(p)`` is ``glob("**/" + p)``."""
        return self.glob("**/" + pattern, root=root)

    def _base_snapshot(self) -> Dict[str, bytes]:
        """Materialize every read-through base file into a key -> bytes map.

        Combines the existing in-memory base with a fresh read of the host
        ``workdir`` (symlinks skipped, matching the no-escape policy), so the
        result can rebuild the base without the workdir on disk.
        """
        base = dict(self._base_files)
        if self._workdir is not None and self._workdir.is_dir():
            for path in self._workdir.rglob("*"):
                if path.is_symlink() or not path.is_file():
                    continue
                key = path.relative_to(self._workdir).as_posix()
                if key not in base:
                    try:
                        base[key] = path.read_bytes()
                    except OSError:
                        pass
        return base

    def get_state(self, snapshot_base: bool = False) -> dict:
        """JSON-safe snapshot of the filesystem.

        Always captures the overlay (writes, tombstones, mkdir'd dirs,
        journal). With ``snapshot_base=True`` it also materializes the
        read-through base (the ``workdir`` files) so :meth:`set_state` can
        restore the whole filesystem **without** the original ``workdir``
        on disk — at the cost of embedding those file contents.
        """
        state = {
            "overlay": {
                k: base64.b64encode(v).decode("ascii") for k, v in self._overlay.items()
            },
            "tombstones": sorted(self._tombstones),
            "dirs": sorted(self._dirs),
            "journal": [dict(entry) for entry in self._journal],
        }
        if snapshot_base:
            state["base"] = {
                k: base64.b64encode(v).decode("ascii")
                for k, v in self._base_snapshot().items()
            }
        return state

    def set_state(self, state: dict) -> None:
        """Restore filesystem state produced by :meth:`get_state`.

        Restores the overlay, and — if the state carries a ``base``
        snapshot (from ``get_state(snapshot_base=True)``) — the base files
        too, so reads succeed even when the host ``workdir`` is absent.
        """
        self._overlay = {
            k: base64.b64decode(v) for k, v in (state.get("overlay") or {}).items()
        }
        self._tombstones = set(state.get("tombstones") or [])
        self._dirs = set(state.get("dirs") or [])
        self._journal = [dict(entry) for entry in (state.get("journal") or [])]
        self._base_files = {
            k: base64.b64decode(v) for k, v in (state.get("base") or {}).items()
        }

    # -- tool methods ---------------------------------------------------
    #
    # Overlay-backed implementations of the base ``Sandbox`` file tools,
    # so an agent given these tools (via ``synalinks.Tool``) can explore
    # and edit the mounted workdir host-safe. They return plain dicts; the
    # caller wraps them.

    async def list_files(
        self, pattern: str = "**/*", offset: int = 1, limit: int = 0
    ) -> dict:
        """List files in the sandbox filesystem matching a glob pattern.

        Searches the merged overlay view (workdir + overlay writes minus
        deletions), the same files sandboxed code sees. Directories are
        omitted; results are paginated.

        Args:
            pattern (str): Glob pattern, e.g. ``'**/*.py'`` (``**`` crosses
                directories). Defaults to ``'**/*'`` (every file).
            offset (int): 1-based index of the first path to return
                (``1`` = the first). Defaults to 1.
            limit (int): Maximum number of paths to return; 0 (the default)
                returns all remaining.

        Returns:
            dict: ``files`` (this page of path strings), ``total``,
            ``offset`` and ``truncated``.
        """
        files = [p for p in self.glob(pattern) if self.path_is_file(p)]
        page, truncated = _paginate(files, offset, limit)
        return {
            "files": page,
            "total": len(files),
            "offset": max(offset, 1),
            "truncated": truncated,
        }

    async def read_file(self, path: str, offset: int = 1, limit: int = 0) -> dict:
        """Read a text file from the sandbox filesystem, by line range.

        Args:
            path (str): Absolute virtual path, e.g. ``'/src/main.py'``.
            offset (int): 1-based line number to start reading from
                (``1`` = the first line, grep convention). Defaults to 1.
            limit (int): Maximum number of lines to return; 0 (the default)
                returns all remaining lines.

        Returns:
            dict: ``content`` (the requested lines), ``start_line`` /
            ``end_line`` (1-based, inclusive), ``total_lines`` and
            ``truncated`` — or ``error`` if the file is missing.
        """
        content = self.read_overlay(path)
        if content is None:
            return {"error": f"file not found: {path}"}
        lines = content.decode("utf-8", errors="replace").splitlines(keepends=True)
        page, truncated = _paginate(lines, offset, limit)
        start = max(offset, 1)
        return {
            "content": "".join(page),
            "start_line": start,
            "end_line": start + len(page) - 1,  # 1-based, inclusive
            "total_lines": len(lines),
            "truncated": truncated,
        }

    async def write_file(self, path: str, content: str) -> dict:
        """Write a text file in the sandbox filesystem.

        The write lands in the in-memory overlay only — the real host
        workdir is never modified.

        Args:
            path (str): Absolute virtual path to write, e.g. ``'/PLAN.md'``.
            content (str): The text to write.

        Returns:
            dict: ``written`` (the path) and ``bytes`` (count written).
        """
        n = self.path_write_text(path, content)
        return {"written": path, "bytes": n}

    async def edit_file(
        self, path: str, old: str, new: str, replace_all: bool = False
    ) -> dict:
        """Replace text in a file in the sandbox filesystem (overlay only).

        Args:
            path (str): Absolute virtual path of the file to edit.
            old (str): The exact text to replace. Must occur exactly once
                unless ``replace_all`` is true.
            new (str): The text to replace it with.
            replace_all (bool): Replace every occurrence instead of
                requiring a unique match. Defaults to false.

        Returns:
            dict: ``path`` and ``replacements`` (count made), or ``error``
            if the file is missing, or ``old`` is empty / absent / not
            unique.
        """
        content = self.read_overlay(path)
        if content is None:
            return {"error": f"file not found: {path}"}
        if not old:
            return {"error": "`old` must be a non-empty string"}
        text = content.decode("utf-8", errors="replace")
        occurrences = text.count(old)
        if occurrences == 0:
            return {"error": f"`old` text not found in {path}"}
        if occurrences > 1 and not replace_all:
            return {
                "error": (
                    f"`old` is not unique in {path} ({occurrences} occurrences); "
                    "add surrounding context or set replace_all=True"
                )
            }
        replacements = occurrences if replace_all else 1
        new_text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        self.path_write_text(path, new_text)
        return {"path": path, "replacements": replacements}

    async def search_files(
        self, pattern: str, glob: str = "**/*", offset: int = 1, limit: int = 100
    ) -> dict:
        """Search file contents for a regex across files matching a glob.

        Combines a glob (which files) with a grep-style regex (matched per
        line over the merged overlay view). Results are paginated.

        Args:
            pattern (str): Regular expression to search for in file contents.
            glob (str): Glob selecting which files to search, e.g.
                ``'**/*.py'``. Defaults to ``'**/*'`` (all files).
            offset (int): 1-based index of the first match to return
                (``1`` = the first). Defaults to 1.
            limit (int): Maximum number of matches to return; 0 returns all.
                Defaults to 100.

        Returns:
            dict: ``matches`` (a page of ``{path, line, text}`` records with
            1-based line numbers), ``total``, ``offset`` and ``truncated``
            — or ``error`` on a bad regex.
        """
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return {"error": f"invalid regex: {exc}"}
        matches: List[Dict[str, Any]] = []
        for path in self.glob(glob):
            if not self.path_is_file(path):
                continue
            content = self.read_overlay(path)
            if content is None:
                continue
            text = content.decode("utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matches.append({"path": path, "line": lineno, "text": line})
        page, truncated = _paginate(matches, offset, limit)
        return {
            "matches": page,
            "total": len(matches),
            "offset": max(offset, 1),
            "truncated": truncated,
        }

    async def run_python_file(self, path: str) -> dict:
        """Run a Python script file from the sandbox filesystem.

        Reads ``path`` (a script written with :meth:`write_file`) from the
        overlay and executes its contents in the sandbox, sharing the REPL
        namespace with prior runs. Use this to run a self-contained script
        you built — the sandbox cannot ``import`` other overlay files, so
        the script must stand alone.

        Args:
            path (str): Absolute virtual path of the ``.py`` file to run.

        Returns:
            dict: ``ok`` (bool), ``stdout`` and ``stderr`` (captured
            output), and ``error`` (a message string, or null on success)
            — or ``error`` if the file is missing.
        """
        content = self.read_overlay(path)
        if content is None:
            return {"error": f"file not found: {path}"}
        return await self.run_python_code(content.decode("utf-8", errors="replace"))
