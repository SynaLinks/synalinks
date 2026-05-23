# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable


@synalinks_export(
    [
        "synalinks.sandboxes.ExecutionResult",
        "synalinks.ExecutionResult",
    ]
)
class ExecutionResult(DataModel):
    """Structured result of running a snippet in a ``Sandbox``."""

    stdout: str = Field(
        default="",
        description="Concatenated bytes written to stdout by the snippet.",
    )
    stderr: str = Field(
        default="",
        description="Concatenated bytes written to stderr by the snippet.",
    )
    result: Optional[Any] = Field(
        default=None,
        description="Value of the snippet's last expression, or null.",
    )
    error: Optional[str] = Field(
        default=None,
        description=(
            "Formatted error message when execution failed, else null. "
            "A non-null `error` means stdout / stderr / result captured "
            "whatever happened before the failure."
        ),
    )

    @property
    def ok(self) -> bool:
        """True when the snippet ran to completion without an error."""
        return self.error is None


@synalinks_export(
    [
        "synalinks.sandboxes.Sandbox",
        "synalinks.Sandbox",
    ]
)
class Sandbox(SynalinksSaveable):
    """Abstract base class for code execution sandboxes.

    !!! warning "Experimental"
        The sandbox API is experimental and may change in a future
        release.

    A sandbox is a **stateful**, **restricted** Python environment:
    subsequent ``run`` calls see variables, imports and function
    definitions from previous runs.

    ## The contract

    A backend (Monty REPL, Pyodide, Docker, subprocess) is defined by
    overriding these primitives:

    - `run` — execute a snippet, return an `ExecutionResult`.
    - `reset` — wipe execution state back to empty.
    - `dump` / `load` — serialize / restore the namespace as
      an opaque byte string.
    - `get_config` / `from_config` / `_obj_type` — the
      JSON-safe round-trip for Synalinks' saving pipeline.

    Everything else here is **provided** machinery that every backend
    shares, so subclasses neither reimplement nor diverge on it:

    - **Run history** (`history`) — an ordered, JSON-safe log of the
      code each `run` executed and its outcome. Implementations
      record an entry by routing their result through `_record_run`,
      and drop it on `reset` via `clear_history`.
    - **Bound functions** (`bind_functions`, `bound_functions`)
      — host callables exposed inside the sandbox, set once and reused on
      every run. Implementations read `_functions` when dispatching.
    - **Tool methods** (`run_python_code`, `run_python_file`,
      `list_files`, `read_file`, `write_file`,
      `edit_file`, `search_files`) — async, dict-returning
      methods with public names a caller can wrap with ``synalinks.Tool``
      to give an agent. ``run_python_code`` works on any backend; the
      filesystem methods (including ``run_python_file``, which runs a script
      file) default to a "no filesystem" error and are overridden by
      backends that mount one.
      Listing / reading / searching are paginated with a 1-based ``offset``
      and a ``limit`` (grep convention — line numbers are 1-based too) so
      large results stay bounded for a language model.

    ## Ownership

    Ownership is the **caller's** responsibility: construct a sandbox,
    hand it to a code-executing module (e.g. ``RecursiveLanguageModelAgent``) across
    successive interactive turns, and build a new one for a fresh
    conversation. The consuming module stays stateless.

    Args:
        timeout (float): Per-snippet execution budget in seconds
            (Default 5). Backends that cannot enforce this should treat
            it as advisory; modules that instantiate sandboxes (e.g.
            ``RecursiveLanguageModelAgent``) pass this through.
        name (str): Optional. Human-readable name for the sandbox.
        external_functions (dict): Optional. ``name -> callable`` mapping
            bound persistently and exposed inside the sandbox on every
            run (see `bind_functions`). How a backend surfaces them
            is backend-specific; the binding itself is shared here.
    """

    # A natural-language description of the sandbox's constraints,
    # intended for inclusion in LM prompts. Consumers (e.g.
    # ``RecursiveLanguageModelAgent``, ``PythonSynthesis``) compose this text into
    # their instructions or schema descriptions so the language model /
    # optimizer knows which stdlib subset, builtins, and language
    # features are available. Subclasses override with a concise,
    # prompt-friendly description of what code they can run.
    description: str = ""

    def __init__(
        self,
        timeout: float = 5.0,
        name: Optional[str] = None,
        *,
        external_functions: Optional[Dict[str, Callable]] = None,
    ):
        self.timeout = float(timeout)
        self.name = name
        self._history: List[Dict[str, Any]] = []
        self._functions: Dict[str, Callable] = dict(external_functions or {})

    # -- execution primitives (abstract) --------------------------------

    async def run(
        self,
        code: str,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        external_functions: Optional[Dict[str, Callable]] = None,
    ) -> ExecutionResult:
        """Execute ``code`` and return a structured result.

        Implementations should route their result through
        `_record_run` so the snippet lands in `history`, and
        expose `_functions` (merged with any per-call
        ``external_functions``) as callables inside the sandbox.

        Args:
            code (str): The Python source to run.
            inputs (dict): Optional. Variables bound into the sandbox
                namespace before execution. Backend-specific rules apply
                (some backends may only honour this on the first call).
            external_functions (dict): Optional. Mapping of name → async
                callable exposed as global functions inside the sandbox
                for this call, on top of the persistently bound set.

        Returns:
            ExecutionResult: stdout, stderr, last-expression result and
            (if any) an error string.
        """
        raise NotImplementedError("Sandbox subclasses must implement `run`.")

    def reset(self) -> None:
        """Wipe execution state and start over with an empty sandbox.

        Implementations should also drop the run history (call
        `clear_history`); bound functions are configuration and
        persist across a reset.
        """
        raise NotImplementedError("Sandbox subclasses must implement `reset`.")

    def dump(self) -> bytes:
        """Serialize the current sandbox state to bytes.

        The blob returned must be self-contained enough that a subsequent
        ``load(blob)`` reconstructs an equivalent sandbox, including its
        namespace (variables, imports, user-defined functions).
        """
        raise NotImplementedError("Sandbox subclasses must implement `dump`.")

    @classmethod
    def load(cls, data: bytes, **kwargs) -> "Sandbox":
        """Restore a sandbox from bytes produced by ``dump()``."""
        raise NotImplementedError("Sandbox subclasses must implement `load`.")

    # -- serialization primitives (abstract) ----------------------------

    def get_config(self) -> dict:
        """Return a JSON-safe config that ``from_config`` can rebuild from."""
        raise NotImplementedError("Sandbox subclasses must implement `get_config`.")

    @classmethod
    def from_config(cls, config: dict) -> "Sandbox":
        """Rebuild a sandbox from a `get_config` dict."""
        raise NotImplementedError("Sandbox subclasses must implement `from_config`.")

    # -- branching (filesystem backends) --------------------------------
    #
    # A git-like contract for isolating filesystem mutations: ``fork`` a
    # sandbox to get an isolated child that sees the parent's files but
    # whose writes never touch the parent, ``diff`` to review what a
    # (forked) sandbox changed, and ``merge`` to fold a child's changes
    # back into a parent. Backends without a filesystem need not implement
    # these; ``MontySandbox`` does, on top of its copy-on-write overlay.

    def fork(self, *, name: Optional[str] = None) -> "Sandbox":
        """Return an isolated copy that shares this sandbox's current state.

        The child starts seeing exactly the files this sandbox sees now,
        but its mutations are isolated: writing, editing or deleting in the
        child never affects the parent (and vice versa). Use this to hand a
        subagent its own branch of the filesystem; review its work with
        `diff` and optionally fold it back with `merge`.
        """
        raise NotImplementedError("This sandbox does not support `fork`.")

    def diff(self) -> dict:
        """Summarize the filesystem changes this sandbox made since its base.

        For a sandbox produced by `fork`, this is exactly what the
        child changed relative to the fork point — the patch `merge`
        would apply. Returns a JSON-safe summary (written paths with a
        ``kind`` / ``size``, and deleted paths).
        """
        raise NotImplementedError("This sandbox does not support `diff`.")

    def merge(
        self,
        other: "Sandbox",
        *,
        paths: Optional[List[str]] = None,
        force: bool = False,
        repl: bool = False,
    ) -> dict:
        """Apply another (typically forked) sandbox's changes onto this one.

        Replays ``other``'s writes and deletions into this sandbox as if
        they were performed here. A *conflicting* path — one this sandbox
        also changed since the fork — is **refused** (left untouched and
        reported) unless ``force`` is set, in which case ``other``'s version
        is applied (last writer wins). ``paths`` optionally restricts the
        merge to a chosen subset of virtual paths. With ``repl=True`` the
        backend also adopts ``other``'s whole execution-state namespace
        (where it has one). Returns a JSON-safe report of what was applied,
        what conflicted, and what was skipped.
        """
        raise NotImplementedError("This sandbox does not support `merge`.")

    # -- run history (provided) -----------------------------------------

    def history(self) -> List[Dict[str, Any]]:
        """Ordered, JSON-safe log of snippets executed via `run`.

        Each entry records the ``code`` that ran and its outcome
        (``ok``, ``stdout``, ``stderr``, ``error``), in execution order,
        for inspection or replay. Returns a defensive copy; cleared by
        `clear_history` / `reset`.
        """
        return [dict(entry) for entry in self._history]

    def clear_history(self) -> None:
        """Drop all recorded run history."""
        self._history = []

    def _record_run(self, code: str, result: ExecutionResult) -> ExecutionResult:
        """Append a standard history entry for a finished run; returns ``result``.

        The raw last-expression ``result.result`` is intentionally not
        stored — it may not be JSON-safe and would break ``get_config``.
        Subclasses call this from `run` and return its value.
        """
        self._history.append(
            {
                "code": code,
                "ok": result.ok,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": result.error,
            }
        )
        return result

    # -- bound functions (provided) -------------------------------------

    @property
    def bound_functions(self) -> Dict[str, Callable]:
        """Copy of the persistently bound ``name -> callable`` mapping."""
        return dict(self._functions)

    def bind_functions(self, functions: Dict[str, Callable]) -> None:
        """Persistently expose ``functions`` inside the sandbox.

        Each ``name -> callable`` is merged into the bound set and made
        available on every subsequent `run`, so a recurring toolset
        need not be re-passed via ``external_functions`` each call.
        Re-binding a name replaces it. Bound functions survive
        `reset` but are not serialized (callables are not JSON-safe).
        """
        self._functions.update(functions)

    # -- tool methods ---------------------------------------------------
    #
    # Async, fully-documented, dict-returning methods with public names —
    # shaped so a caller can hand one straight to ``synalinks.Tool`` (and
    # then a ``FunctionCallingAgent``). The sandbox does not wrap them
    # itself; it just exposes capabilities an agent can be given. The file
    # methods default to a "no filesystem" error here; backends that mount
    # one (e.g. ``MontySandbox`` with a ``workdir``) override them.

    async def run_python_code(self, code: str) -> dict:
        """Execute Python code inside the sandbox and report the outcome.

        State persists across calls (variables, imports and definitions
        from earlier executions are visible).

        Args:
            code (str): The Python source to execute.

        Returns:
            dict: ``ok`` (bool), ``stdout`` and ``stderr`` (captured
            output), and ``error`` (a message string, or null on success).
        """
        result = await self.run(code)
        return {
            "ok": result.ok,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.error,
        }

    async def run_python_file(self, path: str) -> dict:
        """Run a Python script file from the sandbox filesystem.

        Reads ``path`` (a script written with `write_file`) and
        executes its contents in the sandbox. Use this to run a
        self-contained script you built — the sandbox cannot ``import``
        other files from the filesystem, so the script must stand alone.

        Args:
            path (str): Absolute virtual path of the ``.py`` file to run.

        Returns:
            dict: ``ok`` (bool), ``stdout`` and ``stderr`` (captured
            output), and ``error`` (a message string, or null on success)
            — or ``error`` if the file is missing / this sandbox has no
            filesystem.
        """
        return {"error": "this sandbox has no filesystem"}

    async def list_files(
        self, pattern: str = "**/*", offset: int = 1, limit: int = 0
    ) -> dict:
        """List files in the sandbox filesystem matching a glob pattern.

        Args:
            pattern (str): Glob pattern, e.g. ``'**/*.py'`` (``**`` crosses
                directories). Defaults to ``'**/*'`` (every file).
            offset (int): 1-based index of the first path to return
                (``1`` = the first). Defaults to 1.
            limit (int): Maximum number of paths to return; 0 (the default)
                returns all remaining.

        Returns:
            dict: ``files`` (the matching path strings for this page),
            ``total`` (full match count), ``offset`` and ``truncated``
            (whether more remain) — or ``error`` when this sandbox has no
            filesystem.
        """
        return {"error": "this sandbox has no filesystem"}

    async def read_file(self, path: str, offset: int = 1, limit: int = 0) -> dict:
        """Read a text file from the sandbox filesystem, by line range.

        Args:
            path (str): Absolute virtual path, e.g. ``'/src/main.py'``.
            offset (int): 1-based line number to start reading from
                (``1`` = the first line, grep convention). Defaults to 1.
            limit (int): Maximum number of lines to return; 0 (the default)
                returns all remaining lines.

        Returns:
            dict: ``content`` (the requested lines), ``start_line`` and
            ``end_line`` (1-based, inclusive), ``total_lines`` and
            ``truncated`` — or ``error`` if the file is missing / this
            sandbox has no filesystem.
        """
        return {"error": "this sandbox has no filesystem"}

    async def write_file(
        self,
        path: str,
        content: str,
    ) -> dict:
        """Write a text file in the sandbox filesystem.

        Args:
            path (str): Absolute virtual path to write, e.g. ``'/PLAN.md'``.
            content (str): The text to write.

        Returns:
            dict: ``written`` (the path) and ``bytes`` (count written), or
            ``error`` when this sandbox has no filesystem.
        """
        return {"error": "this sandbox has no filesystem"}

    async def edit_file(
        self,
        path: str,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> dict:
        """Replace text in a file in the sandbox filesystem.

        Args:
            path (str): Absolute virtual path of the file to edit.
            old (str): The exact text to replace. Must occur exactly once
                unless ``replace_all`` is true.
            new (str): The text to replace it with.
            replace_all (bool): Replace every occurrence instead of
                requiring a unique match. Defaults to false.

        Returns:
            dict: ``path`` and ``replacements`` (count made), or ``error``
            if the file is missing, ``old`` is absent / not unique, or
            this sandbox has no filesystem.
        """
        return {"error": "this sandbox has no filesystem"}

    async def search_files(
        self,
        pattern: str,
        glob: str = "**/*",
        offset: int = 1,
        limit: int = 100,
    ) -> dict:
        """Search file contents for a regex across files matching a glob.

        Args:
            pattern (str): Regular expression to search for in file contents
                (matched per line).
            glob (str): Glob selecting which files to search, e.g.
                ``'**/*.py'``. Defaults to ``'**/*'`` (all files).
            offset (int): 1-based index of the first match to return
                (``1`` = the first). Defaults to 1.
            limit (int): Maximum number of matches to return; 0 returns all.
                Defaults to 100.

        Returns:
            dict: ``matches`` (a page of ``{path, line, text}`` records with
            1-based line numbers), ``total`` (full match count), ``offset``
            and ``truncated`` — or ``error`` on a bad regex / when this
            sandbox has no filesystem.
        """
        return {"error": "this sandbox has no filesystem"}

    def _obj_type(self):
        return "Sandbox"
