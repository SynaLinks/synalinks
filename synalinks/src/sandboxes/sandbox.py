# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
import warnings
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable

# -- E2B-compatible result types ------------------------------------------
#
# Mirrors of the types E2B's SDK returns (``e2b_code_interpreter.models`` and
# ``e2b.sandbox``), with the same field names, types and helpers, so code
# written against E2B's ``AsyncSandbox`` reads results the same way here.


@synalinks_export(["synalinks.sandboxes.ExecutionError", "synalinks.ExecutionError"])
@dataclass
class ExecutionError:
    """An exception raised by the executed code (E2B's ``ExecutionError``)."""

    name: str
    """The exception class name, e.g. ``"ValueError"``."""
    value: str
    """The exception message."""
    traceback: str
    """The formatted traceback, trimmed to the executed code's own frames."""

    def to_json(self) -> str:
        return json.dumps(
            {"name": self.name, "value": self.value, "traceback": self.traceback}
        )


@synalinks_export(["synalinks.sandboxes.Result", "synalinks.Result"])
@dataclass
class Result:
    """A displayable value produced by the code (E2B's ``Result``).

    The value of the code's last expression is the *main result*: ``text`` is
    its ``repr`` and ``json`` its JSON value when it is JSON-serializable. The
    rich formats (``html``, ``png``, ...) are kept for E2B parity and are
    ``None`` for this sandbox.
    """

    text: Optional[str] = None
    html: Optional[str] = None
    markdown: Optional[str] = None
    svg: Optional[str] = None
    png: Optional[str] = None
    jpeg: Optional[str] = None
    pdf: Optional[str] = None
    latex: Optional[str] = None
    json: Optional[Any] = None
    javascript: Optional[str] = None
    data: Optional[dict] = None
    chart: Optional[Any] = None
    is_main_result: bool = False
    extra: Optional[dict] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def formats(self) -> List[str]:
        """The formats this result carries, e.g. ``["text", "json"]``."""
        names = [
            "text", "html", "markdown", "svg", "png", "jpeg", "pdf",
            "latex", "json", "javascript", "data", "chart",
        ]  # fmt: skip
        return [n for n in names if getattr(self, n)] + list(self.extra or {})

    def __repr__(self) -> str:
        if self.text:
            return f"Result({self.text})"
        return "Result(Formats: " + ", ".join(self.formats()) + ")"

    __str__ = __repr__


@synalinks_export(["synalinks.sandboxes.Logs", "synalinks.Logs"])
@dataclass(repr=False)
class Logs:
    """Output the code printed, as chunks (E2B's ``Logs``)."""

    stdout: List[str] = field(default_factory=list)
    stderr: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"Logs(stdout: {self.stdout}, stderr: {self.stderr})"

    def to_json(self) -> str:
        return json.dumps({"stdout": self.stdout, "stderr": self.stderr})


@synalinks_export(["synalinks.sandboxes.Execution", "synalinks.Execution"])
@dataclass(repr=False)
class Execution:
    """What `Sandbox.run_code` returns (E2B's ``Execution``).

    ``error`` is set when the code raised; ``logs`` and ``results`` then hold
    whatever happened before the failure.
    """

    results: List[Result] = field(default_factory=list)
    logs: Logs = field(default_factory=Logs)
    error: Optional[ExecutionError] = None
    execution_count: Optional[int] = None

    @property
    def text(self) -> Optional[str]:
        """The ``text`` of the main result (the last expression's ``repr``)."""
        for result in self.results:
            if result.is_main_result:
                return result.text
        return None

    def __repr__(self) -> str:
        return (
            f"Execution(Results: {self.results}, Logs: {self.logs}, Error: {self.error})"
        )

    def to_json(self) -> str:
        return json.dumps(
            {
                "results": [
                    {**{k: r[k] for k in r.formats() if k != "chart"}, "text": r.text}
                    for r in self.results
                ],
                "logs": self.logs.to_json(),
                "error": self.error.to_json() if self.error else None,
            }
        )


@synalinks_export(["synalinks.sandboxes.FileType", "synalinks.FileType"])
class FileType(Enum):
    """Kind of a filesystem entry (E2B's ``FileType``)."""

    FILE = "file"
    DIR = "dir"
    SYMLINK = "symlink"


@synalinks_export(["synalinks.sandboxes.WriteInfo", "synalinks.WriteInfo"])
@dataclass
class WriteInfo:
    """What ``sandbox.files.write`` wrote (E2B's ``WriteInfo``)."""

    name: str
    type: Optional[FileType]
    path: str
    metadata: Optional[Dict[str, str]] = field(default=None, kw_only=True)


@synalinks_export(["synalinks.sandboxes.EntryInfo", "synalinks.EntryInfo"])
@dataclass
class EntryInfo(WriteInfo):
    """A file or directory in the sandbox filesystem (E2B's ``EntryInfo``)."""

    size: int
    mode: int
    permissions: str
    owner: str
    group: str
    modified_time: datetime
    symlink_target: Optional[str] = None


@synalinks_export(["synalinks.sandboxes.CommandResult", "synalinks.CommandResult"])
@dataclass
class CommandResult:
    """Result of ``sandbox.commands.run`` (E2B's ``CommandResult``)."""

    stderr: str
    stdout: str
    exit_code: int
    error: Optional[str]


@synalinks_export(["synalinks.sandboxes.SandboxException", "synalinks.SandboxException"])
class SandboxException(Exception):
    """Base class of the errors the sandbox API raises (E2B's name)."""


@synalinks_export(
    ["synalinks.sandboxes.NotFoundException", "synalinks.NotFoundException"]
)
class NotFoundException(SandboxException, FileNotFoundError):
    """A path does not exist. Also a ``FileNotFoundError``."""


@synalinks_export(["synalinks.sandboxes.TimeoutException", "synalinks.TimeoutException"])
class TimeoutException(SandboxException, TimeoutError):
    """A command ran past its timeout. Also a ``TimeoutError``."""


@synalinks_export(
    ["synalinks.sandboxes.CommandExitException", "synalinks.CommandExitException"]
)
@dataclass
class CommandExitException(SandboxException, CommandResult):
    """Raised by ``commands.run`` when the command exits non-zero, as in E2B.

    Carries the full `CommandResult` (``stdout``, ``stderr``, ``exit_code``).
    """

    def __str__(self) -> str:
        return f"Command exited with code {self.exit_code} and error:\n{self.stderr}"


@synalinks_export(
    [
        "synalinks.sandboxes.ExecutionResult",
        "synalinks.ExecutionResult",
    ]
)
class ExecutionResult(DataModel):
    """Result of the deprecated `Sandbox.run`; `run_code` returns `Execution`."""

    stdout: str = Field(default="", description="Everything written to stdout.")
    stderr: str = Field(
        default="", description="Everything written to stderr, then the traceback."
    )
    result: Optional[Any] = Field(
        default=None, description="JSON value of the last expression, or null."
    )
    error: Optional[str] = Field(
        default=None, description="``Name: message`` of the raised error, or null."
    )

    @property
    def ok(self) -> bool:
        return self.error is None


def flat_output(execution: Execution):
    """``(stdout, stderr, error)`` strings of an `Execution`: stderr ends with
    the traceback and ``error`` is ``"Name: message"`` (or ``None``)."""
    error = execution.error
    stderr = "".join(execution.logs.stderr) + (error.traceback if error else "")
    return (
        "".join(execution.logs.stdout),
        stderr,
        f"{error.name}: {error.value}" if error else None,
    )


class Filesystem:
    """The ``sandbox.files`` namespace, named after E2B's ``Filesystem``.

    Backends with a filesystem subclass this and point
    ``Sandbox.filesystem_class`` at the subclass. This default has no
    filesystem, so every method raises ``NotImplementedError``.
    """

    def __init__(self, sandbox: "Sandbox"):
        self.sandbox = sandbox

    def unsupported(self):
        raise NotImplementedError("This sandbox has no filesystem.")

    async def read(
        self, path: str, format: Literal["text", "bytes"] = "text"
    ) -> Union[str, bytearray]:
        """Read the file at ``path`` as text (default) or ``bytearray``.

        Raises `NotFoundException` when ``path`` does not exist.
        """
        self.unsupported()

    async def write(self, path: str, data: Union[str, bytes]) -> WriteInfo:
        """Write ``data`` to ``path``, creating parent directories."""
        self.unsupported()

    async def list(self, path: str = "/", depth: int = 1) -> List[EntryInfo]:
        """List the entries under the directory ``path``, ``depth`` levels deep."""
        self.unsupported()

    async def exists(self, path: str) -> bool:
        """Whether a file or directory exists at ``path``."""
        self.unsupported()

    async def get_info(self, path: str) -> EntryInfo:
        """Describe the entry at ``path``; raises `NotFoundException` if absent."""
        self.unsupported()

    async def remove(self, path: str) -> None:
        """Delete the file or directory (recursively) at ``path``."""
        self.unsupported()

    async def rename(self, old_path: str, new_path: str) -> EntryInfo:
        """Move ``old_path`` to ``new_path``; returns the new entry."""
        self.unsupported()

    async def make_dir(self, path: str) -> bool:
        """Create ``path`` (and parents); ``False`` if it already existed."""
        self.unsupported()


class Commands:
    """The ``sandbox.commands`` namespace, named after E2B's ``Commands``.

    Backends with a shell subclass this and point
    ``Sandbox.commands_class`` at the subclass. This default has no shell.
    """

    def __init__(self, sandbox: "Sandbox"):
        self.sandbox = sandbox

    async def run(self, cmd: str, timeout: Optional[float] = None) -> CommandResult:
        """Run the shell command ``cmd`` and wait for it to finish.

        As in E2B, a non-zero exit raises `CommandExitException` (which
        carries the `CommandResult`) and running past ``timeout`` raises
        `TimeoutException`.

        Args:
            cmd (str): The shell command line.
            timeout (float): Optional. Seconds before the command is killed;
                ``None`` uses the sandbox's ``timeout``, ``0`` means no limit.
        """
        raise NotImplementedError("This sandbox has no shell.")


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
    subsequent ``run_code`` calls see variables, imports and function
    definitions from previous runs.

    ## E2B-compatible surface

    Method names follow the [E2B](https://e2b.dev/docs) ``AsyncSandbox``
    SDK, so code written against one ports to the other:

    ```python
    sandbox = await synalinks.MirageSandbox.create()
    execution = await sandbox.run_code("x = 1 + 1\nx")
    execution.text             # "2"
    execution.logs.stdout      # ["..."] chunks printed to stdout
    execution.error            # ExecutionError(name, value, traceback) or None
    await sandbox.files.write("/hello.txt", "hi")
    text = await sandbox.files.read("/hello.txt")
    entries = await sandbox.files.list("/")
    result = await sandbox.commands.run("ls -l /")
    await sandbox.kill()
    ```

    ## The contract

    A backend (Mirage, Pyodide, Docker, subprocess) is defined by
    overriding these primitives:

    - `run_code`: execute a snippet, return an `Execution`.
    - `reset`: wipe execution state back to empty.
    - `dump` / `load`: serialize / restore the namespace as
      an opaque byte string.
    - `get_config` / `from_config`: the JSON-safe round-trip for
      Synalinks' saving pipeline.

    Everything else here is **provided** machinery that every backend
    shares, so subclasses neither reimplement nor diverge on it:

    - **Run history** (`history`): an ordered, JSON-safe log of the
      code each `run_code` executed and its outcome. Implementations
      record an entry by routing their result through `record_run`,
      and drop it on `reset` via `clear_history`.
    - **Bound functions** (`bind_functions`, `bound_functions`):
      host callables exposed inside the sandbox, set once and reused on
      every run. Implementations read `functions` when dispatching.
    - **Tool methods** (`run_python_code`, `run_python_file`,
      `list_files`, `read_file`, `write_file`,
      `edit_file`, `search_files`): async, dict-returning
      methods with public names a caller can wrap with ``synalinks.Tool``
      to give an agent. ``run_python_code`` works on any backend; the
      filesystem methods (including ``run_python_file``, which runs a script
      file) default to a "no filesystem" error and are overridden by
      backends that mount one.
      Listing / reading / searching are paginated with a 1-based ``offset``
      and a ``limit`` (grep convention; line numbers are 1-based too) so
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

    # The ``files`` / ``commands`` namespaces. Backends with a filesystem or
    # a shell override these with their own subclasses.
    filesystem_class = Filesystem
    commands_class = Commands

    def __init__(
        self,
        timeout: float = 5.0,
        name: Optional[str] = None,
        *,
        external_functions: Optional[Dict[str, Callable]] = None,
    ):
        self.timeout = float(timeout)
        self.name = name
        self.run_history: List[Dict[str, Any]] = []
        self.functions: Dict[str, Callable] = dict(external_functions or {})

    # -- lifecycle (E2B-compatible) -------------------------------------

    @classmethod
    async def create(cls, **kwargs) -> "Sandbox":
        """Construct a sandbox; ``kwargs`` go to the constructor.

        The async factory E2B code expects (``await Sandbox.create()``).
        Calling the constructor directly is equivalent.
        """
        return cls(**kwargs)

    async def kill(self) -> bool:
        """Shut the sandbox down and release its resources.

        Returns ``True``. Backends holding resources (mounts, temp dirs)
        override this to release them.
        """
        self.killed = True
        return True

    async def is_running(self) -> bool:
        """Whether the sandbox is still usable (``kill`` not yet called)."""
        return not getattr(self, "killed", False)

    @property
    def files(self) -> Filesystem:
        """The filesystem namespace: ``read``, ``write``, ``list``, ``exists``,
        ``get_info``, ``remove``, ``rename``, ``make_dir``."""
        return self.filesystem_class(self)

    @property
    def commands(self) -> Commands:
        """The shell namespace: ``run``."""
        return self.commands_class(self)

    # -- execution primitives (abstract) --------------------------------

    async def run_code(
        self,
        code: str,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        external_functions: Optional[Dict[str, Callable]] = None,
    ) -> ExecutionResult:
        """Execute ``code`` and return a structured result.

        Implementations should route their result through
        `record_run` so the snippet lands in `history`, and
        expose `functions` (merged with any per-call
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
            Execution: the printed output as ``logs``, the last expression as
            the main entry of ``results``, and ``error`` if the code raised.
        """
        if type(self).run is Sandbox.run:
            raise NotImplementedError("Sandbox subclasses must implement `run_code`.")
        # A backend written before `run_code` existed implements `run` and
        # returns the old `ExecutionResult`.
        old = await self.run(code, inputs=inputs, external_functions=external_functions)
        error = None
        if old.error is not None:
            name, _, value = old.error.partition(": ")
            error = ExecutionError(name=name, value=value, traceback=old.stderr)
        return Execution(
            results=(
                [Result(text=repr(old.result), json=old.result, is_main_result=True)]
                if old.result is not None
                else []
            ),
            logs=Logs(stdout=[old.stdout] if old.stdout else []),
            error=error,
        )

    async def run(self, code: str, **kwargs) -> ExecutionResult:
        """Deprecated: use `run_code`, which returns an E2B-style `Execution`."""
        warnings.warn(
            "`Sandbox.run` is deprecated, use `Sandbox.run_code` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        execution = await self.run_code(code, **kwargs)
        main = next((r for r in execution.results if r.is_main_result), None)
        stdout, stderr, error = flat_output(execution)
        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            result=main.json if main else None,
            error=error,
        )

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
    # these; ``MirageSandbox`` does, on top of its mounted filesystem.

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
        child changed relative to the fork point: the patch `merge`
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
        they were performed here. A *conflicting* path, one this sandbox
        also changed since the fork, is **refused** (left untouched and
        reported) unless ``force`` is set, in which case ``other``'s version
        is applied (last writer wins). ``paths`` optionally restricts the
        merge to a chosen subset of virtual paths. With ``repl=True`` the
        backend also adopts ``other``'s whole execution-state namespace
        (where it has one). Returns a JSON-safe report of what was applied,
        what conflicted, what was skipped, and what failed to apply.
        """
        raise NotImplementedError("This sandbox does not support `merge`.")

    def save(self, path: str) -> dict:
        """Save the sandbox's current filesystem to a ``.zip`` on the host.

        Exports the virtual files the sandbox exposes (the merged view an
        agent sees, e.g. after mutating a ``workdir``) into a zip archive at
        ``path`` on the real filesystem, so a caller can persist what an agent
        produced. Archive members are the virtual paths with the leading
        ``/`` removed, so unzipping reproduces the tree. Backends without a
        filesystem do not implement this.

        Args:
            path (str): Host path of the archive to write. A ``.zip`` suffix
                is appended when missing.

        Returns:
            dict: ``path`` (the resolved archive path) and ``files`` (the
            number of files written).
        """
        raise NotImplementedError("This sandbox does not support `save`.")

    # -- run history (provided) -----------------------------------------

    def history(self) -> List[Dict[str, Any]]:
        """Ordered, JSON-safe log of snippets executed via `run_code`.

        Each entry records the ``code`` that ran and its outcome
        (``ok``, ``stdout``, ``stderr``, ``error``), in execution order,
        for inspection or replay. Returns a defensive copy; cleared by
        `clear_history` / `reset`.
        """
        return [dict(entry) for entry in self.run_history]

    def clear_history(self) -> None:
        """Drop all recorded run history."""
        self.run_history = []

    def record_run(self, code: str, execution: Execution) -> Execution:
        """Append a history entry for a finished run; returns ``execution``.

        Also numbers the run (``execution_count``, as a notebook kernel
        does, restarting at 1 after `reset`). The results are intentionally
        not stored: they may not be JSON-safe and would break ``get_config``.
        Subclasses call this from `run_code` and return its value.
        """
        stdout, stderr, error = flat_output(execution)
        self.run_history.append(
            {
                "code": code,
                "ok": error is None,
                "stdout": stdout,
                "stderr": stderr,
                "error": error,
            }
        )
        execution.execution_count = len(self.run_history)
        return execution

    # -- bound functions (provided) -------------------------------------

    @property
    def bound_functions(self) -> Dict[str, Callable]:
        """Copy of the persistently bound ``name -> callable`` mapping."""
        return dict(self.functions)

    def bind_functions(self, functions: Dict[str, Callable]) -> None:
        """Persistently expose ``functions`` inside the sandbox.

        Each ``name -> callable`` is merged into the bound set and made
        available on every subsequent `run_code`, so a recurring toolset
        need not be re-passed via ``external_functions`` each call.
        Re-binding a name replaces it. Bound functions survive
        `reset` but are not serialized (callables are not JSON-safe).
        """
        self.functions.update(functions)

    # -- tool methods ---------------------------------------------------
    #
    # Async, fully-documented, dict-returning methods with public names,
    # shaped so a caller can hand one straight to ``synalinks.Tool`` (and
    # then a ``FunctionCallingAgent``). The sandbox does not wrap them
    # itself; it just exposes capabilities an agent can be given. The file
    # methods default to a "no filesystem" error here; backends that mount
    # one (e.g. ``MirageSandbox`` with a ``workdir``) override them.

    async def run_python_code(self, code: str) -> dict:
        """Execute Python code inside the sandbox and report the outcome.

        State persists across calls (variables, imports and definitions
        from earlier executions are visible).

        Args:
            code (str): The Python source to execute.

        Returns:
            dict: ``ok`` (bool), ``stdout`` and ``stderr`` (captured
            output, with the traceback when the code raised), and ``error``
            (``"ErrorName: message"``, or null on success).
        """
        stdout, stderr, error = flat_output(await self.run_code(code))
        return {"ok": error is None, "stdout": stdout, "stderr": stderr, "error": error}

    async def run_python_file(self, path: str) -> dict:
        """Run a Python script file from the sandbox filesystem.

        Reads ``path`` (a script written with `write_file`) and
        executes its contents in the sandbox. Use this to run a
        self-contained script you built: the sandbox cannot ``import``
        other files from the filesystem, so the script must stand alone.

        Args:
            path (str): Absolute virtual path of the ``.py`` file to run.

        Returns:
            dict: ``ok`` (bool), ``stdout`` and ``stderr`` (captured
            output), and ``error`` (a message string, or null on success);
            or ``error`` if the file is missing / this sandbox has no
            filesystem.
        """
        return {"error": "this sandbox has no filesystem"}

    async def run_bash(self, command: str) -> dict:
        """Run a shell command in the sandbox, if it provides a shell.

        Backends with a real shell (e.g. ``MirageSandbox``) override this to
        execute ``command`` against the mounted filesystem; the default has no
        shell and returns an ``error``.

        Args:
            command (str): The shell command line to execute.

        Returns:
            dict: ``ok`` (bool), ``stdout``, ``stderr`` and ``exit_code``, or
            ``error`` when this sandbox has no shell.
        """
        return {"error": "this sandbox has no shell"}

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
            (whether more remain), or ``error`` when this sandbox has no
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
            ``truncated``, or ``error`` if the file is missing / this
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
            and ``truncated``, or ``error`` on a bad regex / when this
            sandbox has no filesystem.
        """
        return {"error": "this sandbox has no filesystem"}

    def _obj_type(self):
        return "Sandbox"
