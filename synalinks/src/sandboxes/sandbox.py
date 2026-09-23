# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
import dataclasses
import inspect
import json
import logging
import os
import resource
import shutil
import sys
import tempfile
import time
import uuid
import warnings
import weakref
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from enum import Enum
from typing import IO
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import TypedDict
from typing import Union

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.python_utils import class_method_variant

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

    @classmethod
    def from_dict(cls, payload: Dict) -> "WriteInfo":
        """Build from a plain dict (``type`` may be a `FileType` or its value)."""
        values = dict(payload)
        if values.get("type") is not None and not isinstance(values["type"], FileType):
            values["type"] = FileType(values["type"])
        if isinstance(values.get("modified_time"), str):
            values["modified_time"] = datetime.fromisoformat(values["modified_time"])
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in values.items() if k in names})


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


@synalinks_export("synalinks.sandboxes.PtySize")
@dataclass
class PtySize:
    """Terminal size for ``sandbox.pty`` (E2B's ``PtySize``)."""

    rows: int
    cols: int


@synalinks_export("synalinks.sandboxes.WriteEntry")
class WriteEntry(TypedDict):
    """One file for ``sandbox.files.write_files`` (E2B's ``WriteEntry``)."""

    path: str
    data: Union[str, bytes, IO]


@synalinks_export("synalinks.sandboxes.FilesystemEventType")
class FilesystemEventType(Enum):
    """What happened to a watched path (E2B's ``FilesystemEventType``)."""

    CHMOD = "chmod"
    CREATE = "create"
    REMOVE = "remove"
    RENAME = "rename"
    WRITE = "write"


@synalinks_export("synalinks.sandboxes.FilesystemEvent")
@dataclass
class FilesystemEvent:
    """A change seen by ``sandbox.files.watch_dir`` (E2B's ``FilesystemEvent``).

    ``name`` is relative to the watched directory; ``entry`` is set when the
    watch asked for ``include_entry`` and the path still exists.
    """

    name: str
    type: FilesystemEventType
    entry: Optional[EntryInfo] = None


@synalinks_export("synalinks.sandboxes.ProcessInfo")
@dataclass
class ProcessInfo:
    """A running background command (E2B's ``ProcessInfo``)."""

    pid: int
    tag: Optional[str]
    cmd: str
    args: List[str]
    envs: Dict[str, str]
    cwd: Optional[str]


@synalinks_export("synalinks.sandboxes.OutputMessage")
@dataclass
class OutputMessage:
    """One line a snippet printed, for ``run_code``'s ``on_stdout`` /
    ``on_stderr`` callbacks (E2B's ``OutputMessage``)."""

    line: str
    timestamp: int
    """Nanoseconds since the epoch."""
    error: bool = False

    def __str__(self) -> str:
        return self.line


@synalinks_export("synalinks.sandboxes.Context")
@dataclass
class Context:
    """An isolated code context: its own namespace and working directory
    (E2B's ``Context``). Create one with ``create_code_context``."""

    id: str
    language: str
    cwd: str

    @classmethod
    def from_json(cls, data: Dict[str, str]) -> "Context":
        return cls(id=data["id"], language=data["language"], cwd=data["cwd"])


@synalinks_export("synalinks.sandboxes.SandboxState")
class SandboxState(str, Enum):
    """Lifecycle state of a sandbox (E2B's ``SandboxState``)."""

    RUNNING = "running"
    PAUSED = "paused"


@synalinks_export("synalinks.sandboxes.SandboxInfo")
@dataclass
class SandboxInfo:
    """What ``get_info`` / ``list`` report about a sandbox (E2B's ``SandboxInfo``).

    A local sandbox has no domain, no remote daemon and no volumes, so those
    fields are ``None`` / empty.
    """

    sandbox_id: str
    sandbox_domain: Optional[str]
    template_id: str
    name: Optional[str]
    metadata: Dict[str, str]
    started_at: datetime
    end_at: Optional[datetime]
    state: SandboxState
    cpu_count: int
    memory_mb: int
    envd_version: Optional[str]
    allow_internet_access: Optional[bool] = None
    network: Optional[Any] = None
    lifecycle: Optional[Any] = None
    volume_mounts: List[Dict[str, str]] = field(default_factory=list)


@synalinks_export("synalinks.sandboxes.SandboxMetrics")
@dataclass
class SandboxMetrics:
    """A resource sample from ``get_metrics`` (E2B's ``SandboxMetrics``)."""

    cpu_count: int
    cpu_used_pct: float
    disk_total: int
    disk_used: int
    mem_total: int
    mem_used: int
    mem_cache: int
    timestamp: datetime


@synalinks_export("synalinks.sandboxes.SnapshotInfo")
@dataclass
class SnapshotInfo:
    """A snapshot made by ``create_snapshot`` (E2B's ``SnapshotInfo``).

    Pass its ``snapshot_id`` as ``create(template=...)`` to start a sandbox
    from it, as on E2B.
    """

    snapshot_id: str
    names: List[str]


@synalinks_export("synalinks.sandboxes.SandboxQuery")
@dataclass
class SandboxQuery:
    """Filter for `Sandbox.list` (E2B's ``SandboxQuery``)."""

    metadata: Optional[Dict[str, str]] = None
    state: Optional[List[SandboxState]] = None
    started_after: Optional[datetime] = None
    template: Optional[str] = None


# -- E2B-compatible errors -------------------------------------------------
#
# E2B's full exception set, with the same hierarchy and constructors, so
# ``except`` clauses written against E2B catch the same failures here. Where it
# helps local callers, an error is also the matching built-in (a missing path
# is a ``FileNotFoundError``, a timeout a ``TimeoutError``).


@synalinks_export(["synalinks.sandboxes.SandboxException", "synalinks.SandboxException"])
class SandboxException(Exception):
    """Base class of the errors the sandbox API raises (E2B's name)."""

    status_code: Optional[int] = None

    def __init__(self, *args, status_code: Optional[int] = None):
        super().__init__(*args)
        self.status_code = status_code


@synalinks_export(["synalinks.sandboxes.TimeoutException", "synalinks.TimeoutException"])
class TimeoutException(SandboxException, TimeoutError):
    """An execution, command or request ran past its timeout."""


@synalinks_export("synalinks.sandboxes.InvalidArgumentException")
class InvalidArgumentException(SandboxException):
    """An argument is not valid, e.g. an unsupported ``language``."""


@synalinks_export("synalinks.sandboxes.NotEnoughSpaceException")
class NotEnoughSpaceException(SandboxException):
    """The sandbox ran out of disk space."""


@synalinks_export(
    ["synalinks.sandboxes.NotFoundException", "synalinks.NotFoundException"]
)
class NotFoundException(SandboxException, FileNotFoundError):
    """Something does not exist. Also a ``FileNotFoundError``."""


@synalinks_export("synalinks.sandboxes.FileNotFoundException")
class FileNotFoundException(NotFoundException):
    """A path does not exist in the sandbox filesystem."""


@synalinks_export("synalinks.sandboxes.SandboxNotFoundException")
class SandboxNotFoundException(NotFoundException):
    """No sandbox has the given ``sandbox_id`` (it was killed, or never existed)."""


@synalinks_export("synalinks.sandboxes.VolumeNotFoundException")
class VolumeNotFoundException(NotFoundException):
    """A volume does not exist."""


@synalinks_export("synalinks.sandboxes.VolumePathNotFoundException")
class VolumePathNotFoundException(NotFoundException):
    """A path does not exist in a volume."""


@synalinks_export("synalinks.sandboxes.AuthenticationException")
class AuthenticationException(Exception):
    """Authentication failed (E2B's API key, git credentials)."""


@synalinks_export("synalinks.sandboxes.GitAuthException")
class GitAuthException(AuthenticationException):
    """Git refused the credentials."""


@synalinks_export("synalinks.sandboxes.GitUpstreamException")
class GitUpstreamException(SandboxException):
    """A git operation needs an upstream branch that is not set."""


@synalinks_export("synalinks.sandboxes.TemplateException")
class TemplateException(SandboxException):
    """A sandbox template is invalid or incompatible."""


@synalinks_export("synalinks.sandboxes.RateLimitException")
class RateLimitException(SandboxException):
    """Too many requests."""

    def __init__(self, *args):
        super().__init__(*args)


@synalinks_export("synalinks.sandboxes.BuildException")
class BuildException(Exception):
    """A template build failed."""


@synalinks_export("synalinks.sandboxes.FileUploadException")
class FileUploadException(BuildException):
    """A file upload during a template build failed."""


@synalinks_export("synalinks.sandboxes.SecretException")
class SecretException(Exception):
    """A secret operation failed."""


@synalinks_export("synalinks.sandboxes.SecretNotFoundException")
class SecretNotFoundException(SecretException):
    """A secret does not exist."""


@synalinks_export("synalinks.sandboxes.ServiceBusyException")
class ServiceBusyException(Exception):
    """The service is too busy to handle the request."""


@synalinks_export("synalinks.sandboxes.VolumeException")
class VolumeException(Exception):
    """A volume operation failed."""


@synalinks_export("synalinks.sandboxes.NotSupportedException")
class NotSupportedException(SandboxException):
    """An E2B feature that only exists on E2B's cloud (MCP, signed URLs, git,
    PTYs, ...). The method is present with E2B's exact signature, so E2B code
    ports unchanged, but calling it on a local sandbox raises this."""


@synalinks_export(
    ["synalinks.sandboxes.CommandExitException", "synalinks.CommandExitException"]
)
@dataclass
class CommandExitException(SandboxException, CommandResult):
    """Raised when a command exits non-zero, as in E2B.

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


async def call_back(callback: Optional[Callable], value: Any) -> None:
    """Invoke an E2B-style callback, awaiting it when it is a coroutine."""
    if callback is None:
        return
    outcome = callback(value)
    if inspect.isawaitable(outcome):
        await outcome


def output_lines(text: str) -> List[str]:
    """``text`` split into lines, each keeping its newline (E2B's line events)."""
    return text.splitlines(keepends=True)


@synalinks_export("synalinks.sandboxes.CommandHandle")
class CommandHandle:
    """A command started with ``commands.run(..., background=True)``
    (E2B's ``AsyncCommandHandle``).

    The command runs as its own task. With ``stdin=True`` it waits for its
    input: data given to `send_stdin` is buffered and fed to the command
    when `close_stdin` is called. Output reaches ``on_stdout`` /
    ``on_stderr`` line by line when the command finishes, until
    `disconnect`.

    Args:
        pid (int): The process id `Commands.list` / ``kill`` refer to.
        execute (callable): ``async (stdin: bytes) -> (stdout, stderr,
            exit_code)`` that runs the command.
        stdin (bool): Keep stdin open for `send_stdin` until `close_stdin`.
        on_stdout (callable): Optional. Called with each stdout line.
        on_stderr (callable): Optional. Called with each stderr line.
        on_exit (callable): Optional. Called with the handle once the
            command is over (to drop it from the process table).
        cmd (str): The command line, for `Commands.list`.
        envs (dict): Its environment variables, for `Commands.list`.
        cwd (str): Its working directory, for `Commands.list`.
    """

    def __init__(
        self,
        pid: int,
        execute: Callable,
        *,
        stdin: bool = False,
        on_stdout: Optional[Callable] = None,
        on_stderr: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
        cmd: str = "",
        envs: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        self.process_id = pid
        self.cmd = cmd
        self.envs = dict(envs or {})
        self.cwd = cwd
        self.on_stdout = on_stdout
        self.on_stderr = on_stderr
        self.on_exit = on_exit
        self.connected = True
        self.stdin_chunks: List[bytes] = []
        self.stdin_closed = asyncio.Event()
        if not stdin:
            self.stdin_closed.set()
        self.result: Optional[CommandResult] = None
        self.task = asyncio.ensure_future(self.run(execute))

    async def run(self, execute: Callable) -> None:
        try:
            await self.stdin_closed.wait()
            stdout, stderr, exit_code = await execute(b"".join(self.stdin_chunks))
            error = None
            if exit_code != 0:
                error = stderr.strip().splitlines()[-1] if stderr.strip() else None
            self.result = CommandResult(
                stderr=stderr, stdout=stdout, exit_code=exit_code, error=error
            )
            if self.connected:
                for line in output_lines(stdout):
                    await call_back(self.on_stdout, line)
                for line in output_lines(stderr):
                    await call_back(self.on_stderr, line)
        except asyncio.CancelledError:
            self.result = CommandResult(
                stderr="", stdout="", exit_code=-1, error="signal: killed"
            )
        finally:
            if self.on_exit is not None:
                self.on_exit(self)

    @property
    def pid(self) -> int:
        """The process id."""
        return self.process_id

    @property
    def stdout(self) -> str:
        """Everything the command printed to stdout (once it finished)."""
        return self.result.stdout if self.result else ""

    @property
    def stderr(self) -> str:
        """Everything the command printed to stderr (once it finished)."""
        return self.result.stderr if self.result else ""

    @property
    def error(self) -> Optional[str]:
        """The error message of a failed command, or ``None``."""
        return self.result.error if self.result else None

    @property
    def exit_code(self) -> Optional[int]:
        """The exit code, or ``None`` while the command is running."""
        return self.result.exit_code if self.result else None

    async def disconnect(self) -> None:
        """Stop receiving the command's output; the command keeps running."""
        self.connected = False

    async def wait(self) -> CommandResult:
        """Wait for the command to finish and return its `CommandResult`.

        Raises `CommandExitException` when it exited non-zero or was killed.
        """
        try:
            await asyncio.shield(self.task)
        except asyncio.CancelledError:
            if not self.task.done():
                raise
        if self.result.exit_code != 0:
            raise CommandExitException(
                stderr=self.result.stderr,
                stdout=self.result.stdout,
                exit_code=self.result.exit_code,
                error=self.result.error,
            )
        return self.result

    async def kill(self) -> bool:
        """Kill the command; ``False`` if it had already finished."""
        if self.task.done():
            return False
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass
        if self.result is None:
            # Cancelled before it ever ran, so `run` could not record the
            # outcome or leave the process table itself.
            self.result = CommandResult(
                stderr="", stdout="", exit_code=-1, error="signal: killed"
            )
            if self.on_exit is not None:
                self.on_exit(self)
        return True

    async def send_stdin(
        self, data: Union[str, bytes], request_timeout: Optional[float] = None
    ) -> None:
        """Send ``data`` to the command's stdin (started with ``stdin=True``)."""
        if self.stdin_closed.is_set():
            raise SandboxException(
                f"stdin of process {self.pid} is closed; start the command with "
                "stdin=True to send input"
            )
        self.stdin_chunks.append(data.encode("utf-8") if isinstance(data, str) else data)

    async def close_stdin(self, request_timeout: Optional[float] = None) -> None:
        """Close stdin: the command runs with everything sent so far."""
        self.stdin_closed.set()


@synalinks_export("synalinks.sandboxes.WatchHandle")
class WatchHandle:
    """A running ``files.watch_dir`` (E2B's ``AsyncWatchHandle``)."""

    def __init__(self, task: "asyncio.Task"):
        self.task = task

    async def stop(self):
        """Stop watching."""
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass


@synalinks_export("synalinks.sandboxes.Paginator")
class Paginator:
    """Pages over a listing (E2B's ``AsyncSandboxPaginator`` /
    ``AsyncSnapshotPaginator``): call `next_items` while `has_next`.

    Args:
        items (callable): Returns the full, ordered listing.
        limit (int): Optional. Page size; ``None`` returns everything at once.
        next_token (str): Optional. Where to resume, from a previous page.
    """

    def __init__(
        self,
        items: Callable[[], List[Any]],
        limit: Optional[int] = None,
        next_token: Optional[str] = None,
    ):
        self.items = items
        self.limit = limit
        self.token = next_token
        self.more = True

    @property
    def has_next(self) -> bool:
        """Whether `next_items` has another page."""
        return self.more

    @property
    def next_token(self) -> Optional[str]:
        """Where the next page starts, to resume a listing later."""
        return self.token

    async def next_items(self, **opts) -> List[Any]:
        """The next page."""
        if not self.more:
            raise SandboxException("No more items to fetch")
        items = self.items()
        start = int(self.token or 0)
        end = len(items) if self.limit is None else start + self.limit
        self.more = end < len(items)
        self.token = str(end) if self.more else None
        return items[start:end]


def not_supported(feature: str) -> NotSupportedException:
    """The error an E2B cloud-only feature raises in a local sandbox."""
    return NotSupportedException(
        f"{feature} is an E2B cloud feature with no local equivalent; it is not "
        "available in this sandbox."
    )


@synalinks_export("synalinks.sandboxes.Git")
class Git:
    """``sandbox.git``, E2B's git API. Every method has E2B's signature and
    raises `NotSupportedException`: the sandbox's shell has no ``git``."""

    def __init__(self, sandbox: "Sandbox"):
        self.sandbox = sandbox

    async def clone(
        self,
        url: str,
        path: Optional[str] = None,
        branch: Optional[str] = None,
        depth: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
        dangerously_store_credentials: bool = False,
    ):
        raise not_supported("git")

    async def init(
        self,
        path: str,
        bare: bool = False,
        initial_branch: Optional[str] = None,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def remote_add(
        self,
        path: str,
        name: str,
        url: str,
        fetch: bool = False,
        overwrite: bool = False,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def remote_get(
        self,
        path: str,
        name: str,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> Optional[str]:
        raise not_supported("git")

    async def status(
        self,
        path: str,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def branches(
        self,
        path: str,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def create_branch(
        self,
        path: str,
        branch: str,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def checkout_branch(
        self,
        path: str,
        branch: str,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def delete_branch(
        self,
        path: str,
        branch: str,
        force: bool = False,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def add(
        self,
        path: str,
        files: Optional[List[str]] = None,
        all: bool = True,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def commit(
        self,
        path: str,
        message: str,
        author_name: Optional[str] = None,
        author_email: Optional[str] = None,
        allow_empty: bool = False,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def reset(
        self,
        path: str,
        mode: Optional[Literal["soft", "mixed", "hard", "merge", "keep"]] = None,
        target: Optional[str] = None,
        paths: Optional[List[str]] = None,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def restore(
        self,
        path: str,
        paths: List[str],
        staged: Optional[bool] = None,
        worktree: Optional[bool] = None,
        source: Optional[str] = None,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def push(
        self,
        path: str,
        remote: Optional[str] = None,
        branch: Optional[str] = None,
        set_upstream: bool = True,
        username: Optional[str] = None,
        password: Optional[str] = None,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def pull(
        self,
        path: str,
        remote: Optional[str] = None,
        branch: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def set_config(
        self,
        key: str,
        value: str,
        scope: str = "global",
        path: Optional[str] = None,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def get_config(
        self,
        key: str,
        scope: str = "global",
        path: Optional[str] = None,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ) -> Optional[str]:
        raise not_supported("git")

    async def dangerously_authenticate(
        self,
        username: str,
        password: str,
        host: str = "github.com",
        protocol: str = "https",
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")

    async def configure_user(
        self,
        name: str,
        email: str,
        scope: str = "global",
        path: Optional[str] = None,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
    ):
        raise not_supported("git")


@synalinks_export("synalinks.sandboxes.Pty")
class Pty:
    """``sandbox.pty``, E2B's pseudo-terminal API. Every method has E2B's
    signature and raises `NotSupportedException`: the sandbox's shell runs
    commands to completion and has no interactive terminal."""

    def __init__(self, sandbox: "Sandbox"):
        self.sandbox = sandbox

    async def kill(self, pid: int, request_timeout: Optional[float] = None) -> bool:
        raise not_supported("pty")

    async def send_stdin(
        self, pid: int, data: bytes, request_timeout: Optional[float] = None
    ) -> None:
        raise not_supported("pty")

    async def create(
        self,
        size: PtySize,
        on_data: Callable[[bytes], Any],
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        envs: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = 60,
        request_timeout: Optional[float] = None,
    ) -> CommandHandle:
        raise not_supported("pty")

    async def connect(
        self,
        pid: int,
        on_data: Callable[[bytes], Any],
        timeout: Optional[float] = 60,
        request_timeout: Optional[float] = None,
    ) -> CommandHandle:
        raise not_supported("pty")

    async def resize(
        self, pid: int, size: PtySize, request_timeout: Optional[float] = None
    ) -> None:
        raise not_supported("pty")


class Filesystem:
    """The ``sandbox.files`` namespace, with E2B's ``Filesystem`` signatures.

    Backends with a filesystem subclass this and point
    ``Sandbox.filesystem_class`` at the subclass. This default has no
    filesystem, so every method raises ``NotImplementedError``. ``user`` and
    ``request_timeout`` are accepted everywhere for E2B compatibility; a
    local sandbox has a single user and makes no requests.
    """

    def __init__(self, sandbox: "Sandbox"):
        self.sandbox = sandbox

    def unsupported(self):
        raise NotImplementedError("This sandbox has no filesystem.")

    async def read(
        self,
        path: str,
        format: Literal["text", "bytes", "stream"] = "text",
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
        gzip: bool = False,
        stream_idle_timeout: Optional[float] = None,
    ):
        """Read the file at ``path``: ``str`` for ``"text"`` (default),
        ``bytearray`` for ``"bytes"``, an async iterator of ``bytes`` chunks for
        ``"stream"``. Raises `FileNotFoundException` when it does not exist."""
        self.unsupported()

    async def write(
        self,
        path: str,
        data: Union[str, bytes, IO],
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
        gzip: bool = False,
        use_octet_stream: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> WriteInfo:
        """Write ``data`` to ``path``, creating parent directories."""
        self.unsupported()

    async def write_files(
        self,
        files: List[WriteEntry],
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
        gzip: bool = False,
        use_octet_stream: Optional[bool] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> List[WriteInfo]:
        """Write several files (``{"path", "data"}`` entries) at once."""
        return [
            await self.write(
                entry["path"],
                entry["data"],
                user=user,
                request_timeout=request_timeout,
                gzip=gzip,
                use_octet_stream=use_octet_stream,
                metadata=metadata,
            )
            for entry in files
        ]

    async def list(
        self,
        path: str,
        depth: Optional[int] = 1,
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> List[EntryInfo]:
        """List the entries under the directory ``path``, ``depth`` levels deep."""
        self.unsupported()

    async def exists(
        self,
        path: str,
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> bool:
        """Whether a file or directory exists at ``path``."""
        self.unsupported()

    async def get_info(
        self,
        path: str,
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> EntryInfo:
        """Describe the entry at ``path``; raises `FileNotFoundException` if absent."""
        self.unsupported()

    async def remove(
        self,
        path: str,
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> None:
        """Delete the file or directory (recursively) at ``path``."""
        self.unsupported()

    async def rename(
        self,
        old_path: str,
        new_path: str,
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> EntryInfo:
        """Move ``old_path`` to ``new_path``; returns the new entry."""
        self.unsupported()

    async def make_dir(
        self,
        path: str,
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> bool:
        """Create ``path`` (and parents); ``False`` if it already existed."""
        self.unsupported()

    async def watch_dir(
        self,
        path: str,
        on_event: Callable[[FilesystemEvent], Any],
        on_exit: Optional[Callable[[Optional[Exception]], Any]] = None,
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
        timeout: Optional[float] = 60,
        recursive: bool = False,
        include_entry: bool = False,
        allow_network_mounts: bool = False,
    ) -> WatchHandle:
        """Call ``on_event`` for every change under ``path`` until the returned
        handle is stopped or ``timeout`` seconds pass (``0``: no limit)."""
        self.unsupported()


class Commands:
    """The ``sandbox.commands`` namespace, with E2B's ``Commands`` signatures.

    Backends with a shell subclass this and point ``Sandbox.commands_class``
    at the subclass. This default has no shell. The process table
    (background commands) lives on the sandbox, so it is shared by every
    ``sandbox.commands`` access.
    """

    def __init__(self, sandbox: "Sandbox"):
        self.sandbox = sandbox

    async def run(
        self,
        cmd: str,
        background: Optional[bool] = None,
        envs: Optional[Dict[str, str]] = None,
        user: Optional[str] = None,
        cwd: Optional[str] = None,
        on_stdout: Optional[Callable[[str], Any]] = None,
        on_stderr: Optional[Callable[[str], Any]] = None,
        stdin: Optional[bool] = None,
        timeout: Optional[float] = 60,
        request_timeout: Optional[float] = None,
    ):
        """Run the shell command ``cmd``.

        In the foreground (default) wait for it and return a `CommandResult`;
        as in E2B a non-zero exit raises `CommandExitException` (which carries
        the result) and running past ``timeout`` seconds (``0``: no limit)
        raises `TimeoutException`. With ``background=True`` return a
        `CommandHandle` at once.
        """
        raise NotImplementedError("This sandbox has no shell.")

    def handle(self, pid: int) -> CommandHandle:
        """The running background command ``pid``, or `NotFoundException`."""
        handle = self.sandbox.processes.get(pid)
        if handle is None:
            raise NotFoundException(f"Process with pid {pid} not found")
        return handle

    async def list(self, request_timeout: Optional[float] = None) -> List[ProcessInfo]:
        """The background commands still running."""
        return [
            ProcessInfo(pid=pid, tag=None, cmd=h.cmd, args=[], envs=h.envs, cwd=h.cwd)
            for pid, h in sorted(self.sandbox.processes.items())
        ]

    async def kill(self, pid: int, request_timeout: Optional[float] = None) -> bool:
        """Kill the background command ``pid``; ``False`` if it is not running."""
        handle = self.sandbox.processes.get(pid)
        return await handle.kill() if handle is not None else False

    async def send_stdin(
        self, pid: int, data: Union[str, bytes], request_timeout: Optional[float] = None
    ) -> None:
        """Send ``data`` to the stdin of the background command ``pid``."""
        await self.handle(pid).send_stdin(data)

    async def close_stdin(
        self, pid: int, request_timeout: Optional[float] = None
    ) -> None:
        """Close the stdin of the background command ``pid``."""
        await self.handle(pid).close_stdin()

    async def connect(
        self,
        pid: int,
        timeout: Optional[float] = 60,
        request_timeout: Optional[float] = None,
        on_stdout: Optional[Callable[[str], Any]] = None,
        on_stderr: Optional[Callable[[str], Any]] = None,
    ) -> CommandHandle:
        """Reattach to the background command ``pid`` (new output callbacks)."""
        handle = self.handle(pid)
        handle.on_stdout = on_stdout or handle.on_stdout
        handle.on_stderr = on_stderr or handle.on_stderr
        handle.connected = True
        return handle


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

    # E2B's class defaults, for code that reads them.
    default_template = "code-interpreter-v1"
    default_mcp_template = "mcp-gateway"
    default_sandbox_timeout = 300
    mcp_port = 50005

    # Every live sandbox of this process, by ``sandbox_id``: what E2B's
    # class-level calls (``Sandbox.kill(sandbox_id)``, ``connect``, ``list``)
    # look up. Weak, so dropping the last reference still frees a sandbox.
    live_sandboxes: "weakref.WeakValueDictionary[str, Sandbox]" = (
        weakref.WeakValueDictionary()
    )
    # Snapshots made by ``create_snapshot``: ``snapshot_id -> {"data",
    # "names", "sandbox_id", "cls"}``. ``create(template=snapshot_id)`` starts
    # a sandbox from one, as on E2B.
    snapshots: Dict[str, Dict[str, Any]] = {}
    # The keyword arguments of E2B's ``ApiParams`` (API key, domain, ...):
    # accepted everywhere E2B accepts ``**opts``, meaningless locally.
    api_params = (
        "api_key",
        "access_token",
        "domain",
        "debug",
        "request_timeout",
        "headers",
        "proxy",
        "sandbox_url",
    )

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
        # E2B's sandbox-level state.
        self.identifier = uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc)
        self.end_at: Optional[datetime] = None
        self.state = SandboxState.RUNNING
        self.template_id = self.default_template
        self.metadata: Dict[str, str] = {}
        # Environment variables every ``run_code`` / ``commands.run`` gets.
        self.envs: Dict[str, str] = {}
        self.logger: Optional[logging.Logger] = None
        self.lifetime_timer: Optional[asyncio.TimerHandle] = None
        # Background commands (``commands.run(background=True)``) by pid.
        self.processes: Dict[int, CommandHandle] = {}
        self.next_pid = 1
        Sandbox.live_sandboxes[self.identifier] = self

    # -- lifecycle (E2B-compatible) -------------------------------------

    @classmethod
    def lookup(cls, sandbox_id: str) -> "Sandbox":
        """The live sandbox ``sandbox_id``, or `SandboxNotFoundException`."""
        sandbox = Sandbox.live_sandboxes.get(sandbox_id)
        if sandbox is None or not isinstance(sandbox, cls):
            raise SandboxNotFoundException(f"Sandbox {sandbox_id} not found")
        return sandbox

    @classmethod
    def network_options(
        cls, allow_internet_access: Optional[bool], network: Optional[Any]
    ) -> Dict[str, Any]:
        """Constructor kwargs for E2B's network options of `create`.

        A backend with network control overrides this; the default has none,
        so asking for network access fails rather than being ignored.
        """
        if allow_internet_access or network:
            raise not_supported("network configuration for this sandbox backend")
        return {}

    @classmethod
    async def create(
        cls,
        template: Optional[str] = None,
        timeout: Optional[int] = None,
        metadata: Optional[Dict[str, str]] = None,
        envs: Optional[Dict[str, str]] = None,
        secure: Optional[bool] = None,
        allow_internet_access: Optional[bool] = None,
        mcp: Optional[Any] = None,
        network: Optional[Any] = None,
        iam: Optional[Any] = None,
        lifecycle: Optional[Any] = None,
        volume_mounts: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
        **opts,
    ) -> "Sandbox":
        """Create a sandbox (E2B's ``AsyncSandbox.create``).

        E2B's parameters keep their meaning where a local sandbox has one:

        - ``template``: ``None`` or E2B's default template, or the
          ``snapshot_id`` of a `create_snapshot` to start from it.
        - ``timeout``: the sandbox's *lifetime* in seconds, after which it is
          killed (see `set_timeout`). Unlike E2B, ``None`` means no lifetime
          limit, as for a sandbox built with the constructor.
        - ``metadata``, ``envs`` (default environment of every execution and
          command) and ``logger``.
        - ``allow_internet_access`` / ``network``: the network, as far as the
          backend supports it (`network_options`). Secure by default: ``None``
          keeps the network cut, where E2B's default is open.

        ``secure`` is always on locally. ``mcp``, ``iam``, ``lifecycle`` and
        ``volume_mounts`` are E2B cloud features and raise
        `NotSupportedException` when set. E2B's API parameters (``api_key``,
        ``domain``, ...) are dropped; any other keyword argument goes to the
        backend's constructor, e.g. ``create(confine=False)``.
        """
        for feature, value in (
            ("mcp", mcp),
            ("iam", iam),
            ("lifecycle", lifecycle),
            ("volume_mounts", volume_mounts),
        ):
            if value:
                raise not_supported(f"create({feature}=...)")
        kwargs = {k: v for k, v in opts.items() if k not in cls.api_params}
        kwargs.update(cls.network_options(allow_internet_access, network))
        snapshot = Sandbox.snapshots.get(template) if template else None
        if snapshot is not None:
            sandbox = snapshot["cls"].load(snapshot["data"], **kwargs)
        elif template in (None, cls.default_template, "code-interpreter", "base"):
            sandbox = cls(**kwargs)
        else:
            raise TemplateException(
                f"Template {template!r} is not available locally: use the default "
                "template or the snapshot_id of a snapshot made in this process."
            )
        sandbox.template_id = template or cls.default_template
        sandbox.metadata = dict(metadata or {})
        sandbox.envs = dict(envs or {})
        sandbox.logger = logger
        if timeout:
            await sandbox.set_timeout(timeout)
        return sandbox

    @property
    def sandbox_id(self) -> str:
        """The sandbox's unique id (what the class-level calls take)."""
        return self.identifier

    @property
    def sandbox_domain(self) -> Optional[str]:
        """``None``: a local sandbox has no domain."""
        return None

    @property
    def envd_api_url(self) -> Optional[str]:
        """``None``: a local sandbox has no remote daemon."""
        return None

    @property
    def envd_direct_url(self) -> Optional[str]:
        """``None``: a local sandbox has no remote daemon."""
        return None

    @property
    def traffic_access_token(self) -> Optional[str]:
        """``None``: a local sandbox serves no public traffic."""
        return None

    @property
    def connection_config(self) -> Optional[Any]:
        """``None``: a local sandbox makes no API connection."""
        return None

    @property
    def files(self) -> Filesystem:
        """The filesystem namespace (E2B's ``Filesystem``)."""
        return self.filesystem_class(self)

    @property
    def commands(self) -> Commands:
        """The shell namespace (E2B's ``Commands``)."""
        return self.commands_class(self)

    @property
    def git(self) -> Git:
        """E2B's git API; not available locally (see `Git`)."""
        return Git(self)

    @property
    def pty(self) -> Pty:
        """E2B's pseudo-terminal API; not available locally (see `Pty`)."""
        return Pty(self)

    def new_pid(self) -> int:
        """A fresh process id for a background command."""
        pid, self.next_pid = self.next_pid, self.next_pid + 1
        return pid

    async def release(self) -> None:
        """Release what the backend holds (mounts, temp dirs). Backends with
        resources override this; `kill` calls it once."""

    @classmethod
    async def class_kill(cls, sandbox_id: str, **opts) -> bool:
        sandbox = Sandbox.live_sandboxes.get(sandbox_id)
        return await sandbox.kill() if isinstance(sandbox, cls) else False

    @class_method_variant("class_kill")
    async def kill(self, **opts) -> bool:
        """Shut the sandbox down and release its resources (E2B's ``kill``).

        Also callable as ``await Sandbox.kill(sandbox_id)``. Returns ``True``,
        or ``False`` when it was already killed.
        """
        if getattr(self, "killed", False):
            return False
        self.killed = True
        if self.lifetime_timer is not None:
            self.lifetime_timer.cancel()
        for handle in list(self.processes.values()):
            await handle.kill()
        Sandbox.live_sandboxes.pop(self.identifier, None)
        await self.release()
        return True

    async def is_running(self, request_timeout: Optional[float] = None) -> bool:
        """Whether the sandbox is still usable (``kill`` not yet called)."""
        return not getattr(self, "killed", False)

    @classmethod
    async def class_set_timeout(cls, sandbox_id: str, timeout: int, **opts) -> None:
        await cls.lookup(sandbox_id).set_timeout(timeout)

    @class_method_variant("class_set_timeout")
    async def set_timeout(self, timeout: int, **opts) -> None:
        """Kill the sandbox ``timeout`` seconds from now (E2B's lifetime).

        Replaces any earlier deadline; call it again to extend the life of a
        sandbox still in use. Also callable as
        ``await Sandbox.set_timeout(sandbox_id, timeout)``.
        """
        if self.lifetime_timer is not None:
            self.lifetime_timer.cancel()
        loop = asyncio.get_running_loop()
        self.lifetime_timer = loop.call_later(
            timeout, lambda: loop.create_task(self.kill())
        )
        self.end_at = datetime.now(timezone.utc) + timedelta(seconds=timeout)

    @classmethod
    async def class_pause(
        cls, sandbox_id: str, keep_memory: Optional[bool] = None, **opts
    ) -> bool:
        return await cls.lookup(sandbox_id).pause(keep_memory)

    @class_method_variant("class_pause")
    async def pause(self, keep_memory: Optional[bool] = None, **opts) -> bool:
        """Mark the sandbox paused (E2B's ``pause``); `connect` resumes it.

        A local sandbox holds no machine to suspend: its files and namespace
        simply stay as they are, so this records the state and stops a
        lifetime deadline, as a paused E2B sandbox does not time out.
        """
        if self.lifetime_timer is not None:
            self.lifetime_timer.cancel()
            self.lifetime_timer = None
        self.state = SandboxState.PAUSED
        return True

    @classmethod
    async def class_beta_pause(
        cls, sandbox_id: str, keep_memory: Optional[bool] = None, **opts
    ) -> bool:
        return await cls.lookup(sandbox_id).beta_pause(keep_memory)

    @class_method_variant("class_beta_pause")
    async def beta_pause(self, keep_memory: Optional[bool] = None, **opts) -> bool:
        """E2B's earlier name for `pause`."""
        return await self.pause(keep_memory)

    @classmethod
    async def class_connect(
        cls,
        sandbox_id: str,
        timeout: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        *,
        on_resume: Literal["restore", "reboot"] = "restore",
        **opts,
    ) -> "Sandbox":
        return await cls.lookup(sandbox_id).connect(timeout, on_resume=on_resume)

    @class_method_variant("class_connect")
    async def connect(
        self,
        timeout: Optional[int] = None,
        *,
        on_resume: Literal["restore", "reboot"] = "restore",
        **opts,
    ) -> "Sandbox":
        """Resume a paused sandbox and return it (E2B's ``connect``).

        Also callable as ``await Sandbox.connect(sandbox_id)``. A ``timeout``
        sets a new lifetime; ``on_resume="reboot"`` also wipes the execution
        state (`reset`), like a cold boot.
        """
        if on_resume == "reboot":
            self.reset()
        self.state = SandboxState.RUNNING
        if timeout:
            await self.set_timeout(timeout)
        return self

    def describe(self) -> SandboxInfo:
        """This sandbox as a `SandboxInfo` (for `get_info` and `list`)."""
        return SandboxInfo(
            sandbox_id=self.sandbox_id,
            sandbox_domain=None,
            template_id=self.template_id,
            name=self.name,
            metadata=dict(self.metadata),
            started_at=self.started_at,
            end_at=self.end_at,
            state=self.state,
            cpu_count=os.cpu_count() or 1,
            memory_mb=int(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**20
            ),
            envd_version=None,
        )

    @classmethod
    async def class_get_info(cls, sandbox_id: str, **opts) -> SandboxInfo:
        return cls.lookup(sandbox_id).describe()

    @class_method_variant("class_get_info")
    async def get_info(self, **opts) -> SandboxInfo:
        """Describe the sandbox (E2B's ``get_info``); CPU and memory are the
        host's. Also callable as ``await Sandbox.get_info(sandbox_id)``."""
        return self.describe()

    @classmethod
    async def class_get_metrics(
        cls,
        sandbox_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        **opts,
    ) -> List[SandboxMetrics]:
        return await cls.lookup(sandbox_id).get_metrics(start, end)

    @class_method_variant("class_get_metrics")
    async def get_metrics(
        self, start: Optional[datetime] = None, end: Optional[datetime] = None, **opts
    ) -> List[SandboxMetrics]:
        """One resource sample of the host running the sandbox (E2B's
        ``get_metrics``). ``start`` / ``end`` are accepted for E2B
        compatibility; there is no metrics history locally."""
        cpu_count = os.cpu_count() or 1
        disk = shutil.disk_usage(tempfile.gettempdir())
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return [
            SandboxMetrics(
                cpu_count=cpu_count,
                cpu_used_pct=min(100.0, os.getloadavg()[0] / cpu_count * 100),
                disk_total=disk.total,
                disk_used=disk.used,
                mem_total=os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
                # ``ru_maxrss`` is bytes on macOS and kilobytes elsewhere.
                mem_used=rss if sys.platform == "darwin" else rss * 1024,
                mem_cache=0,
                timestamp=datetime.now(timezone.utc),
            )
        ]

    @classmethod
    def list(
        cls,
        query: Optional[SandboxQuery] = None,
        limit: Optional[int] = None,
        next_token: Optional[str] = None,
        order: Optional[Literal["asc", "desc"]] = None,
        **opts,
    ) -> Paginator:
        """Page over the live sandboxes of this process (E2B's ``list``),
        optionally filtered by ``query``, ordered by start time."""

        def matching() -> List[SandboxInfo]:
            infos = [
                sandbox.describe()
                for sandbox in list(Sandbox.live_sandboxes.values())
                if isinstance(sandbox, cls)
            ]
            if query is not None:
                infos = [
                    info
                    for info in infos
                    if (query.state is None or info.state in query.state)
                    and (query.template is None or info.template_id == query.template)
                    and (
                        query.started_after is None
                        or info.started_at > query.started_after
                    )
                    and all(
                        info.metadata.get(k) == v
                        for k, v in (query.metadata or {}).items()
                    )
                ]
            return sorted(
                infos, key=lambda info: info.started_at, reverse=order == "desc"
            )

        return Paginator(matching, limit, next_token)

    @classmethod
    async def class_update_network(cls, sandbox_id: str, network: Any, **opts) -> None:
        await cls.lookup(sandbox_id).update_network(network)

    @class_method_variant("class_update_network")
    async def update_network(self, network: Any, **opts) -> None:
        """E2B's live network update; not available locally."""
        raise not_supported("update_network")

    @classmethod
    async def class_create_snapshot(
        cls, sandbox_id: str, name: Optional[str] = None, **opts
    ) -> SnapshotInfo:
        return await cls.lookup(sandbox_id).create_snapshot(name)

    @class_method_variant("class_create_snapshot")
    async def create_snapshot(self, name: Optional[str] = None, **opts) -> SnapshotInfo:
        """Snapshot the sandbox's current state (E2B's ``create_snapshot``).

        Kept in this process; start a sandbox from it with
        ``create(template=snapshot_id)``. Uses `dump`, so a backend without
        `dump` cannot snapshot.
        """
        snapshot_id = uuid.uuid4().hex
        Sandbox.snapshots[snapshot_id] = {
            "data": self.dump(),
            "names": [name] if name else [],
            "sandbox_id": self.sandbox_id,
            "cls": type(self),
        }
        return SnapshotInfo(snapshot_id=snapshot_id, names=[name] if name else [])

    @classmethod
    def class_list_snapshots(
        cls,
        sandbox_id: Optional[str] = None,
        limit: Optional[int] = None,
        next_token: Optional[str] = None,
        name: Optional[str] = None,
        **opts,
    ) -> Paginator:
        def matching() -> List[SnapshotInfo]:
            return [
                SnapshotInfo(snapshot_id=snapshot_id, names=list(snapshot["names"]))
                for snapshot_id, snapshot in list(Sandbox.snapshots.items())
                if (sandbox_id is None or snapshot["sandbox_id"] == sandbox_id)
                and (name is None or name in snapshot["names"])
            ]

        return Paginator(matching, limit, next_token)

    @class_method_variant("class_list_snapshots")
    def list_snapshots(
        self,
        limit: Optional[int] = None,
        next_token: Optional[str] = None,
        name: Optional[str] = None,
        **opts,
    ) -> Paginator:
        """Page over this sandbox's snapshots (E2B's ``list_snapshots``);
        ``Sandbox.list_snapshots()`` pages over every snapshot."""
        return type(self).class_list_snapshots(self.sandbox_id, limit, next_token, name)

    @classmethod
    async def delete_snapshot(cls, snapshot_id: str, **opts) -> bool:
        """Delete a snapshot; ``False`` if there was none with that id."""
        return Sandbox.snapshots.pop(snapshot_id, None) is not None

    @classmethod
    async def class_fork(
        cls,
        sandbox_id: str,
        timeout: Optional[int] = None,
        count: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        **opts,
    ) -> List[Union["Sandbox", Exception]]:
        return await cls.lookup(sandbox_id).fork(timeout, count, **opts)

    @class_method_variant("class_fork")
    async def fork(
        self,
        timeout: Optional[int] = None,
        count: Optional[int] = None,
        **opts,
    ) -> List[Union["Sandbox", Exception]]:
        """Start ``count`` (default 1) sandboxes from this one's current state
        (E2B's ``fork``); returns them, with the exception in place of any
        fork that failed. ``timeout`` sets their lifetime.

        Backends that can branch implement this; their own keyword arguments
        (e.g. ``name``) pass through ``opts``.
        """
        raise NotImplementedError("This sandbox does not support `fork`.")

    def download_url(
        self,
        path: str,
        user: Optional[str] = None,
        use_signature_expiration: Optional[int] = None,
    ) -> str:
        """E2B's signed download URL; not available locally (a local sandbox
        serves no HTTP). Read the file with ``files.read`` instead."""
        raise not_supported("download_url")

    def upload_url(
        self,
        path: str,
        user: Optional[str] = None,
        use_signature_expiration: Optional[int] = None,
    ) -> str:
        """E2B's signed upload URL; not available locally. Write the file with
        ``files.write`` instead."""
        raise not_supported("upload_url")

    def get_host(self, port: int) -> str:
        """E2B's public host for a sandbox port; not available locally."""
        raise not_supported("get_host")

    def get_mcp_url(self) -> str:
        """E2B's MCP gateway URL; not available locally."""
        raise not_supported("MCP")

    async def get_mcp_token(self) -> Optional[str]:
        """``None``: a local sandbox runs no MCP gateway."""
        return None

    # -- code contexts (E2B-compatible) ---------------------------------

    async def create_code_context(
        self,
        cwd: Optional[str] = None,
        language: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> Context:
        """Create an isolated code context (own namespace and ``cwd``) to pass
        as ``run_code(context=...)``. Backends with contexts implement it."""
        raise NotImplementedError("This sandbox does not support code contexts.")

    async def list_code_contexts(self) -> List[Context]:
        """The code contexts, the default one first."""
        raise NotImplementedError("This sandbox does not support code contexts.")

    async def restart_code_context(self, context: Union[Context, str]) -> None:
        """Wipe a context's namespace (its variables, imports, definitions)."""
        raise NotImplementedError("This sandbox does not support code contexts.")

    async def remove_code_context(self, context: Union[Context, str]) -> None:
        """Delete a context."""
        raise NotImplementedError("This sandbox does not support code contexts.")

    async def notify(
        self,
        execution: Execution,
        on_stdout: Optional[Callable] = None,
        on_stderr: Optional[Callable] = None,
        on_result: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> Execution:
        """Deliver ``run_code``'s E2B callbacks for a finished ``execution``:
        each printed line as an `OutputMessage`, each `Result`, the error.
        Returns ``execution``."""
        timestamp = time.time_ns()
        for callback, chunks, error in (
            (on_stdout, execution.logs.stdout, False),
            (on_stderr, execution.logs.stderr, True),
        ):
            for line in output_lines("".join(chunks)):
                await call_back(callback, OutputMessage(line, timestamp, error))
        for result in execution.results:
            await call_back(on_result, result)
        if execution.error is not None:
            await call_back(on_error, execution.error)
        return execution

    # -- execution primitives (abstract) --------------------------------

    async def run_code(
        self,
        code: str,
        language: Optional[str] = None,
        context: Optional[Context] = None,
        on_stdout: Optional[Callable[[OutputMessage], Any]] = None,
        on_stderr: Optional[Callable[[OutputMessage], Any]] = None,
        on_result: Optional[Callable[[Result], Any]] = None,
        on_error: Optional[Callable[[ExecutionError], Any]] = None,
        envs: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        request_timeout: Optional[float] = None,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        external_functions: Optional[Dict[str, Callable]] = None,
    ) -> Execution:
        """Execute ``code`` and return an `Execution` (E2B's ``run_code``).

        Implementations should route their result through `record_run` so the
        snippet lands in `history`, deliver the callbacks through `notify`,
        and expose `functions` (merged with any per-call
        ``external_functions``) as callables inside the sandbox.

        Args:
            code (str): The source to run.
            language (str): Optional. ``None`` or ``"python"``; other languages
                raise `InvalidArgumentException` on a Python-only backend.
            context (Context): Optional. A context from
                `create_code_context` to run in (its own namespace and cwd).
            on_stdout (callable): Optional. Called with each stdout line, as
                an `OutputMessage`.
            on_stderr (callable): Optional. Called with each stderr line.
            on_result (callable): Optional. Called with each `Result`.
            on_error (callable): Optional. Called with the `ExecutionError`.
            envs (dict): Optional. Environment variables for this run, on top
                of the sandbox's.
            timeout (float): Optional. Seconds this run may take; ``None`` uses
                the sandbox's ``timeout``.
            request_timeout (float): Optional. Accepted for E2B compatibility;
                a local sandbox makes no request.
            inputs (dict): Optional. Variables bound into the namespace before
                execution. Backend-specific rules apply.
            external_functions (dict): Optional. Mapping of name to async
                callable exposed as global functions inside the sandbox for
                this call, on top of the persistently bound set.

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
        return await self.notify(
            Execution(
                results=(
                    [Result(text=repr(old.result), json=old.result, is_main_result=True)]
                    if old.result is not None
                    else []
                ),
                logs=Logs(stdout=[old.stdout] if old.stdout else []),
                error=error,
            ),
            on_stdout,
            on_stderr,
            on_result,
            on_error,
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
