# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
import base64
import contextvars
import dataclasses
import inspect
import io
import json
import os
import posixpath
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import warnings
import weakref
import zipfile
from datetime import datetime
from datetime import timezone
from typing import IO
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from synalinks.src.api_export import synalinks_export
from synalinks.src.sandboxes.sandbox import DEFAULT_TIMEOUT
from synalinks.src.sandboxes.sandbox import CommandExitException
from synalinks.src.sandboxes.sandbox import CommandHandle
from synalinks.src.sandboxes.sandbox import CommandResult
from synalinks.src.sandboxes.sandbox import Commands
from synalinks.src.sandboxes.sandbox import Context
from synalinks.src.sandboxes.sandbox import EntryInfo
from synalinks.src.sandboxes.sandbox import Execution
from synalinks.src.sandboxes.sandbox import ExecutionError
from synalinks.src.sandboxes.sandbox import FileNotFoundException
from synalinks.src.sandboxes.sandbox import Filesystem
from synalinks.src.sandboxes.sandbox import FilesystemEvent
from synalinks.src.sandboxes.sandbox import FilesystemEventType
from synalinks.src.sandboxes.sandbox import FileType
from synalinks.src.sandboxes.sandbox import InvalidArgumentException
from synalinks.src.sandboxes.sandbox import Logs
from synalinks.src.sandboxes.sandbox import NotFoundException
from synalinks.src.sandboxes.sandbox import OutputMessage
from synalinks.src.sandboxes.sandbox import Result
from synalinks.src.sandboxes.sandbox import Sandbox
from synalinks.src.sandboxes.sandbox import TimeoutException
from synalinks.src.sandboxes.sandbox import WatchHandle
from synalinks.src.sandboxes.sandbox import WriteInfo
from synalinks.src.sandboxes.sandbox import call_back
from synalinks.src.sandboxes.sandbox import execution_timeout_error
from synalinks.src.sandboxes.sandbox import output_lines
from synalinks.src.sandboxes.sandbox import request_deadline
from synalinks.src.saving.object_registration import register_synalinks_serializable
from synalinks.src.utils.confinement_utils import CONFINE_PROLOGUE_SRC
from synalinks.src.utils.confinement_utils import build_seccomp_filter
from synalinks.src.utils.confinement_utils import confinement_backend
from synalinks.src.utils.egress_utils import make_egress_tool
from synalinks.src.utils.heap_utils import cap_malloc_arenas
from synalinks.src.utils.heap_utils import malloc_trim
from synalinks.src.utils.microvm_utils import GUEST_BINDS
from synalinks.src.utils.microvm_utils import GUEST_MOUNT
from synalinks.src.utils.microvm_utils import GUEST_SOCK_DIR
from synalinks.src.utils.microvm_utils import prepare_microvm
from synalinks.src.utils.microvm_utils import run_in_microvm
from synalinks.src.utils.python_utils import class_method_variant
from synalinks.src.utils.sandbox_bootstrap_utils import BOOTSTRAP
from synalinks.src.utils.sandbox_bootstrap_utils import LAUNCHER
from synalinks.src.utils.sandbox_fs_utils import glob_to_regex
from synalinks.src.utils.sandbox_fs_utils import paginate
from synalinks.src.utils.sandbox_fs_utils import render_patch

try:
    import mirage
    from mirage import MountMode
    from mirage import RAMResource
    from mirage import Workspace
    from mirage.bridge.sync import run_async_from_sync
except (ImportError, OSError):  # pragma: no cover - mirage genuinely unusable
    # Kept as a backstop so an unusable mirage degrades to a clear error at
    # construction time rather than breaking `import synalinks`.
    mirage = None
    MountMode = None
    RAMResource = None
    Workspace = None
    run_async_from_sync = None


DEFAULT_MOUNT = "/"
# The sandbox's own namespace, as E2B lists it among the code contexts.
DEFAULT_CONTEXT = Context(id="default", language="python", cwd="/")

# Mirage-internal views that are not the agent's files: devices, persisted
# sessions and the shell history view (mirage-ai 0.0.6+). Skipped by file
# listings, search, diffs, snapshots and `save`.
INTERNAL_PATHS = ("/dev", "/.sessions", "/.bash_history")


def rmtree(path: Optional[str]) -> None:
    """Best-effort recursive delete of a host temp directory."""
    if path:
        shutil.rmtree(path, ignore_errors=True)


def require_mirage() -> None:
    if mirage is None:
        raise ImportError(
            "MirageSandbox requires the `mirage-ai` package. "
            "Install it with `pip install mirage-ai`."
        )


# Mirage turned several Workspace methods (stat, readdir, copy, load, snapshot,
# close, stdout_str, ...) into coroutines across its 0.0.x releases. These two
# helpers accept either shape so call sites do not branch on the version.
async def maybe_await(value):
    return await value if inspect.isawaitable(value) else value


def resolve_sync(value):
    """Drive ``value`` to completion from sync code if Mirage returned a coroutine."""
    return run_async_from_sync(value) if inspect.isawaitable(value) else value


def is_dir(stat) -> bool:
    """Whether a Mirage ``FileStat`` describes a directory."""
    return str(getattr(stat.type, "value", stat.type)) == "directory"


def release_mountpoint(path: Optional[str]) -> None:
    """Make sure a mountpoint this sandbox created ends up unmounted and removed.

    Mirage's own teardown is not enough on its own: its macOS unmount runs
    ``diskutil unmount force`` without checking the result, and a mount that
    only comes up *after* its 10 s readiness wait timed out is dropped from its
    bookkeeping while the mount thread goes on to mount it anyway, so nothing
    would ever unmount it. Either leaks a mount per occurrence, under load, and
    macFUSE has only 64 devices. The sandbox picks the path itself (see
    `ensure_fuse_mounted`), so it can always finish the job. Never raises.
    """
    if not path:
        return
    if os.path.ismount(path):
        if sys.platform == "darwin":
            commands = (["umount", "-f", path], ["diskutil", "unmount", "force", path])
        else:
            commands = (["fusermount", "-uz", path], ["umount", "-l", path])
        for command in commands:
            try:
                subprocess.run(command, capture_output=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                continue
            if not os.path.ismount(path):
                break
    try:
        os.rmdir(path)  # empty once unmounted; left alone if it is not
    except OSError:
        pass


def sync_close_workspace(ws) -> None:
    """Release a Mirage workspace's FUSE mount, from any context, never raising.

    ``Workspace.close`` is a coroutine, but the actual unmount (``fusermount -u``,
    or ``diskutil unmount`` on macOS) lives in Mirage's *synchronous*
    ``close_sync_parts`` (the awaited part only drains cache tasks). So:

    * with **no running loop** (a finalizer, ``gc.collect``, interpreter exit, or
      a plain sync caller) we drive the full coroutine via ``asyncio.run``:
      unmount *and* cache drain;
    * with a **loop already running** (called from inside async code) we can't
      block on it, so we release the mount synchronously via
      ``close_sync_parts`` and skip the best-effort cache drain.

    Either way the FUSE mount is dropped, which is what prevents the leak that
    exhausts ``mount_max``.
    """
    if ws is None:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    closed = False
    if loop is None:
        try:
            result = ws.close()
            if inspect.isawaitable(result):
                asyncio.run(result)
            closed = True
        except Exception:  # noqa: BLE001 - fall through to a forced unmount
            pass
    if not closed:
        # A loop is running, or the coroutine could not be driven: unmount
        # now. Discarding the `ws.close()` coroutine here instead would leak
        # the mount, which on macOS exhausts macFUSE's 64 devices within one
        # test run.
        try:
            from mirage.workspace.workspace.lifecycle import close_sync_parts

            close_sync_parts(ws)
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass
    release_mountpoint(getattr(ws, "synalinks_mountpoint", None))


def finalize_workspace(state: dict) -> None:
    """``weakref.finalize`` callback: release a dropped sandbox's mount + hostdir.

    Runs when a :class:`MirageSandbox` is garbage-collected without an explicit
    ``close()`` (and, because ``weakref.finalize`` also fires pending callbacks
    at interpreter exit, on process shutdown too). ``state`` is a plain dict that
    never references the sandbox, so it does not keep it alive; ``reset`` /
    ``rebuild_workspace`` mutate it in place so this always releases the
    *current* workspace.
    """
    sync_close_workspace(state.get("ws"))
    rmtree(state.get("hostdir"))


# Confinement config for the *currently executing* shell command of a confined
# sandbox, read by `confine_shell_python`'s wrapper. ``None`` (the default)
# means "not in a confined shell command", so the wrapper passes code through
# untouched. It is a context variable rather than a sandbox attribute because
# ``run_code`` also launches its bootstrap through the same runtime (and
# confines itself), possibly concurrently with a ``run_bash`` on one sandbox.
active_confine: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "mirage_active_confine", default=None
)


def local_runtime_supported() -> bool:
    """Whether the installed Mirage has the ``LocalRuntime.run(self, RunArgs)``
    coroutine `confine_shell_python` wraps (the layout since mirage-ai 0.0.4)."""
    try:
        from mirage.runtime.python.local import LocalRuntime
    except ImportError:
        return False
    return inspect.iscoroutinefunction(getattr(LocalRuntime, "run", None))


def confine_shell_python(ws, vm: Optional[Dict[str, Any]] = None) -> bool:
    """Make the ``python3`` a shell command spawns in ``ws`` confine itself.

    Mirage runs a shell's ``python3`` through the workspace's ``local`` runtime
    (a host CPython subprocess), not through our ``run_code`` bootstrap, so it
    would otherwise run unconfined. This wraps ``run`` on *that workspace's*
    runtime instance to prepend the confinement prologue whenever
    `active_confine` is set. Nothing process-wide is patched: other workspaces,
    and any other Mirage user in the process, are untouched.

    With ``vm`` (the microVM backend's live launch settings) every ``python3``
    of the workspace, the ``run_code`` bootstrap included, runs in a fresh
    microVM instead of on the host, where the prologue confines it again with
    the Linux backend.

    Returns True when ``python3`` is confined (or is not served by ``local`` at
    all, so it never reaches the host); False, with a warning, when the
    runtime does not have the expected shape.
    """
    bindings = getattr(getattr(ws, "_registry", None), "runtime_bindings", None) or {}
    runtime = bindings.get("python3")
    if runtime is None or runtime_name(runtime) != "local":
        return True
    run = getattr(runtime, "run", None)
    if getattr(run, "confines_shell_python", False):
        return True
    if not inspect.iscoroutinefunction(run):
        warnings.warn(
            "MirageSandbox: cannot confine run_bash's python3 (Mirage's local "
            "runtime has no async `run`); shell python stays unconfined.",
            RuntimeWarning,
            stacklevel=3,
        )
        return False

    async def run_confined(args):
        cfg = active_confine.get()
        if cfg is not None:
            # ``repr`` (not ``json.dumps``) so the embedded config is valid
            # Python source (True/False/None, not JSON true/false/null).
            prologue = (
                "import os, sys\n"
                + CONFINE_PROLOGUE_SRC
                + "\n_confine_platform(%r)\n" % (cfg,)
            )
            args = dataclasses.replace(args, code=prologue + args.code)
        if vm is not None:
            return await run_in_microvm(vm, args)
        return await run(args)

    run_confined.confines_shell_python = True
    runtime.run = run_confined  # instance attribute shadows the class method
    return True


# Which Mirage runtime serves ``python3``. This sandbox needs a *real* CPython:
# the snippet bootstrap dill-loads the persisted namespace, imports third-party
# packages and talks to the host over a unix socket for tool RPC. Mirage's own
# default world binds ``python3`` to ``monty`` (a Rust Python subset) and would
# serve none of that, so ``local`` (host CPython subprocess) is selected here
# and *this* module supplies the isolation: `confine_shell_python` prepends
# the namespace/seccomp prologue to every ``python3`` the shell spawns, exactly
# as ``run_code``'s own bootstrap self-confines.
DEFAULT_RUNTIMES = ("local",)

# Runtimes that keep execution inside a sandbox of Mirage's own: ``monty`` (a
# Rust interpreter with "no host filesystem, environment, or network access"),
# ``wasi`` and ``quickjs`` (CPython / quickjs-ng under wasmtime, whose file I/O
# is routed through the workspace dispatch), and ``vfs`` (the workspace's own
# command engine). ``local`` is deliberately absent: it reaches the host, and is
# only safe under confinement once the run-python patch is installed, which the
# call site checks. Deliberately an allowlist: an unrecognized runtime counts as
# host-escaping rather than being trusted by default.
SANDBOXED_RUNTIMES = frozenset({"monty", "wasi", "quickjs", "vfs"})
SANDBOXED_RUNTIMES_HINT = ", ".join(sorted(SANDBOXED_RUNTIMES))


def runtime_name(entry) -> str:
    """Name of a Mirage runtime entry, given either a name or an instance."""
    if isinstance(entry, str):
        return entry
    return str(getattr(entry, "name", "") or entry.__class__.__name__)


def host_escaping_runtimes(runtimes, patched: bool = False) -> List[str]:
    """Names of ``runtimes`` entries that can execute code on the host.

    Keeps an explicit ``workspace_kwargs={"runtimes": ...}`` from quietly
    defeating confinement; see the call site for the policy. ``local`` is
    excused only when ``patched`` says `confine_shell_python` can apply, so
    every ``python3`` it spawns carries the confinement prologue. ``None``
    (meaning this sandbox's own default world) is resolved first, so the default
    is judged by the same rule as anything a caller passes.
    """
    allowed = set(SANDBOXED_RUNTIMES)
    if patched:
        allowed.add("local")
    entries = runtimes if runtimes else DEFAULT_RUNTIMES
    return [
        name for name in (runtime_name(entry) for entry in entries) if name not in allowed
    ]


def adopt_runtimes(ws, runtimes=DEFAULT_RUNTIMES) -> None:
    """Re-apply this sandbox's runtime world to a copied/loaded workspace.

    ``Workspace.copy()`` and ``Workspace.load()`` both rebuild through Mirage's
    ``_from_state``, which does not carry ``runtimes`` over: the new workspace
    silently reverts to Mirage's default world, where ``python3`` is ``monty``
    rather than the host CPython this sandbox's bootstrap needs. Re-adding the
    entry restores it.

    ``add_runtime`` appends, and the first capturer of a command wins, so this
    only takes effect while nothing ahead of it claims ``python3`` (the usual
    case: Mirage skips ``monty`` when its optional extra is absent). If
    something does, the fork would quietly stop being able to run Python, so
    say so rather than degrade in silence.
    """
    for name in runtimes:
        try:
            ws.add_runtime(name)
        except Exception:  # noqa: BLE001 - already present / unknown to Mirage
            continue
    bindings = getattr(getattr(ws, "_registry", None), "runtime_bindings", None)
    if bindings is None:
        return
    bound = runtime_name(bindings.get("python3")) if bindings.get("python3") else ""
    if bound not in runtimes:
        warnings.warn(
            f"MirageSandbox: 'python3' is served by {bound or 'no runtime'} in "
            f"this workspace, not {'/'.join(runtimes)}; the sandbox bootstrap "
            "needs a real CPython, so Python execution will fail. Pass "
            "workspace_kwargs={'runtimes': [...]} to choose explicitly.",
            RuntimeWarning,
            stacklevel=2,
        )


def ensure_fuse_mounted(ws) -> bool:
    """Mount ``ws``'s virtual filesystem via FUSE if not already; return success.

    Both construction and `fork` need this: mirage-ai 0.0.4 dropped the
    ``Workspace(fuse=True)`` constructor flag (FUSE is now declared per mount,
    or requested afterwards), and ``Workspace.copy()`` does not carry a mount
    over to the child either. Returns False (so the caller falls back to
    unconfined) when FUSE is unavailable or the mount fails.
    """
    if getattr(ws, "fuse_mountpoint", None):
        return True
    add_mount = getattr(ws, "add_fuse_mount", None)
    if add_mount is None:
        return False
    # The sandbox picks the path and records it on the workspace, so every
    # teardown can make sure it is gone (`release_mountpoint`), whatever
    # Mirage's own bookkeeping ends up knowing about it.
    mountpoint = tempfile.mkdtemp(prefix="mirage-")
    ws.synalinks_mountpoint = mountpoint
    try:
        add_mount(DEFAULT_MOUNT, mountpoint=mountpoint)
    except Exception:  # noqa: BLE001 - FUSE unavailable / layout changed
        # Removing the directory now also stops a mount still in flight from
        # landing later with no owner.
        release_mountpoint(mountpoint)
        return False
    return bool(getattr(ws, "fuse_mountpoint", None))


def error_from(stderr: str, exit_code: int) -> Optional[str]:
    """Derive a one-line ``error`` string from a failed run's stderr.

    Returns ``None`` on success (exit 0). Otherwise prefers the final
    non-empty stderr line (the ``ExcType: message`` of a traceback) and falls
    back to a generic exit-code message.
    """
    if exit_code == 0:
        return None
    for line in reversed(stderr.splitlines()):
        if line.strip():
            return line.strip()
    return f"process exited with code {exit_code}"


# Stderr signatures of a sandbox *infrastructure* failure (vs. an error in the
# user's snippet): the confinement bootstrap aborting (it prints
# ``confine-error: ...`` and exits 99) and a dead FUSE backing whose userspace
# daemon went away (``OSError(107, 'Transport endpoint is not connected')``,
# ENOTCONN). Once the mount is dead, *every* subsequent run repeats the error,
# so an agent calling ``run_code`` would loop on it until its wall-clock budget runs
# out. `run_code` detects these, rebuilds the workspace, and retries once.
INFRA_FAILURE_MARKERS = (
    "confine-error",
    "Transport endpoint is not connected",
)

# How many times `run_code` will rebuild the workspace and retry on an infrastructure
# failure before giving up and returning the error. One heal clears the observed
# transient (a stale/dead FUSE mount left by an earlier run); a failure that
# survives a fresh mount is a real environment problem, not worth looping on.
MAX_INFRA_HEALS = 1


def entry_info(path: str, st) -> EntryInfo:
    """Build an E2B-style `EntryInfo` from a Mirage ``FileStat``.

    Mirage's virtual filesystem has no owners or permission bits for most
    backends; those fields then carry E2B's defaults (user ``user``, mode
    ``0o755`` for directories and ``0o644`` for files).
    """
    kind = str(getattr(st.type, "value", st.type))
    file_type = {"directory": FileType.DIR, "symlink": FileType.SYMLINK}.get(
        kind, FileType.FILE
    )
    mode = st.mode if st.mode is not None else (0o755 if kind == "directory" else 0o644)
    try:
        modified = datetime.fromisoformat(str(st.modified).replace("Z", "+00:00"))
    except ValueError:
        modified = datetime.fromtimestamp(0, tz=timezone.utc)
    return EntryInfo(
        name=posixpath.basename(path.rstrip("/")) or "/",
        type=file_type,
        path=path,
        size=0 if file_type is FileType.DIR else int(st.size or 0),
        mode=mode,
        permissions="".join(
            flag if mode & bit else "-"
            for flag, bit in zip(
                "rwxrwxrwx", (0o400, 0o200, 0o100, 0o40, 0o20, 0o10, 4, 2, 1)
            )
        ),
        owner=str(st.uid) if st.uid is not None else "user",
        group=str(st.gid) if st.gid is not None else "user",
        modified_time=modified,
    )


class MirageFilesystem(Filesystem):
    """``sandbox.files`` over the Mirage virtual filesystem (E2B signatures)."""

    async def stat(self, path: str):
        try:
            return await maybe_await(self.sandbox.workspace.stat(path))
        except FileNotFoundError as exc:
            raise FileNotFoundException(str(exc) or path) from exc

    async def shell(self, command: str) -> None:
        _, stderr, exit_code = await self.sandbox.execute(command)
        if exit_code != 0:
            raise OSError(stderr.strip() or f"command failed: {command}")

    @request_deadline
    async def read(
        self,
        path: str,
        format: Literal["text", "bytes", "stream"] = "text",
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
        gzip: bool = False,
        stream_idle_timeout: Optional[float] = None,
    ):
        if format == "text":
            data = await self.sandbox.read_text(path)
            if data is None:
                raise FileNotFoundException(path)
            return data
        result = await self.sandbox.workspace.execute(
            "cat -- %s" % shlex.quote(path),
            session_id=self.sandbox.session_id,
            record=False,
        )
        if result.exit_code != 0:
            raise FileNotFoundException(path)
        data = bytearray(await maybe_await(result.stdout))
        if format == "bytes":
            return data

        async def chunks():
            for start in range(0, len(data), 65536):
                yield bytes(data[start : start + 65536])

        return chunks()

    @request_deadline
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
        if hasattr(data, "read"):
            data = data.read()
        error = await self.sandbox.write(path, data)
        if error:
            raise OSError(error)
        return WriteInfo(
            name=posixpath.basename(path),
            type=FileType.FILE,
            path=path,
            metadata=metadata,
        )

    @request_deadline
    async def list(
        self,
        path: str,
        depth: Optional[int] = 1,
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> List[EntryInfo]:
        entries: List[EntryInfo] = []
        stack = [(path, 1)]
        while stack:
            directory, level = stack.pop()
            for child in await maybe_await(self.sandbox.workspace.readdir(directory)):
                entry = entry_info(child, await self.stat(child))
                entries.append(entry)
                if entry.type is FileType.DIR and (depth is None or level < depth):
                    stack.append((child, level + 1))
        return sorted(entries, key=lambda e: e.path)

    @request_deadline
    async def exists(
        self,
        path: str,
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> bool:
        try:
            await self.stat(path)
        except FileNotFoundError:
            return False
        return True

    @request_deadline
    async def get_info(
        self,
        path: str,
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> EntryInfo:
        return entry_info(path, await self.stat(path))

    @request_deadline
    async def remove(
        self,
        path: str,
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> None:
        await self.shell("rm -rf -- %s" % shlex.quote(path))

    @request_deadline
    async def rename(
        self,
        old_path: str,
        new_path: str,
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> EntryInfo:
        parent = posixpath.dirname(new_path.rstrip("/")) or "/"
        await self.shell(
            "mkdir -p %s && mv -- %s %s"
            % (shlex.quote(parent), shlex.quote(old_path), shlex.quote(new_path))
        )
        return await self.get_info(new_path)

    @request_deadline
    async def make_dir(
        self,
        path: str,
        user: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> bool:
        if await self.exists(path):
            return False
        await self.shell("mkdir -p -- %s" % shlex.quote(path))
        return True

    @request_deadline
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
        """Poll ``path`` and report creations, writes and removals.

        The virtual filesystem has no change notifications, so the tree is
        compared with its previous state every 0.1 s: an entry that appears
        is a ``CREATE``, one whose size or modification time changes a
        ``WRITE``, one that disappears a ``REMOVE``.
        """
        await self.get_info(path)  # a missing directory fails now, as on E2B
        root = path.rstrip("/") or "/"

        async def snapshot() -> Dict[str, EntryInfo]:
            entries = await self.list(root, depth=None if recursive else 1)
            return {entry.path: entry for entry in entries}

        # Taken before returning, as E2B's watch is live once this returns: a
        # change made right after the call is seen.
        initial = await snapshot()

        async def watch():
            error = None
            deadline = time.monotonic() + timeout if timeout else None
            before = initial
            try:
                while deadline is None or time.monotonic() < deadline:
                    await asyncio.sleep(0.1)
                    after = await snapshot()
                    changes = (
                        [
                            (p, FilesystemEventType.CREATE)
                            for p in after
                            if p not in before
                        ]
                        + [
                            (p, FilesystemEventType.WRITE)
                            for p in after
                            if p in before
                            and (after[p].size, after[p].modified_time)
                            != (before[p].size, before[p].modified_time)
                        ]
                        + [
                            (p, FilesystemEventType.REMOVE)
                            for p in before
                            if p not in after
                        ]
                    )
                    for changed, kind in changes:
                        await call_back(
                            on_event,
                            FilesystemEvent(
                                name=posixpath.relpath(changed, root),
                                type=kind,
                                entry=after.get(changed) if include_entry else None,
                            ),
                        )
                    before = after
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reported through on_exit
                error = exc
            await call_back(on_exit, error)

        return WatchHandle(asyncio.ensure_future(watch()))


class MirageCommands(Commands):
    """``sandbox.commands`` over the Mirage shell, confined like `run_bash`
    (E2B signatures). ``envs`` and ``cwd`` apply to the one command: it runs
    in a subshell, so they do not leak into the session."""

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
        environment = {**self.sandbox.envs, **(envs or {})}
        for key in environment:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise InvalidArgumentException(f"invalid environment variable {key!r}")
        prefix = "".join(f"export {k}={shlex.quote(v)}; " for k, v in environment.items())
        if cwd:
            prefix += f"cd {shlex.quote(cwd)} && "
        line = f"( {prefix}{cmd}\n)" if prefix else cmd

        async def execute(stdin_data: bytes):
            return await self.sandbox.run_shell(line, timeout, stdin=stdin_data or None)

        if background:
            pid = self.sandbox.new_pid()
            handle = CommandHandle(
                pid,
                execute,
                stdin=bool(stdin),
                on_stdout=on_stdout,
                on_stderr=on_stderr,
                on_exit=lambda done: self.sandbox.processes.pop(done.pid, None),
                cmd=cmd,
                envs=environment,
                cwd=cwd,
            )
            self.sandbox.processes[pid] = handle
            return handle
        stdout, stderr, exit_code = await execute(b"")
        for line_out in output_lines(stdout):
            await call_back(on_stdout, line_out)
        for line_err in output_lines(stderr):
            await call_back(on_stderr, line_err)
        # As in E2B: a timeout and a non-zero exit raise instead of returning.
        if exit_code == 124 and stderr.startswith("TimeoutError: execution exceeded"):
            raise TimeoutException(stderr)
        if exit_code != 0:
            raise CommandExitException(
                stderr=stderr,
                stdout=stdout,
                exit_code=exit_code,
                error=error_from(stderr, exit_code),
            )
        return CommandResult(stderr=stderr, stdout=stdout, exit_code=0, error=None)


@register_synalinks_serializable()
@synalinks_export(
    [
        "synalinks.sandboxes.MirageSandbox",
        "synalinks.MirageSandbox",
    ]
)
class MirageSandbox(Sandbox):
    """A code-execution sandbox backed by `Mirage <https://mirage.strukto.ai>`_.

    !!! warning "Experimental"
        The sandbox API is experimental and may change in a future
        release.

    Wraps a Mirage ``Workspace``: a virtual filesystem that mounts resources
    (RAM, disk, S3, Postgres, SSH, ...) under virtual paths and runs shell
    commands against them. ``run_code`` executes Python through Mirage's ``python3``
    builtin, and the file tools (`read_file`, `write_file`, ...) operate
    on the mounted virtual filesystem.

    Example:

    ```python
    import synalinks

    sandbox = synalinks.MirageSandbox(timeout=10)
    await sandbox.run_code("x = 6 * 7")
    result = await sandbox.run_code("print(x)")
    print(result.stdout)                    # -> "42\\n"

    # Snapshot + restore (workspace + interpreter state)
    blob = sandbox.dump()
    restored = synalinks.MirageSandbox.load(blob)
    ```

    ## Python state persistence

    Mirage spawns a **fresh** ``python3`` process per command, so Python
    variables would not normally survive between ``run_code`` calls. The sandbox
    bridges this by serializing the interpreter namespace (variables, imports,
    user-defined functions and classes) with ``dill`` after each snippet and
    restoring it before the next. State therefore persists across ``run_code`` calls
    just like a REPL (without replaying earlier snippets), and a snippet that
    raises does not wipe the accumulated namespace.

    ## Confinement (the default; `confine=True`, Linux and macOS)

    By default each ``run_code`` (and any ``python3`` spawned by `run_bash`) is
    **confined**: it enters a fresh user / mount / PID / network namespace and
    ``pivot_root``s into the FUSE-mounted virtual filesystem, so the snippet
    sees **only** the virtual sandbox at ``/`` (and, via the PID namespace, only
    its own processes in ``/proc``: no host PIDs). Its own ``open(...)`` /
    ``pathlib`` land on the mount (one filesystem, shared with the file tools),
    the host filesystem is hidden, and the network is cut. A seccomp denylist
    (``seccomp=True``) shrinks the reachable kernel syscall surface, and the
    runtime is bind-mounted **read-only** (venv / stdlib / system lib/bin, so
    confined code cannot poison files the host later imports). It is rootless
    and entirely in-process (no container runtime): the Python subprocess
    sandboxes itself at startup via ``unshare`` + read-only runtime binds +
    ``pivot_root``, with optional ``RLIMIT_*`` caps. Requires Linux,
    ``/dev/fuse``, libfuse and unprivileged user namespaces.

    **Secure by default, and fail-closed**: the sandbox always uses the
    strongest backend the host can run (there is no knob to pick a weaker
    one; `granted_capabilities` reports which, as ``backend``), and where none
    can run it **raises** rather than run code unconfined. Weakening is always
    explicit: ``confine=False`` opts out, and ``require_confinement=False``
    restores a warn-and-run-unconfined fallback.

    ## Confinement on macOS (a Linux microVM, Seatbelt as the floor)

    On Apple silicon with libkrun installed, each ``python3`` runs in a fresh
    **Linux microVM** (libkrun on Hypervisor.framework, ~0.3 s per run), and
    the Linux confinement above runs *inside* it, unchanged: namespaces, PID
    isolation, seccomp, read-only runtime, cut network, behind a guest kernel
    of its own. The guest serves the virtual filesystem itself (a FUSE mount
    bridged over vsock to this workspace, so macFUSE is not needed), sees no
    host files but this sandbox's own state dir, and inherits no host
    environment variables. The helper process that runs the VM confines
    *itself* with a Seatbelt profile first, so escaping the VM through the
    hypervisor layer still lands in a sandbox. ``backend`` is ``"microvm"``.
    ``extra_binds`` are shared into the VM read-only, except under directories
    the Linux image owns (``/usr``, ``/opt``, ...); host packages that are
    compiled macOS binaries cannot run in the Linux guest.

    Where no microVM can run (Intel Macs, a Mac that is itself a VM, libkrun
    missing), macOS confines through **Seatbelt**, the TrustedBSD MAC policy behind the
    App Sandbox, because none of the Linux mechanism exists on XNU: namespaces,
    ``pivot_root`` and seccomp are Linux kernel features. ``sandbox_init``
    applies a default-deny SBPL profile to the process, which **enforces the
    filesystem and network boundaries plus ``RLIMIT_*``**: the snippet reads
    only the Python runtime it needs, writes only the sandbox's own directories
    and the macFUSE-mounted virtual filesystem, and gets no network unless
    ``confine_network=True`` (the host-tool bridge socket stays reachable either
    way). Reads are restricted, not merely writes. The process stands in the
    mount, so a *relative* ``open()`` lands on the virtual filesystem the file
    tools see; an absolute path names the host, where the profile denies it.

    Where macOS allows it the profile grants *less* than the Linux path:

    * **Mach services by name, three of them.** A bare ``(allow mach-lookup)``
      is the real escape route, since a daemon does the work outside the
      caller's profile. The three allowed are what libSystem looks up while a
      freshly exec'd child starts; none brokers process execution.
    * **write xor execute.** The writable area denies ``file-map-executable``,
      so a snippet cannot ``dlopen`` a dylib it wrote itself.
    * **no reach into ``workdir``.** It is a seed, copied in host-side, and
      stays unreadable and unwritable from inside.

    Subprocesses work as on Linux: exec is limited to the read-only runtime set
    (so not a binary the snippet wrote), and a child **inherits the profile**.

    Guarantees from the Linux path that remain **weaker** (``backend`` in
    `granted_capabilities` reports ``"seatbelt"`` so code can tell):

    * **no PID namespace** - XNU has none. The host process table is hidden
      (``process-info`` and the ``kern.proc`` sysctls are denied) and signals
      reach only the snippet's own process tree, but PIDs are shared.
    * **no syscall filter** - Seatbelt gates MAC operations, not syscall
      numbers, so the seccomp denylist has no counterpart, and the snippet
      shares the host kernel.
    * **fork budget, not a count** - ``RLIMIT_NPROC`` is per *user*, so the cap
      is set to the user's running processes plus ``max_processes``.

    ## Windows (run under WSL2)

    Windows has no confinement backend: ``unshare`` / ``pivot_root`` / FUSE /
    seccomp have no native-Windows equivalent, and Seatbelt is macOS-only. The
    supported way to get it on
    Windows is to run synalinks **inside WSL2** (Windows Subsystem for Linux 2):
    it is a real Linux kernel, so confinement works unchanged, and because WSL2
    is itself a lightweight VM, the confined process is additionally separated
    from the Windows host by the VM boundary. On **native** Windows
    confinement is unavailable, so a default ``MirageSandbox()`` raises (with a
    pointer to WSL2) rather than run an LM's code unconfined. (WSL1 also cannot
    confine: it has no ``/dev/fuse``; use WSL2.)

    ## Unconfined (`confine=False`)

    With ``confine=False`` the snippet runs in a **real, unrestricted CPython
    subprocess** with network access, and two distinct filesystems are in play:

    - The interpreter's own OS file I/O (``open(...)``, ``pathlib``) hits the
      **host** filesystem the Python process runs on.
    - The Mirage **virtual** filesystem (the mounted resources) is reached
      through the shell and the sandbox's file tools (`read_file`,
      `write_file`, `list_files`, `search_files`, `edit_file`,
      `run_python_file`), where mounted S3 buckets, disks, etc. live.

    They are separate spaces, so use the file tools / shell for the mounts and
    ``run_code`` for computation. Choose this only for trusted code that must reach
    the host filesystem or network.

    ## Bound tools and mounts are the real boundary

    Confinement hides the host and cuts the network, but **bound functions and
    mounted resources are capabilities you hand in**: a prompt-injected model
    will use whatever you expose. Bind only the tools the task needs, mount only
    what it needs (read-only where possible), prefer ``allowed_hosts`` over
    ``confine_network`` for egress, and call `granted_capabilities` to audit
    the exact surface before trusting a sandbox with untrusted code.

    ## Bound functions

    Host callables (the ``external_functions`` constructor arg /
    `bind_functions`) are exposed inside the sandbox as **plain synchronous**
    global functions: call them directly, ``result = tool_name(...)``. Because
    Mirage runs the snippet in a separate subprocess, each call is bridged over
    a host Unix socket: the in-sandbox stub RPCs to the host (driving the
    bridge on a background event-loop thread), which runs the real (possibly
    LM-calling) tool and
    returns its JSON-marshalable result synchronously.

    Args:
        timeout (float): Per-snippet execution budget in seconds, what
            ``run_code(timeout=None)`` uses (Default 300, as on E2B). Enforced
            around the underlying Mirage command; on expiry ``run_code``
            raises `TimeoutException`, as on E2B.
        name (str): Optional. Human-readable name for the sandbox.
        workdir (str): Optional. Host directory whose files seed the virtual
            filesystem at construction. The files are copied **into** the mount
            (host-safe: the agent's writes/edits never touch the real
            directory). ``None`` for an empty scratch filesystem.
        resources (dict): Optional. Mapping of virtual mount prefix → Mirage
            resource (e.g. ``{"/": RAMResource()}``), or → a ``(resource,
            MountMode)`` tuple to set that mount's access mode explicitly.
            Defaults to a single in-memory ``RAMResource`` mounted at ``/``.
            **Least-privilege default:** the root scratch mount is ``EXEC``
            (writable, runs ``python3``), but any *other* bare mount defaults to
            ``READ`` (read-only) so an external resource (S3, Postgres, disk) you
            mount cannot be written/corrupted by sandboxed code unless you opt in
            with a ``(resource, MountMode.WRITE)`` / ``EXEC`` tuple.
        mode (MountMode): Optional. Overrides the default mount mode for **all**
            bare resources (i.e. opts out of the per-mount least-privilege rule
            above). ``None`` (default) applies the rule; pass e.g.
            ``MountMode.EXEC`` to make every bare mount read/write/exec.
        session_id (str): Optional. Mirage session whose working directory and
            environment persist across commands. Defaults to ``"default"``.
        confine (bool): Optional. When ``True`` (Linux; macOS via Seatbelt, see
            above), each ``run_code`` executes the snippet in a fresh user /
            mount / network namespace pivoted into the FUSE-mounted virtual
            filesystem: the snippet sees **only** the virtual sandbox at ``/``
            (its ``open()`` lands on the mount, unifying it with the file
            tools), the host filesystem is hidden, and the network is cut.
            Rootless, in-process: no container runtime. Requires ``/dev/fuse``
            and unprivileged user namespaces (on macOS: macFUSE and
            libsandbox); on an unsupported host it is disabled with a
            ``RuntimeWarning`` (graceful fallback). **Defaults to ``True``**
            (secure by default): a plain ``MirageSandbox()`` is confined +
            syscall-filtered + network-cut, and refuses to run where it cannot
            confine. Pass ``confine=False`` to opt out (e.g. when the code must
            reach the host filesystem or network).
        require_confinement (bool): Optional. Confinement **fails closed**.
            When ``True`` (implies ``confine=True``), any condition that would
            otherwise silently fall back to *unconfined* execution (confinement
            unavailable on the host, the FUSE mount failing to come up, or a
            ``fork(confine=True)`` that cannot confine) raises instead of
            warning, and a `run_code` / `run_bash` whose confinement is not active is
            refused. Use this for untrusted (e.g. LM-generated) code, where a
            missing isolation boundary must be a hard error rather than a warning
            that scrolls past. Defaults to ``None``: follows ``confine``, so a
            confined sandbox fails closed. ``False`` restores the graceful
            fallback (warn, then run unconfined).
        seccomp (bool): Optional. Apply a seccomp syscall denylist inside a
            confined run, shrinking the kernel attack surface (no eBPF, ptrace,
            userfaultfd, perf, keyrings, module loading, kexec, namespace ops,
            ``open_by_handle_at``, ...; denied calls return ``EPERM``). Active
            only when confined and on a supported arch (x86_64 / aarch64);
            elsewhere it is a no-op. Defaults to ``True``.
        confine_network (bool): Optional. Keep **full** network access inside a
            confined run (skip the network namespace). All-or-nothing and not
            filtered; prefer ``allowed_hosts`` for restricted egress. Defaults
            to ``False`` (no network).
        allowed_hosts (list): Optional. Egress allowlist. When set, a host-
            mediated ``http_fetch`` tool is bound that reaches only these hosts
            (exact, or ``*.example.com`` for subdomains + apex; redirects are
            re-checked). Under confinement the network is cut, so this tool is
            the *only* egress path and the model cannot open raw sockets around
            it (enforced host-side). Without ``confine=True`` it is advisory
            (raw sockets still work) and a ``RuntimeWarning`` is emitted.
            ``None`` (default) binds no egress tool.
        block_private_egress (bool): Optional. When ``True`` (default), the
            ``http_fetch`` egress tool additionally refuses hosts that resolve
            to non-public addresses (loopback, private RFC 1918, link-local
            including the ``169.254.169.254`` cloud-metadata IP, reserved or
            multicast), re-checked on each redirect, and **pins the connection
            to the validated IP** so a host cannot re-resolve to an internal
            address after the check (no DNS-rebinding window; TLS still verifies
            the cert against the hostname). An SSRF guard so an allowlisted name
            pointing inward (CNAME / poisoned DNS) is refused. Set ``False`` to
            allow reaching internal targets. No effect without ``allowed_hosts``.
        memory_limit_mb (int): Optional. Address-space cap (``RLIMIT_AS``) for a
            confined run, in MiB. ``None`` for no limit.
        cpu_limit_seconds (int): Optional. CPU-time cap (``RLIMIT_CPU``) for a
            confined run, in seconds. ``None`` for no limit.
        malloc_arena_max (int): Optional. Cap on the host process's glibc
            malloc arenas, applied once per process when the first sandbox is
            built (``M_ARENA_MAX``; see ``cap_malloc_arenas``). Without it a
            long-lived host retains a 64 MiB arena per worker thread that ever
            allocated, gigabytes after hours of snippets. ``None`` leaves the
            allocator alone. Defaults to ``2``.
        max_processes (int): Optional. Process cap (``RLIMIT_NPROC``) for a
            confined run (fork-bomb guard). Defaults to 64; ``None`` to disable.
            ``RLIMIT_NPROC`` is counted per user, which on Linux
            means the fresh user namespace the sandbox unshares, so the cap
            applies to the sandbox alone (in the macOS microVM too). Under
            Seatbelt, with no namespace to scope it to, the cap is set to the
            user's running processes plus this budget.
        extra_binds (list): Optional. Additional host directories to bind
            **read-only** into the confined root, on top of the auto-detected
            Python install / site-packages / ``PYTHONPATH`` dirs (which already
            make host-installed libraries importable). Use for libraries in
            nonstandard locations, data files or model weights. A bound dir is
            *visible* in the sandbox at its original path; to ``import`` from it
            it must also be on ``sys.path`` (set ``PYTHONPATH`` or
            ``sys.path.insert(...)`` in the snippet). No effect when unconfined
            (there the whole host filesystem is already visible). Read-only.
        workspace_kwargs (dict): Optional. Extra keyword arguments forwarded to
            the Mirage ``Workspace`` constructor (e.g. ``consistency``,
            ``cache_limit``, ``runtimes``). Under confinement the ``runtimes``
            list is restricted to known-sandboxed engines
            (``monty``, ``quickjs``, ``vfs``, ``wasi``); any other entry (such
            as ``local``, which runs code on the **host** interpreter and so
            bypasses confinement) is rejected under ``require_confinement`` and
            stripped with a warning whenever ``confine`` is active. Mirage's
            default runtime world is already sandboxed, so leaving ``runtimes``
            unset needs no such handling.
        external_functions (dict): Optional. ``name -> callable`` mapping bound
            persistently and exposed inside the sandbox on every run (see
            above).
    """

    filesystem_class = MirageFilesystem
    commands_class = MirageCommands

    description: str = (
        "Code runs in a real Python 3 interpreter (CPython) with the full "
        "standard library and any installed third-party packages. State persists "
        "across runs: variables, imports, functions and classes defined in "
        "earlier runs remain available, and an error does not reset the "
        "namespace. Use `print(...)` to emit results to stdout; the value of the "
        "snippet's last expression is also captured (the `result` convention: "
        "end with a `result` expression to return it). Any tools bound to the "
        "sandbox are exposed as global functions; call them directly, "
        "`result = tool_name(...)`. By default the sandbox is "
        "confined: `open(...)` / `pathlib` and the file tools (`read_file`, "
        "`write_file`, `list_files`, ...) operate on one shared virtual "
        "filesystem (use absolute paths like `/work/out.txt`), the host "
        "filesystem is hidden, and outbound network is disabled; reach the "
        "network only through a bound tool such as `http_fetch` when one is "
        "provided."
    )

    # Class-level default so instances built without ``__init__`` (``load``,
    # ``from_config``, forks) behave like a fresh sandbox.
    malloc_arena_max: Optional[int] = 2

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        name: Optional[str] = None,
        *,
        workdir: Optional[str] = None,
        resources: Optional[Dict[str, Any]] = None,
        mode: Optional[Any] = None,
        session_id: str = "default",
        confine: bool = True,
        require_confinement: Optional[bool] = None,
        seccomp: bool = True,
        confine_network: bool = False,
        allowed_hosts: Optional[List[str]] = None,
        block_private_egress: bool = True,
        memory_limit_mb: Optional[int] = None,
        cpu_limit_seconds: Optional[int] = None,
        max_processes: Optional[int] = 64,
        malloc_arena_max: Optional[int] = 2,
        extra_binds: Optional[List[str]] = None,
        workspace_kwargs: Optional[Dict[str, Any]] = None,
        external_functions: Optional[Dict[str, Callable]] = None,
    ):
        require_mirage()
        Sandbox.__init__(
            self, timeout=timeout, name=name, external_functions=external_functions
        )
        # Before the FUSE mount and the tool-RPC thread below exist: a cap only
        # governs arenas created after it.
        if malloc_arena_max is not None:
            cap_malloc_arenas(malloc_arena_max)
        self.malloc_arena_max = malloc_arena_max
        self.session_id = session_id
        self.mode = mode if mode is not None else MountMode.EXEC
        self.workspace_kwargs = dict(workspace_kwargs or {})
        raw_resources = (
            resources if resources is not None else {DEFAULT_MOUNT: RAMResource()}
        )
        # Least-privilege mounts: the root scratch mount stays EXEC (it must be
        # writable and run ``python3``), but any *other* bare mount defaults to
        # READ (read-only), so a destructive command/snippet cannot corrupt a
        # mounted real resource (S3, Postgres, disk) you did not explicitly make
        # writable. Override per mount with a ``(resource, MountMode)`` tuple, or
        # globally by passing ``mode=``.
        self.resources = {
            prefix: (
                value
                if isinstance(value, tuple)
                else (
                    value,
                    (
                        mode
                        if mode is not None
                        else (
                            MountMode.EXEC if prefix == DEFAULT_MOUNT else MountMode.READ
                        )
                    ),
                )
            )
            for prefix, value in raw_resources.items()
        }
        # The host directory the filesystem was seeded from, or ``None``.
        self.workdir = workdir

        # Confinement: when enabled, ``run_code`` executes the snippet in a fresh
        # user/mount/net namespace pivoted into the FUSE-mounted virtual
        # filesystem (no host fs, no network). Gated on platform support; on an
        # unsupported host it is disabled with a warning (graceful fallback).
        self.confine = bool(confine)
        # Whether ``run_bash``'s ``python3`` carries the confinement prologue.
        # False until the patch actually installs, so the runtime guard below
        # judges the boundary by what is in place, not by what was requested.
        self.run_python_patched = False
        # Fail-closed by default: a confined sandbox that cannot confine raises
        # rather than run LM-generated code unconfined behind a warning that
        # scrolls past. ``None`` follows ``confine``; an explicit False restores
        # the graceful fallback, and ``confine=False`` opts out entirely.
        self.require_confinement = (
            self.confine if require_confinement is None else bool(require_confinement)
        )
        if self.require_confinement and not self.confine:
            raise ValueError(
                "MirageSandbox(require_confinement=True) requires confine=True."
            )
        # Extra host directories bound read-only into the confined root, on top
        # of the auto-detected Python install/package dirs.
        self.extra_binds = list(extra_binds or [])
        self.confine_network = bool(confine_network)
        # Seccomp denylist: prebuilt host-side, applied inside a confined run to
        # shrink the kernel syscall surface. ``None`` when disabled or on an arch
        # without a number table. Active only under confinement (a no-op for an
        # unconfined sandbox, where there is no boundary to harden).
        self.seccomp_blob = build_seccomp_filter() if seccomp else None
        # Egress allowlist: when set, a host-mediated ``http_fetch`` tool is bound
        # that only reaches allowlisted hosts. Combined with a cut network
        # (confined, default), that is the *only* path out; the model cannot
        # open raw sockets around it. ``None`` leaves egress unrestricted-by-tool.
        self.allowed_hosts = list(allowed_hosts) if allowed_hosts is not None else None
        self.block_private_egress = bool(block_private_egress)
        if self.allowed_hosts is not None:
            self.functions.setdefault(
                "http_fetch",
                make_egress_tool(
                    self.allowed_hosts, self.timeout, self.block_private_egress
                ),
            )
            if not self.confine:
                warnings.warn(
                    "MirageSandbox(allowed_hosts=...) without confine=True: the "
                    "allowlist is advisory: unconfined code can open raw sockets "
                    "to any host. Set confine=True to enforce it.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self.rlimits: Dict[str, int] = {}
        if memory_limit_mb:
            self.rlimits["as"] = int(memory_limit_mb) * 1024 * 1024
        if cpu_limit_seconds:
            self.rlimits["cpu"] = int(cpu_limit_seconds)
        if max_processes:
            self.rlimits["nproc"] = int(max_processes)
        # Which mechanism confines: the strongest this host can run, never a
        # knob (see `confinement_backend`). ``vm`` holds the microVM's live
        # launch settings; `set_workspace` / `init_host_state` fill in the
        # mountpoint and host dir as they come to exist.
        self.backend: Optional[str] = None
        self.vm: Optional[Dict[str, Any]] = None
        if self.confine:
            backend, reason = confinement_backend()
            if backend is None:
                self.fall_back_unconfined("MirageSandbox(confine=True)", reason)
            else:
                if backend == "microvm":
                    try:
                        self.vm = {
                            **prepare_microvm(),
                            "network": self.confine_network,
                            "extra_binds": list(self.extra_binds),
                            "cpus": 2,
                            # Guest RAM is committed lazily, so the default
                            # costs only what the snippet touches.
                            "memory_mb": (
                                int(memory_limit_mb) + 512 if memory_limit_mb else 2048
                            ),
                        }
                        # The filter runs in the guest kernel: build it for the
                        # guest's arch, not this host's.
                        self.seccomp_blob = (
                            build_seccomp_filter("aarch64") if seccomp else None
                        )
                    except Exception as exc:  # noqa: BLE001 - fall back, still confined
                        backend = "seatbelt"
                        warnings.warn(
                            "MirageSandbox: the microVM backend could not be "
                            f"prepared ({exc}); confining with Seatbelt instead.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                self.backend = backend
                if seccomp and self.seccomp_blob is None:
                    warnings.warn(
                        "MirageSandbox: seccomp unavailable on this architecture; "
                        "confining without a syscall filter.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                # ``python3`` spawned via ``run_bash`` is confined per workspace
                # in `set_workspace`; this probe only decides, before any
                # workspace exists, whether the ``local`` runtime is safe.
                self.run_python_patched = local_runtime_supported()

        # A runtime decides where a command's code actually executes, so under
        # confinement it is part of the boundary. ``local`` (this sandbox's
        # default, and what ``python3`` needs to be real CPython) spawns a host
        # subprocess that "sees the host filesystem and environment, not the
        # workspace mounts" -- safe here only because the patch above prepends
        # the confinement prologue to every snippet it runs. If that patch could
        # not be installed, or a caller swaps in a runtime we cannot vouch for,
        # the boundary has a hole: hard-error under ``require_confinement``,
        # strip (with a warning) whenever ``confine`` is active. Fail-closed on
        # purpose: anything unrecognized counts as host-escaping, so a runtime
        # added by a future Mirage release cannot silently inherit trust it was
        # never audited for.
        escaping = host_escaping_runtimes(
            self.workspace_kwargs.get("runtimes"),
            patched=self.run_python_patched,
        )
        if escaping and (self.confine or self.require_confinement):
            listed = ", ".join(repr(name) for name in escaping)
            if self.require_confinement:
                raise ValueError(
                    "MirageSandbox(require_confinement=True): runtime "
                    f"{listed} can execute code on the host, bypassing "
                    "confinement. Use only sandboxed runtimes "
                    f"({SANDBOXED_RUNTIMES_HINT}), or 'local' on a host where "
                    "the run-python confinement patch applies."
                )
            warnings.warn(
                f"MirageSandbox: runtime {listed} executes code on the host "
                "and bypasses confinement; dropping it because confine=True.",
                RuntimeWarning,
                stacklevel=2,
            )
            kept = [
                entry
                for entry in (self.workspace_kwargs.get("runtimes") or DEFAULT_RUNTIMES)
                if runtime_name(entry) in SANDBOXED_RUNTIMES
            ]
            # Pin the survivors explicitly (even when empty): dropping the key
            # would fall back to this sandbox's ``local`` default, which is the
            # very thing being removed. An empty list leaves Mirage's bare vfs.
            self.workspace_kwargs["runtimes"] = kept

        self.set_workspace(self.new_workspace())
        self.init_host_state()
        if workdir:
            self.fork_base = self.seed_from_workdir(workdir)

    @classmethod
    def bare(cls, *, timeout: float, name: Optional[str]) -> "MirageSandbox":
        """An instance with every config attribute at its unconfined default.

        `fork` and `load` build through here instead of ``__init__`` (they bring
        their own workspace), so the attribute list lives in one place besides
        the constructor. Callers then override what they inherit.
        """
        instance = cls.__new__(cls)
        Sandbox.__init__(instance, timeout=timeout, name=name)
        instance.session_id = "default"
        instance.mode = MountMode.EXEC
        instance.workspace_kwargs = {}
        instance.resources = {}
        instance.workdir = None
        instance.confine = False
        instance.require_confinement = False
        instance.backend = None
        instance.vm = None
        instance.run_python_patched = False
        instance.confine_network = False
        instance.seccomp_blob = None
        instance.allowed_hosts = None
        instance.block_private_egress = True
        instance.extra_binds = []
        instance.rlimits = {}
        return instance

    def init_host_state(self) -> None:
        """Create the per-sandbox host state and arm the GC finalizer.

        The host directory holds the dill state, per-run result/config files
        and the RPC socket; it is bind-mounted into the confined root so those
        paths still resolve after the pivot. ``state_path`` is created lazily
        on the first ``run_code``. ``fork_base`` is the ``{vpath: text}``
        snapshot `diff` / `merge` compare against (the ``workdir`` seed or the
        fork point).
        """
        self.hostdir = os.path.realpath(tempfile.mkdtemp(prefix="mirage_sandbox_"))
        # Code contexts made by `create_code_context`, by id.
        self.contexts: Dict[str, Dict[str, Any]] = {}
        if self.vm is not None:
            self.vm["hostdir"] = self.hostdir
        self.state_path: Optional[str] = None
        self.fork_base: Dict[str, str] = {}
        # Release the mount even if the sandbox is dropped without ``close()``:
        # leaked mounts pile up until ``mount_max`` and new ones fail silently.
        # The finalizer holds a plain dict, never ``self``, so it does not keep
        # the sandbox alive; `set_workspace` keeps the dict current.
        self.workspace_state = {"ws": self.workspace, "hostdir": self.hostdir}
        self.finalizer = weakref.finalize(self, finalize_workspace, self.workspace_state)

    def set_workspace(self, ws, *, when: str = "") -> None:
        """Adopt ``ws`` as the live workspace and re-check the confinement mount.

        Keeps the GC finalizer pointed at the live workspace and refreshes the
        cached FUSE mountpoint (a new workspace gets a new mount). Under
        ``require_confinement`` a missing mount is a hard error.
        """
        self.workspace = ws
        if getattr(self, "workspace_state", None) is not None:
            self.workspace_state["ws"] = ws
        self.fuse_mountpoint = getattr(ws, "fuse_mountpoint", None)
        if self.vm is not None:
            # The guest serves this workspace's filesystem itself, bridged to
            # the same ``MirageFS`` a host FUSE mount would run.
            from mirage.fuse.fs import MirageFS

            self.vm["fs"] = MirageFS(ws.fs, root_prefix=DEFAULT_MOUNT)
        if self.confine:
            # Each workspace has its own runtime instances, so every new one
            # (construction, reset, heal, fork) needs its ``python3`` wrapped.
            self.run_python_patched = confine_shell_python(ws, self.vm)
        if not self.require_confinement:
            return
        if not self.fuse_mountpoint and self.backend == "namespaces":
            problem = f"the FUSE mount was not established{when}"
        elif not self.run_python_patched:
            problem = "run_bash's python3 cannot be confined in this workspace"
        else:
            return
        # Leave nothing half-usable behind: without a mountpoint every later
        # run refuses, and the workspace (and any mount it holds) is released
        # here since no finalizer may own it yet.
        self.fuse_mountpoint = None
        sync_close_workspace(ws)
        if getattr(self, "workspace_state", None) is not None:
            self.workspace_state["ws"] = None
        raise RuntimeError(
            f"MirageSandbox(require_confinement=True): {problem}; cannot confine."
        )

    def start_confined(
        self,
        ws,
        label: str,
        *,
        backend: Optional[str] = None,
        vm: Optional[Dict[str, Any]] = None,
        seccomp_blob: Optional[str] = None,
    ) -> None:
        """Confine a `bare` instance (from `fork` or `load`) before it adopts ``ws``.

        ``backend`` / ``vm`` / ``seccomp_blob`` carry a parent's settings; with
        no ``backend`` the strongest available is picked, as for a new sandbox.
        Falls back (raising under ``require_confinement``) when nothing can
        confine, or when the namespace backend has no FUSE mount to pivot into.
        """
        reason = ""
        if backend is None:
            backend, reason = confinement_backend()
        # Only the namespace backend needs a host FUSE mount (see
        # `new_workspace`); none is made for a backend that will not run, so
        # a refusal below leaves no mount behind.
        mounted = backend in ("namespaces", "seatbelt") and ensure_fuse_mounted(ws)
        if backend is None or (backend == "namespaces" and not mounted):
            self.fall_back_unconfined(label, reason or "FUSE setup failed")
            return
        self.run_python_patched = local_runtime_supported()
        self.confine = True
        self.backend = backend
        self.seccomp_blob = seccomp_blob
        if backend == "microvm":
            base = vm or {
                **prepare_microvm(),
                "network": self.confine_network,
                "extra_binds": list(self.extra_binds),
                "cpus": 2,
                "memory_mb": 2048,
            }
            # This instance's own host dir and filesystem are filled in as they
            # come to exist.
            self.vm = {k: v for k, v in base.items() if k not in ("hostdir", "fs")}
            self.seccomp_blob = seccomp_blob or build_seccomp_filter("aarch64")

    def fall_back_unconfined(self, label: str, reason: str) -> None:
        """Disable confinement with a warning, or raise if it was required."""
        if self.require_confinement:
            raise RuntimeError(
                f"{label} cannot confine: {reason}. Refusing to run code "
                "unconfined; pass confine=False to opt out explicitly (never "
                "for untrusted code)."
            )
        warnings.warn(
            f"{label} disabled: {reason}. Running without confinement.",
            RuntimeWarning,
            stacklevel=3,
        )
        self.confine = False

    # -- internal helpers ----------------------------------------------------

    def new_workspace(self):
        """Build a Mirage ``Workspace`` for this sandbox's mounts and session.

        The single place a ``Workspace`` is constructed, so construction,
        `reset` and `rebuild_workspace` cannot drift apart. Two details Mirage
        will not infer for us:

        * ``session_id`` names the workspace's default session. Mirage seeds
          exactly that one id and ``execute`` raises ``KeyError`` for any other,
          so the sandbox's session name has to be declared up front.
        * The FUSE mount is requested after construction (mirage-ai 0.0.4 moved
          ``fuse`` off the constructor); a confined sandbox needs it so the
          snippet's own ``open()`` lands on the mount.
        """
        kwargs = dict(self.workspace_kwargs)
        kwargs.setdefault("session_id", self.session_id)
        kwargs.setdefault("runtimes", list(DEFAULT_RUNTIMES))
        ws = Workspace(self.resources, mode=self.mode, **kwargs)
        # A host FUSE mount serves the namespace backend's pivot and Seatbelt's
        # working dir; the microVM guest mounts the filesystem itself.
        if self.confine and self.backend != "microvm":
            ensure_fuse_mounted(ws)
        return ws

    def seed_from_workdir(self, workdir: str) -> Dict[str, str]:
        """Copy a host directory's files into the mount; return the snapshot.

        Host-safe: contents are read once and written into the virtual
        filesystem, so subsequent writes/edits never touch the real directory.
        Returns ``{vpath: text}`` of what was loaded (the branch base).
        """
        base: Dict[str, str] = {}
        root = os.path.abspath(workdir)
        if not os.path.isdir(root):
            return base
        for dirpath, _dirs, files in os.walk(root):
            for fname in files:
                host = os.path.join(dirpath, fname)
                if os.path.islink(host):
                    continue
                try:
                    with open(host, "rb") as fh:
                        data = fh.read()
                except OSError:
                    continue
                vpath = "/" + os.path.relpath(host, root).replace(os.sep, "/")
                run_async_from_sync(self.write(vpath, data))
                base[vpath] = data.decode("utf-8", errors="replace")
        return base

    def ensure_state_path(self) -> str:
        # State lives in the per-sandbox host dir (bound into the confined root
        # so the path resolves after a pivot). Absent until the first run.
        if self.state_path is None:
            self.state_path = os.path.join(self.hostdir, "state.pkl")
        return self.state_path

    def granted_capabilities(self) -> Dict[str, Any]:
        """Audit the privilege surface this sandbox actually grants.

        Bound tools and mounts, not the namespace, are the real boundary for
        confined code, so this returns a single JSON-safe view an operator can
        assert on before trusting a sandbox with untrusted (e.g. LM-generated)
        code: whether it is confined / fail-closed / syscall-filtered, the
        effective network reach, the exact bound-tool names, and each mount's
        access mode. Verify it grants only what the task needs.

        Returns:
            dict: ``confined``, ``backend`` (``"namespaces"`` on Linux;
            ``"microvm"`` on macOS, or ``"seatbelt"`` where no microVM can run,
            which has no PID isolation or syscall filter; ``None`` unconfined),
            ``require_confinement``, ``seccomp``
            (filter active), ``read_only_runtime``, ``network`` (``{"mode": ...}``:
            ``host`` unconfined, else ``cut`` / ``full`` / ``allowlist``),
            ``host_runtimes`` (names of configured runtimes that execute code on
            the host rather than in a sandbox; should be empty when confined),
            ``tools`` (sorted bound-callable names, the egress + capability
            surface) and ``mounts`` (``prefix -> mode``, the actual per-mount
            access mode).
        """
        if not self.confine:
            network = {"mode": "host"}  # unconfined: shares the host network
        elif self.allowed_hosts is not None:
            network = {"mode": "allowlist", "allowed_hosts": list(self.allowed_hosts)}
        elif self.confine_network:
            network = {"mode": "full"}
        else:
            network = {"mode": "cut"}

        return {
            "confined": self.confine,
            "backend": self.backend if self.confine else None,
            "require_confinement": self.require_confinement,
            "seccomp": bool(self.seccomp_blob) and self.confine,
            "read_only_runtime": self.confine,
            "network": network,
            "host_runtimes": host_escaping_runtimes(
                self.workspace_kwargs.get("runtimes"),
                patched=self.run_python_patched,
            ),
            "tools": sorted(self.functions),
            # `load` keeps caller-supplied mounts bare; they use the global mode.
            "mounts": {
                prefix: getattr(m, "name", str(m))
                for prefix, value in self.resources.items()
                for m in [value[1] if isinstance(value, tuple) else self.mode]
            },
        }

    async def execute(
        self,
        command: str,
        *,
        stdin: Optional[bytes] = None,
        timeout: Optional[float] = None,
        record: bool = False,
    ):
        """Run one Mirage command; return ``(stdout, stderr, exit_code)``.

        A ``timeout`` (seconds) bounds the whole command and, on expiry,
        surfaces as ``exit_code = 124`` with a ``TimeoutError`` on stderr,
        mirroring the shell ``timeout`` convention. Mirage logs every recorded
        line in the workspace's ``/.bash_history`` view, so only commands the
        agent itself typed are recorded (``record=True``); the bootstrap
        launch and file-tool plumbing would otherwise fill it.
        """
        coro = self.workspace.execute(
            command, session_id=self.session_id, stdin=stdin, record=record
        )
        try:
            if timeout is not None:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro
        except asyncio.TimeoutError:
            return "", f"TimeoutError: execution exceeded {timeout}s", 124
        out = await maybe_await(result.stdout_str())
        err = await maybe_await(result.stderr_str())
        return out, err, result.exit_code

    async def execute_healing(
        self,
        make_command: Callable[[Optional[dict]], str],
        *,
        stdin: Optional[bytes] = None,
        timeout: Optional[float] = None,
        confine_shell: bool = False,
    ):
        """Run a command, rebuilding the workspace once on an infra failure.

        A dead FUSE mount or a confinement-bootstrap abort makes every later
        command repeat the same error, which an agent would loop on until its
        budget runs out, so the workspace (and its mount) is rebuilt and the
        command retried once. ``make_command`` receives the confinement config,
        recomputed per attempt because a rebuild creates a fresh mount.
        ``confine_shell`` marks an agent-typed shell command: its line is
        recorded in history, and the config is advertised through
        ``active_confine`` so any ``python3`` it spawns self-confines.
        """
        heals = 0
        while True:
            confine_cfg = None
            # Only the namespace backend pivots into a host FUSE mount. Seatbelt
            # confines without one (the snippet then just cannot reach the
            # virtual filesystem), and the microVM guest mounts its own.
            if self.confine and (self.fuse_mountpoint or self.backend != "namespaces"):
                confine_cfg = {
                    "confine": True,
                    "network": self.confine_network,
                    "rlimits": dict(self.rlimits),
                }
                if self.seccomp_blob:
                    confine_cfg["seccomp"] = self.seccomp_blob
            if confine_cfg is not None and self.backend == "microvm":
                # Confined again *inside* the guest by the Linux backend, over
                # the guest image's own runtime dirs (the host's are macOS
                # binaries), pivoting into the guest's mount of the virtual
                # filesystem. The host dir is shared in at its host path; RPC
                # sockets are reached through guest-local proxies.
                confine_cfg.update(
                    mp=GUEST_MOUNT,
                    # ``extra_binds`` are shared into the guest at their host
                    # paths (read-only), so they bind in as they do on Linux.
                    # Both spellings of each: the guest aliases /var/folders.
                    binds=list(GUEST_BINDS)
                    + sorted(
                        {
                            path
                            for d in self.extra_binds
                            for path in (os.path.abspath(d), os.path.realpath(d))
                        }
                    ),
                    rw_binds=[self.hostdir, GUEST_SOCK_DIR],
                    sock_dir=GUEST_SOCK_DIR,
                )
            elif confine_cfg is not None:
                # Host directories bound **read-only** into the confined root:
                # the interpreter's install + venv + system lib/bin dirs, the
                # host's package locations (site-packages, ``PYTHONPATH``) and
                # ``extra_binds``, each at its original path so ``sys.path``
                # resolves after the pivot and host libraries import as they do
                # on the host, while confined code cannot write them. Nothing
                # else of the host filesystem is visible (not ``/etc``,
                # ``/home``). Re-read per run so later installs are picked up.
                candidates = [
                    sys.base_prefix,
                    sys.prefix,
                    "/usr",
                    "/lib",
                    "/lib64",
                    "/bin",
                ]
                if sys.platform == "darwin":
                    # dyld resolves nearly every library out of the shared cache
                    # under /System, and the package prefixes hold the
                    # interpreter's own dependency dylibs (a Homebrew or MacPorts
                    # python links against libssl / libffi living beside it,
                    # outside `sys.prefix`). Without these the Seatbelt profile
                    # denies the reads CPython performs before it can execute a
                    # single line of the snippet. `/Library` is deliberately not
                    # here: it holds third-party application state, and a
                    # python.org framework build under it is covered by
                    # `sys.base_prefix` already.
                    candidates += [
                        "/System",
                        "/opt/homebrew",
                        "/opt/local",
                        "/private/var/db/dyld",
                    ]
                try:
                    import site

                    candidates += list(site.getsitepackages())
                    candidates.append(site.getusersitepackages())
                except Exception:  # noqa: BLE001 - site may be restricted/absent
                    pass
                candidates += [
                    p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p
                ]
                binds: List[str] = []
                for d in candidates + self.extra_binds:
                    if not d or not os.path.isdir(d):
                        continue
                    d = os.path.abspath(d)
                    # A dir under an already-bound one is exposed by its parent.
                    if not any(d == b or d.startswith(b + os.sep) for b in binds):
                        binds.append(d)
                if self.backend == "seatbelt":
                    # Seatbelt confines the process where it stands rather
                    # than pivoting it, so the same sets become profile grants:
                    # the runtime read-only, and writable only the host dir
                    # (state / result / RPC socket) plus the FUSE mount, which
                    # the process also stands in so its relative paths land on
                    # the virtual filesystem. `workdir` is deliberately not
                    # granted: it is a seed copied in host-side, and the agent's
                    # writes must never reach the real directory. Seatbelt
                    # matches resolved paths, and Darwin hands temp dirs out
                    # through the `/var` -> `/private/var` symlink.
                    mp = self.fuse_mountpoint and os.path.realpath(self.fuse_mountpoint)
                    tmpdir = os.path.join(self.hostdir, "tmp")
                    os.makedirs(tmpdir, exist_ok=True)
                    confine_cfg.update(
                        read_paths=[os.path.realpath(b) for b in binds],
                        rw_paths=[p for p in (self.hostdir, mp) if p],
                        # The per-run RPC socket gets a random name inside the
                        # host dir, so the profile grants the directory.
                        rpc_socket_dir=self.hostdir,
                        tmpdir=tmpdir,
                        cwd=mp,
                    )
                else:
                    confine_cfg.update(
                        mp=self.fuse_mountpoint,
                        binds=binds,
                        rw_binds=[self.hostdir],
                    )
            elif self.require_confinement:
                # Defense in depth: even if a later state change (a reset, a
                # fork) left the sandbox unable to confine, never run unconfined.
                raise RuntimeError(
                    "MirageSandbox(require_confinement=True): confinement is not "
                    "active; refusing to execute unconfined."
                )
            command = make_command(confine_cfg)
            token = (
                active_confine.set(confine_cfg) if confine_shell and confine_cfg else None
            )
            try:
                stdout, stderr, exit_code = await self.execute(
                    command, stdin=stdin, timeout=timeout, record=confine_shell
                )
            finally:
                if token is not None:
                    active_confine.reset(token)
            infra_failure = exit_code != 0 and any(
                marker in (stderr or "") for marker in INFRA_FAILURE_MARKERS
            )
            if heals < MAX_INFRA_HEALS and infra_failure:
                heals += 1
                self.rebuild_workspace()
                continue
            return stdout, stderr, exit_code

    # -- execution primitives ------------------------------------------------

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
        import dill

        if context is not None and language is not None:
            raise InvalidArgumentException(
                "You can provide context or language, but not both at the same time."
            )
        if language not in (None, "python", "python3"):
            raise InvalidArgumentException(
                f"language {language!r} is not supported: this sandbox runs Python"
            )
        # As on E2B: ``None`` is the default (the sandbox's), ``0`` no limit.
        timeout = None if timeout == 0 else (timeout or self.timeout or None)
        if context is not None and context.id != DEFAULT_CONTEXT.id:
            state_path = self.context_entry(context)["state_path"]
        else:
            state_path = self.ensure_state_path()
        # Persistently bound functions, plus this call's, exposed in-sandbox.
        functions = {**self.functions, **(external_functions or {})}

        server = None
        sock_path = None
        # Result, config and RPC socket live in the per-sandbox host dir so
        # they are reachable after the confinement pivot (which binds that
        # dir in).
        result_fd, result_path = tempfile.mkstemp(
            prefix="result_", suffix=".json", dir=self.hostdir
        )
        os.close(result_fd)
        config_fd, config_path = tempfile.mkstemp(
            prefix="config_", suffix=".json", dir=self.hostdir
        )
        os.close(config_fd)
        try:
            base_config: Dict[str, Any] = {
                "result": result_path,
                "envs": {**self.envs, **(envs or {})},
            }
            if context is not None:
                base_config["cwd"] = context.cwd
                base_config["cwd_root"] = self.virtual_root()
            if inputs:
                base_config["inputs"] = base64.b64encode(dill.dumps(inputs)).decode(
                    "ascii"
                )
            if functions:
                # A host-side Unix-socket server services in-sandbox tool calls
                # concurrently while the snippet subprocess runs. It lives in the
                # host dir (not the workspace), so it survives a workspace heal.
                # Short name on purpose: AF_UNIX paths cap at 104 bytes on
                # macOS, and the temp dir already uses most of that.
                sock_path = os.path.join(self.hostdir, f"rpc_{uuid.uuid4().hex[:8]}.sock")

                # One length-prefixed JSON request ``{"name", "args", "kwargs"}``
                # per connection; the host callable runs (awaited if async) and
                # its result goes back as ``{"ok": true, "result": ...}`` or
                # ``{"ok": false, "error": ...}``.
                async def handle(reader, writer):
                    try:
                        (length,) = struct.unpack(">I", await reader.readexactly(4))
                        request = json.loads(await reader.readexactly(length))
                        func = functions.get(request.get("name"))
                        if func is None:
                            resp = {
                                "ok": False,
                                "error": f"unknown tool: {request.get('name')}",
                            }
                        else:
                            try:
                                result = await maybe_await(
                                    func(
                                        *(request.get("args") or []),
                                        **(request.get("kwargs") or {}),
                                    )
                                )
                                try:
                                    json.dumps(result)
                                except (TypeError, ValueError):
                                    # DataModels expose ``get_json()``; anything
                                    # else crosses the socket as its ``str``.
                                    try:
                                        result = result.get_json()
                                    except Exception:  # noqa: BLE001
                                        result = str(result)
                                resp = {"ok": True, "result": result}
                            except Exception as exc:  # noqa: BLE001 - to snippet
                                resp = {
                                    "ok": False,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                        body = json.dumps(resp).encode("utf-8")
                    except asyncio.IncompleteReadError:
                        return
                    except Exception as exc:  # noqa: BLE001 - protocol failure
                        body = json.dumps({"ok": False, "error": str(exc)}).encode()
                    try:
                        writer.write(struct.pack(">I", len(body)) + body)
                        await writer.drain()
                        writer.close()
                    except Exception:  # noqa: BLE001 - peer may have gone away
                        pass

                server = await asyncio.start_unix_server(handle, path=sock_path)
                base_config["sock"] = sock_path
                base_config["tools"] = sorted(functions)
            b64_boot = base64.b64encode(BOOTSTRAP.encode("utf-8")).decode("ascii")

            def make_command(confine_cfg: Optional[dict]) -> str:
                with open(config_path, "w", encoding="utf-8") as fh:
                    json.dump({**base_config, **(confine_cfg or {})}, fh)
                return 'python3 -c "%s" %s %s %s' % (
                    LAUNCHER,
                    b64_boot,
                    shlex.quote(state_path),
                    shlex.quote(config_path),
                )

            stdout, stderr, exit_code = await self.execute_healing(
                make_command,
                stdin=code.encode("utf-8"),
                timeout=timeout,
            )
        finally:
            if server is not None:
                server.close()
                try:
                    await server.wait_closed()
                except Exception:  # noqa: BLE001 - server teardown is best-effort
                    pass
            # The bootstrap's report: the raised error and the last
            # expression's value, if any (absent when the process died first).
            try:
                with open(result_path) as fh:
                    report = json.loads(fh.read() or "{}")
            except (OSError, json.JSONDecodeError):
                report = {}
            for path in (result_path, sock_path, config_path):
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        error = report.get("error")
        if error is not None:
            error = ExecutionError(**error)
        elif exit_code == 124 and stderr.startswith("TimeoutError: execution exceeded"):
            # As on E2B, a run past its timeout raises rather than returning an
            # `Execution`; it still lands in the history.
            self.record_run(
                code,
                Execution(
                    logs=Logs(stdout=[stdout] if stdout else []),
                    error=ExecutionError(
                        name="TimeoutError", value=stderr.strip(), traceback=""
                    ),
                ),
            )
            raise execution_timeout_error()
        elif exit_code != 0:
            # Died without reaching the bootstrap's handler: a timeout, a
            # confinement abort, ``sys.exit(n)`` or a crash.
            message = error_from(stderr, exit_code)
            match = re.match(r"([A-Za-z_][\w.]*): (.*)", message)
            name, value = match.groups() if match else ("SandboxError", message)
            # stderr *is* the failure report here; keep it in one place.
            error = ExecutionError(name=name, value=value, traceback=stderr)
            stderr = ""
        if self.malloc_arena_max is not None:
            # The run's buffers (inputs blob, namespace, FUSE copies) are freed
            # by now; give their pages back instead of keeping them in the
            # arenas for the next hour.
            malloc_trim()
        results = []
        if report.get("text") is not None:
            results.append(
                Result(text=report["text"], json=report.get("json"), is_main_result=True)
            )
        execution = self.record_run(
            code,
            Execution(
                results=results,
                logs=Logs(
                    stdout=[stdout] if stdout else [],
                    stderr=[stderr] if stderr else [],
                ),
                error=error,
            ),
        )
        return await self.notify(execution, on_stdout, on_stderr, on_result, on_error)

    # -- code contexts (E2B) ------------------------------------------------

    def virtual_root(self) -> Optional[str]:
        """Where the virtual filesystem's "/" is from inside a run, or ``None``.

        ``""`` once confinement pivoted into it (namespaces, microVM), the host
        mount under Seatbelt, and ``None`` unconfined, where the snippet's
        own file I/O does not reach it.
        """
        if not self.confine:
            return None
        if self.backend == "seatbelt":
            return self.fuse_mountpoint and os.path.realpath(self.fuse_mountpoint)
        return ""

    def context_entry(self, context: Union[Context, str]) -> Dict[str, Any]:
        """The record of a context made by `create_code_context`."""
        context_id = context.id if isinstance(context, Context) else context
        entry = self.contexts.get(context_id)
        if entry is None:
            raise NotFoundException(f"Context {context_id} not found")
        return entry

    @request_deadline
    async def create_code_context(
        self,
        cwd: Optional[str] = None,
        language: Optional[str] = None,
        request_timeout: Optional[float] = None,
    ) -> Context:
        """Create a code context: its own Python namespace, and ``cwd`` (a
        virtual path, default ``/``) as working directory for its runs."""
        if language not in (None, "python", "python3"):
            raise InvalidArgumentException(
                f"language {language!r} is not supported: this sandbox runs Python"
            )
        context = Context(id=uuid.uuid4().hex, language="python", cwd=cwd or "/")
        self.contexts[context.id] = {
            "context": context,
            "state_path": os.path.join(self.hostdir, f"context_{context.id}.dill"),
        }
        return context

    async def list_code_contexts(self) -> List[Context]:
        """The default context, then those made by `create_code_context`."""
        return [DEFAULT_CONTEXT] + [entry["context"] for entry in self.contexts.values()]

    async def restart_code_context(self, context: Union[Context, str]) -> None:
        """Wipe a context's namespace; restarting the default one wipes the
        sandbox's own namespace (its variables, imports, definitions)."""
        context_id = context.id if isinstance(context, Context) else context
        if context_id == DEFAULT_CONTEXT.id:
            self.discard_state()
            return
        path = self.context_entry(context)["state_path"]
        if os.path.exists(path):
            os.unlink(path)

    async def remove_code_context(self, context: Union[Context, str]) -> None:
        """Delete a context and its namespace (the default one cannot be)."""
        context_id = context.id if isinstance(context, Context) else context
        if context_id == DEFAULT_CONTEXT.id:
            raise InvalidArgumentException("The default context cannot be removed")
        await self.restart_code_context(context)
        del self.contexts[context_id]

    async def run_bash(self, command: str) -> dict:
        """Run a shell command in the sandbox's isolated Mirage shell.

        Reach for this for shell work: running programs, pipelines and
        process control; prefer the dedicated ``read_file`` / ``list_files`` /
        ``search_files`` / ``write_file`` / ``edit_file`` tools for plain file
        operations. The command executes against the mounted virtual
        filesystem in the sandbox's persistent session (so ``cd`` / ``export``
        carry across calls). Standard bash is supported (pipes, redirects,
        globs, ``&&`` / ``||``, loops), plus ``python3``.

        Args:
            command (str): The shell command line to execute.

        Returns:
            dict: ``ok`` (exit code 0), ``stdout``, ``stderr`` and
            ``exit_code``.
        """
        stdout, stderr, exit_code = await self.run_shell(command)
        return {
            "ok": exit_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
        }

    async def run_shell(
        self,
        command: str,
        timeout: Optional[float] = None,
        stdin: Optional[bytes] = None,
    ):
        """Run a shell command confined and self-healing, like `run_code`.

        Shared by `run_bash` and ``commands.run``; returns
        ``(stdout, stderr, exit_code)``.
        """
        return await self.execute_healing(
            lambda _cfg: command,
            stdin=stdin,
            # ``None`` means the sandbox default; ``0`` means no limit (E2B).
            timeout=self.timeout if timeout is None else (timeout or None),
            confine_shell=True,
        )

    async def write(self, path: str, data) -> Optional[str]:
        """Write ``data`` (text or bytes) to ``path``, creating parent dirs.

        The one write path behind `write_file`, ``files.write``, seeding and
        `merge`. Returns ``None`` on success, else the error message.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        parent = posixpath.dirname(path.rstrip("/")) or "/"
        command = "mkdir -p %s && cat > %s" % (shlex.quote(parent), shlex.quote(path))
        _, stderr, exit_code = await self.execute(command, stdin=bytes(data))
        if exit_code != 0:
            return stderr.strip() or f"failed to write {path}"
        return None

    async def read_tree(self) -> Dict[str, str]:
        """Snapshot the virtual filesystem as ``{vpath: text}`` (text files)."""
        tree: Dict[str, str] = {}
        for path in await self.walk_files():
            text = await self.read_text(path)
            if text is not None:
                tree[path] = text
        return tree

    def reset(self) -> None:
        self.discard_state()
        self.clear_history()
        # Release the current workspace's FUSE mount before replacing it;
        # otherwise every ``reset`` leaks a mount until ``mount_max`` is hit.
        sync_close_workspace(getattr(self, "workspace", None))
        self.set_workspace(self.new_workspace(), when=" after reset")
        self.fork_base = self.seed_from_workdir(self.workdir) if self.workdir else {}

    def rebuild_workspace(self) -> None:
        """Re-establish the Mirage workspace and its FUSE mount in place.

        Self-heals a dead mount mid-session (``Transport endpoint is not
        connected``) or a confinement bootstrap that could not pivot into it.
        The interpreter namespace lives in the host state file (reloaded by the
        bootstrap on the next run) and run history / bound functions are kept,
        so, unlike `reset`, only the workspace and its mount are rebuilt. The
        virtual filesystem's live contents are lost, but the dead mount had
        already made them unreachable.
        """
        # Called from inside `run_code` / `run_bash`, i.e. with a loop running,
        # which `sync_close_workspace` handles (a blocking close would not).
        sync_close_workspace(getattr(self, "workspace", None))
        self.set_workspace(self.new_workspace(), when=" after an infrastructure failure")

    def disarm_finalizer(self) -> None:
        """Cancel the GC/exit finalizer after an explicit close, and forget the
        workspace so it can't be released a second time."""
        fin = getattr(self, "finalizer", None)
        if fin is not None:
            fin.detach()
        if getattr(self, "workspace_state", None) is not None:
            self.workspace_state["ws"] = None

    @classmethod
    def network_options(
        cls, allow_internet_access: Optional[bool], network: Optional[Any]
    ) -> Dict[str, Any]:
        """Map `create`'s E2B network options onto this sandbox's own.

        ``allow_internet_access=True`` opens the network (``confine_network``);
        ``network={"allow_out": [...]}`` becomes the ``allowed_hosts`` egress
        allowlist (only these hosts, through ``http_fetch``). E2B rules with no
        local equivalent (``deny_out``, ...) raise rather than being dropped.
        """
        options: Dict[str, Any] = {}
        if allow_internet_access:
            options["confine_network"] = True
        if network:
            rules = dict(network)
            allow_out = rules.pop("allow_out", None)
            unsupported = [key for key, value in rules.items() if value]
            if unsupported:
                raise InvalidArgumentException(
                    f"network rules {unsupported} are not supported locally; "
                    "use allow_out (an egress allowlist)"
                )
            if allow_out:
                options["allowed_hosts"] = list(allow_out)
        return options

    async def release(self) -> None:
        """Release the FUSE mount, workspace and host state (for ``kill``)."""
        await self.aclose()

    async def aclose(self) -> None:
        """Async release of the workspace and host state directory."""
        self.killed = True
        Sandbox.live_sandboxes.pop(self.identifier, None)
        self.discard_state()
        ws = getattr(self, "workspace", None)
        if ws is not None:
            await maybe_await(ws.close())
            release_mountpoint(getattr(ws, "synalinks_mountpoint", None))
        rmtree(getattr(self, "hostdir", None))
        self.disarm_finalizer()

    def close(self) -> None:
        """Best-effort, synchronous release of the workspace and state dir.

        Safe to call from any context (with or without a running event loop):
        the FUSE mount is released synchronously via Mirage's
        ``close_sync_parts`` when a loop is already running, and via the full async
        ``close`` otherwise. Teardown never raises.
        """
        self.killed = True
        Sandbox.live_sandboxes.pop(getattr(self, "identifier", None), None)
        self.discard_state()
        sync_close_workspace(getattr(self, "workspace", None))
        rmtree(getattr(self, "hostdir", None))
        self.disarm_finalizer()

    def discard_state(self) -> None:
        if self.state_path and os.path.exists(self.state_path):
            try:
                os.unlink(self.state_path)
            except OSError:
                pass
        self.state_path = None

    def read_state_bytes(self) -> Optional[bytes]:
        if self.state_path and os.path.exists(self.state_path):
            with open(self.state_path, "rb") as fh:
                return fh.read()
        return None

    def write_state_bytes(self, data: Optional[bytes]) -> None:
        if not data:
            return
        path = self.ensure_state_path()
        with open(path, "wb") as fh:
            fh.write(data)

    # -- serialization -------------------------------------------------------

    def dump(self) -> bytes:
        """Serialize the workspace + interpreter state to a JSON byte string.

        Bundles a Mirage workspace snapshot (the virtual filesystem), the
        dill-serialized interpreter namespace, and the run history. Mount
        *resources* themselves are not embedded (they may carry credentials);
        ``load`` re-supplies them, defaulting to a fresh in-memory mount.
        """
        state = self.read_state_bytes()
        snapshot = io.BytesIO()
        resolve_sync(self.workspace.snapshot(snapshot))
        payload = {
            "workspace": base64.b64encode(snapshot.getvalue()).decode("ascii"),
            "state": base64.b64encode(state).decode("ascii") if state else None,
            "history": [dict(entry) for entry in self.run_history],
            "timeout": self.timeout,
            "name": self.name,
            "session_id": self.session_id,
        }
        return json.dumps(payload).encode("utf-8")

    @classmethod
    def load(
        cls,
        data: bytes,
        *,
        resources: Optional[Dict[str, Any]] = None,
        mode: Optional[Any] = None,
        workspace_kwargs: Optional[Dict[str, Any]] = None,
        confine: bool = True,
    ) -> "MirageSandbox":
        """Restore a sandbox from bytes produced by `dump`.

        ``resources`` overrides the mounts for the restored workspace (required
        when the original mounts carried redacted credentials); by default a
        fresh in-memory ``RAMResource`` is mounted at ``/`` and the snapshot's
        files are loaded into it. The restored sandbox is confined, and fails
        closed, exactly like a new one; pass ``confine=False`` to opt out.
        """
        require_mirage()
        payload = json.loads(data.decode("utf-8"))
        mounts = resources if resources is not None else {DEFAULT_MOUNT: RAMResource()}
        ws = resolve_sync(
            Workspace.load(
                io.BytesIO(base64.b64decode(payload["workspace"])), resources=mounts
            )
        )
        # Same as `fork`: a loaded workspace comes back without the runtime
        # world, so ``python3`` would not be the CPython the bootstrap needs.
        adopt_runtimes(ws, (workspace_kwargs or {}).get("runtimes") or DEFAULT_RUNTIMES)
        instance = cls.bare(timeout=payload.get("timeout", 5.0), name=payload.get("name"))
        instance.session_id = payload.get("session_id", "default")
        if mode is not None:
            instance.mode = mode
        instance.workspace_kwargs = dict(workspace_kwargs or {})
        instance.resources = mounts
        if confine:
            instance.require_confinement = True
            instance.start_confined(
                ws,
                "MirageSandbox.load(confine=True)",
                seccomp_blob=build_seccomp_filter(),
            )
        instance.set_workspace(ws)
        instance.init_host_state()
        if payload.get("state"):
            instance.write_state_bytes(base64.b64decode(payload["state"]))
        instance.run_history = [dict(entry) for entry in (payload.get("history") or [])]
        return instance

    def get_config(self) -> dict:
        return {
            "timeout": self.timeout,
            "name": self.name,
            "session_id": self.session_id,
            "data": base64.b64encode(self.dump()).decode("ascii"),
        }

    @classmethod
    def from_config(cls, config: dict) -> "MirageSandbox":
        data_b64 = config.get("data")
        if data_b64:
            return cls.load(base64.b64decode(data_b64))
        return cls(
            timeout=config.get("timeout", 5.0),
            name=config.get("name"),
            session_id=config.get("session_id", "default"),
        )

    # -- branching -----------------------------------------------------------

    @class_method_variant("class_fork")
    async def fork(
        self,
        timeout: Optional[int] = None,
        count: Optional[int] = None,
        *,
        name: Optional[str] = None,
        confine: Optional[bool] = None,
        **opts,
    ) -> List[Union["MirageSandbox", Exception]]:
        """Branch ``count`` (default 1) isolated children off this sandbox's
        current state (E2B's ``fork``); returns them in a list, with the
        exception in place of any child that failed.

        Each child gets an isolated copy of the Mirage workspace (its virtual
        filesystem) and of the Python namespace (variables, imports,
        definitions), as an E2B fork copies the whole sandbox: from then on,
        neither side's writes or assignments reach the other.

        Args:
            timeout (int): Optional. Lifetime of the children in seconds
                (E2B's; see `set_timeout`).
            count (int): Optional. How many children; defaults to 1.
            name (str): Optional name for the child (numbered when
                ``count`` > 1).
            confine (bool): Whether the child is confined to **its own fork**
                (its ``run_code`` / ``run_bash`` python sees only the child's
                virtual filesystem, host hidden, network cut; see ``confine``
                on the constructor). ``None`` (default) inherits this
                sandbox's setting; ``True`` / ``False`` override. An explicit
                ``True`` fails closed when confinement cannot be set up.
        """
        children: List[Union["MirageSandbox", Exception]] = []
        base_name = (
            name if name is not None else (f"{self.name}_fork" if self.name else None)
        )
        for index in range(count or 1):
            try:
                child_name = (
                    f"{base_name}_{index}"
                    if base_name and (count or 1) > 1
                    else base_name
                )
                child = self.bare(timeout=self.timeout, name=child_name)
                child.session_id = self.session_id
                child.mode = self.mode
                child.workspace_kwargs = dict(self.workspace_kwargs)
                child.resources = self.resources
                child.workdir = self.workdir
                child.confine_network = self.confine_network
                child.seccomp_blob = self.seccomp_blob
                child.allowed_hosts = self.allowed_hosts
                child.block_private_egress = self.block_private_egress
                child.extra_binds = list(self.extra_binds)
                # Bound functions (incl. the egress ``http_fetch`` tool) carry to the
                # child, so a confined fork keeps the same allowlisted capabilities.
                child.functions = dict(self.functions)
                child.rlimits = dict(self.rlimits)
                ws = resolve_sync(self.workspace.copy())
                # ``copy`` rebuilds through Mirage's ``_from_state``, which drops the
                # runtime world; without this the child's ``python3`` is not CPython.
                adopt_runtimes(
                    ws, self.workspace_kwargs.get("runtimes") or DEFAULT_RUNTIMES
                )
                # Confine the child to its *own* fork: it needs its own FUSE mount
                # (``copy()`` does not carry one over) so the child's snippet pivots
                # into the child's filesystem, not the parent's. A child only requires
                # confinement (fail-closed) when it is asked to confine at all; an
                # explicit ``confine=False`` fork opts out cleanly.
                want = self.confine if confine is None else bool(confine)
                # Inheriting keeps the parent's fail-closed setting; an explicit
                # ``confine=True`` fails closed like a new sandbox.
                child.require_confinement = (
                    self.require_confinement and want if confine is None else want
                )
                if want:
                    # Same backend as the parent; a fork of an unconfined parent picks
                    # the strongest available, as a new sandbox would.
                    child.start_confined(
                        ws,
                        "MirageSandbox.fork(confine=True)",
                        backend=self.backend if self.confine else None,
                        vm=self.vm,
                        seccomp_blob=self.seccomp_blob,
                    )
                child.set_workspace(ws)
                child.init_host_state()
                # The child branches from the parent's *current* tree, so the child's
                # `diff` reports exactly what it changes from here (a clean boundary).
                child.fork_base = await self.read_tree()
                child.write_state_bytes(self.read_state_bytes())
                if timeout:
                    await child.set_timeout(timeout)
                children.append(child)
            except Exception as exc:  # noqa: BLE001 - E2B reports per-child failures
                children.append(exc)
        return children

    def diff(self) -> dict:
        """Filesystem changes this sandbox made relative to its branch base.

        For a sandbox produced by `fork`, this is exactly the patch the
        child introduced since the fork point. Returns ``{"written":
        [{"path", "kind", "size"}, ...], "deleted": [...]}`` where ``kind`` is
        ``"create"`` (new file) or ``"modify"`` (the path existed in the base).
        """
        base = self.fork_base
        current = run_async_from_sync(self.read_tree())
        written = [
            {
                "path": path,
                "kind": "modify" if path in base else "create",
                "size": len(text.encode()),
            }
            for path, text in sorted(current.items())
            if base.get(path) != text
        ]
        deleted = sorted(p for p in base if p not in current)
        return {"written": written, "deleted": deleted}

    def changes(self) -> Dict[str, List[str]]:
        """Summary of filesystem changes relative to the branch base.

        Returns ``{"written": [...], "deleted": [...]}`` with the virtual paths
        created/modified and removed since this sandbox's base (``workdir`` seed
        or fork point).
        """
        summary = self.diff()
        return {
            "written": [entry["path"] for entry in summary["written"]],
            "deleted": summary["deleted"],
        }

    def save(self, path: str) -> dict:
        """Save the sandbox filesystem to a ``.zip`` archive on the host.

        Snapshots the same merged text-file view as `diff` / `patch`
        (internal ``/.sessions`` and ``/dev`` paths excluded) and writes each
        file into a zip at ``path``. Archive members are the virtual paths with
        the leading ``/`` removed, so unzipping reproduces the workdir tree. A
        missing ``.zip`` suffix is appended and parent directories are created.

        Args:
            path (str): Host path of the archive to write.

        Returns:
            dict: ``path`` (the resolved archive path) and ``files`` (the
            number of files written).
        """
        tree = run_async_from_sync(self.read_tree())
        if not path.endswith(".zip"):
            path = path + ".zip"
        path = os.path.abspath(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for vpath in sorted(tree):
                archive.writestr(vpath.lstrip("/"), tree[vpath])
        return {"path": path, "files": len(tree)}

    def patch(self, *, paths: Optional[List[str]] = None) -> str:
        """Git-style unified diff of changes since this sandbox's branch base.

        Unlike `diff` (a structured, whole-file change set), this returns the
        actual line-level hunks as a single ``git diff``-format string (with
        ``diff --git`` headers, ``/dev/null`` for creates/deletes, ``@@`` hunks
        and ``\\ No newline at end of file`` markers), suitable for
        ``git apply`` / ``patch -p1``. Binary (NUL-containing) files collapse to
        a ``Binary files ... differ`` line. ``paths`` restricts output to a
        subset of virtual paths.
        """
        base = self.fork_base
        current = run_async_from_sync(self.read_tree())
        return render_patch(base, current, None if paths is None else set(paths))

    def merge(
        self,
        other: "MirageSandbox",
        *,
        paths: Optional[List[str]] = None,
        force: bool = False,
        repl: bool = False,
    ) -> dict:
        """Fold another (typically forked) sandbox's changes into this one.

        Replays ``other``'s writes and deletions (relative to its fork point)
        onto this sandbox's filesystem. A path this sandbox also changed since
        the fork is a **conflict**: refused (left untouched, reported) unless
        ``force`` applies ``other``'s version. ``paths`` restricts the merge to
        a subset; ``repl=True`` also adopts ``other``'s interpreter state.

        Returns:
            dict: ``{"written", "deleted", "conflicts", "skipped", "failed",
            "repl_adopted"}``. ``failed`` maps each path whose write or delete
            did not go through to the error; it is not listed as merged.
        """
        base = other.fork_base
        other_current = run_async_from_sync(other.read_tree())
        self_current = run_async_from_sync(self.read_tree())
        selected = None if paths is None else set(paths)
        written: List[str] = []
        deleted: List[str] = []
        conflicts: set = set()
        skipped: set = set()
        failed: Dict[str, str] = {}

        for path in sorted(other_current):
            if selected is not None and path not in selected:
                continue
            if path in base and other_current[path] == base[path]:
                continue  # unchanged in `other`; nothing to merge
            # Conflict iff this sandbox diverged from the fork point for `path`.
            if self_current.get(path) != base.get(path):
                conflicts.add(path)
                if not force:
                    skipped.add(path)
                    continue
            error = run_async_from_sync(self.write(path, other_current[path]))
            if error:
                failed[path] = error
            else:
                written.append(path)

        for path in sorted(base):
            if selected is not None and path not in selected:
                continue
            if path in other_current:
                continue  # not deleted in `other`
            if path not in self_current:
                continue  # already gone here
            if self_current.get(path) != base.get(path):
                conflicts.add(path)
                if not force:
                    skipped.add(path)
                    continue
            _, stderr, exit_code = run_async_from_sync(
                self.execute("rm -f -- %s" % shlex.quote(path))
            )
            if exit_code != 0:
                failed[path] = stderr.strip() or f"failed to delete {path}"
            else:
                deleted.append(path)

        if repl:
            self.write_state_bytes(other.read_state_bytes())

        return {
            "written": written,
            "deleted": deleted,
            "conflicts": sorted(conflicts),
            "skipped": sorted(skipped),
            "failed": failed,
            "repl_adopted": bool(repl),
        }

    # -- tool methods --------------------------------------------------------

    async def walk_files(self) -> List[str]:
        """Return every file path in the virtual filesystem (sorted).

        Walks the merged mount view via Mirage ``readdir`` / ``stat``,
        descending into directories and collecting non-directory paths, and
        skipping Mirage's internal views (`INTERNAL_PATHS`).
        """
        files: List[str] = []
        stack = ["/"]
        seen = set(INTERNAL_PATHS)
        while stack:
            directory = stack.pop()
            if directory in seen:
                continue
            seen.add(directory)
            try:
                children = await maybe_await(self.workspace.readdir(directory))
            except Exception:  # noqa: BLE001 - missing/permission dirs are skipped
                continue
            for child in children:
                if child in seen:
                    continue
                try:
                    st = await maybe_await(self.workspace.stat(child))
                except Exception:  # noqa: BLE001
                    continue
                if is_dir(st):
                    stack.append(child)
                else:
                    files.append(child)
        return sorted(files)

    async def read_text(self, path: str) -> Optional[str]:
        """Read a virtual-filesystem file via the shell; ``None`` if missing.

        Returns the decoded text (Mirage decodes bytes as UTF-8, replacing
        undecodable sequences); these tools target text files.
        """
        stdout, _, exit_code = await self.execute("cat -- %s" % shlex.quote(path))
        return stdout if exit_code == 0 else None

    async def list_files(
        self, pattern: str = "**/*", offset: int = 1, limit: int = 0
    ) -> dict:
        """List files in the mounted virtual filesystem matching a glob.

        Reach for this to discover what files exist before reading, searching
        or editing them.

        Args:
            pattern (str): Glob pattern, e.g. ``'**/*.py'`` (``**`` crosses
                directories). Defaults to ``'**/*'`` (every file).
            offset (int): 1-based index of the first path to return. Defaults
                to 1.
            limit (int): Maximum number of paths to return; 0 returns all
                remaining.

        Returns:
            dict: ``files`` (this page of path strings), ``total``, ``offset``
            and ``truncated``.
        """
        regex = glob_to_regex(pattern)
        files = [p for p in await self.walk_files() if regex.match(p.lstrip("/"))]
        page, truncated = paginate(files, offset, limit)
        return {
            "files": page,
            "total": len(files),
            "offset": max(offset, 1),
            "truncated": truncated,
        }

    async def read_file(self, path: str, offset: int = 1, limit: int = 0) -> dict:
        """Read a text file from the mounted virtual filesystem, by line range.

        Reach for this to inspect a file's contents with line numbers and
        pagination, rather than ``cat``-ing it through ``run_bash``.

        Args:
            path (str): Absolute virtual path, e.g. ``'/src/main.py'``.
            offset (int): 1-based line number to start reading from. Defaults
                to 1.
            limit (int): Maximum number of lines to return; 0 returns all
                remaining lines.

        Returns:
            dict: ``content``, ``start_line`` / ``end_line`` (1-based,
            inclusive), ``total_lines`` and ``truncated``, or ``error`` if the
            file is missing.
        """
        content = await self.read_text(path)
        if content is None:
            return {"error": f"file not found: {path}"}
        lines = content.splitlines(keepends=True)
        page, truncated = paginate(lines, offset, limit)
        start = max(offset, 1)
        return {
            "content": "".join(page),
            "start_line": start,
            "end_line": start + len(page) - 1,
            "total_lines": len(lines),
            "truncated": truncated,
        }

    async def write_file(self, path: str, content: str) -> dict:
        """Write a text file in the mounted virtual filesystem.

        Reach for this to create a new file or overwrite one wholesale; use
        ``edit_file`` for a targeted change to existing content. Creates parent
        directories as needed. The write lands in the mounted resource (e.g.
        the in-memory RAM filesystem).

        Args:
            path (str): Absolute virtual path to write, e.g. ``'/PLAN.md'``.
            content (str): The text to write.

        Returns:
            dict: ``written`` (the path) and ``bytes`` (count written), or
            ``error`` on failure.
        """
        error = await self.write(path, content)
        if error:
            return {"error": error}
        return {"written": path, "bytes": len(content.encode("utf-8"))}

    async def edit_file(
        self, path: str, old: str, new: str, replace_all: bool = False
    ) -> dict:
        """Replace text in a file in the mounted virtual filesystem.

        Reach for this to make a targeted in-place change to existing content;
        use ``write_file`` to create a file or replace it wholesale.

        Args:
            path (str): Absolute virtual path of the file to edit.
            old (str): The exact text to replace. Must occur exactly once
                unless ``replace_all`` is true.
            new (str): The text to replace it with.
            replace_all (bool): Replace every occurrence instead of requiring a
                unique match. Defaults to false.

        Returns:
            dict: ``path`` and ``replacements`` (count made), or ``error`` if
            the file is missing, or ``old`` is empty / absent / not unique.
        """
        content = await self.read_text(path)
        if content is None:
            return {"error": f"file not found: {path}"}
        if not old:
            return {"error": "`old` must be a non-empty string"}
        occurrences = content.count(old)
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
        new_text = content.replace(old, new, -1 if replace_all else 1)
        result = await self.write_file(path, new_text)
        if "error" in result:
            return result
        return {"path": path, "replacements": replacements}

    async def search_files(
        self, pattern: str, glob: str = "**/*", offset: int = 1, limit: int = 100
    ) -> dict:
        """Search file contents for a regex across files matching a glob.

        Reach for this to locate where a string or symbol appears across the
        tree before opening individual files.

        Args:
            pattern (str): Regular expression to search for in file contents
                (matched per line).
            glob (str): Glob selecting which files to search, e.g.
                ``'**/*.py'``. Defaults to ``'**/*'`` (all files).
            offset (int): 1-based index of the first match to return. Defaults
                to 1.
            limit (int): Maximum number of matches to return; 0 returns all.
                Defaults to 100.

        Returns:
            dict: ``matches`` (a page of ``{path, line, text}`` records with
            1-based line numbers), ``total``, ``offset`` and ``truncated``, or
            ``error`` on a bad regex.
        """
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return {"error": f"invalid regex: {exc}"}
        glob_regex = glob_to_regex(glob)
        matches: List[Dict[str, Any]] = []
        for path in await self.walk_files():
            if not glob_regex.match(path.lstrip("/")):
                continue
            content = await self.read_text(path)
            if content is None:
                continue
            for lineno, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    matches.append({"path": path, "line": lineno, "text": line})
        page, truncated = paginate(matches, offset, limit)
        return {
            "matches": page,
            "total": len(matches),
            "offset": max(offset, 1),
            "truncated": truncated,
        }

    async def run_python_file(self, path: str) -> dict:
        """Run a Python script file from the mounted virtual filesystem.

        Reads ``path`` from the virtual filesystem and executes its contents in
        the sandbox, sharing the interpreter namespace with prior runs.

        Args:
            path (str): Absolute virtual path of the ``.py`` file to run.

        Returns:
            dict: ``ok`` (bool), ``stdout`` and ``stderr`` (captured output),
            and ``error`` (a message string, or null on success), or ``error``
            if the file is missing.
        """
        content = await self.read_text(path)
        if content is None:
            return {"error": f"file not found: {path}"}
        return await self.run_python_code(content)
