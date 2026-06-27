# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
import base64
import contextvars
import difflib
import inspect
import io
import json
import os
import re
import shlex
import struct
import sys
import tempfile
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from synalinks.src.api_export import synalinks_export
from synalinks.src.sandboxes.sandbox import ExecutionResult
from synalinks.src.sandboxes.sandbox import Sandbox
from synalinks.src.saving.object_registration import register_synalinks_serializable

try:
    import mirage
    from mirage import MountMode
    from mirage import RAMResource
    from mirage import Workspace
    from mirage.bridge.sync import run_async_from_sync
except ImportError:  # pragma: no cover - exercised only without mirage installed
    mirage = None
    MountMode = None
    RAMResource = None
    Workspace = None
    run_async_from_sync = None


# The launcher is what ``python3 -c`` actually runs: it decodes and execs the
# real bootstrap (argv[1]) so the bootstrap source never has to survive shell
# quoting. argv[2] is the dill session-state path; argv[3] is a base64 JSON
# config (per-call ``inputs`` blob, host-tool RPC socket + tool names, and the
# path to write the result value to). Every arg is non-empty on purpose:
# Mirage's shell drops empty ``''`` tokens, which would shift ``argv``.
_LAUNCHER = "import base64,sys;exec(base64.b64decode(sys.argv[1]))"

# The bootstrap runs inside Mirage's real CPython subprocess. Mirage spawns a
# fresh ``python3`` per command, so to make variables/imports/functions persist
# across ``run`` calls we serialize the user namespace with ``dill`` after each
# snippet and restore it before the next — true REPL state without replaying
# earlier snippets (and their side effects). The namespace lives in a dedicated
# dict (``ns``) the snippet runs in, pickled with explicit file I/O:
# ``dill.dump_module`` is deliberately avoided because it embeds its origin path
# in the pickle, so a re-dump of a *copied* state file (a fork) would write back
# to the original, breaking isolation. Unpicklable values are skipped per-key.
#
# Host callables bound to the sandbox can't be injected into this isolated
# subprocess directly, so each is exposed as an async stub that RPCs to a host
# Unix-socket server (length-prefixed JSON, one request/response per call); the
# host runs the real (possibly LM-calling) tool and returns the result. The
# snippet is exec'd under the filename ``<sandbox>`` so tracebacks trim to user
# frames; its last *expression* (the ``result`` convention) is JSON-encoded
# to the result file so the host can surface it as ``ExecutionResult.result``.
# Rootless in-process confinement, shared by the ``run`` bootstrap and the
# ``run_bash`` ``_run_python`` patch: enter fresh user/mount/PID(/net)
# namespaces, fork (PID namespaces only apply to children) so the child is PID 1
# of the new namespace, bind the Python runtime into the FUSE-mounted virtual
# filesystem, then pivot_root into it so the process sees ONLY the virtual
# sandbox at "/" with the host filesystem gone and a fresh /proc that shows only
# namespaced PIDs. Must run first, single-threaded, before the snippet imports
# anything (the runtime binds keep imports working).
_CONFINE_SRC = r"""
def _confine(cfg):
    import ctypes
    import ctypes.util
    import os
    import platform
    import resource

    libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
    sys_pivot = {
        "x86_64": 155,
        "aarch64": 41,
        "armv7l": 218,
        "ppc64le": 203,
        "s390x": 217,
    }.get(platform.machine())
    if sys_pivot is None:
        raise OSError("unsupported architecture for pivot_root: " + platform.machine())
    NEWUSER, NEWNS, NEWNET, NEWIPC, NEWUTS, NEWPID = (
        0x10000000,
        0x20000,
        0x40000000,
        0x8000000,
        0x4000000,
        0x20000000,
    )
    MS_REC, MS_PRIVATE, MS_BIND, MNT_DETACH = 16384, 1 << 18, 4096, 2
    MS_REMOUNT, MS_RDONLY, MS_NOSUID, MS_NODEV = 32, 1, 2, 4

    def _ck(rc, what):
        if rc != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), what)

    def _mount(src, tgt, fs, flags, data=None):
        enc = lambda x: x.encode() if x else None
        _ck(libc.mount(enc(src), enc(tgt), enc(fs), flags, enc(data)), "mount " + tgt)

    def _bind(d, readonly):
        # Bind ``d`` at the same path under the new root. A bind mount inherits
        # the source's writability, so making it read-only needs a *second*
        # remount (MS_BIND | MS_REMOUNT | MS_RDONLY) — a single mount() cannot
        # express a read-only bind. Read-only binds (the Python runtime: venv,
        # stdlib, system lib/bin) stop confined code from poisoning files the
        # *host* later imports outside the sandbox; ``_hostdir`` is bound
        # writable because the dill state / result file / RPC socket live there.
        if not os.path.isdir(d):
            return
        try:
            os.makedirs(mp + d, exist_ok=True)
            _mount(d, mp + d, None, MS_BIND | MS_REC)
        except OSError:
            return  # cannot bind this dir at all — skip it
        if readonly:
            # Lock it read-only and FAIL CLOSED if that can't be done: swallowing
            # the failure would leave the runtime writable while we still claim
            # (and ``granted_capabilities`` reports) read-only — a fail-open that
            # re-opens the venv-poisoning escape. This remount is deliberately
            # OUTSIDE the try/except so an error propagates to ``_confine`` →
            # ``sys.exit(99)``. ``nosuid``/``nodev`` are re-specified so the
            # remount doesn't EPERM trying to drop those locked flags on a
            # hardened source mount (e.g. a venv on a ``nosuid``/``nodev``
            # ``/tmp`` or ``/home``); the runtime needs neither setuid bits nor
            # device nodes.
            _mount(
                d,
                mp + d,
                None,
                MS_BIND | MS_REMOUNT | MS_RDONLY | MS_NOSUID | MS_NODEV,
            )

    uid, gid = os.getuid(), os.getgid()  # capture before unshare (later: overflow id)
    flags = NEWUSER | NEWNS | NEWIPC | NEWUTS | NEWPID
    if not cfg.get("network"):
        flags |= NEWNET
    os.unshare(flags)
    with open("/proc/self/setgroups", "w") as fh:
        fh.write("deny")
    with open("/proc/self/uid_map", "w") as fh:
        fh.write("0 %d 1" % uid)
    with open("/proc/self/gid_map", "w") as fh:
        fh.write("0 %d 1" % gid)
    # NEWPID only takes effect for *children*: the unsharing process stays in the
    # old PID namespace, so we must fork. The child is PID 1 of the new namespace
    # and does the rest of the confinement (mount /proc, pivot_root, seccomp) and
    # runs the snippet; because it mounts a fresh procfs while inside the new
    # PID namespace, that /proc shows ONLY namespaced PIDs — host processes are
    # no longer visible, removing any reliance on the userns credential check to
    # keep /proc/<host-pid>/root and /environ unreadable. The parent waits and
    # propagates the child's exit status (so the host still sees the real code).
    _pid = os.fork()
    if _pid:
        _, _status = os.waitpid(_pid, 0)
        if os.WIFSIGNALED(_status):
            os._exit(128 + os.WTERMSIG(_status))
        os._exit(os.WEXITSTATUS(_status) if os.WIFEXITED(_status) else 1)
    # Child (PID 1): die if the parent is killed (e.g. host timeout-kill), so the
    # snippet can't outlive the process the host is waiting on.
    try:
        libc.prctl(1, 9, 0, 0, 0)  # PR_SET_PDEATHSIG, SIGKILL
    except Exception:
        pass
    mp = cfg["mp"]
    _mount(None, "/", None, MS_REC | MS_PRIVATE)
    _mount(mp, mp, None, MS_BIND | MS_REC)
    for d in cfg.get("binds", []):
        _bind(d, True)
    for d in cfg.get("rw_binds", []):
        _bind(d, False)
    try:
        os.makedirs(mp + "/proc", exist_ok=True)
        _mount("proc", mp + "/proc", "proc", 0)
    except OSError:
        pass
    os.chdir(mp)
    os.makedirs(".oldroot", exist_ok=True)
    _ck(libc.syscall(sys_pivot, b".", b".oldroot"), "pivot_root")
    os.chdir("/")
    libc.umount2(b"/.oldroot", MNT_DETACH)
    try:
        os.rmdir("/.oldroot")
    except OSError:
        pass
    rl = cfg.get("rlimits") or {}
    for key, which in (
        ("as", resource.RLIMIT_AS),
        ("cpu", resource.RLIMIT_CPU),
        ("nproc", resource.RLIMIT_NPROC),
        ("fsize", resource.RLIMIT_FSIZE),
    ):
        if rl.get(key):
            resource.setrlimit(which, (rl[key], rl[key]))
    # Seccomp goes LAST: it denies mount/unshare/pivot_root, which the steps
    # above still need. The filter (a prebuilt classic-BPF ``sock_filter[]``
    # blob, base64'd by the host) shrinks the kernel attack surface a confined
    # process can reach — eBPF, ptrace, userfaultfd, perf, keyrings, module
    # loading, kexec, namespace ops, open_by_handle_at, etc. — returning EPERM.
    sec = cfg.get("seccomp")
    if sec:
        import base64 as _b64

        blob = _b64.b64decode(sec)
        PR_SET_NO_NEW_PRIVS, PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 38, 22, 2
        libc.prctl.restype = ctypes.c_int
        libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
        # No-new-privs is mandatory to load a filter without privilege, and
        # stops a denied-but-setuid path from regaining what the filter drops.
        _ck(libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "no_new_privs")

        class _SockFprog(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]

        buf = ctypes.create_string_buffer(blob, len(blob))
        prog = _SockFprog(len(blob) // 8, ctypes.cast(buf, ctypes.c_void_p))
        _ck(
            libc.prctl(
                PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.addressof(prog), 0, 0
            ),
            "seccomp",
        )
"""

_BOOTSTRAP = (
    r"""
import os, sys, base64, json, ast
import dill
state = sys.argv[2]
config = {}
if len(sys.argv) > 3 and sys.argv[3]:
    try:
        config = json.loads(base64.b64decode(sys.argv[3]).decode("utf-8"))
    except Exception as exc:
        print("config-warn: " + repr(exc), file=sys.stderr)
"""
    + _CONFINE_SRC
    + r"""
if config.get("confine"):
    try:
        _confine(config)
    except Exception as exc:
        print("confine-error: " + repr(exc), file=sys.stderr)
        sys.exit(99)
ns = {"__name__": "__main__", "__builtins__": __builtins__}
if os.path.exists(state):
    try:
        with open(state, "rb") as fh:
            ns.update(dill.load(fh))
    except Exception as exc:
        print("restore-warn: " + repr(exc), file=sys.stderr)
_inputs = config.get("inputs")
if _inputs:
    try:
        ns.update(dill.loads(base64.b64decode(_inputs)))
    except Exception as exc:
        print("inputs-warn: " + repr(exc), file=sys.stderr)
_sock = config.get("sock")
if _sock and config.get("tools"):
    import asyncio, struct, threading
    async def _rpc(_name, **kwargs):
        reader, writer = await asyncio.open_unix_connection(_sock)
        payload = json.dumps({"name": _name, "kwargs": kwargs}).encode("utf-8")
        writer.write(struct.pack(">I", len(payload)) + payload)
        await writer.drain()
        header = await reader.readexactly(4)
        (length,) = struct.unpack(">I", header)
        body = await reader.readexactly(length)
        writer.close()
        resp = json.loads(body.decode("utf-8"))
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "tool call failed"))
        return resp.get("result")
    # Tools are exposed as PLAIN SYNC functions: `result = submit(...)`, with no
    # `await` / `asyncio.run(...)`. Each call's RPC coroutine runs on a dedicated
    # background event-loop thread and the result is returned synchronously, so
    # this works from a flat script and even from inside a snippet that happens
    # to run its own event loop.
    _tool_loop = asyncio.new_event_loop()
    threading.Thread(target=_tool_loop.run_forever, daemon=True).start()
    def _make_stub(_name):
        def _stub(**kwargs):
            return asyncio.run_coroutine_threadsafe(
                _rpc(_name, **kwargs), _tool_loop
            ).result()
        return _stub
    for _tool_name in config["tools"]:
        ns[_tool_name] = _make_stub(_tool_name)
user = sys.stdin.buffer.read().decode("utf-8")
value = None
try:
    tree = ast.parse(user, "<sandbox>", "exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body.pop()
        if tree.body:
            exec(compile(tree, "<sandbox>", "exec"), ns)
        value = eval(compile(ast.Expression(last.value), "<sandbox>", "eval"), ns)
    else:
        exec(compile(tree, "<sandbox>", "exec"), ns)
finally:
    keep = {}
    for key, item in list(ns.items()):
        if key == "__builtins__":
            continue
        try:
            dill.dumps(item)
            keep[key] = item
        except Exception:
            pass
    try:
        with open(state, "wb") as fh:
            dill.dump(keep, fh)
    except Exception as exc:
        print("persist-warn: " + repr(exc), file=sys.stderr)
    result_path = config.get("result")
    if result_path:
        try:
            encoded = json.dumps(value)
        except Exception:
            encoded = "null"
        try:
            with open(result_path, "w") as fh:
                fh.write(encoded)
        except Exception as exc:
            print("result-warn: " + repr(exc), file=sys.stderr)
"""
)

_DEFAULT_MOUNT = "/"


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


def _unified_body(
    from_file: str,
    to_file: str,
    old_lines: List[str],
    new_lines: List[str],
) -> str:
    """Render the ``---``/``+++``/``@@`` body of a unified diff for one file.

    Wraps ``difflib.unified_diff`` and appends git's ``\\ No newline at end of
    file`` marker after any hunk line whose source lacked a trailing newline,
    so the output applies cleanly. Once inside a hunk we never re-classify
    lines by prefix — a deleted line like ``--foo`` becomes ``---foo`` and must
    not be mistaken for a header.
    """
    parts: List[str] = []
    in_hunk = False
    for line in difflib.unified_diff(
        old_lines, new_lines, fromfile=from_file, tofile=to_file, n=3
    ):
        if not in_hunk:
            parts.append(line)  # ``--- ``/``+++ `` headers and first ``@@``
            if line.startswith("@@"):
                in_hunk = True
            continue
        if line.startswith("@@"):
            parts.append(line)
        elif line.endswith("\n"):
            parts.append(line)
        else:
            parts.append(line + "\n\\ No newline at end of file\n")
    return "".join(parts)


def _render_patch(
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
        out.append(_unified_body(from_file, to_file, old_lines, new_lines))
    return "".join(out)


def _glob_to_regex(pattern: str) -> "re.Pattern":
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


def _clean_traceback(stderr: str) -> str:
    """Drop bootstrap/launcher frames from a traceback, keeping user frames.

    The snippet runs under the filename ``<sandbox>``; everything before the
    first ``File "<sandbox>"`` frame is harness scaffolding, so we keep the
    ``Traceback`` header and splice straight to the user frames.
    """
    lines = stderr.splitlines(keepends=True)
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("Traceback")), None
    )
    sandbox_idx = next((i for i, ln in enumerate(lines) if '"<sandbox>"' in ln), None)
    if header_idx is None or sandbox_idx is None or sandbox_idx <= header_idx:
        return stderr
    kept = [lines[header_idx]] + lines[sandbox_idx:]
    return "".join(kept)


def _jsonable(value: Any) -> Any:
    """Coerce a host tool's return into something JSON-serializable.

    Pass-through when already JSON-safe; falls back to a DataModel's
    ``get_json()`` (the convention sandbox tool adapters follow) and finally to
    ``str``, so the RPC response always crosses the socket.
    """
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if hasattr(value, "get_json"):
            try:
                return value.get_json()
            except Exception:  # noqa: BLE001 - fall through to str
                pass
        return str(value)


def _rmtree(path: str) -> None:
    """Best-effort recursive delete of a host temp directory."""
    import shutil

    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:  # noqa: BLE001 - teardown must not raise
        pass


def _is_wsl() -> bool:
    """Whether this process runs inside the Windows Subsystem for Linux."""
    try:
        with open("/proc/sys/kernel/osrelease") as fh:
            return "microsoft" in fh.read().lower()
    except OSError:
        return False


# One-time native-Windows hint. The sandbox's confinement is Linux-only, so on
# native Windows code runs unconfined; steer users to WSL2. Emitted once per
# process when this module is first imported (it is pulled in by ``import
# synalinks`` via the code-running agents). Suppressible / a no-op off Windows.
_WINDOWS_HINTED = False

_WINDOWS_HINT = (
    "synalinks: on native Windows the code-running sandbox confinement "
    "is unavailable, so LM-generated code runs UNCONFINED. Run synalinks inside "
    "WSL2 (Windows Subsystem for Linux) for the secure, confined sandbox. "
    "Set SYNALINKS_NO_WINDOWS_HINT=1 to silence this message."
)


def _maybe_warn_native_windows():
    """Warn once on native Windows that confinement needs WSL2; return the message.

    A no-op (returns ``None``) off Windows, when already emitted this process, or
    when ``SYNALINKS_NO_WINDOWS_HINT`` is set. Inside WSL2 ``platform.system()``
    reports ``"Linux"``, so this correctly stays silent there.
    """
    import platform
    import warnings

    global _WINDOWS_HINTED
    if _WINDOWS_HINTED or platform.system() != "Windows":
        return None
    _WINDOWS_HINTED = True
    if os.environ.get("SYNALINKS_NO_WINDOWS_HINT"):
        return None
    warnings.warn(_WINDOWS_HINT, RuntimeWarning, stacklevel=2)
    return _WINDOWS_HINT


_maybe_warn_native_windows()


def _confinement_available() -> tuple[bool, str]:
    """Whether in-process confinement (FUSE + user-namespace pivot) can run.

    Returns ``(ok, reason)``; ``reason`` explains why not when ``ok`` is False.
    Requires Linux, ``os.unshare`` (Python 3.12+), ``/dev/fuse``, and
    unprivileged user namespaces. The confinement is Linux-only by
    construction (``unshare`` / ``pivot_root`` / FUSE / seccomp have no Windows
    equivalent), so on Windows the intended path is to run inside **WSL2**,
    where it works unchanged — the reason string says so, and
    ``require_confinement=True`` turns that into a hard error rather than a
    silent unconfined run.
    """
    import platform

    system = platform.system()
    if system == "Windows":
        return False, (
            "confinement requires Linux; on Windows, run synalinks inside WSL2 "
            "(Windows Subsystem for Linux), where confinement works unchanged"
        )
    if system != "Linux":
        return False, f"confinement requires Linux (this host is {system})"
    if not hasattr(os, "unshare"):
        return False, "confinement requires os.unshare (Python 3.12+)"
    if not os.path.exists("/dev/fuse"):
        reason = "confinement requires FUSE (/dev/fuse is absent)"
        if _is_wsl():
            # WSL1 has no real kernel / FUSE; WSL2 does. Point the user there.
            reason += "; on WSL use WSL2 (WSL1 cannot confine)"
        return False, reason
    try:
        with open("/proc/sys/kernel/unprivileged_userns_clone") as fh:
            if fh.read().strip() == "0":
                return False, "unprivileged user namespaces are disabled"
    except OSError:
        pass  # absent on distros where userns is enabled by default
    # The checks above are cheap capability probes; they pass on environments
    # that still *forbid* the credential-map handshake the real confinement
    # performs (notably GitHub Actions and many containers deny writing
    # ``/proc/self/setgroups``). Settle it by actually attempting the setup.
    return _confinement_smoke_test()


_confine_smoke_cache: Optional[Tuple[bool, str]] = None


def _confinement_smoke_test() -> "Tuple[bool, str]":
    """Attempt the unprivileged user-namespace credential handshake for real.

    Forks a throwaway child that runs the exact first steps of ``_confine``
    (``unshare`` the namespaces, then write ``setgroups``/``uid_map``/``gid_map``)
    and reports whether they succeeded. The fork keeps the ``unshare`` off the
    host process; the child only ever ``os._exit``s. Without this, environments
    that allow ``unshare`` but deny the map writes (GitHub Actions, some
    containers) pass the cheap probes, so the documented graceful fallback to
    unconfined never fires and every confined ``run`` dies with
    ``confine-error: PermissionError(13) ... /proc/self/setgroups``. Cached:
    the answer is constant for the process lifetime.
    """
    global _confine_smoke_cache
    if _confine_smoke_cache is not None:
        return _confine_smoke_cache
    try:
        pid = os.fork()
    except OSError as exc:
        _confine_smoke_cache = (False, f"cannot fork to probe confinement: {exc}")
        return _confine_smoke_cache
    if pid == 0:  # child: bare-minimum work, never returns to the caller
        try:
            NEWUSER, NEWNS, NEWNET, NEWIPC, NEWUTS, NEWPID = (
                0x10000000,
                0x20000,
                0x40000000,
                0x8000000,
                0x4000000,
                0x20000000,
            )
            uid, gid = os.getuid(), os.getgid()
            os.unshare(NEWUSER | NEWNS | NEWNET | NEWIPC | NEWUTS | NEWPID)
            with open("/proc/self/setgroups", "w") as fh:
                fh.write("deny")
            with open("/proc/self/uid_map", "w") as fh:
                fh.write("0 %d 1" % uid)
            with open("/proc/self/gid_map", "w") as fh:
                fh.write("0 %d 1" % gid)
        except BaseException:
            os._exit(99)
        os._exit(0)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
        _confine_smoke_cache = (True, "")
    else:
        _confine_smoke_cache = (
            False,
            "the kernel/environment forbids unprivileged user-namespace setup "
            "(e.g. writing /proc/self/setgroups is denied, as on GitHub Actions "
            "and inside many containers)",
        )
    return _confine_smoke_cache


# ``AUDIT_ARCH_*`` tokens (EM_* | 64-bit | little-endian) the seccomp filter
# pins on, so a process cannot dodge the filter by issuing a syscall under a
# different ABI (e.g. x86_64's x32). Only arches with a maintained syscall-number
# table below are listed; others get no filter (namespace isolation still holds).
_AUDIT_ARCH = {"x86_64": 0xC000003E, "aarch64": 0xC00000B7}

# Syscalls a confined process is denied (returned EPERM). Not an attempt at a
# minimal allowlist — sandboxed code runs arbitrary Python + packages, so this
# is a denylist of high-risk calls a normal workload never needs: kernel attack
# surface (bpf, userfaultfd, perf_event_open, ptrace, process_vm_*), privilege /
# namespace manipulation (mount, umount2, pivot_root, chroot, unshare, setns),
# module loading + kexec, keyrings (keyctl/add_key/request_key), the
# open_by_handle_at container-escape vector, and host-management calls
# (reboot, swapon/off, acct, quotactl, iopl/ioperm). ``clone``/``clone3`` are
# deliberately *not* denied — modern glibc routes threading/fork through them.
_SECCOMP_DENY = {
    "x86_64": [
        101,
        155,
        161,
        163,
        165,
        166,
        167,
        168,
        169,
        172,
        173,
        175,
        176,
        179,
        246,
        248,
        249,
        250,
        272,
        298,
        303,
        304,
        308,
        310,
        311,
        313,
        320,
        321,
        323,
    ],
    "aarch64": [
        39,
        40,
        41,
        51,
        60,
        89,
        97,
        104,
        105,
        106,
        117,
        142,
        217,
        218,
        219,
        224,
        225,
        241,
        264,
        265,
        268,
        270,
        271,
        273,
        280,
        282,
        294,
    ],
}


def _build_seccomp_filter() -> Optional[str]:
    """Assemble the classic-BPF seccomp denylist for this arch (base64), or None.

    Emits a ``sock_filter[]`` program: pin the audit arch (mismatch → EPERM),
    reject x32 syscalls on x86_64, then EPERM each denied syscall number and
    ALLOW everything else. Built host-side and shipped in the confinement config
    so the subprocess only has to ``prctl(PR_SET_SECCOMP, ...)`` it. Returns
    ``None`` on an arch without a maintained number table (no filter applied).
    """
    import platform
    import struct

    machine = platform.machine()
    arch = _AUDIT_ARCH.get(machine)
    denied = _SECCOMP_DENY.get(machine)
    if arch is None or not denied:
        return None

    BPF_LD, BPF_W, BPF_ABS = 0x00, 0x00, 0x20
    BPF_JMP, BPF_JEQ, BPF_JGE, BPF_K = 0x05, 0x10, 0x30, 0x00
    BPF_RET = 0x06
    ALLOW = 0x7FFF0000  # SECCOMP_RET_ALLOW
    DENY = 0x00050000 | 1  # SECCOMP_RET_ERRNO | EPERM

    # Lay out the program symbolically first so jump distances to the trailing
    # DENY instruction can be resolved once the total length is known.
    plan = ["LD_ARCH", "CK_ARCH", "LD_NR"]
    if machine == "x86_64":
        plan.append("CK_X32")
    plan.extend(("DENY_IF", nr) for nr in denied)
    plan.append("ALLOW")
    plan.append("DENY")
    deny_idx = len(plan) - 1

    out = []
    for i, item in enumerate(plan):
        tag = item[0] if isinstance(item, tuple) else item
        off = deny_idx - i - 1  # instrs to skip to reach the trailing DENY
        if tag == "LD_ARCH":
            out.append((BPF_LD | BPF_W | BPF_ABS, 0, 0, 4))  # A = data.arch
        elif tag == "CK_ARCH":
            out.append((BPF_JMP | BPF_JEQ | BPF_K, 0, off, arch))  # !=arch → DENY
        elif tag == "LD_NR":
            out.append((BPF_LD | BPF_W | BPF_ABS, 0, 0, 0))  # A = data.nr
        elif tag == "CK_X32":
            out.append((BPF_JMP | BPF_JGE | BPF_K, off, 0, 0x40000000))  # x32 → DENY
        elif tag == "DENY_IF":
            out.append((BPF_JMP | BPF_JEQ | BPF_K, off, 0, item[1]))  # nr → DENY
        elif tag == "ALLOW":
            out.append((BPF_RET | BPF_K, 0, 0, ALLOW))
        else:  # DENY
            out.append((BPF_RET | BPF_K, 0, 0, DENY))

    blob = b"".join(struct.pack("=HBBI", *insn) for insn in out)
    return base64.b64encode(blob).decode("ascii")


_EGRESS_MAX_BYTES = 5 * 1024 * 1024


def _host_allowed(host: str, patterns: List[str]) -> bool:
    """Whether ``host`` matches the egress allowlist.

    Case-insensitive, trailing dot ignored. A bare entry matches that host
    exactly; a ``*.example.com`` entry matches any subdomain **and** the apex
    ``example.com``. No entry matches everything — an empty allowlist denies all.
    """
    host = (host or "").lower().rstrip(".")
    for pat in patterns:
        pat = pat.lower().rstrip(".")
        if pat.startswith("*."):
            suffix = pat[2:]
            if host == suffix or host.endswith("." + suffix):
                return True
        elif host == pat:
            return True
    return False


def _reject_private(host: str, port: Optional[int], scheme: str) -> str:
    """Validate ``host`` resolves to a public address; return that pinned IP.

    A second gate beyond the hostname allowlist: refuses loopback, private
    (RFC 1918), link-local (incl. the ``169.254.169.254`` cloud-metadata IP),
    reserved, multicast and unspecified addresses (``PermissionError`` if any
    resolved address is non-public). Returns the first validated IP so the
    caller can **pin** the connection to it — closing the DNS-rebinding window
    where the host could re-resolve to an internal address between this check
    and the connect.
    """
    import ipaddress
    import socket

    port = port or (443 if scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise PermissionError(f"cannot resolve host {host!r}: {exc}")
    chosen = None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise PermissionError(
                f"host {host!r} resolves to non-public address {ip} "
                "(set block_private_egress=False to allow internal targets)"
            )
        if chosen is None:
            chosen = info[4][0]
    if chosen is None:
        raise PermissionError(f"host {host!r} did not resolve to any address")
    return chosen


def _do_fetch(url, method, headers, data, patterns, timeout, max_bytes, block_private):
    """Blocking HTTP(S) fetch, refused unless every hop stays on the allowlist.

    Runs host-side (the sandbox has no raw network under confinement), so the
    allowlist is enforced where the model cannot reach around it. Redirects are
    re-checked, so an allowlisted host cannot bounce the request off-list, and
    (unless ``block_private`` is False) every hop's resolved address must be
    public **and the connection is pinned to that validated IP** — so a host
    cannot re-resolve to an internal address after the check (no DNS-rebinding
    TOCTOU). TLS still validates the cert against the original hostname (SNI).
    """
    import http.client
    import socket
    import urllib.error
    import urllib.parse
    import urllib.request

    # host -> validated public IP, populated by ``_check`` per hop; the pinned
    # connection classes below dial this IP instead of re-resolving the name.
    pinned: Dict[str, str] = {}

    def _check(u):
        parts = urllib.parse.urlsplit(u)
        if parts.scheme not in ("http", "https"):
            raise PermissionError(f"scheme not allowed: {parts.scheme!r}")
        host = parts.hostname or ""
        if not _host_allowed(host, patterns):
            raise PermissionError(f"host not in allowlist: {host!r}")
        if block_private:
            pinned[host] = _reject_private(host, parts.port, parts.scheme)

    _check(url)

    class _PinnedHTTPConnection(http.client.HTTPConnection):
        def connect(self):
            target = pinned.get(self.host, self.host)
            self.sock = socket.create_connection(
                (target, self.port), self.timeout, self.source_address
            )

    class _PinnedHTTPSConnection(http.client.HTTPSConnection):
        def connect(self):
            target = pinned.get(self.host, self.host)
            sock = socket.create_connection(
                (target, self.port), self.timeout, self.source_address
            )
            # server_hostname is the *name*, not the pinned IP, so SNI and cert
            # hostname verification still check the certificate against the host.
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)

    class _PinnedHTTPHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(_PinnedHTTPConnection, req)

    class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(_PinnedHTTPSConnection, req)

    class _Guard(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            _check(newurl)  # re-checks allowlist + re-pins the redirect target
            return super().redirect_request(req, fp, code, msg, hdrs, newurl)

    body = data.encode("utf-8") if isinstance(data, str) else data
    req = urllib.request.Request(
        url, data=body, method=(method or "GET").upper(), headers=headers or {}
    )
    opener = urllib.request.build_opener(_Guard, _PinnedHTTPHandler, _PinnedHTTPSHandler)
    with opener.open(req, timeout=timeout) as resp:
        raw = resp.read(max_bytes + 1)
        truncated = len(raw) > max_bytes
        text = raw[:max_bytes].decode("utf-8", errors="replace")
        return {
            "status": resp.status,
            "headers": dict(resp.headers.items()),
            "body": text,
            "truncated": truncated,
            "url": resp.geturl(),
        }


def _make_egress_tool(patterns: List[str], timeout: float, block_private: bool):
    """Build the bound ``http_fetch`` callable enforcing ``patterns`` host-side.

    Exposed inside the sandbox like any other bound function (over the host RPC
    bridge, which works even when confinement has cut the network), giving
    confined code an allowlisted egress path and *only* that path. When
    ``block_private`` is True, hosts resolving to non-public addresses are also
    refused (SSRF guard).
    """
    import asyncio

    allow = list(patterns)

    async def http_fetch(url, method="GET", headers=None, data=None):
        """Fetch an allowlisted HTTP(S) URL via the host and return the response.

        Args:
            url (str): Absolute ``http(s)://`` URL; its host (and any redirect
                target) must be on the sandbox's egress allowlist.
            method (str): HTTP method (default ``"GET"``).
            headers (dict): Optional request headers.
            data (str): Optional request body.

        Returns:
            dict: ``status``, ``headers``, ``body`` (text, capped), ``truncated``
            and the final ``url``. Raises ``PermissionError`` if the host is not
            allowlisted (or resolves to a non-public address).
        """
        return await asyncio.to_thread(
            _do_fetch,
            url,
            method,
            headers,
            data,
            allow,
            timeout,
            _EGRESS_MAX_BYTES,
            block_private,
        )

    return http_fetch


# Confinement config for the *currently executing* ``run_bash`` of a confined
# sandbox, read by the ``_run_python`` patch below. ``None`` (the default) means
# "not in a confined run_bash", so the patch is a no-op — every other code path,
# including ``run`` (which confines via its own bootstrap) and unconfined
# sandboxes, is unaffected.
_active_confine: contextvars.ContextVar[Optional[Dict[str, Any]]] = contextvars.ContextVar(
    "mirage_active_confine", default=None
)
_run_python_patched = False

# Mirage's internal python runner, by layout, newest first. It moved from
# ``workspace.executor.builtins._run_python`` (older Mirage) to
# ``commands.builtin.general.python._run_python_subprocess`` (mirage-ai 0.0.2+).
# Both are ``async def f(code, stdin_data, args, env)``; we patch whichever the
# installed Mirage actually exposes. Add new locations to the *front* as Mirage
# evolves so the current layout is preferred.
_RUN_PYTHON_TARGETS = (
    ("mirage.commands.builtin.general.python", "_run_python_subprocess"),
    ("mirage.workspace.executor.builtins", "_run_python"),
)


def _install_run_python_patch() -> bool:
    """Patch Mirage's python runner so ``run_bash``'s ``python3`` self-confines.

    Mirage spawns ``python3`` directly (not through our ``run`` bootstrap), so a
    shell ``python3`` would otherwise run unconfined. This wraps that one
    function to prepend the confinement prologue to the executed code whenever
    `_active_confine` is set (i.e. inside a confined ``run_bash``). The runner's
    module/name has changed across Mirage versions, so we probe the known
    locations (`_RUN_PYTHON_TARGETS`). Idempotent; returns False (with a
    warning naming what was tried) if none is found.
    """
    global _run_python_patched
    if _run_python_patched:
        return True

    import importlib

    module = original = None
    attr = ""
    for mod_name, attr_name in _RUN_PYTHON_TARGETS:
        try:
            candidate_mod = importlib.import_module(mod_name)
        except Exception:  # noqa: BLE001 - mirage layout changed / unavailable
            continue
        candidate = getattr(candidate_mod, attr_name, None)
        if candidate is not None and inspect.iscoroutinefunction(candidate):
            module, attr, original = candidate_mod, attr_name, candidate
            break

    if original is None:
        import warnings

        tried = ", ".join(f"{m}.{a}" for m, a in _RUN_PYTHON_TARGETS)
        warnings.warn(
            "MirageSandbox: cannot confine run_bash's python3 (no known mirage "
            f"python-runner internal found; tried {tried}); shell python stays "
            "unconfined.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    async def _patched_run_python(code, stdin_data=None, args=None, env=None):
        cfg = _active_confine.get()
        if cfg is not None:
            # ``repr`` (not ``json.dumps``) so the embedded config is valid
            # Python source (True/False/None, not JSON true/false/null).
            prologue = "import os\n" + _CONFINE_SRC + "\n_confine(" + repr(cfg) + ")\n"
            code = prologue + code
        return await original(code, stdin_data, args=args, env=env)

    _patched_run_python._mirage_sandbox_original = original  # for introspection
    setattr(module, attr, _patched_run_python)
    _run_python_patched = True
    return True


def _ensure_fuse_mounted(ws) -> bool:
    """Mount ``ws``'s virtual filesystem via FUSE if not already; return success.

    ``Workspace.copy()`` (used by `fork`) does not carry over the FUSE mount,
    so a confined fork sets one up here by replaying what ``fuse=True`` does in
    Mirage's ``Workspace.__init__`` (``self._fuse.setup(self)``). Uses a private
    Mirage attribute; returns False (so the caller falls back to unconfined) if
    the layout changed or the mount fails.
    """
    if getattr(ws, "fuse_mountpoint", None):
        return True
    fuse = getattr(ws, "_fuse", None)
    if fuse is None or not hasattr(fuse, "setup"):
        return False
    try:
        fuse.setup(ws)
    except Exception:  # noqa: BLE001 - FUSE unavailable / layout changed
        return False
    return bool(getattr(ws, "fuse_mountpoint", None))


def _error_from(stderr: str, exit_code: int) -> Optional[str]:
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
# so an agent calling ``run`` would loop on it until its wall-clock budget runs
# out. `run` detects these, rebuilds the workspace, and retries once.
_INFRA_FAILURE_MARKERS = (
    "confine-error",
    "Transport endpoint is not connected",
)

# How many times `run` will rebuild the workspace and retry on an infrastructure
# failure before giving up and returning the error. One heal clears the observed
# transient (a stale/dead FUSE mount left by an earlier run); a failure that
# survives a fresh mount is a real environment problem, not worth looping on.
_MAX_INFRA_HEALS = 1


def _is_infra_failure(stderr: str, exit_code: int) -> bool:
    """True when a run failed because the sandbox plumbing (confinement / FUSE
    mount) is broken, rather than because the snippet itself errored."""
    if exit_code == 0:
        return False
    return any(marker in (stderr or "") for marker in _INFRA_FAILURE_MARKERS)


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

    Wraps a Mirage ``Workspace`` — a virtual filesystem that mounts resources
    (RAM, disk, S3, Postgres, SSH, ...) under virtual paths and runs shell
    commands against them. ``run`` executes Python through Mirage's ``python3``
    builtin, and the file tools (`read_file`, `write_file`, ...) operate
    on the mounted virtual filesystem.

    Example:

    ```python
    import synalinks

    sandbox = synalinks.MirageSandbox(timeout=10)
    await sandbox.run("x = 6 * 7")
    result = await sandbox.run("print(x)")
    print(result.stdout)                    # -> "42\\n"

    # Snapshot + restore (workspace + interpreter state)
    blob = sandbox.dump()
    restored = synalinks.MirageSandbox.load(blob)
    ```

    ## Python state persistence

    Mirage spawns a **fresh** ``python3`` process per command, so Python
    variables would not normally survive between ``run`` calls. The sandbox
    bridges this by serializing the interpreter namespace (variables, imports,
    user-defined functions and classes) with ``dill`` after each snippet and
    restoring it before the next. State therefore persists across ``run`` calls
    just like a REPL — without replaying earlier snippets — and a snippet that
    raises does not wipe the accumulated namespace.

    ## Confinement (the default; `confine=True`, Linux)

    By default each ``run`` (and any ``python3`` spawned by `run_bash`) is
    **confined**: it enters a fresh user / mount / PID / network namespace and
    ``pivot_root``s into the FUSE-mounted virtual filesystem, so the snippet
    sees **only** the virtual sandbox at ``/`` (and, via the PID namespace, only
    its own processes in ``/proc`` — no host PIDs). Its own ``open(...)`` /
    ``pathlib`` land on the mount (one filesystem, shared with the file tools),
    the host filesystem is hidden, and the network is cut. A seccomp denylist
    (``seccomp=True``) shrinks the reachable kernel syscall surface, and the
    runtime is bind-mounted **read-only** (venv / stdlib / system lib/bin, so
    confined code cannot poison files the host later imports). It is rootless
    and entirely in-process (no container runtime): the Python subprocess
    sandboxes itself at startup via ``unshare`` + read-only runtime binds +
    ``pivot_root``, with optional ``RLIMIT_*`` caps. Requires Linux,
    ``/dev/fuse`` and unprivileged user namespaces; elsewhere it falls back to
    unconfined execution with a ``RuntimeWarning`` (or set
    ``require_confinement=True`` to make that a hard error).

    ## Windows (run under WSL2)

    Confinement is Linux-only by construction — ``unshare`` / ``pivot_root`` /
    FUSE / seccomp have no native-Windows equivalent. The supported way to get
    it on Windows is to run synalinks **inside WSL2** (Windows Subsystem for
    Linux 2): it is a real Linux kernel, so confinement works unchanged, and
    because WSL2 is itself a lightweight VM, the confined process is additionally
    separated from the Windows host by the VM boundary. On **native** Windows
    confinement is unavailable: a default ``MirageSandbox()`` falls back to
    unconfined with a ``RuntimeWarning`` pointing at WSL2, and
    ``require_confinement=True`` raises rather than running an LM's code
    unconfined. (WSL1 also cannot confine — it has no ``/dev/fuse``; use WSL2.)

    ## Unconfined (`confine=False`)

    With ``confine=False`` the snippet runs in a **real, unrestricted CPython
    subprocess** with network access, and two distinct filesystems are in play:

    - The interpreter's own OS file I/O (``open(...)``, ``pathlib``) hits the
      **host** filesystem the Python process runs on.
    - The Mirage **virtual** filesystem (the mounted resources) is reached
      through the shell and the sandbox's file tools (`read_file`,
      `write_file`, `list_files`, `search_files`, `edit_file`,
      `run_python_file`) — where mounted S3 buckets, disks, etc. live.

    They are separate spaces, so use the file tools / shell for the mounts and
    ``run`` for computation. Choose this only for trusted code that must reach
    the host filesystem or network.

    ## Bound tools and mounts are the real boundary

    Confinement hides the host and cuts the network, but **bound functions and
    mounted resources are capabilities you hand in** — a prompt-injected model
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
        timeout (float): Per-snippet execution budget in seconds (Default 5).
            Enforced around the underlying Mirage command; on expiry the run
            returns a ``TimeoutError`` in ``error``.
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
        confine (bool): Optional. When ``True`` (Linux only), each ``run``
            executes the snippet in a fresh user / mount / network namespace
            pivoted into the FUSE-mounted virtual filesystem: the snippet sees
            **only** the virtual sandbox at ``/`` (its ``open()`` lands on the
            mount, unifying it with the file tools), the host filesystem is
            hidden, and the network is cut. Rootless, in-process — no container
            runtime. Requires ``/dev/fuse`` and unprivileged user namespaces;
            on an unsupported host it is disabled with a ``RuntimeWarning``
            (graceful fallback). **Defaults to ``True``** (secure by default): a
            plain ``MirageSandbox()`` is confined + syscall-filtered + network-cut
            where the platform supports it. Pass ``confine=False`` to opt out
            (e.g. when the code must reach the host filesystem or network), or
            ``require_confinement=True`` to make the fallback a hard error.
        require_confinement (bool): Optional. Make confinement **fail closed**.
            When ``True`` (implies ``confine=True``), any condition that would
            otherwise silently fall back to *unconfined* execution — confinement
            unavailable on the host, the FUSE mount failing to come up, or a
            ``fork(confine=True)`` that cannot confine — raises instead of
            warning, and a `run` / `run_bash` whose confinement is not active is
            refused. Use this for untrusted (e.g. LM-generated) code, where a
            missing isolation boundary must be a hard error rather than a warning
            that scrolls past. Defaults to ``False`` (graceful fallback).
        seccomp (bool): Optional. Apply a seccomp syscall denylist inside a
            confined run, shrinking the kernel attack surface (no eBPF, ptrace,
            userfaultfd, perf, keyrings, module loading, kexec, namespace ops,
            ``open_by_handle_at``, ...; denied calls return ``EPERM``). Active
            only when confined and on a supported arch (x86_64 / aarch64);
            elsewhere it is a no-op. Defaults to ``True``.
        confine_network (bool): Optional. Keep **full** network access inside a
            confined run (skip the network namespace). All-or-nothing and not
            filtered — prefer ``allowed_hosts`` for restricted egress. Defaults
            to ``False`` (no network).
        allowed_hosts (list): Optional. Egress allowlist. When set, a host-
            mediated ``http_fetch`` tool is bound that reaches only these hosts
            (exact, or ``*.example.com`` for subdomains + apex; redirects are
            re-checked). Under confinement the network is cut, so this tool is
            the *only* egress path and the model cannot open raw sockets around
            it — enforced host-side. Without ``confine=True`` it is advisory
            (raw sockets still work) and a ``RuntimeWarning`` is emitted.
            ``None`` (default) binds no egress tool.
        block_private_egress (bool): Optional. When ``True`` (default), the
            ``http_fetch`` egress tool additionally refuses hosts that resolve
            to non-public addresses — loopback, private (RFC 1918), link-local
            (including the ``169.254.169.254`` cloud-metadata IP), reserved or
            multicast — re-checked on each redirect, and **pins the connection
            to the validated IP** so a host cannot re-resolve to an internal
            address after the check (no DNS-rebinding window; TLS still verifies
            the cert against the hostname). An SSRF guard so an allowlisted name
            pointing inward (CNAME / poisoned DNS) is refused. Set ``False`` to
            allow reaching internal targets. No effect without ``allowed_hosts``.
        memory_limit_mb (int): Optional. Address-space cap (``RLIMIT_AS``) for a
            confined run, in MiB. ``None`` for no limit.
        cpu_limit_seconds (int): Optional. CPU-time cap (``RLIMIT_CPU``) for a
            confined run, in seconds. ``None`` for no limit.
        max_processes (int): Optional. Process cap (``RLIMIT_NPROC``) for a
            confined run (fork-bomb guard). Defaults to 64; ``None`` to disable.
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
            ``cache_limit``, ``fuse``). ``native=True`` (which runs ``run_bash``
            on the **host** shell, bypassing the virtual-filesystem sandbox and
            confinement) is rejected under ``require_confinement`` and stripped
            with a warning whenever ``confine`` is active.
        external_functions (dict): Optional. ``name -> callable`` mapping bound
            persistently and exposed inside the sandbox on every run (see
            above).
    """

    description: str = (
        "Code runs in a real Python 3 interpreter (CPython) with the full "
        "standard library and any installed third-party packages. State persists "
        "across runs: variables, imports, functions and classes defined in "
        "earlier runs remain available, and an error does not reset the "
        "namespace. Use `print(...)` to emit results to stdout; the value of the "
        "snippet's last expression is also captured (the `result` convention — "
        "end with a `result` expression to return it). Any tools bound to the "
        "sandbox are exposed as global functions; call them directly, "
        "`result = tool_name(...)`. By default the sandbox is "
        "confined: `open(...)` / `pathlib` and the file tools (`read_file`, "
        "`write_file`, `list_files`, ...) operate on one shared virtual "
        "filesystem (use absolute paths like `/work/out.txt`), the host "
        "filesystem is hidden, and outbound network is disabled — reach the "
        "network only through a bound tool such as `http_fetch` when one is "
        "provided."
    )

    def __init__(
        self,
        timeout: float = 5.0,
        name: Optional[str] = None,
        *,
        workdir: Optional[str] = None,
        resources: Optional[Dict[str, Any]] = None,
        mode: Optional[Any] = None,
        session_id: str = "default",
        confine: bool = True,
        require_confinement: bool = False,
        seccomp: bool = True,
        confine_network: bool = False,
        allowed_hosts: Optional[List[str]] = None,
        block_private_egress: bool = True,
        memory_limit_mb: Optional[int] = None,
        cpu_limit_seconds: Optional[int] = None,
        max_processes: Optional[int] = 64,
        extra_binds: Optional[List[str]] = None,
        workspace_kwargs: Optional[Dict[str, Any]] = None,
        external_functions: Optional[Dict[str, Callable]] = None,
    ):
        if mirage is None:
            raise ImportError(
                "MirageSandbox requires the `mirage-ai` package. "
                "Install it with `pip install mirage-ai`."
            )
        Sandbox.__init__(
            self, timeout=timeout, name=name, external_functions=external_functions
        )
        self._session = session_id
        self._mode = mode if mode is not None else MountMode.EXEC
        self._workspace_kwargs = dict(workspace_kwargs or {})
        raw_resources = (
            resources if resources is not None else {_DEFAULT_MOUNT: RAMResource()}
        )
        # Least-privilege mounts: the root scratch mount stays EXEC (it must be
        # writable and run ``python3``), but any *other* bare mount defaults to
        # READ (read-only), so a destructive command/snippet cannot corrupt a
        # mounted real resource (S3, Postgres, disk) you did not explicitly make
        # writable. Override per mount with a ``(resource, MountMode)`` tuple, or
        # globally by passing ``mode=``.
        self._resources = self._resolve_mount_modes(raw_resources, mode)
        self._workdir = workdir

        # Confinement: when enabled, ``run`` executes the snippet in a fresh
        # user/mount/net namespace pivoted into the FUSE-mounted virtual
        # filesystem (no host fs, no network). Gated on platform support; on an
        # unsupported host it is disabled with a warning (graceful fallback).
        self._confine = bool(confine)
        # Fail-closed: when ``require_confinement`` is set, every path that would
        # otherwise silently fall back to *unconfined* execution raises instead.
        # Use this when the sandbox runs untrusted (e.g. LM-generated) code and a
        # missing boundary must be a hard error, not a warning that scrolls past.
        self._require_confinement = bool(require_confinement)
        if self._require_confinement and not self._confine:
            raise ValueError(
                "MirageSandbox(require_confinement=True) requires confine=True."
            )
        # Extra host directories bound read-only into the confined root (on top of
        # the auto-detected Python install/package dirs), e.g. for libraries in
        # nonstandard locations, data files or model weights.
        self._extra_binds = list(extra_binds or [])
        self._confine_network = bool(confine_network)
        # Seccomp denylist: prebuilt host-side, applied inside a confined run to
        # shrink the kernel syscall surface. ``None`` when disabled or on an arch
        # without a number table. Active only under confinement (a no-op for an
        # unconfined sandbox, where there is no boundary to harden).
        self._seccomp_blob = _build_seccomp_filter() if seccomp else None
        if seccomp and self._confine and self._seccomp_blob is None:
            import warnings

            warnings.warn(
                "MirageSandbox: seccomp unavailable on this architecture; "
                "confining without a syscall filter.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Egress allowlist: when set, a host-mediated ``http_fetch`` tool is bound
        # that only reaches allowlisted hosts. Combined with a cut network
        # (confined, default), that is the *only* path out — the model cannot
        # open raw sockets around it. ``None`` leaves egress unrestricted-by-tool.
        self._allowed_hosts = list(allowed_hosts) if allowed_hosts is not None else None
        self._block_private_egress = bool(block_private_egress)
        if self._allowed_hosts is not None:
            self._functions.setdefault(
                "http_fetch",
                _make_egress_tool(
                    self._allowed_hosts, self.timeout, self._block_private_egress
                ),
            )
            if not bool(confine):
                import warnings

                warnings.warn(
                    "MirageSandbox(allowed_hosts=...) without confine=True: the "
                    "allowlist is advisory — unconfined code can open raw sockets "
                    "to any host. Set confine=True to enforce it.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._rlimits: Dict[str, int] = {}
        if memory_limit_mb:
            self._rlimits["as"] = int(memory_limit_mb) * 1024 * 1024
        if cpu_limit_seconds:
            self._rlimits["cpu"] = int(cpu_limit_seconds)
        if max_processes:
            self._rlimits["nproc"] = int(max_processes)
        if self._confine:
            ok, reason = _confinement_available()
            if not ok:
                if self._require_confinement:
                    raise RuntimeError(
                        f"MirageSandbox(require_confinement=True): confinement "
                        f"unavailable: {reason}."
                    )
                import warnings

                warnings.warn(
                    f"MirageSandbox(confine=True) disabled: {reason}. "
                    "Running without confinement.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._confine = False
            else:
                # Confinement needs the virtual filesystem exposed via FUSE so
                # the snippet's own ``open()`` lands on the mount.
                self._workspace_kwargs.setdefault("fuse", True)
                # Also confine ``python3`` spawned via ``run_bash`` (Mirage's
                # builtin), not just ``run``'s bootstrap.
                _install_run_python_patch()

        # ``native=True`` routes ``run_bash`` through a REAL host shell (Mirage's
        # ``native_exec`` → ``/bin/sh -c``), bypassing both the builtin
        # virtual-filesystem sandbox AND the ``run_bash`` confinement patch
        # (``run`` still self-confines in its bootstrap, but ``run_bash`` would
        # execute host commands directly). It is therefore incompatible with
        # confinement: hard-error under ``require_confinement``, and strip it
        # (with a warning) whenever confinement is active.
        if self._workspace_kwargs.get("native"):
            if self._require_confinement:
                raise ValueError(
                    "MirageSandbox(require_confinement=True) is incompatible with "
                    "workspace_kwargs={'native': True}: native mode runs run_bash "
                    "on the host shell, bypassing confinement."
                )
            if self._confine:
                import warnings

                warnings.warn(
                    "MirageSandbox: workspace_kwargs={'native': True} runs "
                    "run_bash on the host shell and bypasses confinement; "
                    "disabling it because confine=True.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._workspace_kwargs.pop("native", None)

        self._ws = Workspace(self._resources, mode=self._mode, **self._workspace_kwargs)
        self._fuse_mountpoint = getattr(self._ws, "fuse_mountpoint", None)
        if self._require_confinement and not self._fuse_mountpoint:
            raise RuntimeError(
                "MirageSandbox(require_confinement=True): the FUSE mount was not "
                "established, so the snippet cannot be confined to the virtual "
                "filesystem."
            )
        # Per-sandbox host directory holding the dill state, per-run result file
        # and RPC socket. Bind-mounted into the confined root so those paths
        # still resolve after the pivot.
        self._hostdir = tempfile.mkdtemp(prefix="mirage_sandbox_")
        # Host file holding the dill-serialized interpreter namespace; created
        # lazily on the first ``run`` and reused (so state accumulates).
        self._state_path: Optional[str] = None
        # Snapshot of the filesystem at the branch point (``{vpath: text}``).
        # Seeded from ``workdir``; replaced by `fork` with the parent's
        # current tree so `diff` / `merge` report only post-fork changes.
        self._fork_base: Dict[str, str] = {}
        if workdir:
            self._fork_base = self._seed_from_workdir(workdir)

    @property
    def workdir(self) -> Optional[str]:
        """The host directory the filesystem was seeded from, or ``None``."""
        return self._workdir

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _resolve_mount_modes(resources: Dict[str, Any], explicit_mode: Any) -> dict:
        """Resolve each mount to an explicit ``(resource, MountMode)`` pair.

        Honors a per-mount ``(resource, mode)`` tuple as given. For a bare
        resource: uses ``explicit_mode`` when the caller passed ``mode=``;
        otherwise applies the least-privilege default — ``EXEC`` for the root
        scratch mount (it must be writable and run ``python3``) and ``READ``
        (read-only) for any other mount, so external resources are not writable
        unless opted in.
        """
        resolved: dict = {}
        for prefix, value in resources.items():
            if isinstance(value, tuple):
                resolved[prefix] = value  # explicit per-mount mode — honor it
            elif explicit_mode is not None:
                resolved[prefix] = (value, explicit_mode)
            else:
                default = MountMode.EXEC if prefix == _DEFAULT_MOUNT else MountMode.READ
                resolved[prefix] = (value, default)
        return resolved

    def _seed_from_workdir(self, workdir: str) -> Dict[str, str]:
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
                rel = os.path.relpath(host, root).replace(os.sep, "/")
                vpath = "/" + rel
                text = data.decode("utf-8", errors="replace")
                run_async_from_sync(self._write(vpath, text))
                base[vpath] = text
        return base

    def _ensure_state_path(self) -> str:
        # State lives in the per-sandbox host dir (bound into the confined root
        # so the path resolves after a pivot). Absent until the first run.
        if self._state_path is None:
            self._state_path = os.path.join(self._hostdir, "state.pkl")
        return self._state_path

    def _runtime_binds(self) -> List[str]:
        """Host directories to bind **read-only** into the confined root.

        The interpreter's own install + venv + system lib/bin dirs, plus the
        host's installed-package locations (user site-packages, ``PYTHONPATH``
        dirs) and any caller-supplied ``extra_binds`` — each bind-mounted
        read-only at its original absolute path so ``sys.path`` keeps resolving
        after the pivot and **host libraries import inside the sandbox just as
        they do on the host**, while confined code still cannot write them. The
        per-sandbox host dir (state / result / socket) is bound separately and
        writably via ``rw_binds``. Nothing else of the host filesystem is visible
        (notably not ``/etc`` or ``/home``, beyond the package dirs below).
        """
        candidates = [
            sys.base_prefix,
            sys.prefix,
            "/usr",
            "/lib",
            "/lib64",
            "/bin",
        ]
        # Host import locations outside the active prefix, so packages installed
        # in user site / system dist-packages / PYTHONPATH (and editable installs
        # they reference) resolve inside the sandbox. ``site`` re-adds these to
        # the subprocess ``sys.path``, so binding them is enough to import.
        try:
            import site

            candidates += list(site.getsitepackages())
            candidates.append(site.getusersitepackages())
        except Exception:  # noqa: BLE001 - site may be restricted/absent
            pass
        candidates += [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        candidates += list(self._extra_binds)
        # Keep only existing dirs, and drop any nested under an already-included
        # dir (a parent bind already exposes it) so we don't double-mount.
        seen: List[str] = []
        for d in candidates:
            if not d or not os.path.isdir(d):
                continue
            ad = os.path.abspath(d)
            if any(ad == s or ad.startswith(s + os.sep) for s in seen):
                continue
            seen.append(ad)
        return seen

    def _confine_config(self) -> Optional[dict]:
        """Confinement block for the bootstrap config, or ``None`` if disabled."""
        if not self._confine or not self._fuse_mountpoint:
            return None
        config = {
            "confine": True,
            "mp": self._fuse_mountpoint,
            "binds": self._runtime_binds(),
            "rw_binds": [self._hostdir],
            "network": self._confine_network,
            "rlimits": dict(self._rlimits),
        }
        if self._seccomp_blob:
            config["seccomp"] = self._seccomp_blob
        return config

    def _guard_confinement(self, confine_cfg: Optional[dict]) -> None:
        """Refuse to execute unconfined when confinement was required.

        Defense in depth for `require_confinement`: even if some later
        state change (a `reset`, a `fork`) left the sandbox unable to confine,
        a missing config here is a hard error rather than a silent unconfined
        run.
        """
        if self._require_confinement and confine_cfg is None:
            raise RuntimeError(
                "MirageSandbox(require_confinement=True): confinement is not "
                "active; refusing to execute unconfined."
            )

    def granted_capabilities(self) -> Dict[str, Any]:
        """Audit the privilege surface this sandbox actually grants.

        Bound tools and mounts — not the namespace — are the real boundary for
        confined code, so this returns a single JSON-safe view an operator can
        assert on before trusting a sandbox with untrusted (e.g. LM-generated)
        code: whether it is confined / fail-closed / syscall-filtered, the
        effective network reach, the exact bound-tool names, and each mount's
        access mode. Verify it grants only what the task needs.

        Returns:
            dict: ``confined``, ``require_confinement``, ``seccomp`` (filter
            active), ``read_only_runtime``, ``network`` (``{"mode": ...}`` —
            ``host`` unconfined, else ``cut`` / ``full`` / ``allowlist``),
            ``native_shell`` (``run_bash`` runs on the host shell — should be
            ``False`` when confined), ``tools`` (sorted bound-callable names, the
            egress + capability surface) and ``mounts`` (``prefix -> mode``, the
            actual per-mount access mode).
        """
        if not self._confine:
            network = {"mode": "host"}  # unconfined: shares the host network
        elif self._allowed_hosts is not None:
            network = {"mode": "allowlist", "allowed_hosts": list(self._allowed_hosts)}
        elif self._confine_network:
            network = {"mode": "full"}
        else:
            network = {"mode": "cut"}

        def _mode_name(value):
            # Each entry is a (resource, MountMode) pair after _resolve_mount_modes;
            # fall back to the global mode for any bare survivor.
            m = value[1] if isinstance(value, tuple) else self._mode
            return getattr(m, "name", str(m))

        return {
            "confined": self._confine,
            "require_confinement": self._require_confinement,
            "seccomp": bool(self._seccomp_blob) and self._confine,
            "read_only_runtime": self._confine,
            "network": network,
            "native_shell": bool(self._workspace_kwargs.get("native")),
            "tools": sorted(self._functions),
            "mounts": {p: _mode_name(v) for p, v in self._resources.items()},
        }

    async def _execute(
        self,
        command: str,
        *,
        stdin: Optional[bytes] = None,
        timeout: Optional[float] = None,
    ):
        """Run one Mirage command; return ``(stdout, stderr, exit_code)``.

        ``stdout_str`` / ``stderr_str`` are awaited (Mirage returns them as
        coroutines). A ``timeout`` (seconds) bounds the whole command and, on
        expiry, surfaces as ``exit_code = 124`` with a ``TimeoutError`` on
        stderr — mirroring the shell ``timeout`` convention.
        """
        coro = self._ws.execute(command, session_id=self._session, stdin=stdin)
        try:
            if timeout is not None:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro
        except asyncio.TimeoutError:
            return "", f"TimeoutError: execution exceeded {timeout}s", 124
        out = result.stdout_str()
        err = result.stderr_str()
        if inspect.isawaitable(out):
            out = await out
        if inspect.isawaitable(err):
            err = await err
        return out, err, result.exit_code

    # -- execution primitives ------------------------------------------------

    async def run(
        self,
        code: str,
        *,
        inputs: Optional[Dict[str, Any]] = None,
        external_functions: Optional[Dict[str, Callable]] = None,
    ) -> ExecutionResult:
        import dill

        state_path = self._ensure_state_path()
        # Persistently bound functions, plus this call's, exposed in-sandbox.
        functions = {**self._functions, **(external_functions or {})}

        server = None
        sock_path = None
        # Result file + RPC socket live in the per-sandbox host dir so they are
        # reachable after the confinement pivot (which binds that dir in).
        result_fd, result_path = tempfile.mkstemp(
            prefix="result_", suffix=".json", dir=self._hostdir
        )
        os.close(result_fd)
        try:
            base_config: Dict[str, Any] = {"result": result_path}
            if inputs:
                base_config["inputs"] = base64.b64encode(dill.dumps(inputs)).decode(
                    "ascii"
                )
            if functions:
                # A host-side Unix-socket server services in-sandbox tool calls
                # concurrently while the snippet subprocess runs. It lives in the
                # host dir (not the workspace), so it survives a workspace heal.
                sock_path = tempfile.mktemp(
                    prefix="rpc_", suffix=".sock", dir=self._hostdir
                )
                server = await asyncio.start_unix_server(
                    self._make_rpc_handler(functions), path=sock_path
                )
                base_config["sock"] = sock_path
                base_config["tools"] = sorted(functions)
            b64_boot = base64.b64encode(_BOOTSTRAP.encode("utf-8")).decode("ascii")

            # Heal-and-retry loop: a dead FUSE mount / confinement-bootstrap
            # failure makes every run repeat the same infra error, which an
            # agent would loop on until timeout. Rebuild the workspace (fresh
            # mount) and retry once. The confinement config carries the mount
            # point, so it is recomputed each attempt after a rebuild.
            heals = 0
            while True:
                config = dict(base_config)
                confine_cfg = self._confine_config()
                self._guard_confinement(confine_cfg)
                if confine_cfg:
                    config.update(confine_cfg)
                b64_config = base64.b64encode(json.dumps(config).encode("utf-8")).decode(
                    "ascii"
                )
                command = 'python3 -c "%s" %s %s %s' % (
                    _LAUNCHER,
                    b64_boot,
                    shlex.quote(state_path),
                    b64_config,
                )
                stdout, stderr, exit_code = await self._execute(
                    command, stdin=code.encode("utf-8"), timeout=self.timeout
                )
                if heals < _MAX_INFRA_HEALS and _is_infra_failure(stderr, exit_code):
                    heals += 1
                    self._rebuild_workspace()
                    continue
                break
        finally:
            if server is not None:
                server.close()
                try:
                    await server.wait_closed()
                except Exception:  # noqa: BLE001 - server teardown is best-effort
                    pass
            value = self._read_result(result_path)
            for path in (result_path, sock_path):
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        stderr = _clean_traceback(stderr)
        return self._record_run(
            code,
            ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                result=value,
                error=_error_from(stderr, exit_code),
            ),
        )

    @staticmethod
    def _read_result(path: str) -> Optional[Any]:
        """Decode the JSON last-expression value the subprocess wrote, if any."""
        try:
            with open(path, "r") as fh:
                text = fh.read()
        except OSError:
            return None
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _make_rpc_handler(self, functions: Dict[str, Callable]):
        """Build the Unix-socket handler that dispatches in-sandbox tool calls.

        Each connection carries one length-prefixed JSON request
        ``{"name", "kwargs"}``; the named host callable runs (awaited if a
        coroutine) and its JSON-marshalable return is sent back as
        ``{"ok": true, "result": ...}`` (or ``{"ok": false, "error": ...}``).
        """

        async def handle(reader, writer):
            try:
                header = await reader.readexactly(4)
                (length,) = struct.unpack(">I", header)
                request = json.loads((await reader.readexactly(length)).decode("utf-8"))
                name = request.get("name")
                kwargs = request.get("kwargs") or {}
                func = functions.get(name)
                if func is None:
                    resp = {"ok": False, "error": f"unknown tool: {name}"}
                else:
                    try:
                        result = func(**kwargs)
                        if inspect.isawaitable(result):
                            result = await result
                        resp = {"ok": True, "result": _jsonable(result)}
                    except Exception as exc:  # noqa: BLE001 - surfaced to snippet
                        resp = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                body = json.dumps(resp).encode("utf-8")
            except asyncio.IncompleteReadError:
                return
            except Exception as exc:  # noqa: BLE001 - protocol-level failure
                body = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
            try:
                writer.write(struct.pack(">I", len(body)) + body)
                await writer.drain()
                writer.close()
            except Exception:  # noqa: BLE001 - peer may have gone away
                pass

        return handle

    async def run_bash(self, command: str) -> dict:
        """Run a shell command in the sandbox's isolated Mirage shell.

        Reach for this for shell work — running programs, pipelines and
        process control; prefer the dedicated ``read_file`` / ``list_files`` /
        ``search_files`` / ``write_file`` / ``edit_file`` tools for plain file
        operations. The command executes against the mounted virtual
        filesystem in the sandbox's persistent session (so ``cd`` / ``export``
        carry across calls). Standard bash is supported — pipes, redirects,
        globs, ``&&`` / ``||``, loops — plus ``python3``.

        Args:
            command (str): The shell command line to execute.

        Returns:
            dict: ``ok`` (exit code 0), ``stdout``, ``stderr`` and
            ``exit_code``.
        """
        # Heal-and-retry on an infrastructure failure (dead FUSE mount /
        # confinement abort), same as ``run`` — otherwise every shell call would
        # repeat the error and an agent would loop on it. The confinement config
        # (with the mount point) is recomputed each attempt since a rebuild
        # creates a fresh mount; when confined we advertise it via
        # ``_active_confine`` so any ``python3`` the command spawns (through the
        # patched runner) self-confines like ``run`` does.
        heals = 0
        while True:
            confine_cfg = self._confine_config()
            self._guard_confinement(confine_cfg)
            token = _active_confine.set(confine_cfg) if confine_cfg else None
            try:
                stdout, stderr, exit_code = await self._execute(
                    command, timeout=self.timeout
                )
            finally:
                if token is not None:
                    _active_confine.reset(token)
            if heals < _MAX_INFRA_HEALS and _is_infra_failure(stderr, exit_code):
                heals += 1
                self._rebuild_workspace()
                continue
            break
        return {
            "ok": exit_code == 0,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
        }

    async def _write(self, path: str, content: str) -> int:
        """Write ``content`` to ``path`` in the virtual filesystem (mkdir -p)."""
        parent = os.path.dirname(path.rstrip("/")) or "/"
        command = "mkdir -p %s && cat > %s" % (
            shlex.quote(parent),
            shlex.quote(path),
        )
        data = content.encode("utf-8")
        _, _stderr, _exit = await self._execute(command, stdin=data)
        return len(data)

    async def _delete(self, path: str) -> None:
        """Remove ``path`` from the virtual filesystem (best-effort)."""
        await self._execute("rm -f -- %s" % shlex.quote(path))

    async def _read_tree(self) -> Dict[str, str]:
        """Snapshot the virtual filesystem as ``{vpath: text}`` (text files)."""
        tree: Dict[str, str] = {}
        for path in await self._walk_files():
            if path.startswith("/.sessions") or path.startswith("/dev"):
                continue
            text = await self._read_text(path)
            if text is not None:
                tree[path] = text
        return tree

    def reset(self) -> None:
        self._discard_state()
        self.clear_history()
        self._ws = Workspace(self._resources, mode=self._mode, **self._workspace_kwargs)
        # The new workspace gets a fresh FUSE mount; refresh the cached
        # mountpoint so a confined sandbox pivots into the live mount (and
        # ``require_confinement`` still holds) after a reset.
        self._fuse_mountpoint = getattr(self._ws, "fuse_mountpoint", None)
        if self._require_confinement and not self._fuse_mountpoint:
            raise RuntimeError(
                "MirageSandbox(require_confinement=True): the FUSE mount was not "
                "re-established after reset; cannot confine."
            )
        self._fork_base = self._seed_from_workdir(self._workdir) if self._workdir else {}

    def _rebuild_workspace(self) -> None:
        """Re-establish the Mirage workspace and its FUSE mount in place.

        Self-heals a dead mount mid-session (``Transport endpoint is not
        connected``) or a confinement bootstrap that could not pivot into it.
        The interpreter namespace lives in the host state file (reloaded by the
        bootstrap on the next run) and run history / bound functions are kept,
        so — unlike `reset` — only the workspace and its mount are rebuilt. The
        virtual filesystem's live contents are lost, but the dead mount had
        already made them unreachable.
        """
        ws = getattr(self, "_ws", None)
        if ws is not None:
            try:
                result = ws.close()
                if inspect.isawaitable(result):
                    run_async_from_sync(result)
            except Exception:  # noqa: BLE001 - the old mount is already broken
                pass
        self._ws = Workspace(self._resources, mode=self._mode, **self._workspace_kwargs)
        self._fuse_mountpoint = getattr(self._ws, "fuse_mountpoint", None)
        if self._require_confinement and not self._fuse_mountpoint:
            raise RuntimeError(
                "MirageSandbox(require_confinement=True): the FUSE mount could "
                "not be re-established after an infrastructure failure; cannot "
                "confine."
            )

    async def aclose(self) -> None:
        """Async release of the workspace and host state directory."""
        self._discard_state()
        ws = getattr(self, "_ws", None)
        if ws is not None:
            result = ws.close()
            if inspect.isawaitable(result):
                await result
        _rmtree(getattr(self, "_hostdir", "") or "")

    def close(self) -> None:
        """Best-effort, synchronous release of the workspace and state dir.

        Safe to call outside an event loop. When a loop is already running the
        workspace close is scheduled fire-and-forget; otherwise it runs to
        completion. Teardown never raises.
        """
        self._discard_state()
        ws = getattr(self, "_ws", None)
        try:
            if ws is not None:
                result = ws.close()
                if inspect.isawaitable(result):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop is not None and loop.is_running():
                        loop.create_task(result)
                    else:
                        asyncio.run(result)
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass
        _rmtree(getattr(self, "_hostdir", "") or "")

    def _discard_state(self) -> None:
        if self._state_path and os.path.exists(self._state_path):
            try:
                os.unlink(self._state_path)
            except OSError:
                pass
        self._state_path = None

    def _read_state_bytes(self) -> Optional[bytes]:
        if self._state_path and os.path.exists(self._state_path):
            with open(self._state_path, "rb") as fh:
                return fh.read()
        return None

    def _write_state_bytes(self, data: Optional[bytes]) -> None:
        if not data:
            return
        path = self._ensure_state_path()
        with open(path, "wb") as fh:
            fh.write(data)

    def _snapshot_workspace(self) -> bytes:
        buf = io.BytesIO()
        # ``Workspace.snapshot`` became a coroutine in mirage-ai 0.0.2; drive it
        # from this sync method via Mirage's own sync bridge (older versions
        # return None and skip the await).
        result = self._ws.snapshot(buf)
        if inspect.isawaitable(result):
            run_async_from_sync(result)
        return buf.getvalue()

    # -- serialization -------------------------------------------------------

    def dump(self) -> bytes:
        """Serialize the workspace + interpreter state to a JSON byte string.

        Bundles a Mirage workspace snapshot (the virtual filesystem), the
        dill-serialized interpreter namespace, and the run history. Mount
        *resources* themselves are not embedded (they may carry credentials);
        ``load`` re-supplies them, defaulting to a fresh in-memory mount.
        """
        state = self._read_state_bytes()
        payload = {
            "workspace": base64.b64encode(self._snapshot_workspace()).decode("ascii"),
            "state": base64.b64encode(state).decode("ascii") if state else None,
            "history": [dict(entry) for entry in self._history],
            "timeout": self.timeout,
            "name": self.name,
            "session_id": self._session,
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
    ) -> "MirageSandbox":
        """Restore a sandbox from bytes produced by `dump`.

        ``resources`` overrides the mounts for the restored workspace (required
        when the original mounts carried redacted credentials); by default a
        fresh in-memory ``RAMResource`` is mounted at ``/`` and the snapshot's
        files are loaded into it.
        """
        if mirage is None:
            raise ImportError(
                "MirageSandbox requires the `mirage-ai` package. "
                "Install it with `pip install mirage-ai`."
            )
        payload = json.loads(data.decode("utf-8"))
        mounts = resources if resources is not None else {_DEFAULT_MOUNT: RAMResource()}
        ws = Workspace.load(
            io.BytesIO(base64.b64decode(payload["workspace"])), resources=mounts
        )
        instance = cls.__new__(cls)
        Sandbox.__init__(
            instance,
            timeout=payload.get("timeout", 5.0),
            name=payload.get("name"),
        )
        instance._session = payload.get("session_id", "default")
        instance._mode = mode if mode is not None else MountMode.EXEC
        instance._workspace_kwargs = dict(workspace_kwargs or {})
        instance._resources = mounts
        instance._workdir = None
        instance._confine = False
        instance._require_confinement = False
        instance._confine_network = False
        instance._seccomp_blob = None
        instance._allowed_hosts = None
        instance._block_private_egress = True
        instance._extra_binds = []
        instance._rlimits = {}
        instance._ws = ws
        instance._fuse_mountpoint = getattr(ws, "fuse_mountpoint", None)
        instance._hostdir = tempfile.mkdtemp(prefix="mirage_sandbox_")
        instance._state_path = None
        instance._fork_base = {}
        if payload.get("state"):
            instance._write_state_bytes(base64.b64decode(payload["state"]))
        instance._history = [dict(entry) for entry in (payload.get("history") or [])]
        return instance

    def _obj_type(self):
        return "MirageSandbox"

    def get_config(self) -> dict:
        return {
            "timeout": self.timeout,
            "name": self.name,
            "session_id": self._session,
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

    def fork(
        self,
        *,
        name: Optional[str] = None,
        copy_repl: bool = False,
        confine: Optional[bool] = None,
    ) -> "MirageSandbox":
        """Return an isolated child branched off this sandbox's current state.

        The child gets an isolated copy of the Mirage workspace (its virtual
        filesystem), so the child's writes never touch the parent and vice
        versa. By default the child starts from a clean interpreter; pass
        ``copy_repl=True`` to also inherit this sandbox's Python namespace
        (variables, imports, definitions).

        Args:
            name (str): Optional name for the child sandbox.
            copy_repl (bool): Also inherit this sandbox's Python namespace.
            confine (bool): Whether the child is confined to **its own fork**
                (its ``run`` / ``run_bash`` python sees only the child's virtual
                filesystem, host hidden, network cut — see ``confine`` on the
                constructor). ``None`` (default) inherits this sandbox's
                setting; ``True`` / ``False`` override. A confined child gets
                its own FUSE mount; if confinement can't be set up it falls back
                to unconfined with a warning.
        """
        child = MirageSandbox.__new__(MirageSandbox)
        Sandbox.__init__(
            child,
            timeout=self.timeout,
            name=(
                name if name is not None else (f"{self.name}_fork" if self.name else None)
            ),
        )
        child._session = self._session
        child._mode = self._mode
        child._workspace_kwargs = dict(self._workspace_kwargs)
        child._resources = self._resources
        child._workdir = self._workdir
        child._confine_network = self._confine_network
        child._seccomp_blob = self._seccomp_blob
        child._allowed_hosts = self._allowed_hosts
        child._block_private_egress = self._block_private_egress
        child._extra_binds = list(self._extra_binds)
        # Bound functions (incl. the egress ``http_fetch`` tool) carry to the
        # child, so a confined fork keeps the same allowlisted capabilities.
        child._functions = dict(self._functions)
        child._rlimits = dict(self._rlimits)
        # ``Workspace.copy`` became a coroutine in mirage-ai 0.0.2; drive it from
        # this sync method (older versions return a ``Workspace`` directly).
        child._ws = self._ws.copy()
        if inspect.isawaitable(child._ws):
            child._ws = run_async_from_sync(child._ws)
        # Confine the child to its *own* fork: it needs its own FUSE mount
        # (``copy()`` does not carry one over) so the child's snippet pivots
        # into the child's filesystem, not the parent's.
        want = self._confine if confine is None else bool(confine)
        # A child only requires confinement (fail-closed) when it is asked to
        # confine at all — an explicit ``confine=False`` fork opts out cleanly.
        child._require_confinement = self._require_confinement and want
        child._confine = False
        if want:
            ok, reason = _confinement_available()
            if ok and _ensure_fuse_mounted(child._ws):
                _install_run_python_patch()
                child._confine = True
            elif child._require_confinement:
                raise RuntimeError(
                    "MirageSandbox.fork(confine=True) with require_confinement: "
                    f"{reason or 'FUSE setup failed'}."
                )
            else:
                import warnings

                warnings.warn(
                    "MirageSandbox.fork(confine=True) disabled: "
                    f"{reason or 'FUSE setup failed'}. Fork runs unconfined.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        child._fuse_mountpoint = getattr(child._ws, "fuse_mountpoint", None)
        child._hostdir = tempfile.mkdtemp(prefix="mirage_sandbox_")
        child._state_path = None
        # The child branches from the parent's *current* tree, so the child's
        # `diff` reports exactly what it changes from here (a clean boundary).
        child._fork_base = run_async_from_sync(self._read_tree())
        if copy_repl:
            child._write_state_bytes(self._read_state_bytes())
        return child

    def diff(self) -> dict:
        """Filesystem changes this sandbox made relative to its branch base.

        For a sandbox produced by `fork`, this is exactly the patch the
        child introduced since the fork point. Returns ``{"written":
        [{"path", "kind", "size"}, ...], "deleted": [...]}`` where ``kind`` is
        ``"create"`` (new file) or ``"modify"`` (the path existed in the base).
        """
        base = self._fork_base
        current = run_async_from_sync(self._read_tree())
        written = []
        for path in sorted(current):
            if path not in base:
                written.append(
                    {
                        "path": path,
                        "kind": "create",
                        "size": len(current[path].encode()),
                    }
                )
            elif current[path] != base[path]:
                written.append(
                    {
                        "path": path,
                        "kind": "modify",
                        "size": len(current[path].encode()),
                    }
                )
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

    def patch(self, *, paths: Optional[List[str]] = None) -> str:
        """Git-style unified diff of changes since this sandbox's branch base.

        Unlike `diff` (a structured, whole-file change set), this returns the
        actual line-level hunks as a single ``git diff``-format string — with
        ``diff --git`` headers, ``/dev/null`` for creates/deletes, ``@@`` hunks
        and ``\\ No newline at end of file`` markers — suitable for
        ``git apply`` / ``patch -p1``. Binary (NUL-containing) files collapse to
        a ``Binary files ... differ`` line. ``paths`` restricts output to a
        subset of virtual paths.
        """
        base = self._fork_base
        current = run_async_from_sync(self._read_tree())
        return _render_patch(base, current, None if paths is None else set(paths))

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
            dict: ``{"written", "deleted", "conflicts", "skipped",
            "repl_adopted"}``.
        """
        base = other._fork_base
        other_current = run_async_from_sync(other._read_tree())
        self_current = run_async_from_sync(self._read_tree())
        selected = None if paths is None else set(paths)
        written: List[str] = []
        deleted: List[str] = []
        conflicts: set = set()
        skipped: set = set()

        for path in sorted(other_current):
            if selected is not None and path not in selected:
                continue
            if path in base and other_current[path] == base[path]:
                continue  # unchanged in `other` — nothing to merge
            # Conflict iff this sandbox diverged from the fork point for `path`.
            if self_current.get(path) != base.get(path):
                conflicts.add(path)
                if not force:
                    skipped.add(path)
                    continue
            run_async_from_sync(self._write(path, other_current[path]))
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
            run_async_from_sync(self._delete(path))
            deleted.append(path)

        if repl:
            self._write_state_bytes(other._read_state_bytes())

        return {
            "written": written,
            "deleted": deleted,
            "conflicts": sorted(conflicts),
            "skipped": sorted(skipped),
            "repl_adopted": bool(repl),
        }

    # -- tool methods --------------------------------------------------------

    async def _walk_files(self) -> List[str]:
        """Return every file path in the virtual filesystem (sorted).

        Walks the merged mount view via Mirage ``readdir`` / ``stat``,
        descending into directories and collecting non-directory paths.
        """
        files: List[str] = []
        stack = ["/"]
        seen = set()
        while stack:
            directory = stack.pop()
            if directory in seen:
                continue
            seen.add(directory)
            try:
                children = self._ws.readdir(directory)
                if inspect.isawaitable(children):
                    children = await children
            except Exception:  # noqa: BLE001 - missing/permission dirs are skipped
                continue
            for child in children:
                try:
                    st = self._ws.stat(child)
                    if inspect.isawaitable(st):
                        st = await st
                except Exception:  # noqa: BLE001
                    continue
                if str(getattr(st.type, "value", st.type)) == "directory":
                    stack.append(child)
                else:
                    files.append(child)
        return sorted(files)

    async def _read_text(self, path: str) -> Optional[str]:
        """Read a virtual-filesystem file via the shell; ``None`` if missing.

        Returns the decoded text (Mirage decodes bytes as UTF-8, replacing
        undecodable sequences); these tools target text files.
        """
        stdout, stderr, exit_code = await self._execute("cat -- %s" % shlex.quote(path))
        if exit_code != 0:
            return None
        return stdout

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
        regex = _glob_to_regex(pattern)
        files = [p for p in await self._walk_files() if regex.match(p.lstrip("/"))]
        page, truncated = _paginate(files, offset, limit)
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
            inclusive), ``total_lines`` and ``truncated`` — or ``error`` if the
            file is missing.
        """
        content = await self._read_text(path)
        if content is None:
            return {"error": f"file not found: {path}"}
        lines = content.splitlines(keepends=True)
        page, truncated = _paginate(lines, offset, limit)
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
        quoted = shlex.quote(path)
        parent = os.path.dirname(path.rstrip("/")) or "/"
        command = "mkdir -p %s && cat > %s" % (shlex.quote(parent), quoted)
        data = content.encode("utf-8")
        _, stderr, exit_code = await self._execute(command, stdin=data)
        if exit_code != 0:
            return {"error": stderr.strip() or f"failed to write {path}"}
        return {"written": path, "bytes": len(data)}

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
        content = await self._read_text(path)
        if content is None:
            return {"error": f"file not found: {path}"}
        if not old:
            return {"error": "`old` must be a non-empty string"}
        text = content
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
            1-based line numbers), ``total``, ``offset`` and ``truncated`` — or
            ``error`` on a bad regex.
        """
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return {"error": f"invalid regex: {exc}"}
        glob_regex = _glob_to_regex(glob)
        matches: List[Dict[str, Any]] = []
        for path in await self._walk_files():
            if not glob_regex.match(path.lstrip("/")):
                continue
            content = await self._read_text(path)
            if content is None:
                continue
            for lineno, line in enumerate(content.splitlines(), start=1):
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
        """Run a Python script file from the mounted virtual filesystem.

        Reads ``path`` from the virtual filesystem and executes its contents in
        the sandbox, sharing the interpreter namespace with prior runs.

        Args:
            path (str): Absolute virtual path of the ``.py`` file to run.

        Returns:
            dict: ``ok`` (bool), ``stdout`` and ``stderr`` (captured output),
            and ``error`` (a message string, or null on success) — or ``error``
            if the file is missing.
        """
        content = await self._read_text(path)
        if content is None:
            return {"error": f"file not found: {path}"}
        return await self.run_python_code(content)
