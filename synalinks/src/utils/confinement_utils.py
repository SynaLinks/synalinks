# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Confinement: the in-subprocess prologues (Linux namespaces, macOS Seatbelt),
host probes and seccomp."""

import base64
import ctypes.util
import os
import platform
import struct
import warnings
from typing import Optional
from typing import Tuple

# Whether libfuse can be loaded. `/dev/fuse` is the kernel device; the mount
# confinement pivots into also needs the libfuse user-space library, which
# mfusepy (the mirage-ai[fuse] extra) loads at import and raises OSError when
# it is missing. A host can easily have the device and not the library (slim
# containers), so `confinement_available` checks both and fails closed rather
# than passing the probe and then silently running unconfined.
try:
    import mfusepy as _mfusepy  # noqa: F401

    _LIBFUSE_AVAILABLE = True
except (ImportError, OSError):  # pragma: no cover - no mfusepy / no libfuse
    _LIBFUSE_AVAILABLE = False


# Rootless in-process confinement, shared by the ``run_code`` bootstrap and the
# ``run_bash`` ``_run_python`` patch: enter fresh user/mount/PID(/net)
# namespaces, fork (PID namespaces only apply to children) so the child is PID 1
# of the new namespace, bind the Python runtime into the FUSE-mounted virtual
# filesystem, then pivot_root into it so the process sees ONLY the virtual
# sandbox at "/" with the host filesystem gone and a fresh /proc that shows only
# namespaced PIDs. Must run first, single-threaded, before the snippet imports
# anything (the runtime binds keep imports working).
CONFINE_SRC = r"""
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
        # remount (MS_BIND | MS_REMOUNT | MS_RDONLY); a single mount() cannot
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
            return  # cannot bind this dir at all; skip it
        if readonly:
            # Lock it read-only and FAIL CLOSED if that can't be done: swallowing
            # the failure would leave the runtime writable while we still claim
            # (and ``granted_capabilities`` reports) read-only, a fail-open that
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
    # PID namespace, that /proc shows ONLY namespaced PIDs; host processes are
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
    # process can reach (eBPF, ptrace, userfaultfd, perf, keyrings, module
    # loading, kexec, namespace ops, open_by_handle_at, etc.), returning EPERM.
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


# macOS in-process confinement, the Seatbelt counterpart to ``_confine`` below.
#
# The Linux path is unavailable here and cannot be ported: namespaces,
# ``pivot_root``, ``/proc/self/uid_map`` and seccomp are Linux kernel features
# with no XNU equivalent (POSIX never standardized sandboxing, so every kernel
# grew its own). macOS instead exposes Seatbelt, the TrustedBSD MAC policy the
# App Sandbox is built on, which takes a declarative SBPL profile rather than a
# constructed namespace. ``sandbox_init`` applies one to the *calling* process,
# which fits this bootstrap exactly as ``_confine`` does: run first, in-process,
# before the snippet imports anything.
#
# What this DOES enforce, kernel-side and without root:
#   * filesystem: ``(deny default)`` then explicit allows, so the snippet reads
#     only the Python runtime it needs and writes only the sandbox's own dirs.
#     Reads are restricted, not just writes (Codex and Gemini CLI both give up
#     on read restriction; here the runtime read set is known, so we keep it).
#     ``TMPDIR`` is redirected into that writable area, since the host temp dir
#     the snippet would otherwise get is deliberately not granted, and the
#     ancestors of every allowed path are traversable (metadata only) so the
#     ``/var`` -> ``/private/var`` symlink cannot strand the grants.
#   * network: denied outright unless ``network`` is set, with the host-tool RPC
#     socket allowed by literal path so the bridge keeps working either way.
#   * rlimits: ``setrlimit`` is POSIX and applies unchanged.
#
# What it does NOT, and why the docs must not imply Linux parity:
#   * no PID isolation. XNU has no PID namespace, so the host process table
#     stays visible; ``(deny process-info*)`` narrows this but is not the same
#     guarantee as being PID 1 in a fresh namespace.
#   * no syscall filter. Seatbelt gates MAC operations, not syscall numbers,
#     so the seccomp denylist has no counterpart. The two models overlap but
#     neither contains the other.
#
# ``sandbox_init`` is deprecated (Apple ship no replacement for non-App-Store
# process sandboxing) but remains the documented path, still works on current
# macOS, and underpins Apple's own sandboxes; Chrome, Nix, Bazel, Codex and
# Gemini CLI all rely on it today.
LIBSANDBOX = "/usr/lib/libsandbox.1.dylib"

MACOS_CONFINE_SRC = r'''
# Mach services the confined process may look up, by exact global name. See
# `_build_sbpl` for why this is an allowlist and not `(allow mach-lookup)`.
#
# Deliberately minimal, and every entry is here on evidence.
#
# "The profile is applied after startup, so nothing needs these" holds for the
# process that applies it and *only* that process. A child exec'd under the
# profile runs libSystem startup inside it, and a denied bootstrap lookup there
# does not degrade: libxpc aborts the process. Emptying this list did not break
# `run`, which is why it looked safe; it broke every `subprocess`, which died
# on SIGABRT before it could write a word to stderr.
#
# Nothing here brokers process execution. Notably absent, and to stay absent:
# `com.apple.coreservices.launchservicesd` and the launchd family, which exist
# precisely to spawn processes outside the caller's sandbox.
_MACOS_MACH_SERVICES = (
    # libsystem_notify: registered during libSystem startup.
    "com.apple.system.notification_center",
    # os_log / libsystem_trace, touched by the C runtime before main().
    "com.apple.logd",
    # opendirectoryd's libinfo endpoint, backing pwd/getpwuid, which CPython
    # calls from `site` and `os.path.expanduser`.
    "com.apple.system.opendirectoryd.libinfo",
)


def _sbpl_quote(path):
    """Quote a path for SBPL, which is s-expression syntax, not shell."""
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _build_sbpl(cfg):
    """Render the Seatbelt profile for this run."""
    import os

    lines = [
        "(version 1)",
        "(deny default)",
        # Keep the process able to run at all: fork for subprocesses the snippet
        # may spawn, signals to itself, and the sysctl reads CPython performs
        # during startup. These grant no filesystem or network reach.
        "(allow process-fork)",
        "(allow signal (target self))",
        # Its own children too, so `Popen.kill()` and subprocess timeouts work;
        # still nothing outside its own process tree.
        "(allow signal (target children))",
        "(allow sysctl-read)",
        # ...except the process table. `kern.proc.*` (what `ps` reads) and
        # `kern.procargs*` hand over every host process and its arguments,
        # the reconnaissance that `(deny process-info*)` below closes for the
        # libproc route. Last match wins, so this carves out of the allow.
        '(deny sysctl-read (sysctl-name-prefix "kern.proc"))',
        # Deny introspection of other processes. Not a PID namespace, but it
        # removes the obvious host-process reconnaissance path. Self is allowed
        # back, after the deny since SBPL is last-match-wins: a process reading
        # its own proc info is not reconnaissance (it is its own memory), and
        # dyld asks about itself while starting a freshly exec'd image, which a
        # blanket deny turns into an abort.
        "(deny process-info*)",
        "(allow process-info* (target self))",
    ]
    # Mach services, by explicit name. A bare `(allow mach-lookup)` is the
    # classic way to render a Seatbelt profile ineffective: a send right to any
    # service lets the process ask a *daemon* to act for it, and the daemon's
    # child does not inherit this profile, so the filesystem and network rules
    # below simply do not apply to the work it performs. Chrome and the Codex /
    # Gemini CLI profiles all allowlist for this reason.
    #
    # This set is deliberately minimal: the notification and logging services
    # libSystem touches during startup, plus the directory-service lookup that
    # backs `pwd`/`getpwuid` (CPython's `site` and `os.path.expanduser` call
    # it). Nothing here brokers process execution. Notably absent, and to stay
    # absent: `com.apple.coreservices.launchservicesd` and the launchd family,
    # which exist precisely to spawn processes outside the caller's sandbox.
    #
    # Widen only on evidence: a missing service surfaces as the bootstrap
    # failing closed with `confine-error` and exit 99, never as a silent loss
    # of confinement, so growing this list from real failures is safe.
    for service in _MACOS_MACH_SERVICES:
        lines.append('(allow mach-lookup (global-name "%s"))' % service)
    # Character devices CPython needs; harmless and required for randomness,
    # /dev/null redirection and tty probing.
    for dev in ("/dev/null", "/dev/zero", "/dev/random", "/dev/urandom",
                "/dev/dtracehelper", "/dev/tty"):
        lines.append("(allow file-read* file-write-data (literal %s))"
                     % _sbpl_quote(dev))
    # `/bin/sh` on macOS is a shim that reads this symlink to pick the real
    # shell (bash, dash or zsh). Denied, it falls back but prints an error to
    # every shelled-out command's stderr. One literal: the link, not a dir.
    lines.append('(allow file-read* (literal "/private/var/select/sh"))')
    # Path *resolution*, not content. Reaching an allowed path means reading
    # every symlink on the way to it, and `(deny default)` denies that too.
    # This is not a corner case on macOS: the sandbox's own directories live
    # under the per-user temp dir `/var/folders/...`, and `/var` is a symlink
    # into `/private`, so with the traversal denied the `file-write*` grant
    # below never gets the chance to match and every write into the sandbox's
    # own writable area fails EPERM while the profile looks, on paper, correct.
    # Metadata only: this permits `stat`/`readlink` on the ancestors, not
    # reading their contents or listing them.
    #
    # The working directory is in this set for the same reason, and it is not
    # optional: `getcwd` is itself a path operation (the kernel walks the cwd up
    # to the root), so a cwd outside the granted set fails EPERM. CPython calls
    # it on the *first import after confinement*, since `python3 -c` puts "" at
    # the head of `sys.path` and `_path_importer_cache` resolves "" through
    # `getcwd` while catching only `FileNotFoundError`. That surfaces not as a
    # denied file but as a `PermissionError` thrown by the import machinery,
    # several frames away from anything that looks filesystem-related.
    try:
        cwd = os.getcwd()
    except OSError:  # cwd already unreachable; nothing to grant
        cwd = None
    ancestors = ["/"]
    if cwd and cwd != "/":
        ancestors.append(cwd)
    for path in (
        list(cfg.get("read_paths") or [])
        + list(cfg.get("rw_paths") or [])
        + [cfg.get("rpc_socket_dir"), cfg.get("sock"), cfg.get("tmpdir"), cwd]
    ):
        if not path:
            continue
        # Both forms: the host hands over resolved paths, while the snippet may
        # name the unresolved ones (`TMPDIR`, `/tmp`, `/etc`) for the same file.
        for form in (str(path), os.path.realpath(str(path))):
            current = os.path.dirname(form)
            while current and current != "/":
                if current not in ancestors:
                    ancestors.append(current)
                current = os.path.dirname(current)
    # The top-level symlinks into /private, which any absolute path the snippet
    # itself names under them has to traverse.
    for path in ("/var", "/tmp", "/etc"):
        if path not in ancestors:
            ancestors.append(path)
    for path in ancestors:
        lines.append("(allow file-read-metadata (literal %s))" % _sbpl_quote(path))
    # The root directory gets `file-read-data` on top of the metadata above,
    # and it is load-bearing for every subprocess: dyld4's CacheFinder locates
    # the shared-cache cryptex by *reading* `/` (macOS 26 runners), and a
    # denied read there does not degrade, it halts the freshly exec'd child in
    # `ignition_halt` before dyld can write a word to stderr. The parent never
    # trips this because its cache was mapped before the profile applied.
    # `literal` keeps the grant to the directory itself: listing `/` exposes
    # the well-known top-level names and nothing of any file's contents.
    lines.append('(allow file-read-data (literal "/"))')
    # Read-only: the interpreter, its stdlib and the host package dirs, the
    # same set the Linux path bind-mounts read-only. Without these the snippet
    # cannot import anything, including its own bootstrap dependencies.
    for path in cfg.get("read_paths") or []:
        lines.append("(allow file-read* (subpath %s))" % _sbpl_quote(path))
    # Execution is limited to that same read-only runtime set, so the snippet
    # cannot exec a binary it wrote itself: the writable area is not in this
    # set, and `file-map-executable` is denied there besides.
    #
    # It does include `/usr/bin`, so the snippet can shell out to host tools
    # exactly as it can on Linux, where the runtime dirs are bind-mounted and
    # executable. That is deliberate parity, and it is not an escape: a child
    # process inherits the profile it is spawned under, so a `subprocess` here
    # gets the same denied filesystem and network as its parent. The escape
    # route in this area is not exec but `mach-lookup`, where a *daemon* does
    # the work outside the profile, and that allowlist is empty.
    # `file-map-executable` goes with it and is not implied by `file-read*`:
    # it is a separate operation, and an exec'd binary needs its pages, and
    # dyld needs the pages of every dylib it then loads, mapped executable. A
    # profile with `process-exec` but not this one starts the child and kills
    # it in dyld, which surfaces as a subprocess that produces no output and
    # raises nothing in the parent.
    for path in cfg.get("read_paths") or []:
        lines.append("(allow process-exec (subpath %s))" % _sbpl_quote(path))
        lines.append("(allow file-map-executable (subpath %s))" % _sbpl_quote(path))
    # Read-write: the sandbox's own directories (dill state, result file, RPC
    # socket, working tree). This is the only writable surface.
    for path in cfg.get("rw_paths") or []:
        lines.append("(allow file-read* file-write* (subpath %s))"
                     % _sbpl_quote(path))
    # ...but nothing written there may then be *executed* as native code. SBPL
    # is last-match-wins, so this deny follows the grant above and carves
    # `file-map-executable` back out of it: a snippet cannot drop a dylib into
    # its own scratch dir and `dlopen` it. Executable pages therefore come from
    # the read-only runtime set and nowhere else, which is write-xor-execute
    # across the whole profile, and the nearest macOS gets to the seccomp
    # filter it has no equivalent of: native code the snippet brought itself is
    # exactly what a syscall denylist exists to constrain. (`ctypes` against
    # the *system* libc is still reachable; no profile can gate that.)
    for path in cfg.get("rw_paths") or []:
        lines.append("(deny file-map-executable (subpath %s))" % _sbpl_quote(path))
    # The host-tool bridge is a unix socket, which Seatbelt classes as network
    # rather than file I/O, so it needs an explicit allow to survive the
    # network denial below. The socket is created per run under a random name,
    # so this grants the containing directory; the literal is added as well
    # when the host already knows the path, since path filters on
    # `network-outbound` are far better attested for `literal` than `subpath`.
    sock_dir = cfg.get("rpc_socket_dir")
    if sock_dir:
        lines.append("(allow network-outbound (subpath %s))" % _sbpl_quote(sock_dir))
    if cfg.get("sock"):
        lines.append("(allow network-outbound (literal %s))"
                     % _sbpl_quote(cfg["sock"]))
    if cfg.get("network"):
        lines.append("(allow network*)")
    # Nothing else. Loopback is deliberately NOT granted: unlike Linux, where
    # NEWNET gives the process a private network namespace whose loopback is
    # its own, macOS has no namespace and `(local ip)` would be the *host's*
    # 127.0.0.1. That would hand supposedly network-isolated code every service
    # bound to localhost (model servers, databases, the user's own dev APIs),
    # which is exactly what confinement is meant to prevent. The host-tool
    # bridge does not need it either: it is a unix socket, granted by the
    # `rpc_socket_dir` rule above, which Seatbelt classes as network-outbound
    # but matches on path rather than address.
    return "\n".join(lines) + "\n"


def _prepare_macos_process(cfg):
    """Pre-profile setup: rlimits, and standing where the profile can grant.

    Split out of `_confine_macos` because everything here is plain POSIX and
    testable off a Darwin host, unlike the `sandbox_init` call that follows it.
    """
    import os
    import resource
    import sys
    import tempfile

    # Point the snippet's temp dir at the sandbox's own writable area. The
    # Linux path gets this for free: it pivots into the virtual filesystem,
    # where /tmp is the sandbox's. Seatbelt confines in place, so
    # `tempfile.gettempdir()` would otherwise hand back the *host* temp dir,
    # which the profile does not grant, and the most ordinary thing a snippet
    # can do with a scratch file would fail. Set before the profile applies and
    # before anything has called `gettempdir` (which memoizes), and set
    # `tempfile.tempdir` too so the answer does not depend on that ordering.
    tmpdir = cfg.get("tmpdir")
    if tmpdir:
        os.environ["TMPDIR"] = tmpdir
        tempfile.tempdir = tmpdir
    # Move the process into the sandbox rather than widen the profile to
    # wherever it happens to be standing. `getcwd` is a path operation, and a
    # cwd outside the granted set fails EPERM: granting metadata on the cwd is
    # not enough, so the fix is to be somewhere the profile already grants
    # outright. `cwd` is the FUSE mount of the virtual filesystem, the
    # directory `pivot_root` makes "/" on Linux, so a relative path in a
    # snippet lands on the same filesystem the file tools see; the scratch dir
    # is the fallback when no mount is configured.
    for workdir in (cfg.get("cwd"), tmpdir):
        if not workdir:
            continue
        try:
            os.chdir(workdir)
            break
        except OSError:  # dir missing; the profile still applies below
            pass

    # Drop the leading "" from `sys.path`, belt to the chdir's braces. CPython
    # resolves that entry through `getcwd` on the first import *after* the
    # profile applies, catching only `FileNotFoundError`, so an EPERM there
    # surfaces as a `PermissionError` raised by the import machinery, frames
    # away from anything filesystem-shaped, and takes down an import that would
    # otherwise have succeeded from a granted `sys.path` entry. Nothing in the
    # sandbox imports from the working directory: the runtime set is absolute.
    sys.path = [entry for entry in sys.path if entry]

    # rlimits: POSIX, and unaffected by the profile applied below.
    rl = cfg.get("rlimits") or {}
    for key, which in (
        ("as", getattr(resource, "RLIMIT_AS", None)),
        ("cpu", getattr(resource, "RLIMIT_CPU", None)),
        ("fsize", getattr(resource, "RLIMIT_FSIZE", None)),
    ):
        if rl.get(key) and which is not None:
            try:
                resource.setrlimit(which, (rl[key], rl[key]))
            except (ValueError, OSError):
                # An existing limit already tighter than the request is not a
                # failure; the tighter one wins and confinement continues.
                pass

    # RLIMIT_NPROC is counted per *user*. On Linux that lands on the fresh
    # user namespace `_confine` unshares, so `max_processes` is a budget for
    # the sandbox alone. macOS has no namespace: the count includes every
    # process the user already runs, so the bare number would stop the sandbox
    # forking at all. Set it to what is running now *plus* the budget instead.
    # The limit binds only this process and its children (the host keeps its
    # own), so it caps what the sandbox can add; processes the user starts
    # meanwhile eat into that budget, which can only err towards stricter.
    # Counted before the profile applies, since it denies the process table.
    if rl.get("nproc"):
        try:
            import ctypes

            libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
            proc_uid_only = 4  # PROC_UID_ONLY
            size = libproc.proc_listpids(proc_uid_only, os.getuid(), None, 0)
            pids = (ctypes.c_int * (size // 4 + 256))()
            size = libproc.proc_listpids(
                proc_uid_only, os.getuid(), pids, ctypes.sizeof(pids)
            )
            running = sum(1 for pid in pids[: size // 4] if pid)
            limit = running + rl["nproc"]
            _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
            if hard != resource.RLIM_INFINITY:
                limit = min(limit, hard)
            resource.setrlimit(resource.RLIMIT_NPROC, (limit, limit))
        except (OSError, ValueError):
            pass  # no count, no cap: better than a cap that forbids forking


def _confine_macos(cfg):
    import ctypes
    import ctypes.util

    _prepare_macos_process(cfg)

    path = ctypes.util.find_library("sandbox") or "/usr/lib/libsandbox.1.dylib"
    libsandbox = ctypes.CDLL(path, use_errno=True)
    libsandbox.sandbox_init.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    libsandbox.sandbox_init.restype = ctypes.c_int
    err = ctypes.c_char_p()
    # flags=0 means "profile is SBPL source", as opposed to one of the named
    # builtin profiles.
    rc = libsandbox.sandbox_init(_build_sbpl(cfg).encode("utf-8"), 0,
                                 ctypes.byref(err))
    if rc != 0:
        detail = err.value.decode("utf-8", "replace") if err.value else "unknown"
        raise OSError("sandbox_init failed: " + detail)
'''

# The prologue every confined ``python3`` runs first, from the ``run_code``
# bootstrap and from the ``run_bash`` ``_run_python`` patch alike. Both backends
# are emitted and the platform picked at run time, by the process being
# confined rather than trusted from the host's config: it runs on the very host
# it confines, so its own platform is the authority.
CONFINE_PROLOGUE_SRC = CONFINE_SRC + MACOS_CONFINE_SRC + r"""
def _confine_platform(cfg):
    import sys

    if sys.platform == "darwin":
        _confine_macos(cfg)
    else:
        _confine(cfg)
"""


# One-time native-Windows hint. The sandbox's confinement needs Linux or macOS, so on
# native Windows code runs unconfined; steer users to WSL2. Emitted once per
# process when this module is first imported (it is pulled in by ``import
# synalinks`` via the code-running agents). Suppressible / a no-op off Windows.
WINDOWS_HINTED = False

WINDOWS_HINT = (
    "synalinks: on native Windows the code-running sandbox confinement "
    "is unavailable, so LM-generated code runs UNCONFINED. Run synalinks inside "
    "WSL2 (Windows Subsystem for Linux) for the secure, confined sandbox. "
    "Set SYNALINKS_NO_WINDOWS_HINT=1 to silence this message."
)


def maybe_warn_native_windows():
    """Warn once on native Windows that confinement needs WSL2; return the message.

    A no-op (returns ``None``) off Windows, when already emitted this process, or
    when ``SYNALINKS_NO_WINDOWS_HINT`` is set. Inside WSL2 ``platform.system()``
    reports ``"Linux"``, so this correctly stays silent there.
    """

    global WINDOWS_HINTED
    if WINDOWS_HINTED or platform.system() != "Windows":
        return None
    WINDOWS_HINTED = True
    if os.environ.get("SYNALINKS_NO_WINDOWS_HINT"):
        return None
    warnings.warn(WINDOWS_HINT, RuntimeWarning, stacklevel=2)
    return WINDOWS_HINT


maybe_warn_native_windows()


confine_smoke_cache: Optional[Tuple[bool, str]] = None

# ``CLONE_NEW*`` flags for the user-namespace probe in
# `confinement_available`. ``CONFINE_SRC`` carries its
# own copy because it runs as source in a separate interpreter.
CLONE_NEWUSER = 0x10000000
CLONE_NEWNS = 0x20000
CLONE_NEWNET = 0x40000000
CLONE_NEWIPC = 0x8000000
CLONE_NEWUTS = 0x4000000
CLONE_NEWPID = 0x20000000


def confinement_available() -> tuple[bool, str]:
    """Whether in-process confinement can run, by whichever backend fits.

    Returns ``(ok, reason)``; ``reason`` explains why not when ``ok`` is False.
    On Linux this means ``os.unshare`` (Python 3.12+), ``/dev/fuse``, libfuse
    and unprivileged user namespaces; on macOS it means Seatbelt's libsandbox,
    the floor under the microVM backend. Windows has neither, and the intended
    path there is to run synalinks inside **WSL2**, where the Linux backend
    works unchanged. See `confinement_backend` for *which* backend.
    """

    system = platform.system()
    if system == "Windows":
        return False, (
            "confinement requires Linux; on Windows, run synalinks inside WSL2 "
            "(Windows Subsystem for Linux), where confinement works unchanged"
        )
    if system == "Darwin":
        # Seatbelt is the floor on every Mac (built in, nothing to install);
        # `confinement_backend` upgrades to the microVM where it can run.
        # Neither needs macFUSE to *confine*: without the mount the snippet
        # simply cannot reach the virtual filesystem, which is a loss of
        # function, not of isolation.
        if not (ctypes.util.find_library("sandbox") or os.path.exists(LIBSANDBOX)):
            return False, (
                "confinement requires the macOS Seatbelt library (libsandbox), "
                "which could not be found"
            )
        return True, ""
    if system != "Linux":
        return False, f"confinement requires Linux or macOS (this host is {system})"
    if not hasattr(os, "unshare"):
        return False, "confinement requires os.unshare (Python 3.12+)"
    if not os.path.exists("/dev/fuse"):
        reason = "confinement requires FUSE (/dev/fuse is absent)"
        try:
            with open("/proc/sys/kernel/osrelease") as fh:
                if "microsoft" in fh.read().lower():
                    # WSL1 has no real kernel / FUSE; WSL2 does.
                    reason += "; on WSL use WSL2 (WSL1 cannot confine)"
        except OSError:
            pass
        return False, reason
    if not _LIBFUSE_AVAILABLE:
        # `/dev/fuse` above is the kernel device; the mount also needs the
        # libfuse user-space library, which mfusepy loads at import. A host can
        # easily have the first and not the second (slim containers, or the
        # mirage-ai[fuse] extra left uninstalled). Report it here so confinement
        # fails closed with a usable reason, instead of passing this probe and
        # then silently running unconfined when the mount does not come up.
        return False, (
            "confinement requires the libfuse user-space library, which could "
            "not be loaded; install libfuse (Debian/Ubuntu: libfuse3-3 or "
            "libfuse2) and the mirage-ai[fuse] extra"
        )
    try:
        with open("/proc/sys/kernel/unprivileged_userns_clone") as fh:
            if fh.read().strip() == "0":
                return False, "unprivileged user namespaces are disabled"
    except OSError:
        pass  # absent on distros where userns is enabled by default
    # The checks above are cheap capability probes; they pass on environments
    # that still *forbid* the credential-map handshake the real confinement
    # performs (notably GitHub Actions and many containers deny writing
    # ``/proc/self/setgroups``), where every confined run would die with
    # ``confine-error``. Settle it by attempting the first steps of
    # ``_confine`` in a throwaway child (the fork keeps ``unshare`` off this
    # process). Cached: the answer is constant for the process lifetime.
    global confine_smoke_cache
    if confine_smoke_cache is None:
        try:
            pid = os.fork()
        except OSError as exc:
            return False, f"cannot fork to probe confinement: {exc}"
        if pid == 0:  # child: bare-minimum work, never returns to the caller
            try:
                uid, gid = os.getuid(), os.getgid()
                os.unshare(
                    CLONE_NEWUSER
                    | CLONE_NEWNS
                    | CLONE_NEWNET
                    | CLONE_NEWIPC
                    | CLONE_NEWUTS
                    | CLONE_NEWPID
                )
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
            confine_smoke_cache = (True, "")
        else:
            confine_smoke_cache = (
                False,
                "the kernel/environment forbids unprivileged user-namespace setup "
                "(e.g. writing /proc/self/setgroups is denied, as on GitHub "
                "Actions and inside many containers)",
            )
    return confine_smoke_cache


def confinement_backend() -> Tuple[Optional[str], str]:
    """The strongest confinement backend this host can run; ``(backend, reason)``.

    ``"namespaces"`` on Linux. On macOS ``"microvm"`` (a libkrun VM running the
    Linux backend behind its own kernel) where it can run, else ``"seatbelt"``,
    in which case ``reason`` says why the microVM was not used. ``None`` when
    nothing can confine, with ``reason`` saying why.
    """
    ok, reason = confinement_available()
    if not ok:
        return None, reason
    if platform.system() == "Linux":
        return "namespaces", ""
    from synalinks.src.utils.microvm_utils import microvm_available

    vm_ok, vm_reason = microvm_available()
    return ("microvm", "") if vm_ok else ("seatbelt", vm_reason)


# ``AUDIT_ARCH_*`` tokens (EM_* | 64-bit | little-endian) the seccomp filter
# pins on, so a process cannot dodge the filter by issuing a syscall under a
# different ABI (e.g. x86_64's x32). Only arches with a maintained syscall-number
# table below are listed; others get no filter (namespace isolation still holds).
AUDIT_ARCH = {"x86_64": 0xC000003E, "aarch64": 0xC00000B7}

# Syscalls a confined process is denied (returned EPERM). Not an attempt at a
# minimal allowlist: sandboxed code runs arbitrary Python + packages, so this
# is a denylist of high-risk calls a normal workload never needs: kernel attack
# surface (bpf, userfaultfd, perf_event_open, ptrace, process_vm_*), privilege /
# namespace manipulation (mount, umount2, pivot_root, chroot, unshare, setns),
# module loading + kexec, keyrings (keyctl/add_key/request_key), the
# open_by_handle_at container-escape vector, and host-management calls
# (reboot, swapon/off, acct, quotactl, iopl/ioperm). ``clone``/``clone3`` are
# deliberately *not* denied: modern glibc routes threading/fork through them.
SECCOMP_DENY = {
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


def build_seccomp_filter(machine: Optional[str] = None) -> Optional[str]:
    """Assemble the classic-BPF seccomp denylist for an arch (base64), or None.

    ``machine`` defaults to this host's; the microVM backend passes the guest's
    (``aarch64``), since the filter runs in the guest kernel, not this one.

    Emits a ``sock_filter[]`` program: pin the audit arch (mismatch → EPERM),
    reject x32 syscalls on x86_64, then EPERM each denied syscall number and
    ALLOW everything else. Built host-side and shipped in the confinement config
    so the subprocess only has to ``prctl(PR_SET_SECCOMP, ...)`` it. Returns
    ``None`` on an arch without a maintained number table (no filter applied).
    """

    machine = machine or platform.machine()
    arch = AUDIT_ARCH.get(machine)
    denied = SECCOMP_DENY.get(machine)
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
