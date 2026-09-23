# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""macOS microVM backend: run each sandboxed ``python3`` in a libkrun microVM.

Seatbelt confines a process on the *host* kernel, so a kernel bug reachable from
the snippet is a full escape, and it has no PID isolation, syscall filter or
process cap. This backend instead boots a Linux microVM (libkrun, on Apple's
Hypervisor.framework) per Python execution and runs the **Linux** confinement
inside it: namespaces, ``pivot_root`` into the virtual filesystem and seccomp,
behind a separate guest kernel. Mirage spawns a fresh interpreter per command,
and a libkrun VM boots in ~0.15 s, so one VM per execution fits that model.

Pieces:

* ``synalinks-krun``, a small C helper (``krun/krun_helper.c``) that creates
  the VM. It is a separate binary because creating a VM needs the
  ``com.apple.security.hypervisor`` entitlement on the calling executable,
  which the Python interpreter does not carry. Built with the Xcode command
  line tools on first use and ad-hoc signed with that one entitlement.
* A Linux root filesystem: the ``python`` slim image, pinned by digest and
  verified blob by blob, extracted once into the user cache and exposed to the
  guest **read-only**. The guest entry script and ``dill`` are added to it.
* The sandbox's host directory (state, config, result files) is shared into
  the guest over virtiofs **at its host path**, so every path in the bootstrap
  config means the same thing on both sides.
* The virtual filesystem is served *by the guest*: a FUSE daemon there forwards
  each operation over vsock to `FsBridge`, which calls the workspace's
  ``MirageFS``, the same object behind a host FUSE mount. (libkrun's virtiofs
  cannot re-export a macFUSE mount, and this needs no macFUSE at all.)
* Host-tool RPC sockets are mapped onto vsock ports; the guest entry proxies a
  guest-local unix socket of the same name onto each (see ``sock_dir``).

Nothing reaches the guest implicitly: no network (libkrun's transparent socket
proxying is disabled unless the sandbox asked for network), no host
environment variables, and a read-only root.
"""

import asyncio
import ctypes
import ctypes.util
import errno
import fcntl
import glob
import hashlib
import io
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.request
import uuid
import zipfile
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

# The guest image: python 3.12 slim for linux/arm64, pinned by manifest digest
# so every host runs the same bytes and a registry compromise cannot swap them.
IMAGE_REPO = "library/python"
IMAGE_DIGEST = "sha256:950206c37262dd86c55659797f6ee418fee30535072f65a82ed470d985f5cda5"
REGISTRY = "https://registry-1.docker.io/v2"
AUTH = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:{}:pull"
MANIFEST_TYPES = (
    "application/vnd.oci.image.manifest.v1+json,"
    "application/vnd.docker.distribution.manifest.v2+json"
)

GUEST_PYTHON = "/usr/local/bin/python3"
GUEST_SITE = "usr/local/lib/python3.12/site-packages"
GUEST_ENTRY = "/opt/synalinks/entry.py"
# Guest-local directory of RPC proxy sockets (a tmpfs, bound into the confined
# root). The bootstrap connects to ``sock_dir/<basename of the host socket>``.
GUEST_SOCK_DIR = "/run/synalinks"
# Guest directories bound read-only into the confined root: the image's Python
# install and system libraries, the guest counterpart of the host runtime binds.
GUEST_BINDS = ["/usr/local", "/usr", "/lib", "/bin"]
# First vsock port for host-tool RPC sockets (ports below 1024 are reserved).
VSOCK_BASE = 1024
# Where the guest mounts the sandbox's virtual filesystem (a FUSE mount it
# serves itself, bridged to the host's Mirage ops; see `FsBridge`).
GUEST_MOUNT = "/sandbox"
# libfuse3 for the guest FUSE daemon (the slim image has none): the Debian
# trixie arm64 package matching the image, pinned by the sha256 from Debian's
# signed package index.
LIBFUSE_DEB = (
    "https://deb.debian.org/debian/pool/main/f/fuse3/libfuse3-4_3.17.2-3_arm64.deb"
)
LIBFUSE_DEB_SHA256 = "66793b6b9a559e95a5ff853cbbff337076ef900c4b0d927f57fd986b3b1bea09"
GUEST_LIBFUSE = "/usr/lib/aarch64-linux-gnu/libfuse3.so.4"
GUEST_FS = "/opt/synalinks/fs.py"
# Plotting in the guest: matplotlib and what it needs, as Linux arm64 wheels
# for the image's Python 3.12, at the versions synalinks' own lock resolves
# (so the guest runs what the host tests). Pinned by URL and sha256 from PyPI;
# ``run_code`` captures their figures as PNG results.
GUEST_WHEELS = (
    (
        "https://files.pythonhosted.org/packages/88/90/4e10e033d9b66589d8ed98b84c95cdbb57033d57c1f41339d7393dbd2f2e/matplotlib-3.11.1-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl",
        "c52f7ad20ef476806ed212380b1d54d20310c8b86bdc2c9a68b51f0024a44472",
    ),
    (
        "https://files.pythonhosted.org/packages/e5/21/4947e0e9d6c9fc2e2ff15b8949049ee44f63adb9cacc729ab8793f97e712/numpy-2.5.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
        "8ee9c4eeb8454b3660a8b53493563c3e121c2fc94fbd72b848ef814ed7b676a9",
    ),
    (
        "https://files.pythonhosted.org/packages/d4/1c/a12359b9b2ca3a845e8f7f9ac08bdf776114eb931392fcad91743e2ea17b/contourpy-1.3.3-cp312-cp312-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl",
        "92d9abc807cf7d0e047b95ca5d957cf4792fcd04e920ca70d48add15c1a90ea7",
    ),
    (
        "https://files.pythonhosted.org/packages/e7/05/c19819d5e3d95294a6f5947fb9b9629efb316b96de511b418c53d245aae6/cycler-0.12.1-py3-none-any.whl",
        "85cef7cff222d8644161529808465972e51340599459b8ac3ccbac5a854e0d30",
    ),
    (
        "https://files.pythonhosted.org/packages/44/04/0b91d8e916e92ad1fac9e4624760baf0fd5ff2ead614c2f68fb21373f03f/fonttools-4.63.0-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl",
        "ef3048ef05dbb552b89817713d9cac912e00d0fde4a3105c00d29e52e10c89af",
    ),
    (
        "https://files.pythonhosted.org/packages/c8/2f/cebfcdb60fd6a9b0f6b47a9337198bcbad6fbe15e68189b7011fd914911f/kiwisolver-1.5.0-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl",
        "b2af221f268f5af85e776a73d62b0845fc8baf8ef0abfae79d29c77d0e776aaf",
    ),
    (
        "https://files.pythonhosted.org/packages/63/34/ba1c580383c9eada3711951fef0795c80b829a078d72188184bcab9dd527/packaging-26.3-py3-none-any.whl",
        "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c",
    ),
    (
        "https://files.pythonhosted.org/packages/25/27/ac8f99618ffd3dde21db0f4d4b1d2ab00c0880595bfd17df103f7f39fd0c/pillow-12.3.0-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl",
        "d9c7f76c0673154f044e9d78c8655fb4213f6ca31a836df48b40fe5d187717b9",
    ),
    (
        "https://files.pythonhosted.org/packages/10/bd/c038d7cc38edc1aa5bf91ab8068b63d4308c66c4c8bb3cbba7dfbc049f9c/pyparsing-3.3.2-py3-none-any.whl",
        "850ba148bd908d7e2411587e247a1e4f0327839c40e2e5e6d05a007ecc69911d",
    ),
    (
        "https://files.pythonhosted.org/packages/ec/57/56b9bcc3c9c6a792fcbaf139543cee77261f3651ca9da0c93f5c1221264b/python_dateutil-2.9.0.post0-py2.py3-none-any.whl",
        "a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427",
    ),
    (
        "https://files.pythonhosted.org/packages/b7/ce/149a00dd41f10bc29e5921b496af8b574d8413afcd5e30dfa0ed46c2cc5e/six-1.17.0-py2.py3-none-any.whl",
        "4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274",
    ),
)
# MirageFS operations the bridge forwards. The macFUSE-only entry points
# (renamex, setattr_x, ...) are not needed by a Linux guest and not exposed.
FS_OPS = (
    "getattr readdir read write create mkdir readlink symlink unlink rename "
    "rmdir statfs chmod chown utimens access setxattr getxattr listxattr "
    "removexattr flush fsync open release truncate"
).split()

# Wire codec for the filesystem bridge, run on both sides: JSON with bytes
# tagged as base64, and errors sent as errno *names*, because errno numbers
# differ between the Linux guest and the macOS host (ENOTEMPTY is 39 on one
# and 66 on the other). Frames are a 4-byte big-endian length plus JSON.
CODEC_SRC = r"""
import base64 as _b64, json as _json, struct as _struct


def encode(value):
    if isinstance(value, (bytes, bytearray)):
        return {"__b64__": _b64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return None  # e.g. a ctypes fuse_file_info: not meaningful across the wire


def decode(value):
    if isinstance(value, dict):
        if "__b64__" in value:
            return _b64.b64decode(value["__b64__"])
        return {k: decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v) for v in value]
    return value


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed")
        buf += chunk
    return buf


def send_frame(sock, obj):
    body = _json.dumps(obj).encode("utf-8")
    sock.sendall(_struct.pack(">I", len(body)) + body)


def recv_frame(sock):
    (length,) = _struct.unpack(">I", recv_exact(sock, 4))
    return _json.loads(recv_exact(sock, length))
"""

# The guest FUSE daemon: serves GUEST_MOUNT by forwarding every operation over
# vsock to the host's `FsBridge`. One connection per libfuse worker thread.
FS_SRC = CODEC_SRC + r"""
import errno, socket, sys, threading
import mfusepy

PORT, MOUNT = int(sys.argv[1]), sys.argv[2]
_local = threading.local()


def call(op, *args):
    sock = getattr(_local, "sock", None)
    if sock is None:
        sock = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
        sock.connect((2, PORT))
        _local.sock = sock
    send_frame(sock, {"op": op, "args": encode(list(args))})
    reply = recv_frame(sock)
    if "errno" in reply:
        raise mfusepy.FuseOSError(getattr(errno, reply["errno"], errno.EIO))
    return decode(reply["result"])


class Proxy(mfusepy.Operations):
    use_ns = True


for _op in %(ops)r:
    setattr(Proxy, _op, lambda self, *args, _op=_op: call(_op, *args))

mfusepy.FUSE(Proxy(), MOUNT, foreground=True, allow_other=True)
""" % {"ops": FS_OPS}

HELPER_SRC = os.path.join(os.path.dirname(__file__), "krun", "krun_helper.c")
ENTITLEMENTS = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>com.apple.security.hypervisor</key><true/>
</dict></plist>
"""

# Runs as root inside the guest, as the VM's only workload. Mounts the shares
# (the root is read-only, so scratch space is tmpfs), starts the RPC proxies,
# then runs the job's ``python3`` and exits with its code, which libkrun hands
# back to the helper as its own exit code. Everything the job needs arrives in
# the job file: argv travels on the kernel command line, which is short and
# ASCII-only, so it carries only the share list and the job path.
ENTRY_SRC = r"""
import ctypes, json, os, socket, subprocess, sys, threading, time

SOCK_DIR = %(sock_dir)r
MOUNT = %(mount)r
VMADDR_CID_HOST = 2


libc = ctypes.CDLL(None, use_errno=True)
MS_BIND = 4096


def mount_fs(source, target, fstype, flags=0):
    # mount(2) directly: a `mount` process per call costs more than the call.
    if libc.mount(source.encode(), target.encode(), fstype.encode(), flags, None):
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), target)


def pump(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except OSError:
        pass
    finally:
        for s in (src, dst):
            try:
                s.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


def proxy(path, port):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(16)

    def serve():
        while True:
            conn, _ = server.accept()
            upstream = socket.socket(socket.AF_VSOCK, socket.SOCK_STREAM)
            try:
                upstream.connect((VMADDR_CID_HOST, port))
            except OSError:
                conn.close()
                continue
            for a, b in ((conn, upstream), (upstream, conn)):
                threading.Thread(target=pump, args=(a, b), daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()


def main():
    *shares, job_path = sys.argv[1:]
    for path in ("/tmp", "/run"):
        mount_fs("tmpfs", path, "tmpfs")
    # Each share mounts at its host path. The helper added only the path's top
    # level to the read-only root, as an empty virtual dir: a tmpfs on it makes
    # the rest of the path creatable.
    tops = set()
    for item in shares:
        tag, path = item.split("=", 1)
        top = "/" + path.strip("/").split("/")[0]
        if top not in tops:
            try:
                populated = bool(os.listdir(top))
            except OSError:  # libkrun's virtual dirs cannot be listed: ours
                populated = False
            if populated:
                # A share under /usr, /opt, ... would shadow the image's own
                # files; skip it (an extra bind the guest then does not see).
                continue
            mount_fs("tmpfs", top, "tmpfs")
            tops.add(top)
        os.makedirs(path, exist_ok=True)
        mount_fs(tag, path, "virtiofs")
        # macOS reaches /private/var through the /var symlink, and host paths
        # arrive either way (a temp dir is /var/folders/... unresolved). The
        # guest's own /var is not in the confined root, so a tmpfs can stand in
        # for it and carry the alias.
        if path.startswith("/private/var/"):
            alias = path[len("/private"):]
            if "/var" not in tops:
                mount_fs("tmpfs", "/var", "tmpfs")
                tops.add("/var")
            os.makedirs(alias, exist_ok=True)
            mount_fs(path, alias, "none", MS_BIND)
    with open(job_path) as fh:
        job = json.load(fh)
    uid, gid = job["uid"], job["gid"]
    os.makedirs(SOCK_DIR, exist_ok=True)
    for name, port in (job.get("vsock") or {}).items():
        proxy(os.path.join(SOCK_DIR, name), port)
        os.chown(os.path.join(SOCK_DIR, name), uid, gid)
    if job.get("fs_port"):
        # Serve the sandbox's virtual filesystem at MOUNT, bridged to the host.
        subprocess.Popen(
            [%(python)r, %(fs)r, str(job["fs_port"]), MOUNT],
            env={"FUSE_LIBRARY_PATH": %(libfuse)r, "PATH": "/usr/bin:/bin"},
        )
        for _ in range(500):
            with open("/proc/mounts") as fh:
                if any(line.split()[1] == MOUNT for line in fh):
                    break
            time.sleep(0.01)
        else:
            raise SystemExit("the sandbox filesystem did not come up")
    stdin = open(job["stdin"], "rb") if job.get("stdin") else subprocess.DEVNULL
    # The job runs as the host user's uid, not root: the shared host dir is
    # owned by that uid, and the job's own user namespace (the Linux backend)
    # can only map its own uid, so as root it could not write its state there.
    # Nothing in the job needs guest root, so it does not get it.
    proc = subprocess.run(
        [%(python)r, *job["argv"]],
        stdin=stdin,
        env=job["env"],
        user=uid,
        group=gid,
        extra_groups=[],
    )
    sys.exit(proc.returncode)


main()
""" % {
    "sock_dir": GUEST_SOCK_DIR,
    "mount": GUEST_MOUNT,
    "python": GUEST_PYTHON,
    "fs": GUEST_FS,
    "libfuse": GUEST_LIBFUSE,
}


def cache_dir() -> str:
    """Per-user cache holding the built helper and the extracted image."""
    return os.path.join(os.path.expanduser("~/Library/Caches"), "synalinks", "microvm")


def libkrun_dir() -> Optional[str]:
    """Directory holding ``libkrun.dylib`` (and ``libkrunfw``), or ``None``."""
    for prefix in ("/opt/homebrew", "/usr/local"):
        libdir = os.path.join(prefix, "lib")
        if os.path.exists(os.path.join(libdir, "libkrun.dylib")):
            return libdir
    return None


def hypervisor_supported() -> bool:
    """Whether this Mac exposes Hypervisor.framework (``kern.hv_support``)."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    rc = libc.sysctlbyname(
        b"kern.hv_support", ctypes.byref(value), ctypes.byref(size), None, 0
    )
    return rc == 0 and value.value == 1


def microvm_available() -> Tuple[bool, str]:
    """Whether the microVM backend can run on this host; ``(ok, reason)``.

    Cheap: checks the platform, the hypervisor, libkrun and a compiler (or an
    already-built helper). The image is provisioned later, on first use.
    """
    if sys.platform != "darwin":
        return False, "the microVM backend is macOS-only"
    if platform.machine() != "arm64":
        return False, "the microVM backend needs Apple silicon"
    if not hypervisor_supported():
        return False, (
            "Hypervisor.framework is unavailable (a Mac that is itself a VM "
            "usually cannot nest one)"
        )
    if libkrun_dir() is None:
        return False, "libkrun is not installed (brew tap slp/krun; brew install libkrun)"
    if not os.path.exists(helper_path()) and shutil.which("clang") is None:
        return False, "building the microVM helper needs the Xcode command line tools"
    return True, ""


def helper_path() -> str:
    """Where the helper built from the current source lives (keyed by its hash)."""
    with open(HELPER_SRC, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    return os.path.join(cache_dir(), f"synalinks-krun-{digest}")


class CacheLock:
    """Exclusive lock over the cache, so concurrent processes provision once."""

    def __enter__(self):
        os.makedirs(cache_dir(), exist_ok=True)
        self._fh = open(os.path.join(cache_dir(), ".lock"), "w")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()


def build_helper(libdir: str) -> str:
    """Compile and ad-hoc sign the helper if the current source is not built."""
    target = helper_path()
    if os.path.exists(target):
        return target
    prefix = os.path.dirname(libdir)
    with tempfile.TemporaryDirectory(dir=cache_dir()) as tmp:
        binary = os.path.join(tmp, "synalinks-krun")
        entitlements = os.path.join(tmp, "hv.entitlements")
        with open(entitlements, "w") as fh:
            fh.write(ENTITLEMENTS)
        subprocess.run(
            [
                "clang",
                "-O2",
                "-Wno-comment",
                "-I" + os.path.join(prefix, "include"),
                "-L" + libdir,
                "-lkrun",
                HELPER_SRC,
                "-o",
                binary,
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["codesign", "-s", "-", "--entitlements", entitlements, "--force", binary],
            check=True,
            capture_output=True,
        )
        os.replace(binary, target)
    return target


def fetch(url: str, token: Optional[str] = None, accept: Optional[str] = None) -> bytes:
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    if accept:
        request.add_header("Accept", accept)
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def verified(data: bytes, digest: str) -> bytes:
    """``data`` if it matches the content-addressed ``sha256:...`` ``digest``."""
    algo, _, expected = digest.partition(":")
    if algo != "sha256":
        raise ValueError(f"unsupported digest {digest}")
    return verified_sha256(data, expected)


def verified_sha256(data: bytes, expected: str) -> bytes:
    """``data`` if its sha256 is ``expected``, else raise: nothing unverified lands."""
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"download does not match its pinned sha256 {expected}")
    return data


def extract_deb(data: bytes, root: str) -> None:
    """Extract a Debian package's files into ``root`` (no maintainer scripts)."""
    if not data.startswith(b"!<arch>\n"):
        raise ValueError("not a Debian package")
    offset = 8
    while offset < len(data):
        header = data[offset : offset + 60]
        name = header[:16].decode("ascii").strip().rstrip("/")
        size = int(header[48:58].decode("ascii").strip())
        body = data[offset + 60 : offset + 60 + size]
        if name.startswith("data.tar"):
            extract_layer(body, root)
            return
        offset += 60 + size + (size % 2)  # ar members are 2-byte aligned
    raise ValueError("Debian package has no data archive")


def remove(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.remove(path)


def extract_layer(data: bytes, root: str) -> None:
    """Apply one image layer to ``root``: OCI whiteouts, no device nodes."""
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        members = []
        for member in tar.getmembers():
            parent, base = os.path.split(member.name)
            if base == ".wh..wh..opq":  # opaque dir: drop what lower layers put there
                directory = os.path.join(root, parent)
                if os.path.isdir(directory):
                    for entry in os.listdir(directory):
                        remove(os.path.join(directory, entry))
            elif base.startswith(".wh."):  # whiteout: delete one lower-layer entry
                remove(os.path.join(root, parent, base[len(".wh.") :]))
            elif not (member.ischr() or member.isblk() or member.isfifo()):
                members.append(member)
        tar.extractall(root, members=members, filter="tar")


def rootfs_key() -> str:
    """Cache key: the image plus everything this module adds to it."""
    import dill

    extras = (
        ENTRY_SRC
        + FS_SRC
        + dill.__version__
        + IMAGE_DIGEST
        + LIBFUSE_DEB_SHA256
        + "".join(sha for _, sha in GUEST_WHEELS)
    )
    return hashlib.sha256(extras.encode("utf-8")).hexdigest()[:16]


def provision_rootfs(helper: str, libdir: str) -> str:
    """Download, verify and extract the guest image once; return its root."""
    target = os.path.join(cache_dir(), f"rootfs-{rootfs_key()}")
    if os.path.isdir(target):
        return target
    token = json.loads(fetch(AUTH.format(IMAGE_REPO)))["token"]
    manifest = json.loads(
        verified(
            fetch(
                f"{REGISTRY}/{IMAGE_REPO}/manifests/{IMAGE_DIGEST}",
                token,
                MANIFEST_TYPES,
            ),
            IMAGE_DIGEST,
        )
    )
    staging = tempfile.mkdtemp(dir=cache_dir(), prefix="rootfs-staging-")
    try:
        for layer in manifest["layers"]:
            blob = fetch(f"{REGISTRY}/{IMAGE_REPO}/blobs/{layer['digest']}", token)
            extract_layer(verified(blob, layer["digest"]), staging)
        for path, source in ((GUEST_ENTRY, ENTRY_SRC), (GUEST_FS, FS_SRC)):
            target_file = os.path.join(staging, path.lstrip("/"))
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            with open(target_file, "w") as fh:
                fh.write(source)
        os.makedirs(os.path.join(staging, GUEST_MOUNT.lstrip("/")))
        extract_deb(verified_sha256(fetch(LIBFUSE_DEB), LIBFUSE_DEB_SHA256), staging)
        # ``dill`` (the bootstrap) and ``mfusepy`` (the FUSE daemon) are not in
        # the image. Both are pure Python, so the host's copies run unchanged.
        import dill
        import mfusepy

        shutil.copytree(
            os.path.dirname(dill.__file__),
            os.path.join(staging, GUEST_SITE, "dill"),
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy(mfusepy.__file__, os.path.join(staging, GUEST_SITE))
        for url, sha256 in GUEST_WHEELS:
            wheel = verified_sha256(fetch(url), sha256)
            with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
                archive.extractall(os.path.join(staging, GUEST_SITE))
        # Precompile the bytecode, in the guest so it matches the guest's
        # Python whatever the host runs: the root is read-only at run time, so
        # without this every run recompiles each module it imports.
        subprocess.run(
            [
                helper,
                "--root",
                staging,
                "--root-writable",
                "--",
                GUEST_PYTHON,
                "-m",
                "compileall",
                "-q",
                "-j",
                "0",
                "/usr/local/lib/python3.12",
                os.path.dirname(GUEST_ENTRY),
            ],
            check=True,
            capture_output=True,
            env={"DYLD_FALLBACK_LIBRARY_PATH": libdir},
            timeout=600,
        )
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def prepare_microvm() -> Dict[str, Any]:
    """Build the helper and provision the image; return the VM launch settings.

    Slow only the first time on a host (a compile and a ~42 MB download); later
    calls find both in the cache. Raises when either cannot be prepared.
    """
    libdir = libkrun_dir()
    if libdir is None:
        raise RuntimeError("libkrun is not installed")
    with CacheLock():
        helper = build_helper(libdir)
        rootfs = provision_rootfs(helper, libdir)
        prune_cache(keep={helper, rootfs})
    return {"helper": helper, "rootfs": rootfs, "libdir": libdir}


def prune_cache(keep: set, min_age_seconds: float = 86400) -> None:
    """Drop helpers and images from older versions (~150 MB each).

    Only entries a day old or more: another process running an older synalinks
    may still be booting VMs from them.
    """
    import time

    cutoff = time.time() - min_age_seconds
    for pattern in ("synalinks-krun-*", "rootfs-*"):
        for path in glob.glob(os.path.join(cache_dir(), pattern)):
            if path in keep or os.path.getmtime(path) > cutoff:
                continue
            remove(path)


def sbpl(path: str) -> str:
    """Quote a path for SBPL, which is s-expression syntax, not shell."""
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


def ro_shares(vm: Dict[str, Any]) -> list:
    """The sandbox's ``extra_binds`` the guest can mount, resolved.

    Resolved because Seatbelt matches resolved paths and the guest mounts them
    there. A bind whose top-level dir exists in the image (``/usr/share``,
    ``/opt/...``) is left out: mounting it would need that dir replaced, which
    would take the guest's own runtime with it.
    """
    shares = []
    for d in vm.get("extra_binds") or []:
        if not os.path.isdir(d):
            continue
        real = os.path.realpath(d)
        top = real.strip("/").split("/")[0]
        if not os.path.lexists(os.path.join(vm["rootfs"], top)):
            shares.append(real)
    return shares


def helper_profile(vm: Dict[str, Any], hostdir: str) -> str:
    """The Seatbelt profile the helper confines itself with before booting a VM.

    The second wall: a guest that escapes through a bug in libkrun's device
    emulation lands in the helper process, which this profile leaves reading
    only the system and libkrun's libraries plus the image, writing only this
    sandbox's host dir (where nothing may be mapped executable), and reaching
    only that dir's unix sockets, not the user's files, keys or network.
    """
    prefix = os.path.dirname(vm["libdir"])  # e.g. /opt/homebrew
    runtime = ["/System", "/usr/lib", "/private/var/db/dyld"] + [
        # libkrun and its dependencies, not the whole prefix: Homebrew's
        # ``var`` holds databases and service state.
        os.path.join(prefix, part)
        for part in ("Cellar", "opt", "lib")
    ]
    ancestors = {"/"}
    for path in (vm["helper"], vm["rootfs"], hostdir, *runtime, *ro_shares(vm)):
        for form in (path, os.path.realpath(path)):
            parent = os.path.dirname(form)
            while parent not in ancestors and parent != "/":
                ancestors.add(parent)
                parent = os.path.dirname(parent)
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow signal (target self))",
        "(allow process-info* (target self))",
        "(allow sysctl-read)",
        '(allow file-read-data (literal "/"))',
        *(f"(allow file-read-metadata (literal {sbpl(a)}))" for a in sorted(ancestors)),
        *(f"(allow file-read* file-map-executable (subpath {sbpl(p)}))" for p in runtime),
        f"(allow file-read* file-map-executable (literal {sbpl(vm['helper'])}))",
        f"(allow file-read* (subpath {sbpl(vm['rootfs'])}))",
        *(f"(allow file-read* (subpath {sbpl(d)}))" for d in ro_shares(vm)),
        f"(allow file-read* file-write* (subpath {sbpl(hostdir)}))",
        f"(deny file-map-executable (subpath {sbpl(hostdir)}))",
        # vsock ports are backed by unix sockets in the host dir.
        f"(allow network-outbound (subpath {sbpl(hostdir)}))",
    ]
    if vm.get("network"):
        # libkrun's TSI proxies guest connections through host sockets.
        lines.append("(allow network*)")
    return "\n".join(lines) + "\n"


# The bridge's end of the codec, the same source the guest runs.
codec: Dict[str, Any] = {}
exec(CODEC_SRC, codec)


class FsBridge:
    """Serve one run's guest FUSE daemon: ``MirageFS`` calls over a unix socket.

    libkrun maps the socket onto a vsock port, so the guest can reach nothing
    on the host but these filesystem operations on the sandbox's own virtual
    filesystem, the same authority it already has over its files. One thread
    per connection, as libfuse has one connection per worker thread.
    """

    def __init__(self, fs, path: str):
        self._fs = fs
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(path)
        self._server.listen(16)
        threading.Thread(target=self.serve, daemon=True).start()

    def serve(self) -> None:
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:  # closed: the run is over
                return
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def handle(self, conn) -> None:
        with conn:
            while True:
                try:
                    request = codec["recv_frame"](conn)
                except (ConnectionError, OSError, ValueError):
                    return
                op = request.get("op")
                try:
                    if op not in FS_OPS:
                        raise OSError(errno.ENOSYS, os.strerror(errno.ENOSYS))
                    result = getattr(self._fs, op)(*codec["decode"](request["args"]))
                    reply = {"result": codec["encode"](result)}
                except OSError as exc:
                    # By name: errno numbers differ between macOS and Linux.
                    reply = {"errno": errno.errorcode.get(exc.errno or 0, "EIO")}
                except Exception:  # noqa: BLE001 - a bridge error is an I/O error
                    reply = {"errno": "EIO"}
                try:
                    codec["send_frame"](conn, reply)
                except OSError:
                    return

    def close(self) -> None:
        self._server.close()


async def run_in_microvm(vm: Dict[str, Any], args) -> Any:
    """Run one Mirage ``RunArgs`` in a fresh microVM; return a ``RunResult``.

    ``vm`` holds the launch settings from `prepare_microvm` plus the sandbox's
    live ``hostdir``, ``fs`` (the ``MirageFS`` of its workspace), ``network``,
    ``cpus`` and ``memory_mb``.
    """
    from mirage.runtime.python.bootstrap import bootstrap
    from mirage.runtime.python.flags import init_argv
    from mirage.runtime.types import RunResult

    hostdir = vm["hostdir"]
    run_id = uuid.uuid4().hex[:12]
    jobdir = os.path.join(hostdir, "jobs", run_id)
    os.makedirs(jobdir)
    bridge = None
    try:
        stdin_path = None
        if args.stdin:
            stdin_path = os.path.join(jobdir, "stdin")
            with open(stdin_path, "wb") as fh:
                fh.write(args.stdin)
        # Each live host-tool socket gets a vsock port; the guest entry proxies
        # a same-named guest socket onto it.
        sockets = sorted(glob.glob(os.path.join(hostdir, "rpc_*.sock")))
        vsock = {os.path.basename(s): VSOCK_BASE + i for i, s in enumerate(sockets)}
        # The virtual filesystem, served to the guest's FUSE daemon on the next
        # port. Unix socket paths are short (104 bytes on macOS), and the job
        # dir is not, so the socket lives beside the other host sockets.
        fs_sock = os.path.join(hostdir, f"fs_{run_id}.sock")
        fs_port = VSOCK_BASE + len(sockets)
        bridge = FsBridge(vm["fs"], fs_sock)
        job = {
            "argv": [
                *init_argv(args.flags),
                "-c",
                bootstrap(args.code, args.prog),
                *args.args,
            ],
            # The job's own environment only: never the host's.
            "env": {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", **args.env},
            "stdin": stdin_path,
            "vsock": vsock,
            "fs_port": fs_port,
            # virtiofs shows the host dir with the host's own ownership.
            "uid": os.getuid(),
            "gid": os.getgid(),
        }
        job_path = os.path.join(jobdir, "job.json")
        with open(job_path, "w") as fh:
            json.dump(job, fh)
        profile_path = os.path.join(jobdir, "helper.sb")
        with open(profile_path, "w") as fh:
            fh.write(helper_profile(vm, hostdir))

        command = [
            vm["helper"],
            "--profile",
            profile_path,
            "--root",
            vm["rootfs"],
            "--cpus",
            str(vm["cpus"]),
            "--mem",
            str(vm["memory_mb"]),
            "--share",
            f"hostdir={hostdir}",
            "--mountpoint",
            hostdir,
            "--vsock",
            f"{fs_port}={fs_sock}",
        ]
        for sock in sockets:
            command += ["--vsock", f"{vsock[os.path.basename(sock)]}={sock}"]
        # ``extra_binds``: read-only, at their host paths, like on Linux.
        entry_shares = [f"hostdir={hostdir}"]
        for index, directory in enumerate(ro_shares(vm)):
            command += ["--share-ro", f"ro{index}={directory}", "--mountpoint", directory]
            entry_shares.append(f"ro{index}={directory}")
        if vm.get("network"):
            command.append("--net")
        command += ["--", GUEST_PYTHON, GUEST_ENTRY, *entry_shares, job_path]

        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # The helper's only environment: where libkrun finds libkrunfw.
            env={"DYLD_FALLBACK_LIBRARY_PATH": vm["libdir"]},
        )
        try:
            stdout, stderr = await proc.communicate()
        except asyncio.CancelledError:
            proc.kill()  # kills the VM with it
            await proc.wait()
            raise
        return RunResult(
            stdout=stdout,
            stderr=stderr or None,
            exit_code=proc.returncode if proc.returncode is not None else 1,
        )
    finally:
        if bridge is not None:
            bridge.close()
        shutil.rmtree(jobdir, ignore_errors=True)
        try:
            os.remove(os.path.join(hostdir, f"fs_{run_id}.sock"))
        except OSError:
            pass
