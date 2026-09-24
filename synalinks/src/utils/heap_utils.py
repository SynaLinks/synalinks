# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Host heap hygiene for long-lived sandbox hosts (glibc only)."""

import sys
from typing import Any
from typing import Optional

# ---------------------------------------------------------------------------
# Host heap hygiene.
#
# glibc's malloc gives every thread that allocates its own arena (a 64 MiB
# heap mapping, up to 8 x cores of them) and never returns an arena's freed
# memory to the OS. A sandbox host is exactly the process that suffers from
# it: the asyncio default executor, the FUSE bridge, the tool-RPC loop and the
# HTTP client all allocate from worker threads, and every ``run_code`` churns
# multi-megabyte buffers through them (the dill-and-base64 ``inputs`` blob,
# the persisted namespace, file contents crossing FUSE). After an hour of
# snippets a host was measured at 2.2 GB resident with 14 MB of live Python
# objects: 28 threads, 28 fully-resident 64 MiB arenas (``/proc/<pid>/smaps``,
# 2026-08-30). Capping the arenas makes all threads share two, which bounds
# the retained memory at a couple of hundred MB, and ``malloc_trim`` after each
# run hands back what the shared arenas no longer use.
#
# Both are process-wide glibc settings and a no-op elsewhere (musl, macOS).
# They are applied from the process itself so they also hold where no launcher
# environment is available (a notebook kernel, a Kaggle submission); the
# ``MALLOC_ARENA_MAX`` environment variable is the equivalent for a launcher.
M_ARENA_MAX = -8  # glibc <malloc.h>
malloc_arenas_capped: Optional[int] = None


def load_glibc() -> Optional[Any]:
    """The process's C library via ctypes, or ``None`` when it is not glibc."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        # ``mallopt``/``malloc_trim`` exist on glibc; musl has neither.
        libc.mallopt
        libc.malloc_trim
        return libc
    except (OSError, AttributeError):
        return None


def cap_malloc_arenas(max_arenas: int) -> bool:
    """Cap glibc's per-thread malloc arenas for this process.

    Idempotent: the first call wins for the life of the process, later calls
    with the same or a higher cap are no-ops. Arenas that already exist keep
    living; the cap governs the ones threads would create from now on, so call
    it before the process starts its worker threads.

    Args:
        max_arenas (int): The ``M_ARENA_MAX`` value (``2`` is the usual choice
            for a long-running service).

    Returns:
        bool: ``True`` when the cap is in place (now or from an earlier call),
            ``False`` when the platform's allocator is not glibc.
    """
    global malloc_arenas_capped
    if malloc_arenas_capped is not None and malloc_arenas_capped <= max_arenas:
        return True
    libc = load_glibc()
    if libc is None:
        return False
    if libc.mallopt(M_ARENA_MAX, int(max_arenas)) != 1:
        return False
    malloc_arenas_capped = int(max_arenas)
    return True


def malloc_trim() -> bool:
    """Return freed heap pages to the OS (every glibc arena, not just the top).

    Returns:
        bool: ``True`` when memory was released, ``False`` when nothing could be
            or the allocator is not glibc.
    """
    libc = load_glibc()
    if libc is None:
        return False
    try:
        return bool(libc.malloc_trim(0))
    except Exception:  # noqa: BLE001 - never let hygiene break a run
        return False
