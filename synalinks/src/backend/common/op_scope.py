# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Phase scope (`inference` / `reward` / `optimizer`) for operational metrics.

The trainer wraps each phase in ``with op_scope("reward", clock=...):`` so the
LanguageModel / EmbeddingModel can attribute every provider call to the
phase that triggered it, and so throughput can be measured against the
phase's true wall-clock span.

Two pieces of state, deliberately kept separate:

- **The active scope** is a ``contextvars.ContextVar``. Synalinks runs LM/EM
  calls concurrently (``asyncio.gather``) and across greenlets that copy the
  *context* (see ``utils.async_utils``); a ``threading.local`` set on the
  trainer thread would be lost there and calls would be misattributed. A
  ContextVar is copied into each child task/greenlet at creation, so every
  concurrent call reads the scope that was active when it was spawned, and
  sibling phases never clobber each other.

- **Per-phase wall-clock** lives on a `PhaseClock`. Each trainer (program)
  owns one, so programs evaluated concurrently on the same event loop (e.g.
  the cells of a tuner sweep) each measure their own phases instead of
  sharing, and corrupting, one clock. A phase accrues time while at least one
  context has it as its *innermost* scope on that clock, so concurrent
  regions of the same phase count once (their union), and the optimizer
  phase (which wraps reward computation) is credited its *self* time without
  double-counting the nested reward span, mirroring how calls are bucketed.
  `op_scope` without a clock uses a thread-local default clock (reset by
  ``clear_session``).
"""

import contextvars
import time

from synalinks.src.backend.common import global_state

PHASES = ("inference", "reward", "optimizer")

_OP_SCOPE = contextvars.ContextVar("synalinks_op_scope", default=None)

# The (phase, clock) of the innermost `op_scope` in the current context, so a
# nested scope knows which phase it suspends, and on which clock.
_OP_FRAME = contextvars.ContextVar("synalinks_op_frame", default=None)

_DEFAULT_CLOCK_KEY = "op_scope_default_clock"


def current_op_scope():
    """Return the active phase (``inference``/``reward``/``optimizer``) or
    ``None`` when no phase scope is active (e.g. a standalone call)."""
    scope = _OP_SCOPE.get()
    return scope if scope in PHASES else None


class PhaseClock:
    """Cumulated wall-clock seconds per phase for one program.

    A phase accrues time while its active count (the number of contexts in
    which it is the innermost scope on this clock) is positive.
    """

    def __init__(self):
        self._wall = {phase: 0.0 for phase in PHASES}
        self._active = {phase: 0 for phase in PHASES}
        self._last_ts = None

    def _advance(self, now):
        if self._last_ts is not None:
            elapsed = now - self._last_ts
            for phase, count in self._active.items():
                if count > 0:
                    self._wall[phase] += elapsed
        self._last_ts = now

    def _switch(self, leaving, entering):
        """Credit the interval so far, then move one context from the
        `leaving` phase (or none) to the `entering` phase (or none)."""
        self._advance(time.perf_counter())
        if leaving in self._active:
            self._active[leaving] -= 1
        if entering in self._active:
            self._active[entering] += 1

    def read(self, phase):
        """Seconds accrued by ``phase``, including a still-running interval."""
        wall = self._wall.get(phase, 0.0)
        if self._active.get(phase, 0) > 0 and self._last_ts is not None:
            wall += time.perf_counter() - self._last_ts
        return wall

    def add(self, phase, seconds):
        self._wall[phase] = self._wall.get(phase, 0.0) + seconds


def default_phase_clock():
    """The thread-local clock used by `op_scope` when none is given."""
    clock = global_state.get_global_attribute(_DEFAULT_CLOCK_KEY)
    if clock is None:
        clock = PhaseClock()
        global_state.set_global_attribute(_DEFAULT_CLOCK_KEY, clock)
    return clock


def read_phase_wall_clock_s(phase, clock=None):
    """Cumulated wall-clock seconds spent in ``phase`` on ``clock`` (the
    thread-local default clock when ``None``)."""
    return (clock or default_phase_clock()).read(phase)


class op_scope:
    """Context manager marking a region as running in ``phase``.

    Sets the scope ContextVar for the duration (so concurrent calls spawned
    inside inherit it) and accrues the region's wall-clock to ``phase`` on
    ``clock`` (the trainer's own clock; the thread-local default when
    ``None``), excluding any nested scopes.
    """

    def __init__(self, phase, clock=None):
        self.phase = phase
        self.clock = clock
        self._scope_token = None
        self._frame_token = None
        self._suspended = None

    def __enter__(self):
        clock = self.clock or default_phase_clock()
        self.clock = clock
        parent = _OP_FRAME.get()
        # Suspend the enclosing phase only when it runs on the same clock;
        # another program's phase keeps running on its own clock.
        self._suspended = parent[0] if parent and parent[1] is clock else None
        clock._switch(self._suspended, self.phase)
        self._frame_token = _OP_FRAME.set((self.phase, clock))
        self._scope_token = _OP_SCOPE.set(self.phase)
        return self

    def __exit__(self, *exc_info):
        self.clock._switch(self.phase, self._suspended)
        _OP_SCOPE.reset(self._scope_token)
        _OP_FRAME.reset(self._frame_token)
        return False


def _add_phase_wall_clock_s(phase, seconds, clock=None):
    """Test helper: add to a phase's wall-clock accumulator without timing a
    real region."""
    (clock or default_phase_clock()).add(phase, seconds)


# ---------------------------------------------------------------------------
# Trajectory start (for whole-trajectory time-to-first-token of agents)
# ---------------------------------------------------------------------------

_TRAJECTORY_START = contextvars.ContextVar("synalinks_trajectory_start", default=None)


def current_trajectory_start():
    """Return the ``time.perf_counter()`` stamp marking the start of the
    outermost in-flight agent trajectory, or ``None`` when no trajectory scope
    is active.

    Used to measure *whole-trajectory* time-to-first-token: the wall-clock from
    when an agent begins (including every tool-calling round) to the first token
    of its streamed final answer -- distinct from the per-call TTFT, which only
    times the final LM call.
    """
    return _TRAJECTORY_START.get()


class trajectory_scope:
    """Mark the start of an agent trajectory for whole-trajectory TTFT.

    **Set-once**: if a trajectory start is already active in the current context
    a nested (sub-)agent leaves it untouched, so the timestamp always reflects
    the *outermost* agent's start. Like `op_scope`, the start is held in a
    ContextVar, so it is copied into the concurrent tasks/greenlets the agent
    spawns and the final LM call reads the value active when it was spawned.
    """

    def __init__(self):
        self._token = None

    def __enter__(self):
        if _TRAJECTORY_START.get() is None:
            self._token = _TRAJECTORY_START.set(time.perf_counter())
        return self

    def __exit__(self, *exc_info):
        # Only the scope that actually set the start resets it; nested scopes
        # (which found one already active) are no-ops, preserving the outermost.
        if self._token is not None:
            _TRAJECTORY_START.reset(self._token)
        return False
