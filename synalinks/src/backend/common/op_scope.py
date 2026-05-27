# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Phase scope (`inference` / `reward` / `optimizer`) for operational metrics.

The trainer wraps each phase in ``with op_scope("reward"):`` so the
LanguageModel / EmbeddingModel can attribute every provider call to the
phase that triggered it, and so throughput can be measured against the
phase's true wall-clock span.

Two pieces of state, deliberately kept separate:

- **The active scope** is a ``contextvars.ContextVar``. Synalinks runs LM/EM
  calls concurrently (``asyncio.gather``) and across greenlets that copy the
  *context* (see ``utils.async_utils``) — a ``threading.local`` set on the
  trainer thread would be lost there and calls would be misattributed. A
  ContextVar is copied into each child task/greenlet at creation, so every
  concurrent call reads the scope that was active when it was spawned, and
  sibling phases never clobber each other.

- **Per-phase wall-clock** is accumulated on enter/exit with a nesting stack.
  Only the scope currently on top of the stack accrues time, so the optimizer
  phase (which wraps reward computation) is credited its *self* time and does
  not double-count the nested reward span — mirroring how calls are bucketed.
  This lives in ``global_state`` (thread-local, reset by ``clear_session``);
  enter/exit run serially on the trainer's event-loop thread, and that's the
  same thread that reads the totals when computing metrics.
"""

import contextvars
import time

from synalinks.src.backend.common import global_state

PHASES = ("inference", "reward", "optimizer")

_OP_SCOPE = contextvars.ContextVar("synalinks_op_scope", default=None)

_WALL_STATE_KEY = "op_scope_wall_state"


def current_op_scope():
    """Return the active phase (``inference``/``reward``/``optimizer``) or
    ``None`` when no phase scope is active (e.g. a standalone call)."""
    scope = _OP_SCOPE.get()
    return scope if scope in PHASES else None


def _wall_state():
    state = global_state.get_global_attribute(_WALL_STATE_KEY)
    if state is None:
        state = {
            "stack": [],
            "last_ts": None,
            "wall": {phase: 0.0 for phase in PHASES},
        }
        global_state.set_global_attribute(_WALL_STATE_KEY, state)
    return state


def read_phase_wall_clock_s(phase):
    """Cumulated wall-clock seconds spent with ``phase`` on top of the scope
    stack since the last ``clear_session()``."""
    return _wall_state()["wall"].get(phase, 0.0)


def _credit(state, now):
    """Credit the time since the last transition to the phase currently on
    top of the stack (its self-time)."""
    top = state["stack"][-1] if state["stack"] else None
    if top in state["wall"] and state["last_ts"] is not None:
        state["wall"][top] += now - state["last_ts"]


class op_scope:
    """Context manager marking a region as running in ``phase``.

    Sets the scope ContextVar for the duration (so concurrent calls spawned
    inside inherit it) and accrues the region's wall-clock to ``phase``,
    excluding any nested scopes.
    """

    def __init__(self, phase):
        self.phase = phase
        self._token = None

    def __enter__(self):
        now = time.perf_counter()
        state = _wall_state()
        _credit(state, now)  # close out the parent's running interval first
        state["stack"].append(self.phase)
        state["last_ts"] = now
        self._token = _OP_SCOPE.set(self.phase)
        return self

    def __exit__(self, *exc_info):
        now = time.perf_counter()
        state = _wall_state()
        _credit(state, now)  # credit this phase's self-time
        if state["stack"]:
            state["stack"].pop()
        state["last_ts"] = now
        _OP_SCOPE.reset(self._token)
        return False


def _add_phase_wall_clock_s(phase, seconds):
    """Test helper: add to a phase's wall-clock accumulator without timing a
    real region."""
    _wall_state()["wall"][phase] = _wall_state()["wall"].get(phase, 0.0) + seconds
