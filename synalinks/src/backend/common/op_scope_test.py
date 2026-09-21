# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
from unittest import mock

from synalinks.src import testing
from synalinks.src.backend.common import op_scope as op_scope_module
from synalinks.src.backend.common.op_scope import PhaseClock
from synalinks.src.backend.common.op_scope import current_op_scope
from synalinks.src.backend.common.op_scope import current_trajectory_start
from synalinks.src.backend.common.op_scope import op_scope
from synalinks.src.backend.common.op_scope import trajectory_scope


class TrajectoryScopeTest(testing.TestCase):
    def test_inactive_by_default(self):
        self.assertIsNone(current_trajectory_start())

    def test_sets_start_for_the_duration(self):
        with trajectory_scope():
            self.assertIsNotNone(current_trajectory_start())
        self.assertIsNone(current_trajectory_start())

    def test_set_once_outermost_wins(self):
        """A nested (sub-)agent inherits the outermost agent's start rather than
        resetting it, so whole-trajectory TTFT is measured from the top."""
        with trajectory_scope():
            outer = current_trajectory_start()
            self.assertIsNotNone(outer)
            with trajectory_scope():
                # Nested scope is a no-op: the outermost start is preserved.
                self.assertEqual(current_trajectory_start(), outer)
            # ... and still preserved after the nested scope exits.
            self.assertEqual(current_trajectory_start(), outer)
        # Only the outermost scope clears it.
        self.assertIsNone(current_trajectory_start())


class _FakeTime:
    """A controllable `time.perf_counter`."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class PhaseClockTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.time = _FakeTime()
        patcher = mock.patch.object(op_scope_module.time, "perf_counter", self.time)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_accrues_the_region(self):
        clock = PhaseClock()
        with op_scope("inference", clock=clock):
            self.assertEqual(current_op_scope(), "inference")
            self.time.now = 3.0
            # A read mid-region includes the running interval.
            self.assertEqual(clock.read("inference"), 3.0)
        self.time.now = 10.0
        self.assertEqual(clock.read("inference"), 3.0)
        self.assertIsNone(current_op_scope())

    def test_nested_scope_gets_self_time(self):
        clock = PhaseClock()
        with op_scope("optimizer", clock=clock):
            self.time.now = 1.0
            with op_scope("reward", clock=clock):
                self.time.now = 4.0
            self.time.now = 6.0
        self.assertEqual(clock.read("optimizer"), 3.0)
        self.assertEqual(clock.read("reward"), 3.0)

    def test_programs_on_one_loop_keep_their_own_clock(self):
        """Two programs evaluated concurrently on one event loop, with
        interleaved inference and reward phases, each measure only their own
        spans (the tuner-sweep case)."""
        clock_a, clock_b = PhaseClock(), PhaseClock()
        steps = [asyncio.Event() for _ in range(4)]

        async def advance(t, i):
            self.time.now = t
            steps[i].set()
            await asyncio.sleep(0)

        async def program_a():
            with op_scope("inference", clock=clock_a):  # t=0 -> 2
                await steps[0].wait()
            with op_scope("reward", clock=clock_a):  # t=2 -> 5
                await steps[2].wait()

        async def program_b():
            with op_scope("inference", clock=clock_b):  # t=0 -> 3
                await steps[1].wait()
            with op_scope("reward", clock=clock_b):  # t=3 -> 7
                await steps[3].wait()

        async def driver():
            await asyncio.sleep(0)
            await advance(2.0, 0)
            await advance(3.0, 1)
            await advance(5.0, 2)
            await advance(7.0, 3)

        async def main():
            await asyncio.gather(program_a(), program_b(), driver())

        asyncio.run(main())
        self.assertEqual(clock_a.read("inference"), 2.0)
        self.assertEqual(clock_a.read("reward"), 3.0)
        self.assertEqual(clock_b.read("inference"), 3.0)
        self.assertEqual(clock_b.read("reward"), 4.0)

    def test_concurrent_regions_of_one_phase_count_once(self):
        clock = PhaseClock()
        done = asyncio.Event()

        async def region():
            with op_scope("inference", clock=clock):
                await done.wait()

        async def main():
            tasks = [asyncio.ensure_future(region()) for _ in range(3)]
            await asyncio.sleep(0)
            self.time.now = 2.0
            done.set()
            await asyncio.gather(*tasks)

        asyncio.run(main())
        self.assertEqual(clock.read("inference"), 2.0)

    def test_other_programs_phase_keeps_running(self):
        """A program run inside another program's phase does not suspend the
        outer phase, which lives on a different clock."""
        outer, inner = PhaseClock(), PhaseClock()
        with op_scope("reward", clock=outer):
            self.time.now = 1.0
            with op_scope("inference", clock=inner):
                self.time.now = 3.0
            self.time.now = 4.0
        self.assertEqual(outer.read("reward"), 4.0)
        self.assertEqual(inner.read("inference"), 2.0)
