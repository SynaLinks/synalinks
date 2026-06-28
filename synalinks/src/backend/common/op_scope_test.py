# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend.common.op_scope import current_trajectory_start
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
