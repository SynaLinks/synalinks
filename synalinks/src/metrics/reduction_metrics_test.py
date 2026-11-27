# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.metrics.reduction_metrics import Sum


class SumTest(testing.TestCase):
    async def test_sum_accumulates_lists_and_scalars(self):
        metric = Sum()
        self.assertEqual(await metric([1, 3, 5, 7]), 16.0)
        self.assertEqual(await metric(2.0), 18.0)
        metric.reset_state()
        self.assertEqual(metric.result(), 0.0)
