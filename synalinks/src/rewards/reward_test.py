# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src import tree
from synalinks.src.backend import DataModel
from synalinks.src.rewards.reward import apply_masks
from synalinks.src.rewards.reward import reduce_rewards
from synalinks.src.rewards.reward import reduce_values


class ReduceValuesTest(testing.TestCase):
    def test_scalar_mean_passthrough(self):
        self.assertEqual(reduce_values(0.7, "mean"), 0.7)

    def test_scalar_sum_passthrough(self):
        self.assertEqual(reduce_values(0.7, "sum"), 0.7)

    def test_scalar_none_passthrough(self):
        self.assertEqual(reduce_values(0.7, "none"), 0.7)

    def test_list_mean(self):
        self.assertAlmostEqual(reduce_values([0.0, 1.0], "mean"), 0.5)

    def test_list_sum(self):
        self.assertAlmostEqual(reduce_values([0.5, 0.5, 0.5], "sum"), 1.5)

    def test_list_min(self):
        self.assertAlmostEqual(reduce_values([0.2, 0.9, 0.5], "min"), 0.2)

    def test_list_max(self):
        self.assertAlmostEqual(reduce_values([0.2, 0.9, 0.5], "max"), 0.9)

    def test_scalar_min_passthrough(self):
        self.assertEqual(reduce_values(0.7, "min"), 0.7)

    def test_scalar_max_passthrough(self):
        self.assertEqual(reduce_values(0.7, "max"), 0.7)

    def test_list_none_returns_unreduced(self):
        # Regression: previously raised TypeError because float([...]) is
        # not valid. "none" semantics: preserve the per-sample values.
        result = reduce_values([0.0, 1.0, 0.5], "none")
        self.assertEqual(list(result), [0.0, 1.0, 0.5])

    def test_list_none_via_python_none(self):
        result = reduce_values([0.0, 1.0], None)
        self.assertEqual(list(result), [0.0, 1.0])

    def test_empty_returns_zero_for_every_reduction(self):
        # An empty batch has no signal: every reduction must collapse to a
        # scalar 0.0. Regression: "none"/None returned the empty list and
        # min/max raised "zero-size array to reduction operation", both of
        # which break scalar consumers (e.g. a tuner objective expecting a
        # float, not a list).
        for reduction in ("mean", "sum", "min", "max", "none", None):
            result = reduce_values([], reduction)
            self.assertIsInstance(result, float)
            self.assertEqual(result, 0.0)


class ApplyMasksTest(testing.TestCase):
    def test_skips_none_leaves(self):
        # A batch can carry None leaves (e.g. a sample with no gold for a
        # judge-only reward). Masking must skip them instead of calling
        # ``None.in_mask`` / ``None.out_mask`` and raising AttributeError.
        class Answer(DataModel):
            answer: str
            extra: str = ""

        y_true = [Answer(answer="a", extra="x"), None]
        y_pred = [Answer(answer="a", extra="y"), None]

        masked_true, masked_pred = apply_masks(y_true, y_pred, in_mask=["answer"])
        to_json = lambda x: None if x is None else x.get_json()
        self.assertEqual(
            tree.map_structure(to_json, masked_pred), [{"answer": "a"}, None]
        )
        self.assertEqual(
            tree.map_structure(to_json, masked_true), [{"answer": "a"}, None]
        )

    def test_skips_none_leaves_out_mask(self):
        class Answer(DataModel):
            answer: str
            extra: str = ""

        y_pred = [Answer(answer="a", extra="y"), None]
        _, masked_pred = apply_masks(None, y_pred, out_mask=["extra"])
        to_json = lambda x: None if x is None else x.get_json()
        self.assertEqual(
            tree.map_structure(to_json, masked_pred), [{"answer": "a"}, None]
        )


class ReduceRewardsTest(testing.TestCase):
    def test_empty_returns_zero(self):
        self.assertEqual(reduce_rewards([], "mean"), 0.0)

    def test_mean(self):
        self.assertAlmostEqual(reduce_rewards([0.0, 1.0], "mean"), 0.5)

    def test_sum(self):
        self.assertAlmostEqual(reduce_rewards([0.5, 0.5], "sum"), 1.0)

    def test_min(self):
        # Worst-case reduction: smallest per-sample reward in the batch.
        self.assertAlmostEqual(reduce_rewards([0.2, 0.9, 0.5], "min"), 0.2)

    def test_max(self):
        # Best-case reduction: largest per-sample reward in the batch.
        self.assertAlmostEqual(reduce_rewards([0.2, 0.9, 0.5], "max"), 0.9)

    def test_none_falls_back_to_mean(self):
        # Trackers/scoring need a scalar; "none" has no sensible scalar
        # passthrough so it falls back to mean.
        self.assertAlmostEqual(reduce_rewards([0.0, 1.0], "none"), 0.5)

    def test_python_none_falls_back_to_mean(self):
        self.assertAlmostEqual(reduce_rewards([0.0, 1.0], None), 0.5)
