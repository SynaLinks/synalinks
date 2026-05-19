# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.metrics.reduction_metrics import Mean
from synalinks.src.rewards.exact_match import ExactMatch
from synalinks.src.trainers.compile_utils import CompileMetrics
from synalinks.src.trainers.compile_utils import CompileReward
from synalinks.src.trainers.compile_utils import MetricsList
from synalinks.src.trainers.compile_utils import is_function_like


class Answer(DataModel):
    answer: str


class IsFunctionLikeTest(testing.TestCase):
    def test_none_str_callable_pass(self):
        self.assertTrue(is_function_like(None))
        self.assertTrue(is_function_like("mean"))
        self.assertTrue(is_function_like(lambda x: x))

    def test_other_objects_fail(self):
        self.assertFalse(is_function_like(42))
        self.assertFalse(is_function_like({"key": "value"}))


class CompileMetricsValidationTest(testing.TestCase):
    """Cover the error paths in `CompileMetrics.__init__` /
    `_build_metrics_set` that fire before any update_state call."""

    def test_rejects_non_collection_metrics(self):
        with self.assertRaisesRegex(ValueError, "list, tuple, or dict"):
            CompileMetrics(metrics=42)

    def test_unbuilt_variables_returns_empty(self):
        cm = CompileMetrics(metrics=[Mean(name="m")])
        # Before `build()`, variables is the empty list (early-exit branch).
        self.assertEqual(cm.variables, [])
        self.assertEqual(cm.metrics, [])

    def test_unbuilt_reset_state_is_no_op(self):
        cm = CompileMetrics(metrics=[Mean(name="m")])
        # Should not raise even though _flat_metrics doesn't exist yet.
        cm.reset_state()

    def test_result_before_build_raises(self):
        cm = CompileMetrics(metrics=[Mean(name="m")])
        with self.assertRaisesRegex(ValueError, "has not yet been built"):
            cm.result()

    def test_metrics_list_get_config_not_implemented(self):
        ml = MetricsList([Mean(name="m")])
        with self.assertRaises(NotImplementedError):
            ml.get_config()
        with self.assertRaises(NotImplementedError):
            MetricsList.from_config({})

    def test_compile_metrics_get_config_not_implemented(self):
        cm = CompileMetrics(metrics=[Mean(name="m")])
        with self.assertRaises(NotImplementedError):
            cm.get_config()
        with self.assertRaises(NotImplementedError):
            CompileMetrics.from_config({})


class CompileMetricsBuildErrorPathsTest(testing.TestCase):
    """`_build_metrics_set` rejects malformed metric configs at build time."""

    def _build(self, **kw):
        # Build is sync; the y_true/y_pred shape is what triggers the
        # multi-output branches we want to exercise.
        cm = CompileMetrics(**kw)
        return cm

    def test_dict_metrics_with_unknown_output_name_raises(self):
        cm = self._build(
            metrics={"only_a": [Mean(name="m")]},
            output_names=["a", "b"],
        )
        with self.assertRaisesRegex(ValueError, "does not correspond to any"):
            cm.build(y_true=[None, None], y_pred=[None, None])

    def test_list_metrics_wrong_length_raises(self):
        cm = self._build(metrics=[[Mean(name="m1")], [Mean(name="m2")]])
        # 2 metric sublists but 3 outputs → mismatch raises.
        with self.assertRaisesRegex(ValueError, "many entries as the program has"):
            cm.build(y_true=[None, None, None], y_pred=[None, None, None])

    def test_list_metrics_with_non_metric_entry_raises(self):
        cm = self._build(metrics=[[Mean(name="m1")], [42]])
        with self.assertRaisesRegex(ValueError, "should be metric objects"):
            cm.build(y_true=[None, None], y_pred=[None, None])

    def test_dict_metrics_with_non_metric_entry_raises(self):
        cm = self._build(
            metrics={"a": [42]},
            output_names=["a", "b"],
        )
        with self.assertRaisesRegex(ValueError, "should be metric objects"):
            cm.build(y_true=[None, None], y_pred=[None, None])

    def test_dict_metrics_without_output_names_raises(self):
        # Dict-shaped metrics need output names to match against; with
        # `output_names=None` the helper must raise with the intended
        # ValueError (not a TypeError on `None in None`).
        cm = self._build(metrics={"a": Mean(name="m")})
        with self.assertRaisesRegex(ValueError, "can only be provided as a dict"):
            cm.build(y_true=[None, None], y_pred=[None, None])

    def test_single_output_list_of_non_metrics_raises(self):
        cm = self._build(metrics=[42])
        with self.assertRaisesRegex(ValueError, "to be metric objects"):
            cm.build(y_true=None, y_pred=None)


class CompileMetricsResultDedupeTest(testing.TestCase):
    """`result()` deduplicates colliding metric names across outputs."""

    async def test_same_metric_name_across_outputs_gets_indexed(self):
        cm = CompileMetrics(
            metrics=[[Mean(name="reward")], [Mean(name="reward")]],
            output_names=["a", "b"],
        )
        cm.build(y_true=[1.0, 2.0], y_pred=[1.0, 2.0])
        # First output's MetricsList has no `output_name` because we didn't
        # set it; both metrics share the bare name "reward" → second one is
        # renamed "reward_1".
        # Note: with output_names provided, both get output-prefixed names
        # like "a_reward" / "b_reward" — distinct, so no dedupe triggers.
        # Force collision by stripping output_name from one MetricsList.
        cm._flat_metrics[1].output_name = None
        cm._flat_metrics[0].output_name = None
        results = cm.result()
        self.assertIn("reward", results)
        self.assertIn("reward_1", results)


class CompileRewardValidationTest(testing.TestCase):
    """Reward-side mirror of CompileMetricsValidationTest."""

    def test_invalid_reward_weights_type_raises(self):
        with self.assertRaisesRegex(ValueError, "Expected `reward_weights`"):
            CompileReward(reward=ExactMatch(), reward_weights="not-a-weight")

    def test_metrics_property_default_empty(self):
        cr = CompileReward(reward=ExactMatch())
        # Before build, _metrics is the empty list registered with Tracker.
        self.assertEqual(cr.metrics, [])
        self.assertEqual(cr.variables, [])

    async def test_has_batch_rewards_false_before_build(self):
        cr = CompileReward(reward=ExactMatch())
        self.assertFalse(cr.has_batch_rewards)


class CompileRewardMultiOutputTest(testing.TestCase):
    """Cover the multi-output (nested) path of ``CompileReward.call``."""

    async def test_multi_output_dict_call_awaits_each_reward(self):
        compile_reward = CompileReward(
            reward={"a": ExactMatch(), "b": ExactMatch()},
            output_names=["a", "b"],
        )
        y_true = {"a": Answer(answer="x"), "b": Answer(answer="y")}
        y_pred = {"a": Answer(answer="x"), "b": Answer(answer="y")}

        result = await compile_reward(y_true, y_pred)
        # Both outputs match → 1.0 + 1.0 = 2.0.
        self.assertEqual(float(result), 2.0)

    async def test_multi_output_partial_match(self):
        compile_reward = CompileReward(
            reward={"a": ExactMatch(), "b": ExactMatch()},
            output_names=["a", "b"],
        )
        y_true = {"a": Answer(answer="x"), "b": Answer(answer="y")}
        y_pred = {"a": Answer(answer="x"), "b": Answer(answer="WRONG")}

        result = await compile_reward(y_true, y_pred)
        self.assertEqual(float(result), 1.0)
