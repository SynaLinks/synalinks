# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import metrics
from synalinks.src import rewards
from synalinks.src import testing
from synalinks.src.metrics.reduction_metrics import MeanMetricWrapper
from synalinks.src.metrics.reduction_metrics import RewardTracker
from synalinks.src.metrics.regression_metrics import CosineSimilarity

_UP_OPERATIONAL = {
    prefix + name
    for prefix in ("", "Reward", "Optimizer")
    for name in (
        "Throughput",
        "TokensPerSecond",
        "CacheHitRate",
        "CachedTokens",
        "EmbeddingThroughput",
        "EmbeddingTokensPerSecond",
        "EmbeddingVectorsPerSecond",
        "EmbeddingCacheHitRate",
        "EmbeddingCachedTokens",
    )
} | {
    "AvgCachedTokensPerCall",
    "AvgRewardCachedTokensPerCall",
    "AvgOptimizerCachedTokensPerCall",
    "AvgEmbeddingCachedTokensPerCall",
    "AvgRewardEmbeddingCachedTokensPerCall",
    "AvgOptimizerEmbeddingCachedTokensPerCall",
    "ProgramCallsPerSecond",
}

_OPERATIONAL_BASES = (
    metrics.lm_metrics.LMOperationalMetric,
    metrics.em_metrics.EmbeddingModelOperationalMetric,
    metrics.program_metrics.ProgramOperationalMetric,
)


class MetricDirectionsTest(testing.TestCase):
    def test_every_metric_has_a_direction(self):
        for cls in metrics.ALL_OBJECTS:
            self.assertIn(cls.direction, ("up", "down"), cls.__name__)

    def test_operational_metric_directions(self):
        names = {c.__name__ for c in metrics.ALL_OBJECTS}
        self.assertTrue(_UP_OPERATIONAL <= names, _UP_OPERATIONAL - names)
        for cls in metrics.ALL_OBJECTS:
            if not issubclass(cls, _OPERATIONAL_BASES) or cls in _OPERATIONAL_BASES:
                continue
            expected = "up" if cls.__name__ in _UP_OPERATIONAL else "down"
            self.assertEqual(cls.direction, expected, cls.__name__)

    def test_quality_metrics_are_up(self):
        for cls in (
            metrics.Accuracy,
            metrics.F1Score,
            metrics.Precision,
            metrics.Recall,
            metrics.PassAtK,
            metrics.PassHatK,
            CosineSimilarity,
            RewardTracker,
        ):
            self.assertEqual(cls.direction, "up", cls.__name__)
        self.assertEqual(metrics.GapK.direction, "down")

    def test_reward_wrappers_are_up(self):
        for fn in (rewards.exact_match, rewards.cosine_similarity, rewards.ExactMatch()):
            self.assertEqual(MeanMetricWrapper(fn=fn).direction, "up")
