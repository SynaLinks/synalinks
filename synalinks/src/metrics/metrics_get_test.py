# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import metrics
from synalinks.src import testing


class MetricsGetTest(testing.TestCase):
    def test_default_names_and_aliases_resolve(self):
        for identifier, expected in (
            ("fbeta_score", metrics.FBetaScore),
            ("binary_fbeta_score", metrics.BinaryFBetaScore),
            ("list_f1_score", metrics.CategoricalF1Score),
            ("list_fbeta_score", metrics.CategoricalFBetaScore),
            ("f1_score", metrics.F1Score),
            ("FBetaScore", metrics.FBetaScore),
        ):
            self.assertIsInstance(metrics.get(identifier), expected, identifier)
