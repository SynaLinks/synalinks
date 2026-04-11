# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import MagicMock

from synalinks.src import testing
from synalinks.src.metrics.regression_metrics import CosineSimilarity


class CosineSimilarityTest(testing.TestCase):
    def test_init_defaults(self):
        metric = CosineSimilarity()
        self.assertEqual(metric.name, "cosine_similarity")
        self.assertEqual(metric.axis, -1)
        self.assertIsNone(metric.embedding_model)
        self.assertIsNone(metric.in_mask)
        self.assertIsNone(metric.out_mask)

    def test_init_custom_params(self):
        mock_model = MagicMock()
        metric = CosineSimilarity(
            name="my_cosine",
            axis=0,
            embedding_model=mock_model,
            in_mask=["field1"],
            out_mask=["field2"],
        )
        self.assertEqual(metric.name, "my_cosine")
        self.assertEqual(metric.axis, 0)
        self.assertEqual(metric.embedding_model, mock_model)
        self.assertEqual(metric.in_mask, ["field1"])
        self.assertEqual(metric.out_mask, ["field2"])

    def test_get_config(self):
        metric = CosineSimilarity(
            name="test_metric",
            axis=1,
            in_mask=["a"],
            out_mask=["b"],
        )
        config = metric.get_config()
        self.assertEqual(config["name"], "test_metric")
        self.assertEqual(config["axis"], 1)
        self.assertEqual(config["in_mask"], ["a"])
        self.assertEqual(config["out_mask"], ["b"])
        self.assertIn("embedding_model", config)

    def test_fn_kwargs(self):
        mock_model = MagicMock()
        metric = CosineSimilarity(axis=2, embedding_model=mock_model)
        self.assertEqual(metric._fn_kwargs["axis"], 2)
        self.assertEqual(metric._fn_kwargs["embedding_model"], mock_model)
