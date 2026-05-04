# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.metrics.regression_metrics import CosineSimilarity
from synalinks.src.modules.embedding_models import EmbeddingModel


class CosineSimilarityTest(testing.TestCase):
    def test_init_defaults(self):
        metric = CosineSimilarity(embedding_model=EmbeddingModel(model="ollama/mxbai-embed-large"))
        self.assertEqual(metric.name, "cosine_similarity")
        self.assertEqual(metric.axis, -1)
        self.assertIsInstance(metric.embedding_model, EmbeddingModel)
        self.assertIsNone(metric.in_mask)
        self.assertIsNone(metric.out_mask)

    def test_init_custom_params(self):
        em = EmbeddingModel(model="ollama/mxbai-embed-large")
        metric = CosineSimilarity(
            name="my_cosine",
            axis=0,
            embedding_model=em,
            in_mask=["field1"],
            out_mask=["field2"],
        )
        self.assertEqual(metric.name, "my_cosine")
        self.assertEqual(metric.axis, 0)
        self.assertIs(metric.embedding_model, em)
        self.assertEqual(metric.in_mask, ["field1"])
        self.assertEqual(metric.out_mask, ["field2"])

    def test_get_config(self):
        metric = CosineSimilarity(
            name="test_metric",
            axis=1,
            embedding_model=EmbeddingModel(model="ollama/mxbai-embed-large"),
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
        em = EmbeddingModel(model="ollama/mxbai-embed-large")
        metric = CosineSimilarity(axis=2, embedding_model=em)
        self.assertEqual(metric._fn_kwargs["axis"], 2)
        self.assertIs(metric._fn_kwargs["embedding_model"], em)
