# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.modules.embedding_models import EmbeddingModel
from synalinks.src.rewards.cosine_similarity import CosineSimilarity


class CosineSimilarityTest(testing.TestCase):
    @patch("litellm.aembedding")
    async def test_base_function(self, mock_embedding):
        embedding_model = EmbeddingModel(model="ollama/all-minilm")
        expected_value = [0.0, 0.1, 0.2, 0.3, 0.4]
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        class Answer(DataModel):
            answer: str

        y_true = Answer(answer="Paris")
        y_pred = Answer(answer="Paris")

        cosine_similarity = CosineSimilarity(embedding_model=embedding_model)
        reward = await cosine_similarity(y_true, y_pred)
        self.assertEqual(reward, 1.0)

    @patch("litellm.aembedding")
    async def test_multiple_fields(self, mock_embedding):
        embedding_model = EmbeddingModel(model="ollama/all-minilm")
        expected_value = [0.0, 0.1, 0.2, 0.3, 0.4]
        mock_embedding.return_value = {
            "data": [{"embedding": expected_value}, {"embedding": expected_value}]
        }

        class Answer(DataModel):
            thinking: str
            answer: str

        y_true = Answer(thinking="The capital of France is paris", answer="Paris")
        y_pred = Answer(thinking="The capital of France is paris", answer="Paris")

        cosine_similarity = CosineSimilarity(embedding_model=embedding_model)
        reward = await cosine_similarity(y_true, y_pred)
        self.assertEqual(reward, 1.0)
