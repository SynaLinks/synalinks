# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import patch
from unittest.mock import sentinel

import litellm

from synalinks.src import testing
from synalinks.src.backend import Embeddings
from synalinks.src.embedding_models.embedding_model import EmbeddingModel


class EmbeddingModelTest(testing.TestCase):
    @patch("httpx.AsyncClient")
    def test_ollama_model_refreshes_litellm_async_client(self, mock_async_client):
        previous_client = getattr(litellm, "module_level_aclient", None)
        self.addCleanup(setattr, litellm, "module_level_aclient", previous_client)
        mock_async_client.return_value = sentinel.client

        EmbeddingModel(model="ollama/nomic-embed-text")

        self.assertIs(litellm.module_level_aclient, sentinel.client)
        mock_async_client.assert_called_once()
        limits = mock_async_client.call_args.kwargs["limits"]
        self.assertEqual(limits.max_keepalive_connections, 0)
        self.assertEqual(limits.max_connections, 100)

    @patch("httpx.AsyncClient")
    def test_non_ollama_model_keeps_existing_async_client(self, mock_async_client):
        previous_client = getattr(litellm, "module_level_aclient", None)
        self.addCleanup(setattr, litellm, "module_level_aclient", previous_client)
        litellm.module_level_aclient = sentinel.previous_client

        EmbeddingModel(model="gemini/text-embedding-004")

        self.assertIs(litellm.module_level_aclient, sentinel.previous_client)
        mock_async_client.assert_not_called()

    @patch("litellm.aembedding")
    async def test_call_api(self, mock_embedding):
        embedding_model = EmbeddingModel(model="ollama/all-minilm")

        expected_value = [0.0, 0.1, 0.2, 0.3]
        mock_embedding.return_value = {"data": [{"embedding": expected_value}]}

        result = await embedding_model(["What is the capital of France?"])
        self.assertEqual(result, Embeddings(**result).get_json())
        self.assertEqual(result, {"embeddings": [expected_value]})

    @patch("litellm.aembedding")
    async def test_retry_succeeds_on_second_attempt(self, mock_embedding):
        embedding_model = EmbeddingModel(
            model="ollama/all-minilm", retry=3
        )

        expected_value = [0.0, 0.1, 0.2, 0.3]
        mock_embedding.side_effect = [
            Exception("Temporary failure"),
            {"data": [{"embedding": expected_value}]},
        ]

        result = await embedding_model(
            ["What is the capital of France?"]
        )
        self.assertEqual(result, {"embeddings": [expected_value]})
        self.assertEqual(mock_embedding.call_count, 2)

    @patch("litellm.aembedding")
    async def test_retry_exhausted_returns_none(self, mock_embedding):
        embedding_model = EmbeddingModel(
            model="ollama/all-minilm", retry=2
        )

        mock_embedding.side_effect = Exception("Persistent failure")

        result = await embedding_model(
            ["What is the capital of France?"]
        )
        self.assertIsNone(result)
        self.assertEqual(mock_embedding.call_count, 2)

    @patch("litellm.aembedding")
    async def test_retry_exhausted_uses_fallback(self, mock_embedding):
        fallback_model = EmbeddingModel(
            model="ollama/mxbai-embed-large"
        )
        embedding_model = EmbeddingModel(
            model="ollama/all-minilm",
            retry=2,
            fallback=fallback_model,
        )

        expected_value = [0.5, 0.6, 0.7, 0.8]
        # First 2 calls fail (primary), third succeeds (fallback)
        mock_embedding.side_effect = [
            Exception("Fail 1"),
            Exception("Fail 2"),
            {"data": [{"embedding": expected_value}]},
        ]

        result = await embedding_model(
            ["What is the capital of France?"]
        )
        self.assertEqual(result, {"embeddings": [expected_value]})
        self.assertEqual(mock_embedding.call_count, 3)
