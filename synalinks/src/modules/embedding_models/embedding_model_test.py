# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from litellm.types.utils import Embedding
from litellm.types.utils import EmbeddingResponse
from litellm.types.utils import PromptTokensDetailsWrapper
from litellm.types.utils import Usage

from synalinks.src import testing
from synalinks.src.backend import Embeddings
from synalinks.src.backend.common import global_state
from synalinks.src.modules.embedding_models.embedding_model import EmbeddingModel


class EmbeddingModelTest(testing.TestCase):
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
        embedding_model = EmbeddingModel(model="ollama/all-minilm", retry=3)

        expected_value = [0.0, 0.1, 0.2, 0.3]
        mock_embedding.side_effect = [
            Exception("Temporary failure"),
            {"data": [{"embedding": expected_value}]},
        ]

        result = await embedding_model(["What is the capital of France?"])
        self.assertEqual(result, {"embeddings": [expected_value]})
        self.assertEqual(mock_embedding.call_count, 2)

    @patch("litellm.aembedding")
    async def test_retry_exhausted_returns_none(self, mock_embedding):
        embedding_model = EmbeddingModel(model="ollama/all-minilm", retry=2)

        mock_embedding.side_effect = Exception("Persistent failure")

        result = await embedding_model(["What is the capital of France?"])
        self.assertIsNone(result)
        self.assertEqual(mock_embedding.call_count, 2)

    @patch("litellm.aembedding")
    async def test_retry_exhausted_uses_fallback(self, mock_embedding):
        fallback_model = EmbeddingModel(model="ollama/mxbai-embed-large")
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

        result = await embedding_model(["What is the capital of France?"])
        self.assertEqual(result, {"embeddings": [expected_value]})
        self.assertEqual(mock_embedding.call_count, 3)


def _em_response(prompt_tokens, n_vectors, cost=None):
    resp = EmbeddingResponse(
        model="text-embedding-3-small",
        data=[
            Embedding(embedding=[0.1, 0.2], index=i, object="embedding")
            for i in range(n_vectors)
        ],
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )
    if cost is not None:
        resp._hidden_params = {"response_cost": cost}
    return resp


def _set_scope(value):
    global_state.set_global_attribute("synalinks_op_scope", value)


class EMCounterPopulationTest(testing.TestCase):
    """End-to-end checks that LiteLLM-shaped responses populate operational
    counters correctly. Guards against drift in LiteLLM's response schema —
    in particular `response["usage"]["prompt_tokens"]`,
    `response._hidden_params["response_cost"]`, and `response["data"]`.
    """

    @patch("litellm.aembedding")
    async def test_em_populates_inference_counters(self, mock_embedding):
        mock_embedding.return_value = _em_response(
            prompt_tokens=24, n_vectors=3, cost=0.00007
        )
        em = EmbeddingModel(model="openai/text-embedding-3-small")
        _set_scope("inference")
        try:
            await em(["a", "b", "c"])
        finally:
            _set_scope(None)
        self.assertEqual(em.cumulated_calls, 1)
        self.assertEqual(em.cumulated_prompt_tokens, 24)
        self.assertEqual(em.cumulated_tokens, 24)
        self.assertEqual(em.cumulated_vectors, 3)
        self.assertAlmostEqual(em.cumulated_cost, 0.00007)
        self.assertEqual(em.inference_cumulated_calls, 1)
        self.assertEqual(em.inference_cumulated_prompt_tokens, 24)
        self.assertEqual(em.inference_cumulated_vectors, 3)
        self.assertAlmostEqual(em.inference_cumulated_cost, 0.00007)
        self.assertEqual(em.reward_cumulated_calls, 0)
        self.assertEqual(em.optimizer_cumulated_calls, 0)
        self.assertEqual(em.last_call_vectors, 3)
        self.assertGreater(em.last_call_elapsed_s, 0.0)

    @patch("litellm.aembedding")
    async def test_em_routes_each_scope(self, mock_embedding):
        mock_embedding.return_value = _em_response(
            prompt_tokens=10, n_vectors=2, cost=0.0001
        )
        em = EmbeddingModel(model="openai/text-embedding-3-small")
        for scope, expected_phase in (
            ("inference", "inference"),
            ("reward", "reward"),
            ("optimizer", "optimizer"),
        ):
            _set_scope(scope)
            try:
                await em(["x", "y"])
            finally:
                _set_scope(None)
            self.assertEqual(
                getattr(em, f"{expected_phase}_cumulated_vectors"),
                2,
                f"scope={scope} did not bump {expected_phase}_cumulated_vectors",
            )
        self.assertEqual(em.cumulated_calls, 3)
        self.assertEqual(em.inference_cumulated_calls, 1)
        self.assertEqual(em.reward_cumulated_calls, 1)
        self.assertEqual(em.optimizer_cumulated_calls, 1)

    @patch("litellm.aembedding")
    async def test_em_cached_tokens(self, mock_embedding):
        """OpenAI batch-embedding caches; cached_tokens populates."""
        resp = _em_response(prompt_tokens=200, n_vectors=4, cost=0.0001)
        resp.usage.prompt_tokens_details = PromptTokensDetailsWrapper(cached_tokens=150)
        mock_embedding.return_value = resp
        em = EmbeddingModel(model="openai/text-embedding-3-small")
        _set_scope("inference")
        try:
            await em(["a", "b", "c", "d"])
        finally:
            _set_scope(None)
        self.assertEqual(em.cumulated_cached_tokens, 150)
        self.assertEqual(em.inference_cumulated_cached_tokens, 150)
        self.assertEqual(em.reward_cumulated_cached_tokens, 0)

    @patch("litellm.aembedding")
    async def test_em_missing_usage(self, mock_embedding):
        """Some providers may return data without usage. Vectors still
        count (we infer from data length); tokens stay 0.
        """
        resp = EmbeddingResponse(
            model="ollama/mxbai-embed-large",
            data=[
                Embedding(embedding=[0.1], index=0, object="embedding"),
                Embedding(embedding=[0.2], index=1, object="embedding"),
            ],
            usage=None,
        )
        mock_embedding.return_value = resp
        em = EmbeddingModel(model="ollama/mxbai-embed-large")
        _set_scope("inference")
        try:
            await em(["a", "b"])
        finally:
            _set_scope(None)
        self.assertEqual(em.cumulated_calls, 1)
        self.assertEqual(em.cumulated_prompt_tokens, 0)
        self.assertEqual(em.cumulated_tokens, 0)
        self.assertEqual(em.cumulated_vectors, 2)
        self.assertEqual(em.inference_cumulated_vectors, 2)
