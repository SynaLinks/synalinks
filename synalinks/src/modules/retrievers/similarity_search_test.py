# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""End-to-end tests for `SimilaritySearch` against a real DuckDB store.

The DuckDB knowledge base — including its HNSW vector index — is real.
Only the external APIs are mocked: ``litellm.aembedding`` yields
deterministic keyword-based vectors so nearest-neighbour ranking is
verifiable, and ``litellm.acompletion`` returns the search query.
"""

import os
import tempfile
from typing import List
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.knowledge_bases import KnowledgeBase
from synalinks.src.modules import Input
from synalinks.src.modules.embedding_models import EmbeddingModel
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.modules.retrievers.similarity_search import SimilaritySearch
from synalinks.src.programs import Program


class Document(DataModel):
    id: str = Field(description="The document id")
    text: str = Field(description="The document content")
    # DuckDB stores a pre-computed embedding column (upstream EmbedKnowledge
    # step); the adapter does not embed on insert.
    embedding: List[float] = Field(default=[], description="The embedding vector")


class Query(DataModel):
    question: str = Field(description="The user question")


def _keyword_vector(text):
    text = (text or "").lower()
    if "fox" in text:
        return [1.0, 0.0, 0.0]
    if "dog" in text:
        return [0.0, 1.0, 0.0]
    if "bird" in text:
        return [0.0, 0.0, 1.0]
    return [0.33, 0.33, 0.33]


def _fake_aembedding(*args, **kwargs):
    texts = kwargs.get("input")
    if isinstance(texts, str):
        texts = [texts]
    return {"data": [{"embedding": _keyword_vector(t)} for t in texts]}


def _lm_returns(content):
    def _fake(*args, **kwargs):
        return {"choices": [{"message": {"content": content}}]}

    return _fake


class SimilaritySearchTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = "duckdb://" + os.path.join(self.temp_dir, "sim.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def _kb(self):
        kb = KnowledgeBase(
            uri=self.db_path,
            data_models=[Document],
            embedding_model=EmbeddingModel(model="gemini/text-embedding-004"),
        )
        await kb.update(
            [
                JsonDataModel(
                    data_model=Document(
                        id="d1", text="The brown fox runs", embedding=[1.0, 0.0, 0.0]
                    )
                ),
                JsonDataModel(
                    data_model=Document(
                        id="d2", text="A lazy dog sleeps", embedding=[0.0, 1.0, 0.0]
                    )
                ),
                JsonDataModel(
                    data_model=Document(
                        id="d3", text="A small bird sings", embedding=[0.0, 0.0, 1.0]
                    )
                ),
            ]
        )
        return kb

    async def test_nearest_neighbour_ranks_first(self):
        with patch("litellm.aembedding", side_effect=_fake_aembedding):
            kb = await self._kb()
            lm = LanguageModel(model="ollama/mistral")
            with patch(
                "litellm.acompletion",
                side_effect=_lm_returns('{"similarity_search": ["fox"]}'),
            ):
                inputs = Input(data_model=Query)
                outputs = await SimilaritySearch(
                    knowledge_base=kb,
                    language_model=lm,
                    data_model=Document,
                    k=3,
                    name="sim",
                )(inputs)
                result = await Program(inputs=inputs, outputs=outputs)(
                    Query(question="something about a fox")
                )
        rows = result.get("result")
        self.assertEqual(rows[0]["id"], "d1")  # fox doc is nearest

    async def test_respects_k(self):
        with patch("litellm.aembedding", side_effect=_fake_aembedding):
            kb = await self._kb()
            lm = LanguageModel(model="ollama/mistral")
            with patch(
                "litellm.acompletion",
                side_effect=_lm_returns('{"similarity_search": ["dog"]}'),
            ):
                inputs = Input(data_model=Query)
                outputs = await SimilaritySearch(
                    knowledge_base=kb,
                    language_model=lm,
                    data_model=Document,
                    k=1,
                    name="sim",
                )(inputs)
                result = await Program(inputs=inputs, outputs=outputs)(
                    Query(question="dogs")
                )
        rows = result.get("result")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "d2")

    async def test_return_inputs_and_query_threaded(self):
        with patch("litellm.aembedding", side_effect=_fake_aembedding):
            kb = await self._kb()
            lm = LanguageModel(model="ollama/mistral")
            with patch(
                "litellm.acompletion",
                side_effect=_lm_returns('{"similarity_search": ["bird"]}'),
            ):
                inputs = Input(data_model=Query)
                outputs = await SimilaritySearch(
                    knowledge_base=kb,
                    language_model=lm,
                    data_model=Document,
                    return_inputs=True,
                    return_query=True,
                    name="sim",
                )(inputs)
                result = await Program(inputs=inputs, outputs=outputs)(
                    Query(question="birds")
                )
        json = result.get_json()
        self.assertIn("result", json)
        self.assertIn("question", json)
        self.assertIn("similarity_search", json)

    async def test_none_input_returns_none(self):
        with patch("litellm.aembedding", side_effect=_fake_aembedding):
            kb = await self._kb()
            lm = LanguageModel(model="ollama/mistral")
            mod = SimilaritySearch(
                knowledge_base=kb, language_model=lm, data_model=Document, name="sim"
            )
            self.assertIsNone(await mod(None))

    async def test_infers_table_name_when_not_provided(self):
        # With no table_name/schema/data_model, the LM infers the table from an
        # enum of the KB's actual tables, concatenated onto the query schema.
        with patch("litellm.aembedding", side_effect=_fake_aembedding):
            kb = await self._kb()
            lm = LanguageModel(model="ollama/mistral")
            mod = SimilaritySearch(knowledge_base=kb, language_model=lm, k=3, name="sim")
            self.assertIsNone(mod.table_name)
            # the query schema gained a table_name enum of the KB's tables
            gen_schema = mod.query_generator.schema
            self.assertIn("table_name", gen_schema["properties"])
            self.assertEqual(gen_schema["properties"]["table_name"]["enum"], ["Document"])
            with patch(
                "litellm.acompletion",
                side_effect=_lm_returns(
                    '{"similarity_search": ["fox"], "table_name": "Document"}'
                ),
            ):
                inputs = Input(data_model=Query)
                outputs = await mod(inputs)
                result = await Program(inputs=inputs, outputs=outputs)(
                    Query(question="something about a fox")
                )
        rows = result.get("result")
        self.assertEqual(rows[0]["id"], "d1")
