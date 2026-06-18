# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""End-to-end tests for `HybridFTSSearch` against a real DuckDB store.

Exercises the real vector (HNSW) + BM25 full-text fusion path. Docs
carry pre-computed embeddings (DuckDB does not embed on insert); only
``litellm.aembedding`` (query vector) and ``litellm.acompletion``
(query generation) are mocked.
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
from synalinks.src.modules.retrievers.hybrid_fts_search import HybridFTSSearch
from synalinks.src.programs import Program


class Document(DataModel):
    id: str = Field(description="The document id")
    text: str = Field(description="The document content")
    embedding: List[float] = Field(default=[], description="The embedding vector")


class Query(DataModel):
    question: str = Field(description="The user question")


def _keyword_vector(text):
    text = (text or "").lower()
    if "fox" in text:
        return [1.0, 0.0, 0.0]
    if "dog" in text:
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


def _fake_aembedding(*args, **kwargs):
    texts = kwargs.get("input")
    if isinstance(texts, str):
        texts = [texts]
    return {"data": [{"embedding": _keyword_vector(t)} for t in texts]}


def _lm_returns(content):
    def _fake(*args, **kwargs):
        return {"choices": [{"message": {"content": content}}]}

    return _fake


class HybridFTSSearchTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = "duckdb://" + os.path.join(self.temp_dir, "hyb.db")

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
                        id="d1", text="The brown fox runs fast", embedding=[1.0, 0.0, 0.0]
                    )
                ),
                JsonDataModel(
                    data_model=Document(
                        id="d2", text="A lazy dog sleeps", embedding=[0.0, 1.0, 0.0]
                    )
                ),
                JsonDataModel(
                    data_model=Document(
                        id="d3", text="The fox and the hound", embedding=[1.0, 0.0, 0.0]
                    )
                ),
            ]
        )
        return kb

    async def test_fuses_vector_and_fulltext(self):
        with patch("litellm.aembedding", side_effect=_fake_aembedding):
            kb = await self._kb()
            lm = LanguageModel(model="ollama/mistral")
            with patch(
                "litellm.acompletion",
                side_effect=_lm_returns(
                    '{"similarity_search": ["fox"], "keywords": ["fox"]}'
                ),
            ):
                inputs = Input(data_model=Query)
                outputs = await HybridFTSSearch(
                    knowledge_base=kb,
                    language_model=lm,
                    data_model=Document,
                    k=5,
                    name="hyb",
                )(inputs)
                result = await Program(inputs=inputs, outputs=outputs)(
                    Query(question="tell me about the fox")
                )
        ids = {r["id"] for r in result.get("result")}
        # both fox docs surface via vector and/or BM25; dog doc ranks last/out
        self.assertIn("d1", ids)
        self.assertIn("d3", ids)

    async def test_return_inputs_and_query_threaded(self):
        with patch("litellm.aembedding", side_effect=_fake_aembedding):
            kb = await self._kb()
            lm = LanguageModel(model="ollama/mistral")
            with patch(
                "litellm.acompletion",
                side_effect=_lm_returns(
                    '{"similarity_search": ["fox"], "keywords": ["fox"]}'
                ),
            ):
                inputs = Input(data_model=Query)
                outputs = await HybridFTSSearch(
                    knowledge_base=kb,
                    language_model=lm,
                    data_model=Document,
                    return_inputs=True,
                    return_query=True,
                    name="hyb",
                )(inputs)
                result = await Program(inputs=inputs, outputs=outputs)(
                    Query(question="fox?")
                )
        json = result.get_json()
        self.assertIn("result", json)
        self.assertIn("question", json)
        self.assertIn("similarity_search", json)

    async def test_none_input_returns_none(self):
        with patch("litellm.aembedding", side_effect=_fake_aembedding):
            kb = await self._kb()
            lm = LanguageModel(model="ollama/mistral")
            mod = HybridFTSSearch(
                knowledge_base=kb, language_model=lm, data_model=Document, name="hyb"
            )
            self.assertIsNone(await mod(None))

    async def test_infers_table_name_when_not_provided(self):
        # With no table_name/schema/data_model, the LM infers the table from an
        # enum of the KB's actual tables, concatenated onto the query schema.
        with patch("litellm.aembedding", side_effect=_fake_aembedding):
            kb = await self._kb()
            lm = LanguageModel(model="ollama/mistral")
            mod = HybridFTSSearch(knowledge_base=kb, language_model=lm, k=5, name="hyb")
            self.assertIsNone(mod.table_name)
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
                    Query(question="tell me about the fox")
                )
        ids = {r["id"] for r in result.get("result")}
        self.assertIn("d1", ids)
        self.assertIn("d3", ids)
