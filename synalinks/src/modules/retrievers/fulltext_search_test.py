# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""End-to-end tests for `FullTextSearch` against a real DuckDB store.

Only the language model (search-query generation) is mocked; the
DuckDB knowledge base — including BM25 full-text indexing — is real.
"""

import os
import tempfile
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.knowledge_bases import KnowledgeBase
from synalinks.src.modules import Input
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.modules.retrievers.fulltext_search import FullTextSearch
from synalinks.src.programs import Program


class Document(DataModel):
    id: str = Field(description="The document id")
    text: str = Field(description="The document content")


class Query(DataModel):
    question: str = Field(description="The user question")


def _lm_returns(content):
    def _fake(*args, **kwargs):
        return {"choices": [{"message": {"content": content}}]}

    return _fake


class FullTextSearchTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = "duckdb://" + os.path.join(self.temp_dir, "fts.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def _kb(self):
        kb = KnowledgeBase(uri=self.db_path, data_models=[Document])
        await kb.update(
            [
                JsonDataModel(
                    data_model=Document(id="d1", text="The quick brown fox jumps")
                ),
                JsonDataModel(data_model=Document(id="d2", text="A lazy dog sleeps")),
                JsonDataModel(
                    data_model=Document(id="d3", text="The fox outsmarts the hound")
                ),
            ]
        )
        return kb

    async def test_returns_matching_documents(self):
        kb = await self._kb()
        lm = LanguageModel(model="ollama/mistral")
        with patch(
            "litellm.acompletion",
            side_effect=_lm_returns('{"fulltext_search": ["fox"]}'),
        ):
            inputs = Input(data_model=Query)
            outputs = await FullTextSearch(
                knowledge_base=kb, language_model=lm, data_model=Document, k=5, name="fts"
            )(inputs)
            program = Program(inputs=inputs, outputs=outputs)
            result = await program(Query(question="tell me about the fox"))

        rows = result.get("result")
        ids = {r["id"] for r in rows}
        self.assertIn("d1", ids)
        self.assertIn("d3", ids)
        self.assertNotIn("d2", ids)  # no "fox" -> no BM25 match

    async def test_respects_k(self):
        kb = await self._kb()
        lm = LanguageModel(model="ollama/mistral")
        with patch(
            "litellm.acompletion",
            side_effect=_lm_returns('{"fulltext_search": ["the"]}'),
        ):
            inputs = Input(data_model=Query)
            outputs = await FullTextSearch(
                knowledge_base=kb, language_model=lm, data_model=Document, k=1, name="fts"
            )(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="anything")
            )
        self.assertLessEqual(len(result.get("result")), 1)

    async def test_return_inputs_and_query_threaded(self):
        kb = await self._kb()
        lm = LanguageModel(model="ollama/mistral")
        with patch(
            "litellm.acompletion",
            side_effect=_lm_returns('{"fulltext_search": ["fox"]}'),
        ):
            inputs = Input(data_model=Query)
            outputs = await FullTextSearch(
                knowledge_base=kb,
                language_model=lm,
                data_model=Document,
                return_inputs=True,
                return_query=True,
                name="fts",
            )(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="about the fox")
            )
        json = result.get_json()
        self.assertIn("result", json)
        self.assertIn("question", json)  # inputs threaded
        self.assertIn("fulltext_search", json)  # generated query threaded

    async def test_no_match_returns_empty(self):
        kb = await self._kb()
        lm = LanguageModel(model="ollama/mistral")
        with patch(
            "litellm.acompletion",
            side_effect=_lm_returns('{"fulltext_search": ["zzzznotpresent"]}'),
        ):
            inputs = Input(data_model=Query)
            outputs = await FullTextSearch(
                knowledge_base=kb, language_model=lm, data_model=Document, name="fts"
            )(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="nothing matches")
            )
        self.assertEqual(result.get("result"), [])

    async def test_none_input_returns_none(self):
        kb = await self._kb()
        lm = LanguageModel(model="ollama/mistral")
        mod = FullTextSearch(
            knowledge_base=kb, language_model=lm, data_model=Document, name="fts"
        )
        self.assertIsNone(await mod(None))
