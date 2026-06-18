# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""End-to-end tests for `RegexSearch` against a real DuckDB store.

Only the language model (regex-pattern generation) is mocked; DuckDB's
RE2 regex matching is real.
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
from synalinks.src.modules.retrievers.regex_search import RegexSearch
from synalinks.src.programs import Program


class LogLine(DataModel):
    id: str = Field(description="The log id")
    text: str = Field(description="The log line")


class Query(DataModel):
    question: str = Field(description="The user question")


def _lm_returns(content):
    def _fake(*args, **kwargs):
        return {"choices": [{"message": {"content": content}}]}

    return _fake


class RegexSearchTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = "duckdb://" + os.path.join(self.temp_dir, "rgx.db")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        super().tearDown()

    async def _kb(self):
        kb = KnowledgeBase(uri=self.db_path, data_models=[LogLine])
        await kb.update(
            [
                JsonDataModel(data_model=LogLine(id="l1", text="ERROR 500 timeout")),
                JsonDataModel(data_model=LogLine(id="l2", text="INFO request ok")),
                JsonDataModel(data_model=LogLine(id="l3", text="ERROR 404 not found")),
            ]
        )
        return kb

    async def test_matches_pattern(self):
        kb = await self._kb()
        lm = LanguageModel(model="ollama/mistral")
        with patch(
            "litellm.acompletion",
            side_effect=_lm_returns(r'{"regex_search": "ERROR \\d+"}'),
        ):
            inputs = Input(data_model=Query)
            outputs = await RegexSearch(
                knowledge_base=kb, language_model=lm, data_model=LogLine, k=5, name="rgx"
            )(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="find errors")
            )
        ids = {r["id"] for r in result.get("result")}
        self.assertEqual(ids, {"l1", "l3"})  # only ERROR-with-code lines

    async def test_case_insensitive(self):
        kb = await self._kb()
        lm = LanguageModel(model="ollama/mistral")
        with patch(
            "litellm.acompletion",
            side_effect=_lm_returns('{"regex_search": "error"}'),
        ):
            inputs = Input(data_model=Query)
            outputs = await RegexSearch(
                knowledge_base=kb,
                language_model=lm,
                data_model=LogLine,
                case_sensitive=False,
                name="rgx",
            )(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="errors")
            )
        ids = {r["id"] for r in result.get("result")}
        self.assertEqual(ids, {"l1", "l3"})

    async def test_no_match_returns_empty(self):
        kb = await self._kb()
        lm = LanguageModel(model="ollama/mistral")
        with patch(
            "litellm.acompletion",
            side_effect=_lm_returns('{"regex_search": "CRITICAL"}'),
        ):
            inputs = Input(data_model=Query)
            outputs = await RegexSearch(
                knowledge_base=kb, language_model=lm, data_model=LogLine, name="rgx"
            )(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="critical")
            )
        self.assertEqual(result.get("result"), [])

    async def test_return_query_threaded(self):
        kb = await self._kb()
        lm = LanguageModel(model="ollama/mistral")
        with patch(
            "litellm.acompletion",
            side_effect=_lm_returns('{"regex_search": "ERROR"}'),
        ):
            inputs = Input(data_model=Query)
            outputs = await RegexSearch(
                knowledge_base=kb,
                language_model=lm,
                data_model=LogLine,
                return_query=True,
                name="rgx",
            )(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="errors")
            )
        self.assertIn("regex_search", result.get_json())

    async def test_infers_table_name_when_not_provided(self):
        # With no table_name/schema/data_model, the LM infers the table from an
        # enum of the KB's actual tables, concatenated onto the query schema.
        kb = await self._kb()
        lm = LanguageModel(model="ollama/mistral")
        mod = RegexSearch(knowledge_base=kb, language_model=lm, k=5, name="rgx")
        self.assertIsNone(mod.table_name)
        gen_schema = mod.query_generator.schema
        self.assertIn("table_name", gen_schema["properties"])
        self.assertEqual(gen_schema["properties"]["table_name"]["enum"], ["LogLine"])
        with patch(
            "litellm.acompletion",
            side_effect=_lm_returns(
                r'{"regex_search": "ERROR \\d+", "table_name": "LogLine"}'
            ),
        ):
            inputs = Input(data_model=Query)
            outputs = await mod(inputs)
            result = await Program(inputs=inputs, outputs=outputs)(
                Query(question="find errors")
            )
        ids = {r["id"] for r in result.get("result")}
        self.assertEqual(ids, {"l1", "l3"})
