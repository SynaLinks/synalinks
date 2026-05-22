# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.knowledge_bases.knowledge_base import KnowledgeBase
from synalinks.src.modules.knowledge.text2sql import DatabaseSchema
from synalinks.src.modules.knowledge.text2sql import SQLQueryResult
from synalinks.src.modules.knowledge.text2sql import Text2SQL
from synalinks.src.modules.knowledge.text2sql import _format_schema
from synalinks.src.modules.knowledge.text2sql import default_text2sql_instructions
from synalinks.src.modules.language_models import LanguageModel


class Customer(DataModel):
    id: str = Field(description="Customer ID")
    name: str = Field(description="Customer name")
    country: str = Field(description="Customer country")


class Query(DataModel):
    query: str = Field(description="A natural language question")


def _sql_response(sql_query):
    """Shape litellm.acompletion returns for a generated SQLQuery."""
    return {
        "choices": [{"message": {"content": '{"sql_query": ' + f'"{sql_query}"' + "}"}}]
    }


class Text2SQLHelpersTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def _make_kb(self):
        db = os.path.join(self.tmp, "t2s.db")
        return KnowledgeBase(
            uri=f"duckdb://{db}",
            data_models=[Customer],
            wipe_on_start=True,
        )

    def test_default_instructions_mention_select_only(self):
        text = default_text2sql_instructions()
        self.assertIn("SELECT", text)
        # Read-only contract: the destructive verbs are called out as rejected.
        self.assertIn("read-only", text)

    def test_format_schema_lists_tables_and_columns(self):
        kb = self._make_kb()
        rendered = _format_schema(kb)
        self.assertIn("Customer", rendered)
        self.assertIn("name", rendered)
        self.assertIn("country", rendered)


class Text2SQLConstructionTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def _make_kb(self):
        db = os.path.join(self.tmp, "t2s.db")
        return KnowledgeBase(
            uri=f"duckdb://{db}", data_models=[Customer], wipe_on_start=True
        )

    def test_invalid_output_format_raises(self):
        with pytest.raises(ValueError, match="output_format"):
            Text2SQL(
                knowledge_base=self._make_kb(),
                language_model=LanguageModel(model="ollama/mistral"),
                output_format="xml",
            )

    def test_invalid_k_raises(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")
        with pytest.raises(ValueError, match="`k` must be"):
            Text2SQL(knowledge_base=kb, language_model=lm, k=0)
        with pytest.raises(ValueError, match="`k` must be"):
            Text2SQL(knowledge_base=kb, language_model=lm, k=-3)

    def test_default_instructions_used_when_omitted(self):
        module = Text2SQL(
            knowledge_base=self._make_kb(),
            language_model=LanguageModel(model="ollama/mistral"),
        )
        self.assertEqual(module.instructions, default_text2sql_instructions())

    def test_get_config_from_config_roundtrip(self):
        module = Text2SQL(
            knowledge_base=self._make_kb(),
            language_model=LanguageModel(model="ollama/mistral"),
            k=10,
            output_format="csv",
            instructions="custom",
            name="t2s",
        )
        config = module.get_config()
        self.assertEqual(config["k"], 10)
        self.assertEqual(config["output_format"], "csv")
        rebuilt = Text2SQL.from_config(config)
        self.assertEqual(rebuilt.k, 10)
        self.assertEqual(rebuilt.output_format, "csv")
        self.assertEqual(rebuilt.instructions, "custom")


class Text2SQLCallTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    async def _make_kb(self):
        db = os.path.join(self.tmp, "t2s.db")
        kb = KnowledgeBase(
            uri=f"duckdb://{db}", data_models=[Customer], wipe_on_start=True
        )
        await kb.update(
            [
                Customer(id="C1", name="Alice", country="USA"),
                Customer(id="C2", name="Bob", country="USA"),
                Customer(id="C3", name="Carlos", country="Spain"),
            ]
        )
        return kb

    def _module(self, kb, **kwargs):
        return Text2SQL(
            knowledge_base=kb,
            language_model=LanguageModel(model="ollama/mistral"),
            **kwargs,
        )

    async def test_none_input_returns_none(self):
        kb = await self._make_kb()
        module = self._module(kb)
        self.assertIsNone(await module(None))

    @patch("litellm.acompletion")
    async def test_generates_and_executes_select(self, mock_completion):
        mock_completion.return_value = _sql_response(
            "SELECT name FROM Customer WHERE country = 'USA' ORDER BY name"
        )
        kb = await self._make_kb()
        module = self._module(kb)
        result = await module(Query(query="USA customers?"))
        data = result.get_json()
        self.assertIn("SELECT", data["sql_query"])
        names = [row["name"] for row in data["result"]]
        self.assertEqual(names, ["Alice", "Bob"])

    @patch("litellm.acompletion")
    async def test_outer_limit_caps_rows(self, mock_completion):
        # Unbounded SELECT * is wrapped in `... LIMIT k`, so k caps the rows
        # even though the LM's own query has no LIMIT.
        mock_completion.return_value = _sql_response("SELECT * FROM Customer")
        kb = await self._make_kb()
        module = self._module(kb, k=2)
        result = await module(Query(query="all customers"))
        self.assertEqual(len(result.get_json()["result"]), 2)

    @patch("litellm.acompletion")
    async def test_k_none_runs_query_as_is(self, mock_completion):
        mock_completion.return_value = _sql_response("SELECT id FROM Customer")
        kb = await self._make_kb()
        module = self._module(kb, k=None)
        result = await module(Query(query="all ids"))
        self.assertEqual(len(result.get_json()["result"]), 3)

    @patch("litellm.acompletion")
    async def test_blank_sql_yields_empty_result(self, mock_completion):
        mock_completion.return_value = _sql_response("")
        kb = await self._make_kb()
        module = self._module(kb)
        result = await module(Query(query="?"))
        data = result.get_json()
        self.assertEqual(data["sql_query"], "")
        self.assertEqual(data["result"], [])

    @patch("litellm.acompletion")
    async def test_execution_error_is_captured_in_result(self, mock_completion):
        mock_completion.return_value = _sql_response("SELECT * FROM NoSuchTable")
        kb = await self._make_kb()
        module = self._module(kb)
        result = await module(Query(query="bad"))
        rows = result.get_json()["result"]
        self.assertEqual(len(rows), 1)
        self.assertIn("error", rows[0])

    @patch("litellm.acompletion")
    async def test_return_inputs_concatenates_original_query(self, mock_completion):
        mock_completion.return_value = _sql_response("SELECT id FROM Customer")
        kb = await self._make_kb()
        module = self._module(kb, return_inputs=True)
        result = await module(Query(query="keep me"))
        data = result.get_json()
        # Original input field is carried through alongside the output.
        self.assertEqual(data["query"], "keep me")
        self.assertIn("sql_query", data)

    async def test_compute_output_spec_shape(self):
        kb = await self._make_kb()
        module = self._module(kb)
        spec = await module.compute_output_spec(Query.to_symbolic_data_model())
        self.assertEqual(spec.get_schema(), SQLQueryResult.get_schema())

    def test_database_schema_data_model_shape(self):
        # The schema-snapshot field the module concats onto the inputs.
        self.assertIn("database_schema", DatabaseSchema.get_schema()["properties"])
