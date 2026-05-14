# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import tempfile

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.knowledge_bases.knowledge_base import KnowledgeBase
from synalinks.src.modules.agents.sql_agent import SQLAgent
from synalinks.src.modules.agents.sql_agent import _build_tools
from synalinks.src.modules.agents.sql_agent import get_default_instructions
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Program


async def calculator(expression: str):
    """Compute a simple arithmetic expression.

    Args:
        expression (str): A safe arithmetic expression like '2 + 2'.
    """
    return {"result": eval(expression, {"__builtins__": {}}, {})}


class Customer(DataModel):
    id: str = Field(description="Customer ID")
    name: str = Field(description="Customer name")
    country: str = Field(description="Customer country")


class Query(DataModel):
    query: str = Field(description="A natural language question")


class SQLAnswer(DataModel):
    answer: str = Field(description="Natural-language answer")
    sql_query: str = Field(description="The SQL that produced the answer")


class SQLAgentToolsTest(testing.TestCase):
    """Tool-level tests: exercise the closures without invoking an LM."""

    def _make_kb(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_tmpdir, tmpdir)
        db_path = os.path.join(tmpdir, "sql_agent_test.db")
        return KnowledgeBase(
            uri=f"duckdb://{db_path}",
            data_models=[Customer],
            wipe_on_start=True,
            name="sql_agent_test_kb",
        )

    def _cleanup_tmpdir(self, tmpdir):
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    def _tools(self, kb, output_format="csv", k=50):
        get_schema, get_sample, run_sql = _build_tools(
            kb, output_format=output_format, k=k
        )
        return get_schema, get_sample, run_sql

    async def test_get_database_schema_tool(self):
        kb = self._make_kb()
        await kb.update(
            Customer(id="C1", name="Alice", country="USA").to_json_data_model()
        )

        get_schema, _, _ = self._tools(kb)
        result = await get_schema()

        self.assertEqual(result["table_count"], 1)
        self.assertIn("Customer", result["schema"])
        self.assertIn("id (string)", result["schema"])
        self.assertIn("country (string)", result["schema"])

    async def test_get_table_sample_csv_pagination(self):
        kb = self._make_kb()
        await kb.update(
            [
                Customer(id="C1", name="Alice", country="USA").to_json_data_model(),
                Customer(id="C2", name="Bob", country="UK").to_json_data_model(),
                Customer(id="C3", name="Carol", country="USA").to_json_data_model(),
            ]
        )

        _, get_sample, _ = self._tools(kb, output_format="csv")
        result = await get_sample(table_name="Customer", limit=2, offset=0)

        self.assertEqual(result["table"], "Customer")
        self.assertEqual(result["output_format"], "csv")
        self.assertEqual(result["row_count"], 2)
        self.assertIsInstance(result["sample_data"], str)
        self.assertIn("name", result["sample_data"])

        # Page 2 — offset past the first 2 rows returns the remaining 1.
        result = await get_sample(table_name="Customer", limit=2, offset=2)
        self.assertEqual(result["row_count"], 1)

    async def test_get_table_sample_json_format(self):
        kb = self._make_kb()
        await kb.update(
            Customer(id="C1", name="Alice", country="USA").to_json_data_model()
        )

        _, get_sample, _ = self._tools(kb, output_format="json")
        result = await get_sample(table_name="Customer", limit=5, offset=0)

        self.assertEqual(result["output_format"], "json")
        self.assertEqual(result["row_count"], 1)
        self.assertIsInstance(result["sample_data"], list)
        self.assertEqual(result["sample_data"][0]["name"], "Alice")

    async def test_get_table_sample_clamps_limit_to_k(self):
        kb = self._make_kb()
        await kb.update(
            [
                Customer(id=f"C{i}", name=f"N{i}", country="USA").to_json_data_model()
                for i in range(10)
            ]
        )

        # k=3 caps the LM's request even though it asked for 50.
        _, get_sample, _ = self._tools(kb, output_format="json", k=3)
        result = await get_sample(table_name="Customer", limit=50, offset=0)

        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["limit"], 3)
        self.assertTrue(result["limit_capped"])

    async def test_get_table_sample_passes_limit_under_cap(self):
        kb = self._make_kb()
        await kb.update(
            [
                Customer(id=f"C{i}", name=f"N{i}", country="USA").to_json_data_model()
                for i in range(5)
            ]
        )

        # LM asks for 2 — under the k=10 cap — so request is honored verbatim.
        _, get_sample, _ = self._tools(kb, output_format="json", k=10)
        result = await get_sample(table_name="Customer", limit=2, offset=0)

        self.assertEqual(result["row_count"], 2)
        self.assertEqual(result["limit"], 2)
        self.assertFalse(result["limit_capped"])

    async def test_get_table_sample_unknown_table(self):
        kb = self._make_kb()
        _, get_sample, _ = self._tools(kb)
        result = await get_sample(table_name="DoesNotExist", limit=5, offset=0)

        self.assertIn("error", result)
        self.assertIn("DoesNotExist", result["error"])

    async def test_run_sql_query_csv_default(self):
        kb = self._make_kb()
        await kb.update(
            [
                Customer(id="C1", name="Alice", country="USA").to_json_data_model(),
                Customer(id="C2", name="Bob", country="UK").to_json_data_model(),
                Customer(id="C3", name="Carol", country="USA").to_json_data_model(),
            ]
        )

        _, _, run_sql = self._tools(kb, output_format="csv")
        result = await run_sql(
            sql_query="SELECT country, COUNT(*) AS n FROM Customer GROUP BY country"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["output_format"], "csv")
        self.assertEqual(result["row_count"], 2)
        self.assertIsInstance(result["results"], str)
        self.assertIn("country", result["results"])
        self.assertIn("USA", result["results"])

    async def test_run_sql_query_json_format(self):
        kb = self._make_kb()
        await kb.update(
            [
                Customer(id="C1", name="Alice", country="USA").to_json_data_model(),
                Customer(id="C2", name="Bob", country="UK").to_json_data_model(),
            ]
        )

        _, _, run_sql = self._tools(kb, output_format="json")
        result = await run_sql(sql_query="SELECT COUNT(*) AS n FROM Customer")

        self.assertTrue(result["success"])
        self.assertEqual(result["output_format"], "json")
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["results"][0]["n"], 2)

    async def test_run_sql_query_caps_at_k(self):
        kb = self._make_kb()
        await kb.update(
            [
                Customer(id=f"C{i}", name=f"N{i}", country="USA").to_json_data_model()
                for i in range(10)
            ]
        )

        # k=3 caps even though SELECT * would return all 10.
        _, _, run_sql = self._tools(kb, output_format="json", k=3)
        result = await run_sql(sql_query="SELECT * FROM Customer")

        self.assertTrue(result["success"])
        self.assertEqual(result["row_count"], 3)
        self.assertEqual(result["row_cap"], 3)
        self.assertTrue(result["may_have_more"])

    async def test_run_sql_query_inner_limit_takes_precedence(self):
        kb = self._make_kb()
        await kb.update(
            [
                Customer(id=f"C{i}", name=f"N{i}", country="USA").to_json_data_model()
                for i in range(10)
            ]
        )

        # k=50, user asks for LIMIT 2 — inner limit wins.
        _, _, run_sql = self._tools(kb, output_format="json", k=50)
        result = await run_sql(sql_query="SELECT * FROM Customer LIMIT 2")

        self.assertEqual(result["row_count"], 2)
        self.assertFalse(result["may_have_more"])

    async def test_run_sql_query_aggregate_under_cap(self):
        kb = self._make_kb()
        await kb.update(
            [
                Customer(id=f"C{i}", name=f"N{i}", country="USA").to_json_data_model()
                for i in range(5)
            ]
        )

        # Aggregate returns 1 row — well under cap, may_have_more is False.
        _, _, run_sql = self._tools(kb, output_format="json", k=10)
        result = await run_sql(sql_query="SELECT COUNT(*) AS n FROM Customer")

        self.assertEqual(result["row_count"], 1)
        self.assertFalse(result["may_have_more"])
        self.assertEqual(result["results"][0]["n"], 5)

    async def test_run_sql_query_trailing_semicolon_handled(self):
        kb = self._make_kb()
        await kb.update(
            Customer(id="C1", name="Alice", country="USA").to_json_data_model()
        )

        _, _, run_sql = self._tools(kb, output_format="json", k=10)
        result = await run_sql(sql_query="SELECT * FROM Customer;")

        self.assertTrue(result["success"])
        self.assertEqual(result["row_count"], 1)

    async def test_run_sql_query_empty_query_rejected(self):
        kb = self._make_kb()
        _, _, run_sql = self._tools(kb)

        result = await run_sql(sql_query="   ")
        self.assertFalse(result["success"])
        self.assertIn("Empty", result["error"])

    async def test_run_sql_query_rejects_writes(self):
        kb = self._make_kb()
        _, _, run_sql = self._tools(kb)
        result = await run_sql(sql_query="DROP TABLE Customer")

        self.assertFalse(result["success"])
        self.assertIn("error", result)

    async def test_run_sql_query_rejects_multi_statement(self):
        kb = self._make_kb()
        _, _, run_sql = self._tools(kb)
        result = await run_sql(sql_query="SELECT 1; DROP TABLE Customer")

        self.assertFalse(result["success"])


class SQLAgentInstantiationTest(testing.TestCase):
    """End-to-end instantiation tests — wires SQLAgent but doesn't run LM."""

    def _make_kb(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_tmpdir, tmpdir)
        db_path = os.path.join(tmpdir, "sql_agent_instantiation.db")
        return KnowledgeBase(
            uri=f"duckdb://{db_path}",
            data_models=[Customer],
            wipe_on_start=True,
            name="sql_agent_instantiation_kb",
        )

    def _cleanup_tmpdir(self, tmpdir):
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    async def test_default_instructions_includes_tables(self):
        instructions = get_default_instructions(["Customer", "Product"])
        self.assertIn("Customer", instructions)
        self.assertIn("Product", instructions)
        self.assertIn("SELECT", instructions)

    async def test_agent_instantiation_with_data_model(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await SQLAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=SQLAnswer,
            name="sql_agent_test",
        )(inputs)

        program = Program(inputs=inputs, outputs=outputs, name="sql_agent_program")
        self.assertIsNotNone(program)

    async def test_agent_wires_three_tools(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        agent = SQLAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=SQLAnswer,
            name="three_tools",
        )

        tool_names = set(agent.agent.tools.keys())
        self.assertEqual(
            tool_names,
            {"get_database_schema", "get_table_sample", "run_sql_query"},
        )

    async def test_agent_requires_knowledge_base(self):
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            SQLAgent(language_model=lm, data_model=SQLAnswer)

    async def test_agent_custom_instructions_preserved(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        custom = "You are a SQL bot. Be terse."
        agent = SQLAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=SQLAnswer,
            instructions=custom,
            name="custom_instr",
        )

        self.assertEqual(agent.instructions, custom)
        self.assertEqual(agent.agent.instructions, custom)

    async def test_agent_default_output_format_is_csv(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")
        agent = SQLAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=SQLAnswer,
            name="default_fmt",
        )
        self.assertEqual(agent.output_format, "csv")

    async def test_agent_rejects_invalid_output_format(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            SQLAgent(
                knowledge_base=kb,
                language_model=lm,
                data_model=SQLAnswer,
                output_format="xml",
            )

    async def test_agent_appends_user_tools(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        agent = SQLAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=SQLAnswer,
            tools=[Tool(calculator)],
            name="with_extra",
        )

        tool_names = set(agent.agent.tools.keys())
        self.assertEqual(
            tool_names,
            {
                "get_database_schema",
                "get_table_sample",
                "run_sql_query",
                "calculator",
            },
        )

    async def test_agent_accepts_plain_async_functions_as_tools(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        agent = SQLAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=SQLAnswer,
            tools=[calculator],  # not wrapped in Tool
            name="bare_fn",
        )

        self.assertIn("calculator", agent.agent.tools)

    async def test_agent_tool_name_collision_raises(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        async def run_sql_query(sql_query: str):
            """Shadowing the built-in tool.

            Args:
                sql_query (str): unused.
            """
            return {"oops": True}

        with self.assertRaises(ValueError):
            SQLAgent(
                knowledge_base=kb,
                language_model=lm,
                data_model=SQLAnswer,
                tools=[run_sql_query],
            )

    async def test_agent_rejects_leading_underscore_tool(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        async def _private_helper(x: str):
            """Has a leading underscore — should be rejected.

            Args:
                x (str): unused.
            """
            return {"x": x}

        with self.assertRaises(ValueError):
            SQLAgent(
                knowledge_base=kb,
                language_model=lm,
                data_model=SQLAnswer,
                tools=[_private_helper],
            )

    async def test_agent_k_default_and_storage(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")
        agent = SQLAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=SQLAnswer,
            name="default_k",
        )
        self.assertEqual(agent.k, 50)

        agent2 = SQLAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=SQLAnswer,
            k=7,
            name="custom_k",
        )
        self.assertEqual(agent2.k, 7)

    async def test_agent_rejects_invalid_k(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            SQLAgent(
                knowledge_base=kb,
                language_model=lm,
                data_model=SQLAnswer,
                k=0,
            )
        with self.assertRaises(ValueError):
            SQLAgent(
                knowledge_base=kb,
                language_model=lm,
                data_model=SQLAnswer,
                k=-5,
            )

    async def test_agent_forwards_max_iterations(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        agent = SQLAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=SQLAnswer,
            max_iterations=7,
            name="max_iter",
        )

        self.assertEqual(agent.max_iterations, 7)
        self.assertEqual(agent.agent.max_iterations, 7)
