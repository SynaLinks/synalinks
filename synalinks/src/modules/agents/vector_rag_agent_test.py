# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import shutil
import tempfile

from synalinks.src import testing
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.knowledge_bases.knowledge_base import KnowledgeBase
from synalinks.src.modules.agents.vector_rag_agent import VectorRAGAgent
from synalinks.src.modules.agents.vector_rag_agent import _build_tools
from synalinks.src.modules.agents.vector_rag_agent import get_default_instructions
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.embedding_models import EmbeddingModel
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Program


async def calculator(expression: str):
    """Compute a simple arithmetic expression.

    Args:
        expression (str): A safe arithmetic expression like '2 + 2'.
    """
    return {"result": eval(expression, {"__builtins__": {}}, {})}


class Document(DataModel):
    id: str = Field(description="Document id")
    title: str = Field(description="Document title")
    content: str = Field(description="Document body")


class Query(DataModel):
    query: str = Field(description="A natural language question")


class RAGAnswer(DataModel):
    answer: str = Field(description="The answer grounded in retrieved docs")


class VectorRAGAgentToolsTest(testing.TestCase):
    """Tool-level tests: exercise the closures without invoking an LM.

    Uses fulltext search (no embedding model needed) so tests stay
    fast and don't require a network round-trip.
    """

    def _make_kb(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        db_path = os.path.join(tmpdir, "vector_rag_agent_test.db")
        return KnowledgeBase(
            uri=f"duckdb://{db_path}",
            data_models=[Document],
            wipe_on_start=True,
            name="vector_rag_agent_test_kb",
        )

    def _tools(self, kb, search_type="fulltext", output_format="csv", k=5):
        get_schema, search, get_record = _build_tools(
            kb, search_type=search_type, k=k, output_format=output_format
        )
        return get_schema, search, get_record

    async def test_get_knowledge_base_schema_tool(self):
        kb = self._make_kb()
        await kb.update(
            Document(
                id="D1", title="PTO", content="20 days per year"
            ).to_json_data_model()
        )

        get_schema, _, _ = self._tools(kb)
        result = await get_schema()

        self.assertEqual(result["table_count"], 1)
        self.assertIn("Document", result["schema"])
        self.assertIn("title (string)", result["schema"])
        self.assertIn("content (string)", result["schema"])

    async def test_search_fulltext_returns_csv(self):
        kb = self._make_kb()
        await kb.update(
            [
                Document(
                    id="D1",
                    title="PTO Policy",
                    content="Employees receive 20 days of PTO per year",
                ).to_json_data_model(),
                Document(
                    id="D2",
                    title="Remote Work",
                    content="Employees may work remotely up to 3 days per week",
                ).to_json_data_model(),
            ]
        )

        _, search, _ = self._tools(kb, search_type="fulltext", output_format="csv")
        result = await search(table_name="Document", query="PTO days")

        self.assertEqual(result["table"], "Document")
        self.assertEqual(result["search_type"], "fulltext")
        self.assertEqual(result["output_format"], "csv")
        self.assertIsInstance(result["results"], str)
        self.assertIn("PTO", result["results"])
        self.assertGreaterEqual(result["row_count"], 1)

    async def test_search_fulltext_returns_json(self):
        kb = self._make_kb()
        await kb.update(
            Document(
                id="D1", title="Vacation", content="20 days PTO annually"
            ).to_json_data_model()
        )

        _, search, _ = self._tools(kb, search_type="fulltext", output_format="json")
        result = await search(table_name="Document", query="vacation")

        self.assertEqual(result["output_format"], "json")
        self.assertIsInstance(result["results"], list)
        self.assertEqual(result["row_count"], len(result["results"]))

    async def test_search_unknown_table_returns_error(self):
        kb = self._make_kb()
        _, search, _ = self._tools(kb)
        result = await search(
            table_name="DoesNotExist", query="anything"
        )

        self.assertIn("error", result)
        self.assertIn("DoesNotExist", result["error"])

    async def test_get_record_by_id_found(self):
        kb = self._make_kb()
        await kb.update(
            Document(
                id="D1", title="Policy", content="Full content here"
            ).to_json_data_model()
        )

        _, _, get_record = self._tools(kb)
        result = await get_record(table_name="Document", record_id="D1")

        self.assertTrue(result["found"])
        self.assertEqual(result["record"]["title"], "Policy")
        self.assertEqual(result["record"]["content"], "Full content here")

    async def test_get_record_by_id_missing(self):
        kb = self._make_kb()
        _, _, get_record = self._tools(kb)
        result = await get_record(table_name="Document", record_id="ghost")

        self.assertFalse(result["found"])

    async def test_get_record_by_id_unknown_table(self):
        kb = self._make_kb()
        _, _, get_record = self._tools(kb)
        result = await get_record(table_name="Nope", record_id="x")

        self.assertIn("error", result)


class VectorRAGAgentInstantiationTest(testing.TestCase):
    """End-to-end instantiation tests — wires VectorRAGAgent but doesn't run LM."""

    def _make_kb(self, with_embedding_model=False):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        db_path = os.path.join(tmpdir, "vector_rag_agent_inst.db")
        kwargs = dict(
            uri=f"duckdb://{db_path}",
            data_models=[Document],
            wipe_on_start=True,
            name="vector_rag_agent_inst_kb",
        )
        if with_embedding_model:
            kwargs["embedding_model"] = EmbeddingModel(
                model="ollama/all-minilm",
            )
        return KnowledgeBase(**kwargs)

    async def test_default_instructions_includes_tables_and_mode(self):
        instructions = get_default_instructions(["Document", "FAQ"], "hybrid_fts")
        self.assertIn("Document", instructions)
        self.assertIn("FAQ", instructions)
        self.assertIn("hybrid_fts", instructions)

    async def test_default_instructions_per_search_type(self):
        sim = get_default_instructions(["T"], "similarity")
        fts = get_default_instructions(["T"], "fulltext")
        hyb = get_default_instructions(["T"], "hybrid_fts")
        # Each mode produces a distinct guidance string.
        self.assertNotEqual(sim, fts)
        self.assertNotEqual(fts, hyb)
        self.assertNotEqual(sim, hyb)

    async def test_agent_instantiation_with_data_model(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await VectorRAGAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=RAGAnswer,
            search_type="fulltext",
            name="rag_test",
        )(inputs)

        program = Program(inputs=inputs, outputs=outputs, name="rag_program")
        self.assertIsNotNone(program)

    async def test_agent_instantiation_with_chat_messages(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=ChatMessages)
        outputs = await VectorRAGAgent(
            knowledge_base=kb,
            language_model=lm,
            search_type="fulltext",
            name="rag_chat",
        )(inputs)

        program = Program(inputs=inputs, outputs=outputs, name="rag_chat_program")
        self.assertIsNotNone(program)

    async def test_agent_wires_three_tools(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        agent = VectorRAGAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=RAGAnswer,
            search_type="fulltext",
            name="three_tools",
        )

        tool_names = set(agent.agent.tools.keys())
        self.assertEqual(
            tool_names,
            {
                "get_knowledge_base_schema",
                "search_knowledge_base",
                "get_record_by_id",
            },
        )

    async def test_agent_requires_knowledge_base(self):
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            VectorRAGAgent(language_model=lm, data_model=RAGAnswer)

    async def test_agent_rejects_invalid_search_type(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            VectorRAGAgent(
                knowledge_base=kb,
                language_model=lm,
                data_model=RAGAnswer,
                search_type="bogus",
            )

    async def test_agent_rejects_invalid_output_format(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            VectorRAGAgent(
                knowledge_base=kb,
                language_model=lm,
                data_model=RAGAnswer,
                search_type="fulltext",
                output_format="xml",
            )

    async def test_agent_defaults(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        agent = VectorRAGAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=RAGAnswer,
            search_type="fulltext",
            name="defaults",
        )

        self.assertEqual(agent.search_type, "fulltext")
        self.assertEqual(agent.output_format, "csv")
        self.assertEqual(agent.k, 5)
        self.assertEqual(agent.max_iterations, 5)

    async def test_agent_appends_user_tools(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        agent = VectorRAGAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=RAGAnswer,
            search_type="fulltext",
            tools=[Tool(calculator)],
            name="with_extra",
        )

        tool_names = set(agent.agent.tools.keys())
        self.assertEqual(
            tool_names,
            {
                "get_knowledge_base_schema",
                "search_knowledge_base",
                "get_record_by_id",
                "calculator",
            },
        )

    async def test_agent_accepts_plain_async_functions_as_tools(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        agent = VectorRAGAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=RAGAnswer,
            search_type="fulltext",
            tools=[calculator],  # not wrapped in Tool
            name="bare_fn",
        )

        self.assertIn("calculator", agent.agent.tools)

    async def test_agent_tool_name_collision_raises(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        async def search_knowledge_base(query: str):
            """Shadowing the built-in tool.

            Args:
                query (str): unused.
            """
            return {"oops": True}

        with self.assertRaises(ValueError):
            VectorRAGAgent(
                knowledge_base=kb,
                language_model=lm,
                data_model=RAGAnswer,
                search_type="fulltext",
                tools=[search_knowledge_base],
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
            VectorRAGAgent(
                knowledge_base=kb,
                language_model=lm,
                data_model=RAGAnswer,
                search_type="fulltext",
                tools=[_private_helper],
            )

    async def test_agent_custom_instructions_preserved(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        custom = "Be concise."
        agent = VectorRAGAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=RAGAnswer,
            instructions=custom,
            search_type="fulltext",
            name="custom_instr",
        )

        self.assertEqual(agent.instructions, custom)
        self.assertEqual(agent.agent.instructions, custom)
