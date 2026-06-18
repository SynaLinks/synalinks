# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import tempfile

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend.pydantic.knowledge import Entity
from synalinks.src.backend.pydantic.knowledge import Relation
from synalinks.src.knowledge_bases.knowledge_base import KnowledgeBase
from synalinks.src.modules.agents.cypher_agent import CypherAgent
from synalinks.src.modules.agents.cypher_agent import _build_tools
from synalinks.src.modules.agents.cypher_agent import get_default_instructions
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


class Person(Entity):
    label: str = Field(default="Person", description="The entity label")
    name: str = Field(description="Person name")


class City(Entity):
    label: str = Field(default="City", description="The entity label")
    name: str = Field(description="City name")


class LivesIn(Relation):
    label: str = Field(default="LivesIn", description="The relation label")
    subj: Person
    obj: City


class Query(DataModel):
    query: str = Field(description="A natural language question")


class CypherAnswer(DataModel):
    answer: str = Field(description="Natural-language answer")
    cypher_query: str = Field(description="The Cypher that produced the answer")


class CypherAgentToolsTest(testing.TestCase):
    """Tool-level tests: exercise the closures without invoking an LM.

    The graph KnowledgeBase is built without an embedding model, so the tools
    (schema discovery + read-only Cypher) run with no network calls.
    """

    def _make_kb(self):
        return KnowledgeBase(
            graph_uri="ladybug://:memory:",
            entity_models=[Person, City],
            relation_models=[LivesIn],
            name="cypher_agent_test_kb",
        )

    def _tools(self, kb, output_format="csv", k=50):
        get_schema, get_sample, run_cypher = _build_tools(
            kb, output_format=output_format, k=k
        )
        return get_schema, get_sample, run_cypher

    async def test_get_graph_schema_tool(self):
        kb = self._make_kb()
        await kb.update_relations(
            LivesIn(subj=Person(name="Alice"), obj=City(name="Paris"))
        )

        get_schema, _, _ = self._tools(kb)
        result = await get_schema()

        self.assertEqual(result["node_count"], 2)
        self.assertEqual(result["relation_count"], 1)
        self.assertIn("Person", result["schema"])
        self.assertIn("City", result["schema"])
        self.assertIn("LivesIn", result["schema"])

    async def test_get_node_sample_json(self):
        kb = self._make_kb()
        await kb.update_relations(
            [
                LivesIn(subj=Person(name="Alice"), obj=City(name="Paris")),
                LivesIn(subj=Person(name="Bob"), obj=City(name="London")),
            ]
        )

        _, get_sample, _ = self._tools(kb, output_format="json")
        result = await get_sample(label="Person", limit=5, offset=0)

        self.assertEqual(result["label"], "Person")
        self.assertEqual(result["output_format"], "json")
        self.assertEqual(result["row_count"], 2)
        self.assertIsInstance(result["sample_data"], list)

    async def test_get_node_sample_clamps_limit_to_k(self):
        kb = self._make_kb()
        await kb.update_relations(
            [LivesIn(subj=Person(name=f"P{i}"), obj=City(name="Paris")) for i in range(6)]
        )

        _, get_sample, _ = self._tools(kb, output_format="json", k=3)
        result = await get_sample(label="Person", limit=50, offset=0)

        self.assertEqual(result["limit"], 3)
        self.assertTrue(result["limit_capped"])

    async def test_get_node_sample_unknown_label(self):
        kb = self._make_kb()
        _, get_sample, _ = self._tools(kb)
        result = await get_sample(label="DoesNotExist", limit=5, offset=0)

        self.assertIn("error", result)
        self.assertIn("DoesNotExist", result["error"])

    async def test_run_cypher_query_returns_rows(self):
        kb = self._make_kb()
        await kb.update_relations(
            LivesIn(subj=Person(name="Alice"), obj=City(name="Paris"))
        )

        _, _, run_cypher = self._tools(kb, output_format="json")
        result = await run_cypher(
            "MATCH (p:Person)-[:LivesIn]->(c:City) RETURN p.name AS person"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["row_count"], 1)

    async def test_run_cypher_query_rejects_writes(self):
        kb = self._make_kb()
        _, _, run_cypher = self._tools(kb)
        result = await run_cypher("CREATE (n:Person {name: 'Mallory'}) RETURN n")

        # Read-only enforcement happens in the engine; the tool surfaces it
        # as a failed result rather than raising.
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    async def test_run_cypher_query_empty_rejected(self):
        kb = self._make_kb()
        _, _, run_cypher = self._tools(kb)
        result = await run_cypher("   ")

        self.assertFalse(result["success"])
        self.assertIn("error", result)


class CypherAgentInstantiationTest(testing.TestCase):
    """Instantiation tests — wires CypherAgent but doesn't run an LM."""

    def _make_kb(self):
        return KnowledgeBase(
            graph_uri="ladybug://:memory:",
            entity_models=[Person, City],
            relation_models=[LivesIn],
            name="cypher_agent_instantiation_kb",
        )

    def _make_sql_only_kb(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(self._cleanup_tmpdir, tmpdir)
        db_path = os.path.join(tmpdir, "cypher_agent_sql_only.db")
        return KnowledgeBase(
            uri=f"duckdb://{db_path}",
            data_models=[Query],
            wipe_on_start=True,
            name="cypher_agent_sql_only_kb",
        )

    def _cleanup_tmpdir(self, tmpdir):
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    async def test_default_instructions_include_labels(self):
        instructions = get_default_instructions(["Person", "City"], ["LivesIn"])
        self.assertIn("Person", instructions)
        self.assertIn("City", instructions)
        self.assertIn("LivesIn", instructions)
        self.assertIn("read-only", instructions)

    async def test_agent_requires_knowledge_base(self):
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            CypherAgent(language_model=lm, data_model=CypherAnswer)

    async def test_agent_requires_graph_adapter(self):
        """A SQL-only KnowledgeBase (no graph adapter) must be rejected at
        construction (fail-fast), rather than only erroring later inside the
        agent loop. ``_require_graph_adapter`` surfaces this as a
        ``NotImplementedError``."""
        kb = self._make_sql_only_kb()
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(NotImplementedError):
            CypherAgent(knowledge_base=kb, language_model=lm, data_model=CypherAnswer)

    async def test_agent_instantiation_with_data_model(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await CypherAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=CypherAnswer,
            name="cypher_agent_test",
        )(inputs)

        program = Program(inputs=inputs, outputs=outputs, name="cypher_agent_program")
        self.assertIsNotNone(program)

    async def test_agent_rejects_invalid_output_format(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            CypherAgent(
                knowledge_base=kb,
                language_model=lm,
                data_model=CypherAnswer,
                output_format="xml",
            )

    async def test_agent_tool_name_collision_raises(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        async def run_cypher_query(cypher_query: str):
            """Collides with a built-in Cypher tool.

            Args:
                cypher_query (str): ignored.
            """
            return {"result": None}

        with self.assertRaises(ValueError):
            CypherAgent(
                knowledge_base=kb,
                language_model=lm,
                data_model=CypherAnswer,
                tools=[Tool(run_cypher_query)],
            )

    async def test_agent_appends_user_tools(self):
        kb = self._make_kb()
        lm = LanguageModel(model="ollama/mistral")

        agent = CypherAgent(
            knowledge_base=kb,
            language_model=lm,
            data_model=CypherAnswer,
            tools=[Tool(calculator)],
        )
        tool_names = set(agent.tools.keys())
        self.assertIn("calculator", tool_names)
        self.assertIn("get_graph_schema", tool_names)
        self.assertIn("run_cypher_query", tool_names)
