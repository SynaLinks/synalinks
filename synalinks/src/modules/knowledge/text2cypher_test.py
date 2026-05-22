# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

import pytest

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend.pydantic.knowledge import Entity
from synalinks.src.backend.pydantic.knowledge import Relation
from synalinks.src.knowledge_bases.knowledge_base import KnowledgeBase
from synalinks.src.modules.knowledge.text2cypher import CypherQueryResult
from synalinks.src.modules.knowledge.text2cypher import GraphSchema
from synalinks.src.modules.knowledge.text2cypher import Text2Cypher
from synalinks.src.modules.knowledge.text2cypher import _format_graph_schema
from synalinks.src.modules.knowledge.text2cypher import default_text2cypher_instructions
from synalinks.src.modules.language_models import LanguageModel


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


def _cypher_response(cypher_query):
    """Shape litellm.acompletion returns for a generated CypherQuery."""
    return {
        "choices": [
            {"message": {"content": json.dumps({"cypher_query": cypher_query})}}
        ]
    }


def _make_kb(name="t2c_kb"):
    return KnowledgeBase(
        graph_uri="ladybug://:memory:",
        entity_models=[Person, City],
        relation_models=[LivesIn],
        name=name,
    )


class Text2CypherHelpersTest(testing.TestCase):
    def test_default_instructions_include_limit_hint_when_k_set(self):
        text = default_text2cypher_instructions(50)
        self.assertIn("LIMIT 50", text)
        self.assertIn("RETURN", text)
        self.assertIn("read-only", text)

    def test_default_instructions_drop_limit_hint_when_k_none(self):
        text = default_text2cypher_instructions(None)
        self.assertNotIn("LIMIT", text)
        self.assertIn("RETURN", text)

    def test_format_graph_schema_lists_entities_and_relations(self):
        kb = _make_kb("schema_kb")
        rendered = _format_graph_schema(kb)
        self.assertIn("# Entities", rendered)
        self.assertIn("# Relations", rendered)
        self.assertIn("Person", rendered)
        self.assertIn("City", rendered)
        # Relations are rendered as Cypher ASCII-art with the endpoints.
        self.assertIn("(:Person)-[:LivesIn]->(:City)", rendered)


class Text2CypherConstructionTest(testing.TestCase):
    def test_invalid_output_format_raises(self):
        with pytest.raises(ValueError, match="output_format"):
            Text2Cypher(
                knowledge_base=_make_kb(),
                language_model=LanguageModel(model="ollama/mistral"),
                output_format="xml",
            )

    def test_invalid_k_raises(self):
        kb = _make_kb()
        lm = LanguageModel(model="ollama/mistral")
        with pytest.raises(ValueError, match="`k` must be"):
            Text2Cypher(knowledge_base=kb, language_model=lm, k=0)

    def test_default_instructions_reflect_k(self):
        module = Text2Cypher(
            knowledge_base=_make_kb(),
            language_model=LanguageModel(model="ollama/mistral"),
            k=7,
        )
        self.assertEqual(
            module.instructions, default_text2cypher_instructions(7)
        )
        self.assertIn("LIMIT 7", module.instructions)

    def test_get_config_from_config_roundtrip(self):
        module = Text2Cypher(
            knowledge_base=_make_kb(),
            language_model=LanguageModel(model="ollama/mistral"),
            k=10,
            output_format="csv",
            instructions="custom",
            name="t2c",
        )
        config = module.get_config()
        self.assertEqual(config["k"], 10)
        self.assertEqual(config["output_format"], "csv")
        rebuilt = Text2Cypher.from_config(config)
        self.assertEqual(rebuilt.k, 10)
        self.assertEqual(rebuilt.output_format, "csv")
        self.assertEqual(rebuilt.instructions, "custom")


class Text2CypherCallTest(testing.TestCase):
    async def _make_populated_kb(self, name="t2c_call_kb"):
        kb = _make_kb(name)
        await kb.update_relations(
            [
                LivesIn(subj=Person(name="Alice"), obj=City(name="Paris")),
                LivesIn(subj=Person(name="Bob"), obj=City(name="Paris")),
            ]
        )
        return kb

    def _module(self, kb, **kwargs):
        return Text2Cypher(
            knowledge_base=kb,
            language_model=LanguageModel(model="ollama/mistral"),
            **kwargs,
        )

    async def test_none_input_returns_none(self):
        kb = await self._make_populated_kb("none_kb")
        self.assertIsNone(await self._module(kb)(None))

    @patch("litellm.acompletion")
    async def test_generates_and_executes_cypher(self, mock_completion):
        mock_completion.return_value = _cypher_response(
            "MATCH (p:Person)-[:LivesIn]->(c:City {name: 'Paris'}) "
            "RETURN p.name AS person ORDER BY person"
        )
        kb = await self._make_populated_kb("exec_kb")
        result = await self._module(kb)(Query(query="Who lives in Paris?"))
        data = result.get_json()
        self.assertIn("MATCH", data["cypher_query"])
        people = [row["person"] for row in data["result"]]
        self.assertEqual(people, ["Alice", "Bob"])

    @patch("litellm.acompletion")
    async def test_blank_cypher_yields_empty_result(self, mock_completion):
        mock_completion.return_value = _cypher_response("")
        kb = await self._make_populated_kb("blank_kb")
        result = await self._module(kb)(Query(query="?"))
        data = result.get_json()
        self.assertEqual(data["cypher_query"], "")
        self.assertEqual(data["result"], [])

    @patch("litellm.acompletion")
    async def test_write_query_rejected_and_captured(self, mock_completion):
        # read_only=True makes the adapter reject any write keyword; the
        # raised error is captured into the result rather than propagated.
        mock_completion.return_value = _cypher_response(
            "CREATE (p:Person {name: 'Mallory'}) RETURN p"
        )
        kb = await self._make_populated_kb("write_kb")
        result = await self._module(kb)(Query(query="add a person"))
        rows = result.get_json()["result"]
        self.assertEqual(len(rows), 1)
        self.assertIn("error", rows[0])

    @patch("litellm.acompletion")
    async def test_return_inputs_concatenates_original_query(self, mock_completion):
        mock_completion.return_value = _cypher_response(
            "MATCH (p:Person) RETURN p.name AS person"
        )
        kb = await self._make_populated_kb("ret_kb")
        result = await self._module(kb, return_inputs=True)(Query(query="keep me"))
        data = result.get_json()
        self.assertEqual(data["query"], "keep me")
        self.assertIn("cypher_query", data)

    async def test_compute_output_spec_shape(self):
        kb = await self._make_populated_kb("spec_kb")
        spec = await self._module(kb).compute_output_spec(
            Query.to_symbolic_data_model()
        )
        self.assertEqual(spec.get_schema(), CypherQueryResult.get_schema())

    def test_graph_schema_data_model_shape(self):
        self.assertIn("graph_schema", GraphSchema.get_schema()["properties"])
