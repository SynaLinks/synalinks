# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend.common.json_data_model import JsonDataModel


class JsonDataModelTest(testing.TestCase):
    def test_init_with_data_model(self):
        class Query(DataModel):
            query: str

        json_data_model = JsonDataModel(
            data_model=Query(query="What is the capital of France?")
        )

        expected_schema = {
            "additionalProperties": False,
            "properties": {"query": {"title": "Query", "type": "string"}},
            "required": ["query"],
            "title": "Query",
            "type": "object",
        }
        expected_json = {"query": "What is the capital of France?"}

        self.assertEqual(json_data_model.get_json(), expected_json)
        self.assertEqual(json_data_model.get_schema(), expected_schema)

    def test_init_with_data_model_non_instanciated(self):
        class Query(DataModel):
            query: str

        with self.assertRaisesRegex(ValueError, "Couldn't get the JSON data"):
            _ = JsonDataModel(data_model=Query)

    def test_init_with_data_model_non_instanciated_and_value(self):
        class Query(DataModel):
            query: str

        expected_schema = {
            "additionalProperties": False,
            "properties": {"query": {"title": "Query", "type": "string"}},
            "required": ["query"],
            "title": "Query",
            "type": "object",
        }
        expected_json = {"query": "What is the capital of France?"}

        json_data_model = JsonDataModel(
            data_model=Query,
            json=expected_json,
        )

        self.assertEqual(json_data_model.get_json(), expected_json)
        self.assertEqual(json_data_model.get_schema(), expected_schema)

    def test_init_with_dict(self):
        schema = {
            "additionalProperties": False,
            "properties": {"query": {"title": "Query", "type": "string"}},
            "required": ["query"],
            "title": "Query",
            "type": "object",
        }
        value = {"query": "What is the capital of France?"}

        json_data_model = JsonDataModel(schema=schema, json=value)

        self.assertEqual(json_data_model.get_json(), value)
        self.assertEqual(json_data_model.get_schema(), schema)

    def test_representation(self):
        class Query(DataModel):
            query: str

        json_data_model = JsonDataModel(
            data_model=Query(query="What is the capital of France?")
        )

        expected_schema = {
            "additionalProperties": False,
            "properties": {"query": {"title": "Query", "type": "string"}},
            "required": ["query"],
            "title": "Query",
            "type": "object",
        }
        expected_json = {"query": "What is the capital of France?"}

        self.assertEqual(
            str(json_data_model),
            f"<JsonDataModel schema={expected_schema}, json={expected_json}>",
        )

    def test_contains_json_data_model(self):
        class Foo(DataModel):
            foo: str

        class FooBar(DataModel):
            foo: str
            bar: str

        class Bar(DataModel):
            bar: str

        foo_json = JsonDataModel(data_model=Foo(foo="a"))
        foobar_json = JsonDataModel(data_model=FooBar(foo="a", bar="b"))
        bar_json = JsonDataModel(data_model=Bar(bar="c"))

        self.assertTrue(foo_json in foobar_json)
        self.assertFalse(bar_json in foo_json)

    def test_contains_string_key_json_data_model(self):
        class FooBar(DataModel):
            foo: str
            bar: str

        foobar_json = JsonDataModel(data_model=FooBar(foo="a", bar="b"))

        self.assertTrue("foo" in foobar_json)
        self.assertTrue("bar" in foobar_json)
        self.assertFalse("baz" in foobar_json)

    def test_not_json_data_model(self):
        class Foo(DataModel):
            foo: str

        foo_json = JsonDataModel(data_model=Foo(foo="a"))

        not_foo = ~foo_json

        self.assertTrue(not_foo is None)

    def test_get_nested_entity(self):
        from typing import Literal

        from synalinks.src.backend.pydantic.knowledge import Entity
        from synalinks.src.backend.pydantic.knowledge import Relation

        class Chunk(Entity):
            label: Literal["Chunk"]
            text: str

        class Document(Entity):
            label: Literal["Document"]
            title: str

        class IsPartOf(Relation):
            label: Literal["IsPartOf"]
            subj: Chunk
            obj: Document

        rel_json = IsPartOf(
            subj=Chunk(label="Chunk", text="abc"),
            label="IsPartOf",
            obj=Document(label="Document", title="paper-1"),
        ).to_json_data_model()

        subj = rel_json.get_nested_entity("subj")
        obj = rel_json.get_nested_entity("obj")

        self.assertIsNotNone(subj)
        self.assertIsNotNone(obj)
        self.assertEqual(subj.get_schema(), Chunk.get_schema())
        self.assertEqual(obj.get_schema(), Document.get_schema())
        self.assertEqual(subj.get_json(), {"label": "Chunk", "text": "abc"})

    def test_get_nested_entity_missing_key_returns_none(self):
        from typing import Literal

        from synalinks.src.backend.pydantic.knowledge import Entity

        class Doc(Entity):
            label: Literal["Doc"]

        class Holder(DataModel):
            child: Doc

        holder_json = Holder(child=Doc(label="Doc")).to_json_data_model()
        self.assertIsNone(holder_json.get_nested_entity("missing"))

    def test_get_nested_entity_no_label_returns_none(self):
        class Inner(DataModel):
            name: str

        class Outer(DataModel):
            child: Inner

        outer_json = Outer(child=Inner(name="x")).to_json_data_model()
        self.assertIsNone(outer_json.get_nested_entity("child"))

    def test_get_nested_entity_list(self):
        from typing import List
        from typing import Literal

        from synalinks.src.backend.pydantic.knowledge import Entity

        class Chunk(Entity):
            label: Literal["Chunk"]
            text: str

        class Bag(DataModel):
            entities: List[Chunk]

        bag_json = Bag(
            entities=[
                Chunk(label="Chunk", text="a"),
                Chunk(label="Chunk", text="b"),
            ]
        ).to_json_data_model()

        items = bag_json.get_nested_entity_list("entities")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].get_schema(), Chunk.get_schema())
        self.assertEqual(items[0].get_json(), {"label": "Chunk", "text": "a"})
        self.assertEqual(items[1].get_json(), {"label": "Chunk", "text": "b"})

    def test_get_nested_entity_list_empty(self):
        from typing import List
        from typing import Literal

        from synalinks.src.backend.pydantic.knowledge import Entity

        class Chunk(Entity):
            label: Literal["Chunk"]
            text: str

        class Bag(DataModel):
            entities: List[Chunk] = []

        bag_json = Bag(entities=[]).to_json_data_model()
        self.assertEqual(bag_json.get_nested_entity_list("entities"), [])

    def test_get_nested_entity_list_free_form_labels(self):
        """Free-form: the items' ``label`` values aren't $defs keys (the
        class is generic ``Node``, labels are open data), so resolution
        falls back to the field's declared item schema."""
        from typing import List

        from synalinks.src.backend.pydantic.knowledge import Entity
        from synalinks.src.backend.pydantic.knowledge import KnowledgeGraph

        class Node(Entity):
            name: str

        class FreeGraph(KnowledgeGraph):
            entities: List[Node]

        graph_json = FreeGraph(
            entities=[
                Node(label="Person", name="Ada"),
                Node(label="Field", name="Computing"),
            ],
            relations=[],
        ).to_json_data_model()

        items = graph_json.get_nested_entity_list("entities")
        self.assertEqual(len(items), 2)
        # Open label survives as data; the Node schema (with ``name``)
        # is used even though "Person"/"Field" aren't $defs keys.
        self.assertEqual(items[0].get_json(), {"label": "Person", "name": "Ada"})
        self.assertEqual(items[1].get_json(), {"label": "Field", "name": "Computing"})

    def test_get_nested_entity_free_form_label(self):
        """Single nested entity with an open label falls back to the
        field's declared $ref schema."""
        from synalinks.src.backend.pydantic.knowledge import Entity
        from synalinks.src.backend.pydantic.knowledge import Relation

        class Node(Entity):
            name: str

        class Edge(Relation):
            subj: Node
            obj: Node

        rel_json = Edge(
            subj=Node(label="Person", name="Ada"),
            label="PIONEER_OF",
            obj=Node(label="Field", name="Computing"),
        ).to_json_data_model()

        subj = rel_json.get_nested_entity("subj")
        obj = rel_json.get_nested_entity("obj")
        self.assertEqual(subj.get_json(), {"label": "Person", "name": "Ada"})
        self.assertEqual(obj.get_json(), {"label": "Field", "name": "Computing"})
