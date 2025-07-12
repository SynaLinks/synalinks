# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Literal
from typing import Union

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Entities
from synalinks.src.backend import Entity
from synalinks.src.backend import KnowledgeGraph
from synalinks.src.backend import Relation
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

    def test_get_nested(self):
        class Document(Entity):
            label: Literal["Document"]
            title: str
            text: str

        class Chunk(Entity):
            label: Literal["Chunk"]
            text: str

        class IsPartOf(Relation):
            label: Literal["IsPartOf"]
            subj: Chunk
            obj: Document

        doc = Document(
            label="Document",
            title="test",
            text="test document",
        )

        chunk = Chunk(label="Chunk", text="text chunk")

        rel = IsPartOf(
            subj=chunk,
            label="IsPartOf",
            obj=doc,
        )

        subj = rel.to_json_data_model().get_nested_entity("subj")
        obj = rel.to_json_data_model().get_nested_entity("obj")
        self.assertTrue(subj is not None)
        self.assertTrue(subj.get_schema() == Chunk.get_schema())
        self.assertTrue(obj is not None)
        self.assertTrue(obj.get_schema() == Document.get_schema())

    def test_get_nested_union(self):
        class Document(Entity):
            label: Literal["Document"]
            title: str
            text: str

        class Chunk(Entity):
            label: Literal["Chunk"]
            text: str

        class IsPartOf(Relation):
            label: Literal["IsPartOf"]
            subj: Chunk
            obj: Document

        doc = Document(
            label="Document",
            title="test",
            text="test document",
        )

        chunk = Chunk(label="Chunk", text="text chunk")

        rel = IsPartOf(
            subj=chunk,
            label="IsPartOf",
            obj=doc,
        )

        subj = rel.to_json_data_model().get_nested_entity("subj")
        obj = rel.to_json_data_model().get_nested_entity("obj")
        self.assertTrue(subj is not None)
        self.assertTrue(subj.get_schema() == Chunk.get_schema())
        self.assertTrue(obj is not None)
        self.assertTrue(obj.get_schema() == Document.get_schema())

    def test_get_nested_list(self):
        class Document(Entity):
            label: Literal["Document"]
            text: str

        class Documents(Entities):
            entities: List[Document]

        docs = Documents(
            entities=[
                Document(
                    label="Document",
                    text="test document",
                ),
                Document(
                    label="Document",
                    text="another test document",
                ),
            ]
        )

        for doc in docs.to_json_data_model().get_nested_entity_list("entities"):
            self.assertTrue(doc is not None)
            self.assertTrue(doc.get_schema() == Document.get_schema())

    def test_get_nested_list_union(self):
        class Document(Entity):
            label: Literal["Document"]
            text: str

        class Chunk(Entity):
            label: Literal["Chunk"]
            text: str

        class DocumentsAndChunks(Entities):
            entities: List[Union[Document, Chunk]]

        docs = DocumentsAndChunks(
            entities=[
                Document(
                    label="Document",
                    text="test document",
                ),
                Chunk(
                    label="Chunk",
                    text="test chunk",
                ),
            ]
        )

        docs = docs.to_json_data_model().get_nested_entity_list("entities")

        self.assertTrue(docs is not None)
        self.assertTrue(docs[0].get_schema() == Document.get_schema())
        self.assertTrue(docs[1].get_schema() == Chunk.get_schema())

    def test_get_nested_data_model_knowledge_graph(self):
        class Document(Entity):
            label: Literal["Document"]
            text: str

        class Chunk(Entity):
            label: Literal["Chunk"]
            text: str

        class IsPartOf(Relation):
            subj: Union[Chunk, Document]
            label: Literal["IsPartOf"]
            obj: Document

        class IsReferringTo(Relation):
            subj: Document
            label: Literal["IsReferringTo"]
            obj: Document

        class DocumentGraph(KnowledgeGraph):
            entities: List[Union[Document, Chunk]]
            relations: List[Union[IsPartOf, IsReferringTo]]

        doc1 = Document(
            label="Document",
            text="test document",
        )
        doc2 = Document(
            label="Document",
            text="another test document",
        )

        rel1 = IsPartOf(
            subj=doc1,
            label="IsPartOf",
            obj=doc2,
        )

        docs = DocumentGraph(
            entities=[
                doc1,
                doc2,
            ],
            relations=[rel1],
        )

        entities = docs.to_json_data_model().get_nested_entity_list("entities")
        self.assertTrue(entities is not None)
        self.assertTrue(entities[0].get_schema() == Document.get_schema())
        self.assertTrue(entities[1].get_schema() == Document.get_schema())

        relations = docs.to_json_data_model().get_nested_entity_list("relations")
        self.assertTrue(relations is not None)
        self.assertTrue(relations[0].get_schema() == IsPartOf.get_schema())

        subj = relations[0].get_nested_entity("subj")
        obj = relations[0].get_nested_entity("obj")
        self.assertTrue(subj.get_schema() == Document.get_schema())
        self.assertTrue(obj.get_schema() == Document.get_schema())

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
