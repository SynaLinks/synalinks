# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend.common.json_schema_utils import is_schema_equal
from synalinks.src.backend.common.json_schema_utils import standardize_schema
from synalinks.src.backend.pydantic.core import DataModel
from synalinks.src.backend.pydantic.core import is_meta_class


class CoreTest(testing.TestCase):
    def test_non_instanciated_data_model_schema(self):
        class TestDataModel(DataModel):
            foo: str

        expected_schema = {
            "additionalProperties": False,
            "properties": {"foo": {"title": "Foo", "type": "string"}},
            "required": ["foo"],
            "title": "TestDataModel",
            "type": "object",
        }

        self.assertEqual(TestDataModel.get_schema(), expected_schema)

    def test_instanciated_data_model_schema(self):
        class TestDataModel(DataModel):
            foo: str

        expected_schema = {
            "additionalProperties": False,
            "properties": {"foo": {"title": "Foo", "type": "string"}},
            "required": ["foo"],
            "title": "TestDataModel",
            "type": "object",
        }

        self.assertEqual(TestDataModel(foo="bar").get_schema(), expected_schema)

    def test_instanciated_data_model_json(self):
        class TestDataModel(DataModel):
            foo: str

        expected_json = {"foo": "bar"}

        self.assertEqual(TestDataModel(foo="bar").get_json(), expected_json)

    def test_concatenate_meta_data_model(self):
        class Foo(DataModel):
            foo: str

        class Bar(DataModel):
            bar: str

        class Result(DataModel):
            foo: str
            bar: str

        x = Foo + Bar

        schema = x.get_schema()
        expected_schema = standardize_schema(Result.get_schema())
        self.assertTrue(is_schema_equal(schema, expected_schema))

    def test_and_meta_data_model(self):
        class Foo(DataModel):
            foo: str

        class Bar(DataModel):
            bar: str

        class Result(DataModel):
            foo: str
            bar: str

        x = Foo & Bar

        schema = x.get_schema()
        expected_schema = standardize_schema(Result.get_schema())
        self.assertTrue(is_schema_equal(schema, expected_schema))

    def test_or_meta_data_model(self):
        class Foo(DataModel):
            foo: str

        class Bar(DataModel):
            bar: str

        class Result(DataModel):
            foo: str
            bar: str

        x = Foo | Bar

        schema = x.get_schema()
        expected_schema = standardize_schema(Foo.get_schema())
        self.assertEqual(schema, expected_schema)

    def test_xor_meta_data_model(self):
        class Foo(DataModel):
            foo: str

        class Bar(DataModel):
            bar: str

        class Result(DataModel):
            foo: str
            bar: str

        x = Foo ^ Bar

        schema = x.get_schema()
        expected_schema = standardize_schema(Foo.get_schema())
        self.assertEqual(schema, expected_schema)

    def test_not_meta_data_model(self):
        class Foo(DataModel):
            foo: str

        x = ~Foo

        schema = x.get_schema()
        expected_schema = standardize_schema(Foo.get_schema())
        self.assertEqual(schema, expected_schema)

    def test_is_meta_class(self):
        class Query(DataModel):
            query: str

        self.assertTrue(is_meta_class(Query))

    def test_is_not_meta_class(self):
        class Query(DataModel):
            query: str

        self.assertFalse(is_meta_class(Query(query="What is the French capital?")))

    def test_contains_meta_class(self):
        class Foo(DataModel):
            foo: str

        class FooBar(DataModel):
            foo: str
            bar: str

        class Bar(DataModel):
            bar: str

        self.assertTrue(Foo in FooBar)
        self.assertFalse(Bar in Foo)

    def test_contains_string_key_meta_class(self):
        class FooBar(DataModel):
            foo: str
            bar: str

        self.assertTrue("foo" in FooBar)
        self.assertTrue("bar" in FooBar)
        self.assertFalse("baz" in FooBar)

    def test_contains_data_model_instance(self):
        class Foo(DataModel):
            foo: str

        class FooBar(DataModel):
            foo: str
            bar: str

        class Bar(DataModel):
            bar: str

        foo_instance = Foo(foo="a")
        foobar_instance = FooBar(foo="a", bar="b")
        bar_instance = Bar(bar="c")

        self.assertTrue(foo_instance in foobar_instance)
        self.assertFalse(bar_instance in foo_instance)

    def test_contains_string_key_data_model_instance(self):
        class FooBar(DataModel):
            foo: str
            bar: str

        foobar_instance = FooBar(foo="a", bar="b")

        self.assertTrue("foo" in foobar_instance)
        self.assertTrue("bar" in foobar_instance)
        self.assertFalse("baz" in foobar_instance)

    def test_get_nested_entity_on_data_model_instance(self):
        from typing import Literal

        from synalinks.src.backend.pydantic.base import Entity
        from synalinks.src.backend.pydantic.base import Relation

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

        rel = IsPartOf(
            subj=Chunk(label="Chunk", text="abc"),
            label="IsPartOf",
            obj=Document(label="Document", title="paper-1"),
        )

        subj = rel.get_nested_entity("subj")
        obj = rel.get_nested_entity("obj")

        self.assertIsNotNone(subj)
        self.assertIsNotNone(obj)
        self.assertEqual(subj.get_schema(), Chunk.get_schema())
        self.assertEqual(obj.get_schema(), Document.get_schema())

    def test_get_nested_entity_list_on_data_model_instance(self):
        from typing import List
        from typing import Literal

        from synalinks.src.backend.pydantic.base import Entity

        class Chunk(Entity):
            label: Literal["Chunk"]
            text: str

        class Bag(DataModel):
            entities: List[Chunk]

        bag = Bag(
            entities=[
                Chunk(label="Chunk", text="a"),
                Chunk(label="Chunk", text="b"),
            ]
        )

        items = bag.get_nested_entity_list("entities")
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].get_schema(), Chunk.get_schema())
        self.assertEqual(
            [i.get_json() for i in items],
            [{"label": "Chunk", "text": "a"}, {"label": "Chunk", "text": "b"}],
        )
