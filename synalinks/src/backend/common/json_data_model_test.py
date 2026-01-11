# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

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
