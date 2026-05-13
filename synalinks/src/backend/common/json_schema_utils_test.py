# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Literal
from typing import Union

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import standardize_schema
from synalinks.src.backend.common.json_schema_utils import _py_concatenate_schema
from synalinks.src.backend.common.json_schema_utils import _py_factorize_schema
from synalinks.src.backend.common.json_schema_utils import _py_in_mask_schema
from synalinks.src.backend.common.json_schema_utils import _py_out_mask_schema
from synalinks.src.backend.common.json_schema_utils import _py_prefix_schema
from synalinks.src.backend.common.json_schema_utils import _py_suffix_schema
from synalinks.src.backend.common.json_schema_utils import concatenate_schema
from synalinks.src.backend.common.json_schema_utils import contains_schema
from synalinks.src.backend.common.json_schema_utils import factorize_schema
from synalinks.src.backend.common.json_schema_utils import in_mask_schema
from synalinks.src.backend.common.json_schema_utils import is_array
from synalinks.src.backend.common.json_schema_utils import is_object
from synalinks.src.backend.common.json_schema_utils import is_schema_equal
from synalinks.src.backend.common.json_schema_utils import out_mask_schema
from synalinks.src.backend.common.json_schema_utils import prefix_schema
from synalinks.src.backend.common.json_schema_utils import suffix_schema


class JsonSchemaConcatenateTest(testing.TestCase):
    def test_concatenate_identical_schemas(self):
        class Input(DataModel):
            foo: str

        class Result(DataModel):
            foo: str
            foo_1: str

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = concatenate_schema(schema, schema)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_concatenate_schemas_with_different_properties(self):
        class Input1(DataModel):
            foo: str

        class Input2(DataModel):
            bar: str

        class Result(DataModel):
            foo: str
            bar: str

        schema1 = standardize_schema(Input1.get_schema())
        schema2 = standardize_schema(Input2.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = concatenate_schema(schema1, schema2)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_concatenate_similar_entities(self):
        class City(DataModel):
            label: Literal["City"]
            name: str

        class Cities(DataModel):
            entities: List[City]

        class Result(DataModel):
            entities: List[City]
            entities_1: List[City]

        schema1 = standardize_schema(Cities.get_schema())
        schema2 = standardize_schema(Cities.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = concatenate_schema(schema1, schema2)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_concatenate_different_entities(self):
        class City(DataModel):
            label: Literal["City"]
            name: str

        class Event(DataModel):
            label: Literal["Event"]
            name: str

        class Cities(DataModel):
            entities: List[City]

        class Events(DataModel):
            entities: List[Event]

        class Result(DataModel):
            entities: List[City]
            entities_1: List[Event]

        schema1 = standardize_schema(Cities.get_schema())
        schema2 = standardize_schema(Events.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = concatenate_schema(schema1, schema2)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_concatenate_schema_multiple_times(self):
        class Input(DataModel):
            foo: str

        class Result(DataModel):
            foo: str
            foo_1: str
            foo_2: str

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = concatenate_schema(schema, schema)
        result_schema = concatenate_schema(result_schema, schema)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))


class JsonSchemaFactorizeTest(testing.TestCase):
    def test_factorize_schema_with_identical_properties(self):
        class Input(DataModel):
            foo: str
            foo_1: str

        class Result(DataModel):
            foos: List[str]

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = factorize_schema(schema)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_factorize_schema_with_multiple_identical_properties(self):
        class Input(DataModel):
            foo: str
            foo_1: str
            foo_2: str

        class Result(DataModel):
            foos: List[str]

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = factorize_schema(schema)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_factorize_schema_with_different_properties(self):
        class Input(DataModel):
            foo: str
            foo_1: str
            foo_2: str

        class Result(DataModel):
            foos: List[str]

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = factorize_schema(schema)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_factorize_schema_with_mixed_properties(self):
        class Input(DataModel):
            foo: str
            foo_1: str
            bar: str

        class Result(DataModel):
            foos: List[str]
            bar: str

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = factorize_schema(schema)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_factorize_schema_with_existing_array_property(self):
        class Input(DataModel):
            foos: List[str]
            foo: str

        class Result(DataModel):
            foos: List[str]

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = factorize_schema(schema)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_factorize_schema_with_existing_array_property_and_additional_properties(
        self,
    ):
        class Input(DataModel):
            foos: List[str]
            foo: str
            foo_1: str

        class Result(DataModel):
            foos: List[str]

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = factorize_schema(schema)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_factorize_schema_with_multiple_groups_of_properties(self):
        class Input(DataModel):
            foo: str
            foo_1: str
            bar: str
            bar_1: str

        class Result(DataModel):
            foos: List[str]
            bars: List[str]

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = factorize_schema(schema)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_factorize_similar_entities(self):
        class City(DataModel):
            label: Literal["City"]
            name: str

        class Input(DataModel):
            entities: List[City]
            entities_1: List[City]

        class Result(DataModel):
            entities: List[City]

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = factorize_schema(schema)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_factorize_different_entities(self):
        class City(DataModel):
            label: Literal["City"]
            name: str

        class Event(DataModel):
            label: Literal["Event"]
            name: str

        class Input(DataModel):
            entities: List[City]
            entities_1: List[Event]

        class Result(DataModel):
            entities: List[Union[City, Event]]

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = factorize_schema(schema)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))


class JsonSchemaOutMaskTest(testing.TestCase):
    def test_mask_basic(self):
        class Input(DataModel):
            foo: str
            bar: str

        class Result(DataModel):
            bar: str

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = out_mask_schema(schema, mask=["foo"])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_mask_multiple_fields_with_same_base_name(self):
        class Input(DataModel):
            foo: str
            foo_1: str
            bar: str
            bar_1: str

        class Result(DataModel):
            bar: str
            bar_1: str

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = out_mask_schema(schema, mask=["foo"])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_mask_nested(self):
        class BarObject(DataModel):
            foo: str
            bar: str

        class Input(DataModel):
            foo: str
            foo_1: str
            bar: BarObject

        schema = standardize_schema(Input.get_schema())

        class BarObject(DataModel):
            bar: str

        class Result(DataModel):
            bar: BarObject

        expected_schema = standardize_schema(Result.get_schema())
        result_schema = out_mask_schema(schema, mask=["foo"])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_mask_deeply_nested(self):
        class BooObject(DataModel):
            foo: str
            boo: str

        class BarObject(DataModel):
            boo: BooObject

        class Input(DataModel):
            foo: str
            bar: BarObject

        schema = standardize_schema(Input.get_schema())

        class BooObject(DataModel):
            boo: str

        class BarObject(DataModel):
            boo: BooObject

        class Result(DataModel):
            bar: BarObject

        expected_schema = standardize_schema(Result.get_schema())
        result_schema = out_mask_schema(schema, mask=["foo"])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_mask_array(self):
        class BarObject(DataModel):
            foo: str
            bar: str

        class Input(DataModel):
            bars: List[BarObject]

        schema = standardize_schema(Input.get_schema())

        class BarObject(DataModel):
            bar: str

        class Result(DataModel):
            bars: List[BarObject]

        expected_schema = standardize_schema(Result.get_schema())

        result_schema = out_mask_schema(schema, mask=["foo"])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_mask_empty_schema(self):
        class Input(DataModel):
            pass

        class Result(DataModel):
            pass

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = out_mask_schema(schema, mask=["foo"])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_empty_mask_list(self):
        class Input(DataModel):
            foo: str
            bar: str

        class Result(DataModel):
            foo: str
            bar: str

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = out_mask_schema(schema, mask=[])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_mask_non_recursive(self):
        class BarObject(DataModel):
            foo: str
            bar: str

        class Input(DataModel):
            foo: str
            foo_1: str
            bar: BarObject

        class Result(DataModel):
            bar: BarObject

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = out_mask_schema(schema, mask=["foo"], recursive=False)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))


class JsonSchemaInMaskTest(testing.TestCase):
    def test_mask_basic(self):
        class Input(DataModel):
            foo: str
            bar: str

        class Result(DataModel):
            foo: str

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = in_mask_schema(schema, mask=["foo"])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_mask_multiple_fields_with_same_base_name(self):
        class Input(DataModel):
            foo: str
            foo_1: str
            bar: str
            bar_1: str

        class Result(DataModel):
            foo: str
            foo_1: str

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = in_mask_schema(schema, mask=["foo"])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_mask_nested(self):
        class BarObject(DataModel):
            foo: str
            bar: str
            boo: str

        class Input(DataModel):
            foo: str
            foo_1: str
            bar: BarObject
            boo: str

        schema = standardize_schema(Input.get_schema())

        class BarObject(DataModel):
            foo: str
            bar: str

        class Result(DataModel):
            foo: str
            foo_1: str
            bar: BarObject

        expected_schema = standardize_schema(Result.get_schema())
        result_schema = in_mask_schema(schema, mask=["foo", "bar"])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_mask_deeply_nested(self):
        class BooObject(DataModel):
            foo: str
            boo: str

        class BarObject(DataModel):
            boo: BooObject

        class Input(DataModel):
            foo: str
            bar: BarObject

        schema = standardize_schema(Input.get_schema())

        class BooObject(DataModel):
            foo: str

        class BarObject(DataModel):
            pass

        class Result(DataModel):
            foo: str
            bar: BarObject

        expected_schema = standardize_schema(Result.get_schema())
        result_schema = in_mask_schema(schema, mask=["foo", "bar"])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_mask_array(self):
        class BarObject(DataModel):
            foo: str
            bar: str

        class Input(DataModel):
            bars: List[BarObject]

        schema = standardize_schema(Input.get_schema())

        class BarObject(DataModel):
            bar: str

        class Result(DataModel):
            bars: List[BarObject]

        expected_schema = standardize_schema(Result.get_schema())

        result_schema = in_mask_schema(schema, mask=["bar"])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_mask_empty_schema(self):
        class Input(DataModel):
            pass

        class Result(DataModel):
            pass

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = in_mask_schema(schema, mask=["foo"])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_mask_empty_mask_list(self):
        class Input(DataModel):
            foo: str
            bar: str

        class Result(DataModel):
            pass

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = in_mask_schema(schema, mask=[])
        self.assertTrue(is_schema_equal(result_schema, expected_schema))

    def test_mask_non_recursive(self):
        class BarObject(DataModel):
            foo: str
            bar: str
            boo: str

        class Input(DataModel):
            foo: str
            foo_1: str
            bar: BarObject
            boo: str

        class Result(DataModel):
            foo: str
            foo_1: str
            bar: BarObject

        schema = standardize_schema(Input.get_schema())
        expected_schema = standardize_schema(Result.get_schema())

        result_schema = in_mask_schema(schema, mask=["foo", "bar"], recursive=False)
        self.assertTrue(is_schema_equal(result_schema, expected_schema))


class JsonSchemaContainsTest(testing.TestCase):
    def test_contains_same_schema(self):
        class Input1(DataModel):
            foo: str
            bar: str

        class Input2(DataModel):
            foo: str
            bar: str

        schema1 = standardize_schema(Input1.get_schema())
        schema2 = standardize_schema(Input2.get_schema())

        self.assertTrue(contains_schema(schema1, schema2))

    def test_contains_subset_schema(self):
        class Input1(DataModel):
            foo: str
            bar: str

        class Input2(DataModel):
            foo: str

        schema1 = standardize_schema(Input1.get_schema())
        schema2 = standardize_schema(Input2.get_schema())

        self.assertTrue(contains_schema(schema1, schema2))

    def test_contains_different_schema(self):
        class Input1(DataModel):
            bar: str

        class Input2(DataModel):
            foo: str

        schema1 = standardize_schema(Input1.get_schema())
        schema2 = standardize_schema(Input2.get_schema())

        self.assertFalse(contains_schema(schema1, schema2))


class JsonSchemaPrefixSuffixTest(testing.TestCase):
    def test_prefix_schema(self):
        class Input(DataModel):
            query: str
            answer: str

        schema = standardize_schema(Input.get_schema())
        result = prefix_schema(schema, prefix="user")
        self.assertIn("user_query", result["properties"])
        self.assertIn("user_answer", result["properties"])
        self.assertEqual(set(result["required"]), {"user_query", "user_answer"})
        self.assertEqual(result["properties"]["user_query"]["title"], "User Query")

    def test_suffix_schema(self):
        class Input(DataModel):
            query: str
            answer: str

        schema = standardize_schema(Input.get_schema())
        result = suffix_schema(schema, suffix="raw")
        self.assertIn("query_raw", result["properties"])
        self.assertIn("answer_raw", result["properties"])
        self.assertEqual(result["properties"]["query_raw"]["title"], "Query Raw")

    def test_py_prefix_schema_uses_key_as_title_when_missing(self):
        schema = {"properties": {"a_b": {"type": "string"}}, "required": ["a_b"]}
        result = _py_prefix_schema(schema, prefix="x")
        self.assertEqual(result["properties"]["x_a_b"]["title"], "X A B")

    def test_py_suffix_schema_uses_key_as_title_when_missing(self):
        schema = {"properties": {"a_b": {"type": "string"}}, "required": ["a_b"]}
        result = _py_suffix_schema(schema, suffix="y")
        self.assertEqual(result["properties"]["a_b_y"]["title"], "A B Y")


class JsonSchemaPredicateTest(testing.TestCase):
    def test_is_object_true(self):
        self.assertTrue(is_object({"type": "object"}))

    def test_is_object_false(self):
        self.assertFalse(is_object({"type": "array"}))
        self.assertFalse(is_object({"properties": {}}))
        self.assertFalse(is_object("not a dict"))

    def test_is_array_true(self):
        self.assertTrue(is_array({"type": "array"}))

    def test_is_array_false(self):
        self.assertFalse(is_array({"type": "object"}))
        self.assertFalse(is_array(None))


class JsonSchemaUnequalLengthTest(testing.TestCase):
    def test_is_schema_equal_different_lengths(self):
        class Input1(DataModel):
            foo: str

        class Input2(DataModel):
            foo: str
            bar: str

        schema1 = standardize_schema(Input1.get_schema())
        schema2 = standardize_schema(Input2.get_schema())
        self.assertFalse(is_schema_equal(schema1, schema2))


# The "_py_*" tests below explicitly exercise the Python fallback path, even
# when the Rust `synaops` extension is installed.
class PyConcatenateSchemaTest(testing.TestCase):
    def test_py_concatenate_basic(self):
        s1 = {
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "title": "S1",
            "type": "object",
        }
        s2 = {
            "properties": {"b": {"type": "string"}},
            "required": ["b"],
            "title": "S2",
            "type": "object",
        }
        result = _py_concatenate_schema(s1, s2)
        self.assertIn("a", result["properties"])
        self.assertIn("b", result["properties"])
        self.assertEqual(result["required"], ["a", "b"])

    def test_py_concatenate_collision_renames(self):
        schema = {
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "title": "S",
            "type": "object",
        }
        result = _py_concatenate_schema(schema, schema)
        self.assertIn("a", result["properties"])
        self.assertIn("a_1", result["properties"])
        self.assertEqual(result["required"], ["a", "a_1"])

    def test_py_concatenate_merges_defs(self):
        s1 = {
            "$defs": {"X": {"type": "object"}},
            "properties": {},
            "required": [],
            "type": "object",
        }
        s2 = {
            "$defs": {"Y": {"type": "object"}},
            "properties": {},
            "required": [],
            "type": "object",
        }
        result = _py_concatenate_schema(s1, s2)
        self.assertIn("X", result["$defs"])
        self.assertIn("Y", result["$defs"])

    def test_py_concatenate_only_one_side_has_defs(self):
        s1 = {"$defs": {"X": {"type": "object"}}, "properties": {}, "type": "object"}
        s2 = {"properties": {}, "type": "object"}
        result = _py_concatenate_schema(s1, s2)
        self.assertIn("X", result["$defs"])

    def test_py_concatenate_drops_defs_when_empty(self):
        s1 = {"properties": {}, "type": "object"}
        s2 = {"properties": {}, "type": "object"}
        result = _py_concatenate_schema(s1, s2)
        self.assertNotIn("$defs", result)

    def test_py_concatenate_required_when_only_second_has_it(self):
        s1 = {"properties": {"a": {"type": "string"}}, "type": "object"}
        s2 = {
            "properties": {"b": {"type": "string"}},
            "required": ["b"],
            "type": "object",
        }
        result = _py_concatenate_schema(s1, s2)
        self.assertIn("b", result["required"])
        self.assertNotIn("a", result["required"])


class PyFactorizeSchemaTest(testing.TestCase):
    def test_py_factorize_groups_scalars(self):
        schema = {
            "properties": {
                "foo": {"type": "string"},
                "foo_1": {"type": "string"},
            },
            "required": ["foo", "foo_1"],
            "type": "object",
        }
        result = _py_factorize_schema(schema)
        self.assertIn("foos", result["properties"])
        self.assertEqual(result["properties"]["foos"]["type"], "array")
        self.assertIn("foos", result["required"])

    def test_py_factorize_groups_arrays_with_compatible_items(self):
        schema = {
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
                "tags_1": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tags", "tags_1"],
            "type": "object",
        }
        result = _py_factorize_schema(schema)
        self.assertEqual(result["properties"]["tags"]["items"], {"type": "string"})

    def test_py_factorize_keeps_singular_when_no_group(self):
        schema = {
            "properties": {"only": {"type": "string"}},
            "required": ["only"],
            "type": "object",
        }
        result = _py_factorize_schema(schema)
        self.assertEqual(result["properties"], {"only": {"type": "string"}})

    def test_py_factorize_drops_defs_when_empty(self):
        schema = {
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "type": "object",
        }
        result = _py_factorize_schema(schema)
        self.assertNotIn("$defs", result)

    def test_py_factorize_property_without_type_falls_back_to_string(self):
        schema = {
            "properties": {
                "thing": {"description": "no-type"},
                "thing_1": {"description": "no-type"},
            },
            "required": ["thing", "thing_1"],
            "type": "object",
        }
        result = _py_factorize_schema(schema)
        self.assertEqual(result["properties"]["things"]["items"], {"type": "string"})


class PyMaskSchemaTest(testing.TestCase):
    def test_py_out_mask_no_args_returns_input(self):
        schema = {"properties": {"a": {"type": "string"}}, "type": "object"}
        result = _py_out_mask_schema(schema)
        self.assertEqual(result, schema)

    def test_py_in_mask_no_args_returns_empty_skeleton(self):
        schema = {
            "properties": {"a": {"type": "string"}},
            "title": "S",
            "type": "object",
        }
        result = _py_in_mask_schema(schema)
        self.assertEqual(result["properties"], {})
        self.assertEqual(result["title"], "S")

    def test_py_out_mask_with_pattern(self):
        schema = {
            "properties": {
                "input_query": {"type": "string"},
                "output_answer": {"type": "string"},
            },
            "required": ["input_query", "output_answer"],
            "type": "object",
        }
        result = _py_out_mask_schema(schema, pattern="^input_")
        self.assertNotIn("input_query", result["properties"])
        self.assertIn("output_answer", result["properties"])
        self.assertEqual(result["required"], ["output_answer"])

    def test_py_in_mask_with_pattern(self):
        schema = {
            "properties": {
                "input_query": {"type": "string"},
                "output_answer": {"type": "string"},
            },
            "required": ["input_query", "output_answer"],
            "type": "object",
        }
        result = _py_in_mask_schema(schema, pattern="^input_")
        self.assertIn("input_query", result["properties"])
        self.assertNotIn("output_answer", result["properties"])

    def test_py_in_mask_drops_required_when_empty(self):
        schema = {
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
            "type": "object",
        }
        result = _py_in_mask_schema(schema, mask=["b"])
        self.assertNotIn("required", result)

    def test_py_out_mask_cleans_unused_defs(self):
        schema = {
            "$defs": {"Used": {"type": "object"}, "Unused": {"type": "object"}},
            "properties": {
                "thing": {"$ref": "#/$defs/Used"},
                "drop": {"$ref": "#/$defs/Unused"},
            },
            "required": ["thing", "drop"],
            "type": "object",
        }
        result = _py_out_mask_schema(schema, mask=["drop"])
        self.assertIn("Used", result["$defs"])
        self.assertNotIn("Unused", result["$defs"])
