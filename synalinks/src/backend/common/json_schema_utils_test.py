# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List
from typing import Literal
from typing import Union

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import standardize_schema
from synalinks.src.backend.common.json_schema_utils import concatenate_schema
from synalinks.src.backend.common.json_schema_utils import contains_schema
from synalinks.src.backend.common.json_schema_utils import factorize_schema
from synalinks.src.backend.common.json_schema_utils import in_mask_schema
from synalinks.src.backend.common.json_schema_utils import is_schema_equal
from synalinks.src.backend.common.json_schema_utils import out_mask_schema


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
