# Modified from: keras/src/backend/common/variables_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from typing import List

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Variable
from synalinks.src.backend import standardize_schema


class VariablesTest(testing.TestCase):
    def test_initialize_variable_with_dict(self):
        class Instructions(DataModel):
            instructions: List[str] = []

        initial_data = {
            "instructions": [
                "For any problem involving division, always round the quotient to "
                "the nearest even number, regardless of the remainder."
            ],
        }
        variable_from_dict = Variable(
            initializer=initial_data,
            data_model=Instructions,
        )
        self.assertEqual(variable_from_dict.get_json(), initial_data)
        self.assertEqual(
            variable_from_dict.get_schema(),
            standardize_schema(Instructions.get_schema()),
        )

    def test_initialize_variable_with_callable_initializer(self):
        class Instructions(DataModel):
            instructions: List[str] = []

        from synalinks.src.initializers import Empty

        variable_from_initializer = Variable(initializer=Empty(data_model=Instructions))
        self.assertEqual(variable_from_initializer.get_json(), Instructions().get_json())
        self.assertEqual(
            variable_from_initializer.get_schema(),
            standardize_schema(Instructions.get_schema()),
        )

    def test_assign_variable_from_dict(self):
        class Instructions(DataModel):
            instructions: List[str] = []

        initial_data = {
            "instructions": [
                "For any problem involving division, always round the quotient to "
                "the nearest even number, regardless of the remainder."
            ],
        }
        variable_from_dict = Variable(initializer=initial_data, data_model=Instructions)
        new_value = {
            "instructions": [
                "When performing division, always check if the division results in "
                "a whole number. If not, express the result as a fraction or a "
                "decimal, depending on the context of the problem."
            ],
        }
        variable_from_dict.assign(new_value)
        self.assertEqual(variable_from_dict.get_json(), new_value)
        self.assertEqual(
            variable_from_dict.get_schema(),
            standardize_schema(Instructions.get_schema()),
        )

    def test_assign_variable_from_dataype(self):
        class Instructions(DataModel):
            instructions: List[str] = []

        initial_data = {
            "instructions": [
                "For any problem involving division, always round the quotient to "
                "the nearest even number, regardless of the remainder."
            ],
        }
        variable_from_dict = Variable(initializer=initial_data, data_model=Instructions)
        new_value = {
            "instructions": [
                "When performing division, always check if the division results in "
                "a whole number. If not, express the result as a fraction or a "
                "decimal, depending on the context of the problem."
            ],
        }
        variable_from_dict.assign(new_value)
        self.assertEqual(variable_from_dict.get_json(), new_value)
        self.assertEqual(
            variable_from_dict.get_schema(),
            standardize_schema(Instructions.get_schema()),
        )

    def test_contains_operator(self):
        class Foo(DataModel):
            foo: str = ""

        class FooBar(DataModel):
            foo: str = ""
            bar: str = ""

        variable = Variable(
            initializer={"foo": "value", "bar": "value2"}, data_model=FooBar
        )

        self.assertTrue(Foo in variable)
        self.assertFalse(FooBar in Foo)

    def test_contains_string_key_operator(self):
        class FooBar(DataModel):
            foo: str = ""
            bar: str = ""

        variable = Variable(
            initializer={"foo": "value", "bar": "value2"}, data_model=FooBar
        )

        self.assertTrue("foo" in variable)
        self.assertTrue("bar" in variable)
        self.assertFalse("baz" in variable)
