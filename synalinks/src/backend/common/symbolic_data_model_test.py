# Modified from: keras/src/backend/common/keras_tensor_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import standardize_schema
from synalinks.src.backend.common import SymbolicDataModel


class SymbolicDataModelTest(testing.TestCase):
    def test_constructor_with_schema(self):
        class Query(DataModel):
            query: str

        x = SymbolicDataModel(schema=Query.get_schema())
        self.assertEqual(
            x.get_schema(),
            standardize_schema(Query.get_schema()),
        )

    def test_constructor_with_datatype(self):
        class Query(DataModel):
            query: str

        x = SymbolicDataModel(data_model=Query)
        self.assertEqual(
            x.get_schema(),
            standardize_schema(Query.get_schema()),
        )

    def test_constructor_without_args(self):
        with self.assertRaisesRegex(
            ValueError,
            "You should specify at least one argument between `data_model` or `schema`",
        ):
            _ = SymbolicDataModel()

    def test_representation(self):
        class Query(DataModel):
            query: str

        x = SymbolicDataModel(schema=Query.get_schema())
        self.assertIn(
            f"<SymbolicDataModel schema={standardize_schema(Query.get_schema())}",
            repr(x),
        )

    def test_contains_symbolic_data_model(self):
        class Foo(DataModel):
            foo: str

        class FooBar(DataModel):
            foo: str
            bar: str

        class Bar(DataModel):
            bar: str

        foo_symbolic = SymbolicDataModel(data_model=Foo)
        foobar_symbolic = SymbolicDataModel(data_model=FooBar)
        bar_symbolic = SymbolicDataModel(data_model=Bar)

        self.assertTrue(foo_symbolic in foobar_symbolic)
        self.assertFalse(bar_symbolic in foo_symbolic)
