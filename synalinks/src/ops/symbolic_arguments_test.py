# Modified from: keras/src/ops/symbolic_arguments_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.ops.symbolic_arguments import SymbolicArguments


class SymbolicArgumentsTest(testing.TestCase):
    # Testing multiple args and empty kwargs
    def test_args(self):
        class Test(DataModel):
            foo1: str
            foo2: str

        a = SymbolicDataModel(data_model=Test)
        b = SymbolicDataModel(data_model=Test)
        args = SymbolicArguments(
            (
                a,
                b,
            ),
            {},
        )

        self.assertEqual(args.symbolic_data_models, [a, b])
        self.assertEqual(args._flat_arguments, [a, b])
        self.assertEqual(args._single_positional_data_model, None)

    # Testing single arg and single position data_model
    def test_args_single_arg(self):
        class Test(DataModel):
            foo1: str
            foo2: str

        a = SymbolicDataModel(data_model=Test)
        args = SymbolicArguments((a))

        self.assertEqual(args.symbolic_data_models, [a])
        self.assertEqual(args._flat_arguments, [a])
        self.assertEqual(len(args.kwargs), 0)
        self.assertEqual(isinstance(args.args[0], SymbolicDataModel), True)
        self.assertEqual(args._single_positional_data_model, a)

    # Testing kwargs
    def test_kwargs(self):
        class Test(DataModel):
            foo1: str
            foo2: str

        a = SymbolicDataModel(data_model=Test)
        b = SymbolicDataModel(data_model=Test)
        c = SymbolicDataModel(data_model=Test)
        args = SymbolicArguments(
            (
                a,
                b,
            ),
            {1: c},
        )

        self.assertEqual(args.symbolic_data_models, [a, b, c])
        self.assertEqual(args._flat_arguments, [a, b, c])
        self.assertEqual(args._single_positional_data_model, None)

    # Testing fill in function with single args only
    def test_fill_in_single_arg(self):
        class Test(DataModel):
            foo1: str
            foo2: str

        a = SymbolicDataModel(data_model=Test)

        json_data_model = JsonDataModel(data_model=Test(foo1="foo1", foo2="foo2"))

        data_model_dict = {id(a): json_data_model}
        sym_args = SymbolicArguments((a))

        # Call the method to be tested
        result, _ = sym_args.fill_in(data_model_dict)

        self.assertEqual(result, (json_data_model,))

    # Testing fill in function with multiple args
    def test_fill_in_multiple_arg(self):
        class Test(DataModel):
            foo1: str
            foo2: str

        a = SymbolicDataModel(data_model=Test)
        b = SymbolicDataModel(data_model=Test)

        json_data_model = JsonDataModel(data_model=Test(foo1="foo1", foo2="foo2"))

        data_model_dict = {id(b): json_data_model}
        sym_args = SymbolicArguments((a, b))

        # Call the method to be tested
        result, _ = sym_args.fill_in(data_model_dict)
        self.assertEqual(result, ((None, json_data_model),))

    # Testing fill in function for args and kwargs
    def test_fill_in(self):
        class Test1(DataModel):
            foo: str
            bar: int

        class Test2(DataModel):
            foo: int
            bar: str

        a = SymbolicDataModel(schema=Test1.get_schema())
        b = SymbolicDataModel(schema=Test2.get_schema())
        c = SymbolicDataModel(schema=Test2.get_schema())

        json_data_model1 = JsonDataModel(data_model=Test1(foo="foo", bar=2))
        json_data_model2 = JsonDataModel(data_model=Test2(foo=3, bar="foo"))

        dictionary = {id(a): json_data_model1, id(c): json_data_model2}

        sym_args = SymbolicArguments(
            (
                a,
                b,
            ),
            {"1": c},
        )

        values, _ = sym_args.fill_in(dictionary)
        self.assertEqual(values, ((json_data_model1, None), {"1": json_data_model2}))
