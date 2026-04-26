# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_schema_equal
from synalinks.src.backend import standardize_schema
from synalinks.src.modules import Input
from synalinks.src.ops.json import concat
from synalinks.src.ops.json import decompose
from synalinks.src.ops.json import factorize
from synalinks.src.ops.json import in_mask
from synalinks.src.ops.json import logical_and
from synalinks.src.ops.json import logical_or
from synalinks.src.ops.json import logical_xor
from synalinks.src.ops.json import out_mask
from synalinks.src.programs import Program
from synalinks.src.utils.nlp_utils import remove_numerical_suffix


class SchemaEqualTest(testing.TestCase):
    def test_is_schema_equal(self):
        class Test(DataModel):
            foo: str
            foo_1: str

        x = SymbolicDataModel(data_model=Test)
        y = SymbolicDataModel(data_model=Test)
        self.assertEqual(x.get_schema(), standardize_schema(Test.get_schema()))
        self.assertEqual(y.get_schema(), standardize_schema(Test.get_schema()))
        self.assertTrue(is_schema_equal(x.get_schema(), y.get_schema()))

    def test_schema_not_equal(self):
        class Test1(DataModel):
            foo: str

        class Test2(DataModel):
            foo: str
            foo_1: str

        x = SymbolicDataModel(data_model=Test1)
        y = SymbolicDataModel(data_model=Test2)
        self.assertEqual(x.get_schema(), standardize_schema(Test1.get_schema()))
        self.assertEqual(y.get_schema(), standardize_schema(Test2.get_schema()))
        self.assertFalse(is_schema_equal(x.get_schema(), y.get_schema()))

    def test_not_equal_but_same_names(self):
        class Test1(DataModel):
            foo: str

        class Test2(DataModel):
            foo: int

        x = SymbolicDataModel(data_model=Test1)
        y = SymbolicDataModel(data_model=Test2)
        self.assertEqual(x.get_schema(), standardize_schema(Test1.get_schema()))
        self.assertEqual(y.get_schema(), standardize_schema(Test2.get_schema()))
        self.assertFalse(is_schema_equal(x.get_schema(), y.get_schema()))


class ConcatenateTest(testing.TestCase):
    async def test_concat_schema_same_property(self):
        class Test1(DataModel):
            foo: str

        class Test2(DataModel):
            foo: str

        class Result(DataModel):
            foo: str
            foo_1: str

        x = SymbolicDataModel(data_model=Test1)
        y = SymbolicDataModel(data_model=Test2)
        expected = SymbolicDataModel(data_model=Result)

        result = await concat(x, y)
        self.assertTrue(is_schema_equal(result.get_schema(), expected.get_schema()))

    async def test_concat_schema_different_property(self):
        class Test1(DataModel):
            foo: str

        class Test2(DataModel):
            bar: str

        class Result(DataModel):
            foo: str
            bar: str

        x = SymbolicDataModel(data_model=Test1)
        y = SymbolicDataModel(data_model=Test2)
        expected = SymbolicDataModel(data_model=Result)
        result = await concat(x, y)
        self.assertTrue(is_schema_equal(result.get_schema(), expected.get_schema()))

    async def test_concat_serialization(self):
        class Test1(DataModel):
            foo: str

        class Test2(DataModel):
            bar: str

        i0 = Input(data_model=Test1)
        i1 = Input(data_model=Test2)
        x0 = await concat(i0, i1)

        program = Program(
            inputs=[i0, i1],
            outputs=x0,
        )

        config = program.get_config()
        new_program = Program.from_config(config)
        for var1 in program.variables:
            for var2 in new_program.variables:
                if remove_numerical_suffix(var2.path) == var1.path:
                    self.assertEqual(var2.get_json(), var1.get_json())


class FactorizeTest(testing.TestCase):
    async def test_factorize_schema_same_property(self):
        class Test(DataModel):
            foo: str
            foo_1: str

        class Result(DataModel):
            foos: List[str]

        x = SymbolicDataModel(data_model=Test)
        expected = SymbolicDataModel(data_model=Result)
        result = await factorize(x)
        self.assertTrue(is_schema_equal(result.get_schema(), expected.get_schema()))

    async def test_factorize_schema_properties_multiple_times(self):
        class Test(DataModel):
            foos: List[str]
            foo: str

        class Result(DataModel):
            foos: List[str]

        x = SymbolicDataModel(data_model=Test)
        expected = SymbolicDataModel(data_model=Result)
        result = await factorize(x)
        self.assertTrue(is_schema_equal(result.get_schema(), expected.get_schema()))

    async def test_factorize_schema_different_property(self):
        class Test(DataModel):
            foo: str
            bar: str

        class Result(DataModel):
            foo: str
            bar: str

        x = SymbolicDataModel(data_model=Test)
        expected = SymbolicDataModel(data_model=Result)
        result = await factorize(x)
        self.assertTrue(is_schema_equal(result.get_schema(), expected.get_schema()))

    async def test_factorize_serialization(self):
        class Test(DataModel):
            foos: List[str]
            foo: str

        i0 = Input(data_model=Test)
        x0 = await factorize(i0)

        program = Program(
            inputs=i0,
            outputs=x0,
        )

        config = program.get_config()
        new_program = Program.from_config(config)
        for var1 in program.variables:
            for var2 in new_program.variables:
                if remove_numerical_suffix(var2.path) == var1.path:
                    self.assertEqual(var2.get_json(), var1.get_json())


class LogicalAndTest(testing.TestCase):
    async def test_and_table(self):
        class Answer(DataModel):
            answer: str

        class Result(DataModel):
            answer: str
            answer_1: str

        i0 = Input(data_model=Answer)
        i1 = Input(data_model=Answer)

        x0 = await logical_and(i0, i1)

        program = Program(
            inputs=[i0, i1],
            outputs=x0,
        )

        result = await program([Answer(answer="Paris"), Answer(answer="Toulouse")])
        self.assertEqual(
            result.get_json(),
            Result(
                answer="Paris",
                answer_1="Toulouse",
            ).get_json(),
        )

        result = await program([Answer(answer="Paris"), None])
        self.assertEqual(result, None)

        result = await program([None, Answer(answer="Toulouse")])
        self.assertEqual(result, None)

        result = await program([None, None])
        self.assertEqual(result, None)


class LogicalOrTest(testing.TestCase):
    async def test_or_table(self):
        class Answer(DataModel):
            answer: str

        class Result(DataModel):
            answer: str
            answer_1: str

        i0 = Input(data_model=Answer)
        i1 = Input(data_model=Answer)

        x0 = await logical_or(i0, i1)

        program = Program(
            inputs=[i0, i1],
            outputs=x0,
        )

        result = await program([Answer(answer="Paris"), Answer(answer="Toulouse")])
        self.assertEqual(
            result.get_json(),
            Result(
                answer="Paris",
                answer_1="Toulouse",
            ).get_json(),
        )

        result = await program([Answer(answer="Paris"), None])
        self.assertEqual(result.get_json(), Answer(answer="Paris").get_json())

        result = await program([None, Answer(answer="Toulouse")])
        self.assertEqual(result.get_json(), Answer(answer="Toulouse").get_json())

        result = await program([None, None])
        self.assertEqual(result, None)


class LogicalXorTest(testing.TestCase):
    async def test_xor_table(self):
        class Answer(DataModel):
            answer: str

        class Result(DataModel):
            answer: str
            answer_1: str

        i0 = Input(data_model=Answer)
        i1 = Input(data_model=Answer)

        x0 = await logical_xor(i0, i1)

        program = Program(
            inputs=[i0, i1],
            outputs=x0,
        )

        result = await program([Answer(answer="Paris"), Answer(answer="Toulouse")])
        self.assertEqual(result, None)

        result = await program([Answer(answer="Paris"), None])
        self.assertEqual(result.get_json(), Answer(answer="Paris").get_json())

        result = await program([None, Answer(answer="Toulouse")])
        self.assertEqual(result.get_json(), Answer(answer="Toulouse").get_json())

        result = await program([None, None])
        self.assertEqual(result, None)


class OutMaskTest(testing.TestCase):
    async def test_out_mask_serialization(self):
        class Test(DataModel):
            foos: List[str]
            foo: str
            bar: str

        i0 = Input(data_model=Test)
        x0 = await out_mask(i0, mask=["foo"])

        program = Program(
            inputs=i0,
            outputs=x0,
        )

        config = program.get_config()
        new_program = Program.from_config(config)
        for var1 in program.variables:
            for var2 in new_program.variables:
                if remove_numerical_suffix(var2.path) == var1.path:
                    self.assertEqual(var2.get_json(), var1.get_json())


class InMaskTest(testing.TestCase):
    async def test_in_mask_serialization(self):
        class Test(DataModel):
            foos: List[str]
            foo: str
            bar: str

        i0 = Input(data_model=Test)
        x0 = await in_mask(i0, mask=["bar"])

        program = Program(
            inputs=i0,
            outputs=x0,
        )

        config = program.get_config()
        new_program = Program.from_config(config)
        for var1 in program.variables:
            for var2 in new_program.variables:
                if remove_numerical_suffix(var2.path) == var1.path:
                    self.assertEqual(var2.get_json(), var1.get_json())


class DecomposeTest(testing.TestCase):
    async def test_decompose_schema_list_property(self):
        class Test(DataModel):
            foos: List[str]

        class Expected(DataModel):
            foo: str
            foo_1: str

        i0 = Input(data_model=Test)
        x0 = await decompose(i0)
        self.assertTrue(
            is_schema_equal(
                standardize_schema(x0.get_schema()),
                standardize_schema(Expected.get_schema()),
            )
        )

    async def test_decompose_schema_non_list_property(self):
        class Test(DataModel):
            foo: str
            bar: str

        i0 = Input(data_model=Test)
        x0 = await decompose(i0)
        self.assertTrue(
            is_schema_equal(
                standardize_schema(x0.get_schema()),
                standardize_schema(Test.get_schema()),
            )
        )

    async def test_decompose_serialization(self):
        class Test(DataModel):
            foos: List[str]

        i0 = Input(data_model=Test)
        x0 = await decompose(i0)

        program = Program(
            inputs=i0,
            outputs=x0,
        )

        config = program.get_config()
        new_program = Program.from_config(config)
        for var1 in program.variables:
            for var2 in new_program.variables:
                if remove_numerical_suffix(var2.path) == var1.path:
                    self.assertEqual(var2.get_json(), var1.get_json())
