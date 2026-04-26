# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.merging.logical_xor import Xor
from synalinks.src.programs.program import Program


class LogicalXorTest(testing.TestCase):
    async def test_logical_xor_module_all_not_none(self):
        class Query(DataModel):
            query: str

        i0 = Input(data_model=Query)
        i1 = Input(data_model=Query)
        i2 = Input(data_model=Query)
        output = await Xor()([i0, i1, i2])

        program = Program(
            inputs=[i0, i1, i2],
            outputs=output,
        )

        result = await program(
            [
                Query(query="a"),
                Query(query="b"),
                Query(query="c"),
            ]
        )

        self.assertEqual(result, None)

    async def test_logical_xor_module_with_two_not_none(self):
        class Query(DataModel):
            query: str

        i0 = Input(data_model=Query)
        i1 = Input(data_model=Query)
        i2 = Input(data_model=Query)
        output = await Xor()([i0, i1, i2])

        program = Program(
            inputs=[i0, i1, i2],
            outputs=output,
        )

        result = await program(
            [
                None,
                Query(query="b"),
                Query(query="c"),
            ]
        )

        self.assertEqual(result, None)

    async def test_logical_xor_module_none(self):
        class Query(DataModel):
            query: str

        i0 = Input(data_model=Query)
        i1 = Input(data_model=Query)
        i2 = Input(data_model=Query)
        output = await Xor()([i0, i1, i2])

        program = Program(
            inputs=[i0, i1, i2],
            outputs=output,
        )

        result = await program(
            [
                None,
                Query(query="b"),
                None,
            ]
        )

        expected_json = {
            "query": "b",
        }

        self.assertEqual(result.get_json(), expected_json)
