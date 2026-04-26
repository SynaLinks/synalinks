# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.modules.core.identity import Identity
from synalinks.src.modules.core.input_module import Input
from synalinks.src.programs.program import Program


class IdentityTest(testing.TestCase):
    async def test_single_inputs(self):
        class Query(DataModel):
            query: str

        inputs = Input(data_model=Query)
        outputs = await Identity()(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
        )

        result = await program(Query(query="a"))

        expected_json = {
            "query": "a",
        }

        self.assertEqual(result.get_json(), expected_json)

    async def test_tuple_inputs(self):
        class Query(DataModel):
            query: str

        inputs = (
            Input(data_model=Query),
            Input(data_model=Query),
            Input(data_model=Query),
        )
        outputs = await Identity()(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
        )

        result = await program(
            (
                Query(query="a"),
                Query(query="b"),
                Query(query="c"),
            )
        )

        expected_json_a = {
            "query": "a",
        }
        expected_json_b = {
            "query": "b",
        }
        expected_json_c = {
            "query": "c",
        }

        self.assertEqual(result[0].get_json(), expected_json_a)
        self.assertEqual(result[1].get_json(), expected_json_b)
        self.assertEqual(result[2].get_json(), expected_json_c)

    async def test_list_inputs(self):
        class Query(DataModel):
            query: str

        inputs = [
            Input(data_model=Query),
            Input(data_model=Query),
            Input(data_model=Query),
        ]
        outputs = await Identity()(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
        )

        result = await program(
            [
                Query(query="a"),
                Query(query="b"),
                Query(query="c"),
            ]
        )

        expected_json_a = {
            "query": "a",
        }
        expected_json_b = {
            "query": "b",
        }
        expected_json_c = {
            "query": "c",
        }

        self.assertEqual(result[0].get_json(), expected_json_a)
        self.assertEqual(result[1].get_json(), expected_json_b)
        self.assertEqual(result[2].get_json(), expected_json_c)

    async def test_dict_inputs(self):
        class Query(DataModel):
            query: str

        inputs = {
            "a": Input(data_model=Query),
            "b": Input(data_model=Query),
            "c": Input(data_model=Query),
        }
        outputs = await Identity()(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
        )

        result = await program(
            {
                "a": Query(query="a"),
                "b": Query(query="b"),
                "c": Query(query="c"),
            }
        )

        expected_json_a = {
            "query": "a",
        }
        expected_json_b = {
            "query": "b",
        }
        expected_json_c = {
            "query": "c",
        }

        self.assertEqual(result["a"].get_json(), expected_json_a)
        self.assertEqual(result["b"].get_json(), expected_json_b)
        self.assertEqual(result["c"].get_json(), expected_json_c)
