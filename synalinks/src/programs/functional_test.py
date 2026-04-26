# Modified from: keras/src/models/functional_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import JsonDataModel
from synalinks.src.modules import Input
from synalinks.src.programs import Functional
from synalinks.src.programs import Program


class FunctionalTest(testing.TestCase):
    async def test_basic_flow_multi_input(self):
        class Query(DataModel):
            query: str

        class Answer(DataModel):
            answer: str

        input_a = Input(data_model=Query, name="input_a")
        input_b = Input(data_model=Answer, name="input_b")
        outputs = input_a + input_b
        program = Functional([input_a, input_b], outputs, name="basic")

        self.assertEqual(program.name, "basic")
        self.assertIsInstance(program, Functional)
        self.assertIsInstance(program, Program)

        # Eager call
        in_val = [
            JsonDataModel(
                json={"query": "What is the capital of France?"},
                schema=Query.get_schema(),
            ),
            JsonDataModel(json={"answer": "Paris"}, schema=Answer.get_schema()),
        ]
        out_val = await program(in_val)
        expected_schema = {
            "properties": {
                "query": {"title": "Query", "type": "string"},
                "answer": {"title": "Answer", "type": "string"},
            },
            "required": ["query", "answer"],
            "title": "Query",
            "type": "object",
            "additionalProperties": False,
        }
        expected_value = {
            "query": "What is the capital of France?",
            "answer": "Paris",
        }
        self.assertIsInstance(out_val, JsonDataModel)
        self.assertEqual(out_val.get_schema(), expected_schema)
        self.assertEqual(out_val.get_json(), expected_value)

        # Eager call with data_models
        in_val = [
            Query(query="What is the capital of France?"),
            Answer(answer="Paris"),
        ]
        out_val = await program(in_val)
        self.assertIsInstance(out_val, JsonDataModel)
        self.assertEqual(out_val.get_schema(), expected_schema)
        self.assertEqual(out_val.get_json(), expected_value)

        # Symbolic call
        input_a_2 = Input(data_model=Query, name="input_a_2")
        input_b_2 = Input(data_model=Answer, name="input_b_2")
        in_val = [input_a_2, input_b_2]
        out_val = await program(in_val)
        self.assertEqual(out_val.get_schema(), expected_schema)

    async def test_basic_flow_multi_output(self):
        class Query(DataModel):
            query: str

        class Answer(DataModel):
            answer: str

        input_a = Input(data_model=Query, name="input_a")
        input_b = Input(data_model=Answer, name="input_b")
        output_a = input_a
        output_b = input_a + input_b

        program = Functional([input_a, input_b], [output_a, output_b])

        # Eager call
        in_val = [
            JsonDataModel(
                json={"query": "What is the capital of France?"},
                schema=Query.get_schema(),
            ),
            JsonDataModel(json={"answer": "Paris"}, schema=Answer.get_schema()),
        ]
        out_val = await program(in_val)

        expected_schema_a = {
            "properties": {
                "query": {"title": "Query", "type": "string"},
            },
            "required": ["query"],
            "title": "Query",
            "type": "object",
            "additionalProperties": False,
        }

        expected_schema_b = {
            "properties": {
                "query": {"title": "Query", "type": "string"},
                "answer": {"title": "Answer", "type": "string"},
            },
            "required": ["query", "answer"],
            "title": "Query",
            "type": "object",
            "additionalProperties": False,
        }
        self.assertIsInstance(out_val, list)
        self.assertEqual(len(out_val), 2)
        self.assertEqual(out_val[0].get_schema(), expected_schema_a)
        self.assertEqual(out_val[1].get_schema(), expected_schema_b)

        # Eager call with data_models
        in_val = [
            Query(query="What is the capital of France?"),
            Answer(answer="Paris"),
        ]
        out_val = await program(in_val)
        self.assertIsInstance(out_val, list)
        self.assertEqual(len(out_val), 2)
        self.assertEqual(out_val[0].get_schema(), expected_schema_a)
        self.assertEqual(out_val[1].get_schema(), expected_schema_b)

        # Symbolic call
        input_a_2 = Input(data_model=Query, name="input_a_2")
        input_b_2 = Input(data_model=Answer, name="input_b_2")
        out_val = await program([input_a_2, input_b_2])
        self.assertIsInstance(out_val, list)
        self.assertEqual(len(out_val), 2)
        self.assertEqual(out_val[0].get_schema(), expected_schema_a)
        self.assertEqual(out_val[1].get_schema(), expected_schema_b)

    async def test_basic_flow_dict_io(self):
        class Query(DataModel):
            query: str

        class Answer(DataModel):
            answer: str

        input_a = Input(data_model=Query, name="a")
        input_b = Input(data_model=Answer, name="b")

        outputs = input_a + input_b

        with self.assertRaisesRegex(
            ValueError, "All `inputs` values must be SymbolicDataModels"
        ):
            program = Functional({"a": "input_a", "b": input_b}, outputs)

        with self.assertRaisesRegex(
            ValueError, "All `outputs` values must be SymbolicDataModels"
        ):
            program = Functional({"a": input_a, "b": input_b}, "outputs")

        program = Functional({"a": input_a, "b": input_b}, outputs)

        # Eager call
        in_val = {
            "a": Query(query="What is the capital of France?"),
            "b": Answer(answer="Paris"),
        }
        out_val = await program(in_val)
        expected_schema = {
            "properties": {
                "query": {"title": "Query", "type": "string"},
                "answer": {"title": "Answer", "type": "string"},
            },
            "required": ["query", "answer"],
            "title": "Query",
            "type": "object",
            "additionalProperties": False,
        }
        self.assertIsInstance(out_val, JsonDataModel)
        self.assertEqual(out_val.get_schema(), expected_schema)

    async def test_representation(self):
        class Query(DataModel):
            query: str

        class Answer(DataModel):
            answer: str

        input_a = Input(data_model=Query, name="a")
        input_b = Input(data_model=Answer, name="b")
        outputs = input_a + input_b

        program = Functional(
            [input_a, input_b],
            outputs,
            name="concat_basic",
            description="Concatenate two data_models",
        )
        self.assertEqual(
            str(program),
            "<Functional name=concat_basic, "
            "description='Concatenate two data_models', built=True>",
        )
