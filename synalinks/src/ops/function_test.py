# Modified from: keras/src/ops/function_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.ops import function
from synalinks.src.ops.json import concat


class FunctionTest(testing.TestCase):
    async def test_define_and_call(self):
        class Query(DataModel):
            query: str

        class Answer(DataModel):
            answer: str

        input_1 = SymbolicDataModel(data_model=Query)
        input_2 = SymbolicDataModel(data_model=Answer)
        output = await concat(input_1, input_2)
        fn = function.Function(
            inputs=[input_1, input_2],
            outputs=output,
            name="test_function",
        )
        self.assertEqual(fn.name, "test_function")

        # Eager call
        input_1_val = JsonDataModel(
            data_model=Query(query="What is the capital of France?")
        )
        input_2_val = JsonDataModel(data_model=Answer(answer="Paris"))
        output_val = await fn([input_1_val, input_2_val])

        expected_value = {
            "query": "What is the capital of France?",
            "answer": "Paris",
        }

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

        self.assertIsInstance(output_val, JsonDataModel)
        self.assertEqual(output_val.get_json(), expected_value)
        self.assertEqual(output_val.get_schema(), expected_schema)

        # Symbolic call
        x1_alt = SymbolicDataModel(data_model=Query)
        x2_alt = SymbolicDataModel(data_model=Answer)
        y_val = await fn([x1_alt, x2_alt])
        self.assertIsInstance(y_val, SymbolicDataModel)

        # Recursion
        fn = function.Function(inputs=[x1_alt, x2_alt], outputs=y_val)
        y_val = await fn(
            [
                JsonDataModel(data_model=Query(query="What is the capital of France?")),
                JsonDataModel(data_model=Answer(answer="Paris")),
            ]
        )

        self.assertIsInstance(output_val, JsonDataModel)
        self.assertEqual(output_val.get_json(), expected_value)
        self.assertEqual(output_val.get_schema(), expected_schema)

    async def test_invalid_inputs_error(self):
        class Query(DataModel):
            query: str

        class Answer(DataModel):
            answer: str

        class Question(DataModel):
            question: str

        input_1 = SymbolicDataModel(data_model=Query)
        input_2 = SymbolicDataModel(data_model=Answer)
        input_3 = SymbolicDataModel(data_model=Question)

        output = await concat(input_1, input_2)
        fn = function.Function(
            inputs=[input_1, input_2],
            outputs=output,
            name="test_function",
        )
        self.assertEqual(fn.name, "test_function")

        # Bad structure
        with self.assertRaisesRegex(ValueError, "invalid input structure"):
            _ = await fn([input_1, input_2, input_3])

        # Bad schema
        with self.assertRaisesRegex(ValueError, "incompatible inputs"):
            _ = await fn([input_1, input_3])

    def test_graph_disconnected_error(self):
        # TODO
        pass

    def test_function_with_empty_outputs(self):
        class Query(DataModel):
            query: str

        x = SymbolicDataModel(data_model=Query)
        with self.assertRaisesRegex(ValueError, "`outputs` argument cannot be empty"):
            _ = function.Function(inputs=x, outputs=[])

    def test_function_with_empty_inputs(self):
        class Query(DataModel):
            query: str

        x = SymbolicDataModel(data_model=Query)
        with self.assertRaisesRegex(ValueError, "`inputs` argument cannot be empty"):
            _ = function.Function(inputs=[], outputs=x)
