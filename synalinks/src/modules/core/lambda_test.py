# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import JsonDataModel
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.lambda_module import Lambda
from synalinks.src.programs.program import Program


class Query(DataModel):
    query: str


class UppercaseQuery(DataModel):
    query: str


class LambdaTest(testing.TestCase):
    async def test_async_function_with_dict_output(self):
        async def uppercase(inputs):
            data = inputs.get_json()
            return {"query": data["query"].upper()}

        inputs = Input(data_model=Query)
        outputs = await Lambda(
            function=uppercase,
            data_model=UppercaseQuery,
        )(inputs)

        program = Program(inputs=inputs, outputs=outputs)
        result = await program(Query(query="hello"))

        self.assertEqual(result.get_json(), {"query": "HELLO"})

    async def test_sync_function_with_dict_output(self):
        def uppercase(inputs):
            data = inputs.get_json()
            return {"query": data["query"].upper()}

        inputs = Input(data_model=Query)
        outputs = await Lambda(
            function=uppercase,
            data_model=UppercaseQuery,
        )(inputs)

        program = Program(inputs=inputs, outputs=outputs)
        result = await program(Query(query="hello"))

        self.assertEqual(result.get_json(), {"query": "HELLO"})

    async def test_python_lambda_function(self):
        inputs = Input(data_model=Query)
        outputs = await Lambda(
            function=lambda x: {"query": x.get_json()["query"].upper()},
            data_model=UppercaseQuery,
        )(inputs)

        program = Program(inputs=inputs, outputs=outputs)
        result = await program(Query(query="hello"))

        self.assertEqual(result.get_json(), {"query": "HELLO"})

    async def test_function_returning_data_model(self):
        async def uppercase(inputs):
            data = inputs.get_json()
            return UppercaseQuery(query=data["query"].upper())

        inputs = Input(data_model=Query)
        outputs = await Lambda(
            function=uppercase,
            data_model=UppercaseQuery,
        )(inputs)

        program = Program(inputs=inputs, outputs=outputs)
        result = await program(Query(query="hello"))

        self.assertEqual(result.get_json(), {"query": "HELLO"})

    async def test_function_returning_json_data_model(self):
        async def uppercase(inputs):
            data = inputs.get_json()
            return JsonDataModel(
                json={"query": data["query"].upper()},
                schema=UppercaseQuery.get_schema(),
            )

        inputs = Input(data_model=Query)
        outputs = await Lambda(
            function=uppercase,
            data_model=UppercaseQuery,
        )(inputs)

        program = Program(inputs=inputs, outputs=outputs)
        result = await program(Query(query="hello"))

        self.assertEqual(result.get_json(), {"query": "HELLO"})

    async def test_function_returning_none(self):
        async def returns_none(inputs):
            return None

        inputs = Input(data_model=Query)
        outputs = await Lambda(
            function=returns_none,
            data_model=UppercaseQuery,
        )(inputs)

        program = Program(inputs=inputs, outputs=outputs)
        result = await program(Query(query="hello"))

        self.assertEqual(result, None)

    async def test_schema_argument(self):
        async def uppercase(inputs):
            data = inputs.get_json()
            return {"query": data["query"].upper()}

        inputs = Input(data_model=Query)
        outputs = await Lambda(
            function=uppercase,
            schema=UppercaseQuery.get_schema(),
        )(inputs)

        program = Program(inputs=inputs, outputs=outputs)
        result = await program(Query(query="hello"))

        self.assertEqual(result.get_json(), {"query": "HELLO"})

    async def test_requires_output_spec(self):
        async def noop(inputs):
            return inputs.get_json()

        with self.assertRaises(ValueError):
            Lambda(function=noop)

    async def test_requires_callable(self):
        with self.assertRaises(TypeError):
            Lambda(function="not a callable", data_model=UppercaseQuery)

    async def test_invalid_return_type_raises(self):
        async def returns_string(inputs):
            return "not a dict"

        inputs = Input(data_model=Query)
        module = Lambda(function=returns_string, data_model=UppercaseQuery)

        with self.assertRaises(TypeError):
            outputs = await module(inputs)
            program = Program(inputs=inputs, outputs=outputs)
            await program(Query(query="hello"))
