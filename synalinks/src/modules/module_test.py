# Modified from: keras/src/layers/layer_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import backend
from synalinks.src import modules
from synalinks.src import testing


class ModuleTest(testing.TestCase):
    async def test_compute_output_spec(self):
        class Query(backend.DataModel):
            query: str

        # Case: single output
        class TestModule(modules.Module):
            async def call(self, x):
                assert False  # Should never be called.

            async def compute_output_spec(self, inputs):
                return backend.SymbolicDataModel(data_model=inputs)

        module = TestModule()
        self.assertEqual(
            (await module(backend.SymbolicDataModel(data_model=Query))).get_schema(),
            backend.standardize_schema(Query.get_schema()),
        )

        # Case: tuple output
        class TestModule(modules.Module):
            async def call(self, x):
                assert False  # Should never be called.

            async def compute_output_spec(self, inputs):
                return (
                    backend.SymbolicDataModel(data_model=inputs),
                    backend.SymbolicDataModel(data_model=inputs),
                )

        module = TestModule()
        out = await module(backend.SymbolicDataModel(data_model=Query))
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertEqual(
            out[0].get_schema(), backend.standardize_schema(Query.get_schema())
        )
        self.assertEqual(
            out[1].get_schema(), backend.standardize_schema(Query.get_schema())
        )

        # Case: list output
        class TestModule(modules.Module):
            async def call(self, x):
                assert False  # Should never be called.

            async def compute_output_spec(self, inputs):
                return [
                    backend.SymbolicDataModel(data_model=inputs),
                    backend.SymbolicDataModel(data_model=inputs),
                ]

        module = TestModule()
        out = await module(backend.SymbolicDataModel(data_model=Query))

        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 2)
        self.assertEqual(
            out[0].get_schema(), backend.standardize_schema(Query.get_schema())
        )
        self.assertEqual(
            out[1].get_schema(), backend.standardize_schema(Query.get_schema())
        )

        # Case: dict output
        class TestModule(modules.Module):
            async def call(self, x):
                assert False  # Should never be called.

            async def compute_output_spec(self, inputs):
                return {
                    "1": backend.SymbolicDataModel(data_model=inputs),
                    "2": backend.SymbolicDataModel(data_model=inputs),
                }

        module = TestModule()
        out = await module(backend.SymbolicDataModel(data_model=Query))

        self.assertIsInstance(out, dict)
        self.assertEqual(len(out), 2)
        self.assertEqual(
            out["1"].get_schema(), backend.standardize_schema(Query.get_schema())
        )
        self.assertEqual(
            out["2"].get_schema(), backend.standardize_schema(Query.get_schema())
        )
