# Modified from: keras/src/layers/layer_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio

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

    async def test_concurrent_calls_have_isolated_call_context(self):
        # Regression test for GH#63: the per-call `CallContext` used to live in
        # `threading.local` global state. An asyncio event loop runs many
        # coroutines on one thread, so two modules awaited concurrently
        # (`asyncio.gather`) shared — and corrupted — a single context between
        # `await` points: `call_id`/`parent_call_id` were overwritten and the
        # first call to return detached the other's still-running context.
        # The context now lives in a `contextvars.ContextVar`, so each
        # concurrent call keeps a private, stable context for its lifetime.

        class Query(backend.DataModel):
            x: int

        class CtxProbe(modules.Module):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.observed = {}

            async def call(self, inputs, training=False):
                ctx_before = self._get_call_context()
                id_before = ctx_before.call_id if ctx_before else None
                # Yield to the sibling coroutine so a shared context would be
                # clobbered here.
                await asyncio.sleep(0.02)
                ctx_after = self._get_call_context()
                id_after = ctx_after.call_id if ctx_after else None
                self.observed = {
                    "same_ctx_object": ctx_before is ctx_after,
                    "call_id_stable": id_before == id_after,
                    "training_seen": training,
                }
                return inputs

            async def compute_output_spec(self, inputs, training=False):
                return inputs

        a = CtxProbe(name="a")
        b = CtxProbe(name="b")
        await asyncio.gather(
            a(Query(x=1).to_json_data_model(), training=True),
            b(Query(x=2).to_json_data_model(), training=False),
        )

        for probe in (a, b):
            self.assertTrue(probe.observed["same_ctx_object"])
            self.assertTrue(probe.observed["call_id_stable"])
        # `training` is captured per-call and must not leak between siblings.
        self.assertTrue(a.observed["training_seen"])
        self.assertFalse(b.observed["training_seen"])
