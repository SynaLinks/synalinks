# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.sandboxes.monty_sandbox import MontySandbox
from synalinks.src.sandboxes.sandbox import ExecutionResult
from synalinks.src.sandboxes.sandbox import Sandbox


class MontySandboxTest(testing.TestCase):
    async def test_is_sandbox_subclass(self):
        sandbox = MontySandbox()
        self.assertIsInstance(sandbox, Sandbox)

    async def test_run_returns_execution_result(self):
        sandbox = MontySandbox()
        result = await sandbox.run("print('hello')")
        self.assertIsInstance(result, ExecutionResult)
        self.assertIn("hello", result.stdout)
        self.assertTrue(result.ok)
        self.assertIsNone(result.error)

    async def test_run_captures_error(self):
        sandbox = MontySandbox()
        result = await sandbox.run("1 / 0")
        self.assertFalse(result.ok)
        self.assertIn("ZeroDivisionError", result.error)

    async def test_state_persists_across_run_calls(self):
        sandbox = MontySandbox()
        await sandbox.run("x = 7")
        result = await sandbox.run("print(x * 6)")
        self.assertIn("42", result.stdout)

    async def test_reset_clears_state(self):
        sandbox = MontySandbox()
        await sandbox.run("x = 7")
        sandbox.reset()
        result = await sandbox.run("print(x)")
        self.assertFalse(result.ok)
        self.assertIn("NameError", result.error)

    async def test_dump_and_load_round_trip(self):
        sandbox = MontySandbox()
        await sandbox.run("y = 123")
        blob = sandbox.dump()
        self.assertIsInstance(blob, bytes)

        restored = MontySandbox.load(blob)
        result = await restored.run("print(y)")
        self.assertIn("123", result.stdout)

    async def test_get_config_embeds_state(self):
        sandbox = MontySandbox(timeout=9.0, name="session_1")
        await sandbox.run("greeting = 'hello'")
        config = sandbox.get_config()

        self.assertEqual(config["timeout"], 9.0)
        self.assertEqual(config["name"], "session_1")
        self.assertIsInstance(config["state"], str)
        self.assertGreater(len(config["state"]), 0)

    async def test_from_config_round_trip(self):
        sandbox = MontySandbox(timeout=4.5, name="s")
        await sandbox.run("data = [1, 2, 3]")
        config = sandbox.get_config()

        restored = MontySandbox.from_config(config)
        self.assertEqual(restored.timeout, 4.5)
        self.assertEqual(restored.name, "s")

        result = await restored.run("print(sum(data))")
        self.assertIn("6", result.stdout)

    async def test_from_config_without_state_builds_fresh_sandbox(self):
        sandbox = MontySandbox.from_config({"timeout": 3.0, "name": "fresh"})
        self.assertEqual(sandbox.timeout, 3.0)
        self.assertEqual(sandbox.name, "fresh")
        result = await sandbox.run("print(undefined_var)")
        self.assertFalse(result.ok)
        self.assertIn("NameError", result.error)

    async def test_inputs_bind_into_namespace(self):
        sandbox = MontySandbox()
        result = await sandbox.run(
            "print(greeting)",
            inputs={"greeting": "hello world"},
        )
        self.assertIn("hello world", result.stdout)

    async def test_timeout_is_per_snippet_not_cumulative(self):
        """Each `run` call gets a fresh `timeout` budget. Monty's native
        REPL treats `max_duration_secs` as cumulative across calls; the
        sandbox rolls it over so the user-facing contract is per-snippet."""
        # 2s budget. A single CPU-bound snippet near the full budget is ok,
        # and a second one must also succeed — with raw Monty the second
        # call would fail because the elapsed clock carried over.
        sandbox = MontySandbox(timeout=2.0)

        heavy = "import math\nsum(math.sqrt(i) for i in range(3_000_000))"
        r1 = await sandbox.run(heavy)
        self.assertTrue(r1.ok, f"first heavy run failed: {r1.error}")
        r2 = await sandbox.run(heavy)
        self.assertTrue(
            r2.ok,
            f"second heavy run failed (budget would have been cumulative): {r2.error}",
        )

    async def test_external_function_is_exposed_as_global(self):
        async def triple(x: int):
            return x * 3

        sandbox = MontySandbox()
        result = await sandbox.run(
            "import asyncio\n"
            "async def main():\n"
            "    return await triple(x=4)\n"
            "value = asyncio.run(main())\n"
            "print(value['result'] if isinstance(value, dict) else value)",
            external_functions={"triple": triple},
        )
        self.assertIn("12", result.stdout)


class SandboxAbstractTest(testing.TestCase):
    async def test_abstract_methods_raise_not_implemented(self):
        base = Sandbox()
        with self.assertRaises(NotImplementedError):
            await base.run("pass")
        with self.assertRaises(NotImplementedError):
            base.reset()
        with self.assertRaises(NotImplementedError):
            base.dump()
        with self.assertRaises(NotImplementedError):
            Sandbox.load(b"")
