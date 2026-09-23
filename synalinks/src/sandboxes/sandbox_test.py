# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.sandboxes.sandbox import Execution
from synalinks.src.sandboxes.sandbox import ExecutionError
from synalinks.src.sandboxes.sandbox import ExecutionResult
from synalinks.src.sandboxes.sandbox import Logs
from synalinks.src.sandboxes.sandbox import Result
from synalinks.src.sandboxes.sandbox import Sandbox


class _EchoSandbox(Sandbox):
    """Minimal backend: implements only `run_code`."""

    async def run_code(self, code, *, inputs=None, external_functions=None):
        if code == "boom":
            error = ExecutionError("ValueError", "boom", "Traceback ...\n")
            return self.record_run(code, Execution(error=error))
        return self.record_run(
            code,
            Execution(
                results=[Result(text=repr(code), json=code, is_main_result=True)],
                logs=Logs(stdout=[code]),
            ),
        )


class SandboxBaseTest(testing.TestCase):
    async def test_create_builds_the_subclass(self):
        sandbox = await _EchoSandbox.create(timeout=2.0, name="echo")
        self.assertIsInstance(sandbox, _EchoSandbox)
        self.assertEqual((sandbox.timeout, sandbox.name), (2.0, "echo"))

    async def test_kill_stops_running(self):
        sandbox = _EchoSandbox()
        self.assertTrue(await sandbox.is_running())
        self.assertTrue(await sandbox.kill())
        self.assertFalse(await sandbox.is_running())

    async def test_run_is_a_deprecated_alias_of_run_code(self):
        sandbox = _EchoSandbox()
        with self.assertWarns(DeprecationWarning):
            result = await sandbox.run("x")
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual((result.stdout, result.result, result.ok), ("x", "x", True))
        self.assertEqual(len(sandbox.history()), 1)
        with self.assertWarns(DeprecationWarning):
            failed = await sandbox.run("boom")
        self.assertEqual(failed.error, "ValueError: boom")
        self.assertIn("Traceback", failed.stderr)

    async def test_run_python_code_uses_run_code(self):
        result = await _EchoSandbox().run_python_code("y")
        self.assertEqual(result["stdout"], "y")
        failed = await _EchoSandbox().run_python_code("boom")
        self.assertEqual((failed["ok"], failed["error"]), (False, "ValueError: boom"))
        self.assertIn("Traceback", failed["stderr"])

    async def test_execution_matches_e2b_shape(self):
        execution = await _EchoSandbox().run_code("x")
        self.assertEqual((execution.text, execution.execution_count), ("'x'", 1))
        self.assertEqual(execution.logs.stdout, ["x"])
        self.assertIn('"logs"', execution.to_json())

    async def test_default_namespaces_are_unsupported(self):
        sandbox = _EchoSandbox()
        with self.assertRaises(NotImplementedError):
            await sandbox.files.read("/a.txt")
        with self.assertRaises(NotImplementedError):
            await sandbox.files.list()
        with self.assertRaises(NotImplementedError):
            await sandbox.commands.run("ls")

    async def test_legacy_backend_overriding_run_still_works(self):
        class _Legacy(Sandbox):
            async def run(self, code, **kwargs):
                return ExecutionResult(stdout="legacy:" + code)

        sandbox = _Legacy()
        self.assertEqual((await sandbox.run_code("1")).logs.stdout, ["legacy:1"])
        self.assertEqual((await sandbox.run_python_code("2"))["stdout"], "legacy:2")
