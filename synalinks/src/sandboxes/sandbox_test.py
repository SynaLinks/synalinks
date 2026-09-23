# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.sandboxes.sandbox import ExecutionResult
from synalinks.src.sandboxes.sandbox import Sandbox


class _EchoSandbox(Sandbox):
    """Minimal backend: implements only `run_code`."""

    async def run_code(self, code, *, inputs=None, external_functions=None):
        return self._record_run(code, ExecutionResult(stdout=code))


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
        self.assertEqual(result.stdout, "x")
        self.assertEqual(len(sandbox.history()), 1)

    async def test_run_python_code_uses_run_code(self):
        result = await _EchoSandbox().run_python_code("y")
        self.assertEqual(result["stdout"], "y")

    async def test_default_namespaces_are_unsupported(self):
        sandbox = _EchoSandbox()
        with self.assertRaises(NotImplementedError):
            await sandbox.files.read("/a.txt")
        with self.assertRaises(NotImplementedError):
            await sandbox.files.list()
        with self.assertRaises(NotImplementedError):
            await sandbox.commands.run("ls")
