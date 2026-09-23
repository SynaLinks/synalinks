# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import gc
import os
import sys
import tempfile
import unittest

from synalinks.src import testing
from synalinks.src.sandboxes.mirage_sandbox import MirageSandbox
from synalinks.src.sandboxes.mirage_sandbox import cap_malloc_arenas
from synalinks.src.sandboxes.mirage_sandbox import malloc_trim
from synalinks.src.sandboxes.sandbox import CommandExitException
from synalinks.src.sandboxes.sandbox import CommandResult
from synalinks.src.sandboxes.sandbox import EntryInfo
from synalinks.src.sandboxes.sandbox import Execution
from synalinks.src.sandboxes.sandbox import ExecutionResult
from synalinks.src.sandboxes.sandbox import FileType
from synalinks.src.sandboxes.sandbox import NotFoundException
from synalinks.src.sandboxes.sandbox import Sandbox
from synalinks.src.sandboxes.sandbox import TimeoutException
from synalinks.src.utils.confinement_utils import MACOS_CONFINE_SRC
from synalinks.src.utils.confinement_utils import confinement_available
from synalinks.src.utils.confinement_utils import confinement_backend

_TIMEOUT = 30.0


def _stdout(execution):
    """Everything an `Execution` printed to stdout, as one string."""
    return "".join(execution.logs.stdout)


_CONFINE_OK, _CONFINE_REASON = confinement_available()
_BACKEND, _BACKEND_REASON = confinement_backend()
# The namespace suite asserts the Linux backend's behaviour (PID namespaces,
# seccomp, the pivot_root filesystem swap). It runs where that backend runs:
# on Linux, and inside the macOS microVM, which confines with the very same
# code behind a guest kernel.
_NAMESPACES_OK = _BACKEND in ("namespaces", "microvm")
_NS_REASON = f"namespace confinement unavailable: {_CONFINE_REASON or _BACKEND_REASON}"
_MICROVM_OK = _BACKEND == "microvm"
_VM_REASON = f"microVM backend unavailable: {_BACKEND_REASON or sys.platform}"
# Seatbelt is the macOS floor; its suite forces it even where a microVM runs.
_SEATBELT_OK = _CONFINE_OK and sys.platform == "darwin"
_SB_REASON = f"seatbelt confinement unavailable: {_CONFINE_REASON or sys.platform}"


class _SandboxTestCase(testing.TestCase):
    """Base for sandbox tests.

    Most tests build a ``MirageSandbox`` (and forks) as locals and never call
    ``close()``. Each live sandbox holds a FUSE mount, so without cleanup the
    suite accumulates mounts until ``mount_max`` (default 1000) is hit and new
    mounts fail *silently*; the confinement tests then read empty output and
    fail. Forcing a GC pass after every test runs each dropped sandbox's
    finalizer, which releases its mount deterministically between tests.
    """

    def tearDown(self):
        super().tearDown()
        gc.collect()


class PinnedNamesTest(_SandboxTestCase):
    """``__rlm_pinned__`` re-asserts environment names on every run.

    The RLM binds its module input as ``inputs`` once per call; before the
    pin, a snippet assigning over the name (LLM-written code does) poisoned
    every later snippet of the call — ``inputs`` stayed ``None`` and every
    function reading it kept failing until the next call re-bound it."""

    async def test_pinned_name_heals_on_the_next_run(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code(
            "inputs = _rlm_inputs\n__rlm_pinned__ = {'inputs': _rlm_inputs}",
            inputs={"_rlm_inputs": {"frames": [1, 2, 3]}},
        )
        # The clobbering snippet breaks itself...
        broken = await sandbox.run_code("inputs = None\nprint(inputs['frames'])")
        self.assertIsNotNone(broken.error)
        # ...and only itself: the next run reads the pinned value again.
        healed = await sandbox.run_code("print(inputs['frames'])")
        self.assertIsNone(healed.error)
        self.assertIn("[1, 2, 3]", _stdout(healed))

    async def test_functions_reading_the_pinned_name_recover_too(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code(
            "inputs = _rlm_inputs\n__rlm_pinned__ = {'inputs': _rlm_inputs}",
            inputs={"_rlm_inputs": {"frames": [7]}},
        )
        await sandbox.run_code("def first_frame():\n    return inputs['frames'][0]")
        await sandbox.run_code("inputs = 'garbage'")
        result = await sandbox.run_code("print(first_frame())")
        self.assertIsNone(result.error)
        self.assertIn("7", _stdout(result))

    async def test_per_run_binding_still_wins_over_the_pin(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code(
            "inputs = _rlm_inputs\n__rlm_pinned__ = {'inputs': _rlm_inputs}",
            inputs={"_rlm_inputs": {"frames": ["old"]}},
        )
        result = await sandbox.run_code(
            "print(inputs['frames'])", inputs={"inputs": {"frames": ["fresh"]}}
        )
        self.assertIsNone(result.error)
        self.assertIn("fresh", _stdout(result))

    async def test_rebinding_the_pin_replaces_it(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        for payload in ({"n": 1}, {"n": 2}):
            await sandbox.run_code(
                "inputs = _rlm_inputs\n__rlm_pinned__ = {'inputs': _rlm_inputs}",
                inputs={"_rlm_inputs": payload},
            )
        await sandbox.run_code("inputs = None")
        result = await sandbox.run_code("print(inputs['n'])")
        self.assertIsNone(result.error)
        self.assertIn("2", _stdout(result))


class MallocHygieneTest(_SandboxTestCase):
    """The host-side glibc settings a sandbox applies to its own process."""

    _GLIBC = sys.platform.startswith("linux") and "glibc" in (
        getattr(__import__("platform"), "libc_ver", lambda: ("", ""))()[0] or ""
    )

    def test_cap_is_idempotent_and_reports_platform(self):
        first = cap_malloc_arenas(2)
        second = cap_malloc_arenas(2)
        self.assertEqual(first, second)
        if self._GLIBC:
            self.assertTrue(first)

    def test_trim_never_raises(self):
        released = malloc_trim()
        self.assertIsInstance(released, bool)
        if not self._GLIBC:
            self.assertFalse(released)

    async def test_sandbox_caps_arenas_by_default_and_can_opt_out(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        self.assertEqual(sandbox.malloc_arena_max, 2)
        untouched = MirageSandbox(timeout=_TIMEOUT, malloc_arena_max=None)
        self.assertIsNone(untouched.malloc_arena_max)
        result = await untouched.run_code("print('still runs')")
        self.assertIsNone(result.error)

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "glibc arenas are a Linux matter"
    )
    def test_cap_bounds_arena_mappings(self):
        """Worker threads churning big buffers must not each keep a 64 MiB heap.

        Runs in a child interpreter and counts how many 64 MiB-class anonymous
        mappings (what a glibc arena looks like in ``/proc/self/maps``) the
        churning threads add; imports create arenas of their own before the
        cap, so only the growth is judged.
        """
        import subprocess

        script = (
            "import json, sys, threading\n"
            "from synalinks.src.sandboxes.mirage_sandbox import cap_malloc_arenas\n"
            "def arenas():\n"
            "    n = 0\n"
            "    for line in open('/proc/self/maps'):\n"
            "        parts = line.split()\n"
            "        if len(parts) != 5:\n"
            "            continue\n"
            "        lo, hi = (int(x, 16) for x in parts[0].split('-'))\n"
            "        n += (60 << 20) <= hi - lo <= (70 << 20)\n"
            "    return n\n"
            "if sys.argv[1] == 'cap':\n"
            "    assert cap_malloc_arenas(2)\n"
            "before = arenas()\n"
            "payload = {'rows': [list(range(4096)) for _ in range(300)]}\n"
            "def churn():\n"
            "    for _ in range(10):\n"
            "        json.loads(json.dumps(payload))\n"
            "threads = [threading.Thread(target=churn) for _ in range(24)]\n"
            "[t.start() for t in threads]\n"
            "[t.join() for t in threads]\n"
            "print(arenas() - before)\n"
        )

        def arenas(mode):
            out = subprocess.run(
                [sys.executable, "-c", script, mode],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
            return int(out.stdout.strip().splitlines()[-1])

        if not self._GLIBC:
            self.skipTest("not glibc")
        # Only the capped bound is a guarantee. How many arenas the *default*
        # allocator adds depends on how often the threads' mallocs overlap
        # (the GIL serialises most Python-level churn; the FUSE bridge and
        # ctypes calls that bloat a real host run outside it), so it is not
        # asserted here.
        self.assertLessEqual(arenas("cap"), 2)


class E2BCompatTest(_SandboxTestCase):
    """The E2B-named surface: create / files / commands / kill."""

    async def test_create_and_kill(self):
        sandbox = await MirageSandbox.create(timeout=_TIMEOUT, confine=False)
        self.assertIsInstance(sandbox, MirageSandbox)
        self.assertTrue(await sandbox.is_running())
        self.assertTrue(await sandbox.kill())
        self.assertFalse(await sandbox.is_running())

    async def test_files_roundtrip(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        info = await sandbox.files.write("/src/a.txt", "hello")
        self.assertEqual(info.path, "/src/a.txt")
        self.assertEqual(info.name, "a.txt")
        self.assertEqual(await sandbox.files.read("/src/a.txt"), "hello")
        self.assertTrue(await sandbox.files.exists("/src/a.txt"))
        self.assertFalse(await sandbox.files.exists("/src/missing.txt"))
        with self.assertRaises(FileNotFoundError):
            await sandbox.files.read("/src/missing.txt")

    async def test_files_bytes_are_exact(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        payload = bytes([0, 255, 1, 2, 128])
        await sandbox.files.write("/b.bin", payload)
        self.assertEqual(await sandbox.files.read("/b.bin", format="bytes"), payload)

    async def test_files_list_and_info(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        await sandbox.files.write("/d/a.txt", "abc")
        await sandbox.files.write("/d/sub/b.txt", "b")
        top = await sandbox.files.list("/d")
        self.assertEqual([e.path for e in top], ["/d/a.txt", "/d/sub"])
        self.assertEqual([e.type for e in top], [FileType.FILE, FileType.DIR])
        deep = await sandbox.files.list("/d", depth=2)
        self.assertIn("/d/sub/b.txt", [e.path for e in deep])
        info = await sandbox.files.get_info("/d/a.txt")
        self.assertIsInstance(info, EntryInfo)
        self.assertEqual((info.type, info.size), (FileType.FILE, 3))
        self.assertEqual((info.mode, info.permissions), (0o644, "rw-r--r--"))

    async def test_files_rename_remove_make_dir(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        await sandbox.files.write("/a.txt", "x")
        moved = await sandbox.files.rename("/a.txt", "/sub/b.txt")
        self.assertEqual(moved.path, "/sub/b.txt")
        self.assertFalse(await sandbox.files.exists("/a.txt"))
        self.assertEqual(await sandbox.files.read("/sub/b.txt"), "x")
        await sandbox.files.remove("/sub")
        self.assertFalse(await sandbox.files.exists("/sub/b.txt"))
        self.assertTrue(await sandbox.files.make_dir("/new"))
        self.assertFalse(await sandbox.files.make_dir("/new"))
        self.assertEqual((await sandbox.files.get_info("/new")).type, FileType.DIR)
        with self.assertRaises(NotFoundException):
            await sandbox.files.get_info("/missing")

    async def test_commands_run(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        await sandbox.files.write("/hello.txt", "hi")
        result = await sandbox.commands.run("cat /hello.txt")
        self.assertIsInstance(result, CommandResult)
        self.assertEqual((result.stdout, result.exit_code, result.error), ("hi", 0, None))
        # As in E2B, a non-zero exit raises, carrying the CommandResult.
        with self.assertRaises(CommandExitException) as caught:
            await sandbox.commands.run("cat /nope.txt")
        self.assertIsInstance(caught.exception, CommandResult)
        self.assertNotEqual(caught.exception.exit_code, 0)
        self.assertIsNotNone(caught.exception.error)
        with self.assertRaises(TimeoutException):
            await sandbox.commands.run("sleep 5", timeout=0.2)

    async def test_internal_commands_stay_out_of_history(self):
        # Mirage 0.0.6+ records executed lines in a /.bash_history view. Only
        # what the agent typed belongs there, not the bootstrap or file I/O.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        await sandbox.run_code("x = 1")
        await sandbox.write_file("/a.txt", "a")
        await sandbox.run_bash("echo typed-by-agent")
        history = await sandbox.read_text("/.bash_history")
        self.assertIn("echo typed-by-agent", history)
        self.assertNotIn("base64", history)
        self.assertNotIn("cat >", history)
        # Internal views are not the agent's files.
        listed = (await sandbox.list_files())["files"]
        self.assertEqual(listed, ["/a.txt"])
        self.assertEqual([e["path"] for e in sandbox.diff()["written"]], ["/a.txt"])

    async def test_commands_run_timeout_zero_means_no_limit(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        result = await sandbox.commands.run("echo ok", timeout=0)
        self.assertEqual((result.exit_code, result.stdout.strip()), (0, "ok"))

    async def test_run_is_a_deprecated_alias(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        with self.assertWarns(DeprecationWarning):
            result = await sandbox.run("1 + 1")
        # The deprecated alias keeps returning the old flat shape.
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual((result.result, result.ok), (2, True))


class MirageSandboxTest(_SandboxTestCase):
    async def test_is_sandbox_subclass(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        self.assertIsInstance(sandbox, Sandbox)

    async def test_run_returns_execution_result(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        result = await sandbox.run_code("print('hello')\n6 * 7")
        self.assertIsInstance(result, Execution)
        self.assertEqual(result.logs.stdout, ["hello\n"])
        self.assertIsNone(result.error)
        # The last expression is the main result: repr as text, value as json.
        self.assertEqual(result.text, "42")
        self.assertEqual(len(result.results), 1)
        self.assertTrue(result.results[0].is_main_result)
        self.assertEqual(result.results[0].formats(), ["text", "json"])
        self.assertEqual(result.execution_count, 1)
        self.assertEqual((await sandbox.run_code("None")).results, [])

    async def test_run_binds_large_inputs(self):
        """An `inputs` payload past the execve per-argument limit
        (MAX_ARG_STRLEN, 128KiB) must still bind: the run config travels
        by file, never on argv. Regression: `Argument list too long`
        left the RLM's `inputs` variable silently unbound on long
        documents.
        """
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        payload = {"log": "x" * 300_000}
        bind = await sandbox.run_code("inputs = _in", inputs={"_in": payload})
        self.assertIsNone(bind.error)
        result = await sandbox.run_code("print(len(inputs['log']))")
        self.assertIn("300000", _stdout(result))

    async def test_run_captures_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        result = await sandbox.run_code("1 / 0")
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.name, "ZeroDivisionError")

    async def test_traceback_is_trimmed_to_user_frames(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        result = await sandbox.run_code("1 / 0")
        # Bootstrap/launcher frames are hidden; only the user frame remains.
        self.assertIn('"<sandbox>"', result.error.traceback)
        self.assertNotIn("dill", result.error.traceback)
        self.assertNotIn("base64", result.error.traceback)

    async def test_state_persists_across_run_calls(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code("x = 7")
        result = await sandbox.run_code("print(x * 6)")
        self.assertIn("42", _stdout(result))

    async def test_functions_and_classes_persist(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code("def sq(n):\n    return n * n")
        await sandbox.run_code("import math\nclass P:\n    v = 3")
        result = await sandbox.run_code("print(sq(4), math.floor(2.7), P.v)")
        self.assertIn("16 2 3", _stdout(result))

    async def test_many_functions_persist(self):
        """Persisting must not be quadratic in the number of definitions.

        dill pickles each function together with a copy of its globals, so
        pickling the namespace item by item re-pickles every sibling function
        once per function: 120 definitions took ~4 s and a few hundred hit
        `RecursionError`, silently losing the whole namespace. One dump of the
        namespace memoizes the shared globals instead."""
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code(
            "\n".join(f"def f{i}(x):\n    return x + {i}" for i in range(120))
        )
        result = await sandbox.run_code("print(f0(1), f119(1))")
        self.assertIn("1 120", _stdout(result))

    async def test_unpicklable_value_does_not_drop_the_namespace(self):
        """One unpicklable object must not take the definitions with it.

        A generator (an open socket, a live handle) cannot be pickled. The
        fallback drops it and pickles the rest with `recurse=True` so each
        function carries only the globals it references, instead of the
        offending object along with the whole namespace."""
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code(
            "gen = (i for i in range(3))\ndef survivor(x):\n    return x * 3"
        )
        result = await sandbox.run_code("print(survivor(5))")
        self.assertIn("15", _stdout(result))
        self.assertIsNone(result.error)

    async def test_function_sees_names_defined_in_later_runs(self):
        """A restored function shares the live namespace, like a real REPL.

        dill pickles a function with a private copy of its globals; without
        re-homing restored functions onto the live ``ns``, ``f`` here keeps a
        ghost namespace frozen at definition time and raises NameError even
        though ``helper`` is defined (top-down design: call first, fill in
        after)."""
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code("def f(x):\n    return helper(x) + 1")
        await sandbox.run_code("def helper(x):\n    return x * 10")
        result = await sandbox.run_code("print(f(4))")
        self.assertIn("41", _stdout(result))

    async def test_method_sees_names_defined_in_later_runs(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code("class G:\n    def area(self):\n        return helper(3)")
        await sandbox.run_code("def helper(x):\n    return x * 10")
        result = await sandbox.run_code("print(G().area())")
        self.assertIn("30", _stdout(result))

    async def test_rehomed_function_keeps_closure_and_kwdefaults(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code(
            "def make_adder(n):\n"
            "    def add(x, *, scale=2):\n"
            "        return (x + n) * scale\n"
            "    return add\n"
            "add5 = make_adder(5)"
        )
        result = await sandbox.run_code("print(add5(1), add5(1, scale=1))")
        self.assertIn("12 6", _stdout(result))

    async def test_imported_function_keeps_its_own_module_globals(self):
        """Re-homing must not touch functions imported from a module.

        An imported function closes over its *module's* globals and reaches
        names in them at call time. Rebuilding it on the sandbox namespace
        strips those: ``os.path.join`` re-homed this way loses ``sep``."""
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code("from os.path import join")
        result = await sandbox.run_code("print(join('a', 'b'))")
        self.assertIn("a/b", _stdout(result))
        self.assertIsNone(result.error)

    async def test_imported_class_methods_keep_their_module_globals(self):
        """Same for the methods of an imported class.

        ``Counter.update`` reads ``_collections_abc`` and ``_count_elements``
        from ``collections``' own globals, so re-homing it onto the sandbox
        namespace made ``Counter('aa')`` raise NameError on every run after
        the one that imported it. Worse, the class is re-homed *in place*, so
        a later plain ``import collections`` inherited the damage."""
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code("from collections import Counter")
        result = await sandbox.run_code("print(Counter('aab')['a'])")
        self.assertIn("2", _stdout(result))
        self.assertIsNone(result.error)
        fresh = await sandbox.run_code(
            "import collections\nprint(collections.Counter('aab')['a'])"
        )
        self.assertIn("2", _stdout(fresh))
        self.assertIsNone(fresh.error)

    async def test_state_survives_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code("keep = 11")
        await sandbox.run_code("raise ValueError('boom')")
        result = await sandbox.run_code("print(keep)")
        self.assertIn("11", _stdout(result))

    async def test_inputs_are_bound(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        result = await sandbox.run_code("print(given + 1)", inputs={"given": 41})
        self.assertIn("42", _stdout(result))
        self.assertIsNone(result.error)

    async def test_last_expression_is_captured_as_result(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        result = await sandbox.run_code("a = 10\nb = 32\na + b")
        self.assertEqual(result.results[0].json, 42)

    async def test_result_variable_convention(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        result = await sandbox.run_code("result = {'k': 5}\nresult")
        self.assertEqual(result.results[0].json, {"k": 5})

    async def test_external_functions_bridge_into_sandbox(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)

        async def adder(x, y):
            return {"sum": x + y}

        # Bound tools are called *synchronously* inside the sandbox: no
        # `await` / `asyncio.run(...)` ceremony.
        code = "out = adder(x=3, y=4)\nprint(out['sum'])\n"
        result = await sandbox.run_code(code, external_functions={"adder": adder})
        self.assertIsNone(result.error)
        self.assertIn("7", _stdout(result))

    async def test_sync_tool_call_returns_value_directly(self):
        # A bare synchronous call returns the host tool's value directly, usable
        # in the same expression, and it actually invokes the host callable.
        called = {"n": 0}

        async def adder(x, y):
            called["n"] += 1
            return {"sum": x + y}

        sandbox = MirageSandbox(timeout=_TIMEOUT)
        # Last expression is the bare sync call result (the `result` convention).
        result = await sandbox.run_code(
            "adder(x=2, y=5)['sum']",
            external_functions={"adder": adder},
        )
        self.assertIsNone(result.error)
        self.assertEqual(result.results[0].json, 7)
        self.assertEqual(called["n"], 1)

    async def test_external_function_accepts_positional_args(self):
        # Regression: the sandbox's tool-call RPC bridge (bootstrap stub ->
        # Unix socket -> host dispatcher) used to marshal only `kwargs`, so a
        # positional call raised `TypeError: _stub() takes 0 positional
        # arguments but 1 was given` even though a plain Python call and
        # `Tool.__call__` both support positional args fine.
        async def adder(x, y):
            return {"sum": x + y}

        sandbox = MirageSandbox(timeout=_TIMEOUT)
        result = await sandbox.run_code(
            "out = adder(3, 4)\nprint(out['sum'])\n",
            external_functions={"adder": adder},
        )
        self.assertIsNone(result.error)
        self.assertIn("7", _stdout(result))

    async def test_bound_functions_persist_across_runs(self):
        captured = {}

        async def submit(result):
            captured["value"] = result
            return {"submitted": result}

        sandbox = MirageSandbox(timeout=_TIMEOUT, external_functions={"submit": submit})
        result = await sandbox.run_code("submit(result={'answer': 'done'})")
        self.assertIsNone(result.error)
        self.assertEqual(captured["value"], {"answer": "done"})

    async def test_run_bash(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        out = await sandbox.run_bash("echo hi > /t.txt && cat /t.txt")
        self.assertTrue(out["ok"])
        self.assertEqual(out["exit_code"], 0)
        self.assertIn("hi", out["stdout"])

    async def test_run_bash_failure_returns_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        out = await sandbox.run_bash("python3 -c 'import sys; sys.exit(7)'")
        self.assertFalse(out["ok"])
        self.assertEqual(out["exit_code"], 7)

    async def test_bash_session_state_persists(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_bash("mkdir -p /work && cd /work")
        out = await sandbox.run_bash("pwd")
        self.assertIn("/work", out["stdout"])

    async def test_run_python_code_tool(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        out = await sandbox.run_python_code("print(2 + 2)")
        self.assertTrue(out["ok"])
        self.assertIn("4", out["stdout"])
        self.assertIsNone(out["error"])

    async def test_run_python_code_tool_invalid_code_reports_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        out = await sandbox.run_python_code("def broken(")
        self.assertFalse(out["ok"])
        self.assertIsNotNone(out["error"])
        self.assertIn("SyntaxError", out["error"])

    async def test_history_records_runs(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code("a = 1")
        await sandbox.run_code("print(a)")
        history = sandbox.history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["code"], "a = 1")
        self.assertTrue(history[1]["ok"])

    async def test_reset_clears_state_and_history(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code("secret = 123")
        sandbox.reset()
        self.assertEqual(sandbox.history(), [])
        result = await sandbox.run_code("print(secret)")
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.name, "NameError")

    async def test_timeout_surfaces_as_error(self):
        sandbox = MirageSandbox(timeout=1.0)
        result = await sandbox.run_code("import time\ntime.sleep(5)")
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.name, "TimeoutError")

    async def test_require_confinement_needs_confine(self):
        # require_confinement is incompatible with an explicit confine=False.
        with self.assertRaises(ValueError):
            MirageSandbox(timeout=_TIMEOUT, require_confinement=True, confine=False)

    async def test_require_confinement_fails_closed_when_unavailable(self):
        # When confinement cannot be established, require_confinement turns the
        # silent unconfined fallback into a hard error.
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        with mock.patch.object(
            mirage_sandbox,
            "confinement_backend",
            return_value=(None, "simulated: unavailable"),
        ):
            with self.assertRaises(RuntimeError):
                MirageSandbox(timeout=_TIMEOUT, confine=True, require_confinement=True)

    async def test_default_sandbox_fails_closed_when_unavailable(self):
        # Secure by default: a plain MirageSandbox() refuses to run LM code
        # unconfined; nobody reads the warning a graceful fallback would emit.
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        with mock.patch.object(
            mirage_sandbox,
            "confinement_backend",
            return_value=(None, "simulated: unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "confine=False"):
                MirageSandbox(timeout=_TIMEOUT)

    async def test_graceful_fallback_needs_an_explicit_opt_in(self):
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        with mock.patch.object(
            mirage_sandbox,
            "confinement_backend",
            return_value=(None, "simulated: unavailable"),
        ):
            with self.assertWarns(RuntimeWarning):
                sandbox = MirageSandbox(timeout=_TIMEOUT, require_confinement=False)
        self.addCleanup(sandbox.close)
        self.assertFalse(sandbox.granted_capabilities()["confined"])

    async def test_confine_false_is_an_explicit_opt_out(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        self.addCleanup(sandbox.close)
        caps = sandbox.granted_capabilities()
        self.assertFalse(caps["confined"])
        self.assertFalse(caps["require_confinement"])
        self.assertIsNone(caps["backend"])

    async def test_load_is_confined_by_default(self):
        # A restored sandbox used to come back unconfined: `load` built it
        # through `bare`. It now confines, and fails closed, like a new one.
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        source = MirageSandbox(timeout=_TIMEOUT, confine=False)
        self.addCleanup(source.close)
        data = source.dump()
        with mock.patch.object(
            mirage_sandbox,
            "confinement_backend",
            return_value=(None, "simulated: unavailable"),
        ):
            with self.assertRaises(RuntimeError):
                MirageSandbox.load(data)
        unconfined = MirageSandbox.load(data, confine=False)
        self.addCleanup(unconfined.close)
        self.assertFalse(unconfined.granted_capabilities()["confined"])

    async def test_explicit_confined_fork_fails_closed(self):
        # fork(confine=True) of an unconfined parent must not inherit the
        # parent's graceful fallback.
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        parent = MirageSandbox(timeout=_TIMEOUT, confine=False)
        self.addCleanup(parent.close)
        with mock.patch.object(
            mirage_sandbox,
            "confinement_backend",
            return_value=(None, "simulated: unavailable"),
        ):
            # As in E2B, a failed child comes back in place of the sandbox.
            (child,) = await parent.fork(confine=True)
            self.assertIsInstance(child, RuntimeError)

    # -- security surface ---------------------------------------------------

    def test_host_allowlist_matching(self):
        from synalinks.src.utils.egress_utils import host_allowed

        self.assertTrue(host_allowed("api.openai.com", ["api.openai.com"]))
        self.assertFalse(host_allowed("evil.com", ["api.openai.com"]))
        # wildcard matches subdomains and the apex, case/trailing-dot insensitive
        self.assertTrue(host_allowed("a.b.example.com", ["*.example.com"]))
        self.assertTrue(host_allowed("EXAMPLE.com.", ["*.example.com"]))
        self.assertFalse(host_allowed("notexample.com", ["*.example.com"]))
        # an empty allowlist denies everything
        self.assertFalse(host_allowed("anything.com", []))

    async def test_egress_tool_refuses_offlist_host(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, allowed_hosts=["api.example.com"])
        fetch = sandbox.bound_functions["http_fetch"]
        with self.assertRaises(PermissionError):
            await fetch(url="http://evil.com/")
        with self.assertRaises(PermissionError):
            await fetch(url="file:///etc/passwd")

    def test_reject_private_blocks_internal_addresses(self):
        from synalinks.src.utils.egress_utils import reject_private

        # loopback / private / link-local (incl. cloud-metadata IP) are refused
        for host in ("127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254"):
            with self.assertRaises(PermissionError):
                reject_private(host, None, "http")
        # a public address passes and is returned (numeric, so no DNS lookup)
        self.assertEqual(reject_private("8.8.8.8", None, "https"), "8.8.8.8")

    async def test_egress_pins_validated_ip(self):
        # The connection must dial the IP validated by reject_private, not
        # re-resolve the hostname (which would reopen the DNS-rebinding window).
        # We allowlist a non-resolving `.invalid` host and pin it to a local
        # server: the fetch only succeeds if the pinned IP is used.
        import http.server
        import threading
        from unittest import mock

        from synalinks.src.utils import egress_utils

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"pinned-ok")

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            sandbox = MirageSandbox(
                timeout=_TIMEOUT, allowed_hosts=["pinned.invalid"], confine=False
            )
            fetch = sandbox.bound_functions["http_fetch"]
            with mock.patch.object(
                egress_utils, "reject_private", return_value="127.0.0.1"
            ):
                resp = await fetch(url=f"http://pinned.invalid:{port}/")
            self.assertEqual(resp["status"], 200)
            self.assertEqual(resp["body"], "pinned-ok")
        finally:
            server.shutdown()
            server.server_close()

    async def test_egress_tool_blocks_private_target_by_default(self):
        # An allowlisted host that resolves to a private IP is still refused.
        sandbox = MirageSandbox(
            timeout=_TIMEOUT, allowed_hosts=["127.0.0.1"], confine=False
        )
        fetch = sandbox.bound_functions["http_fetch"]
        with self.assertRaises(PermissionError):
            await fetch(url="http://127.0.0.1/")

    async def test_block_private_egress_false_bypasses_ssrf_gate(self):
        # With the SSRF gate off, a private target gets past the check and fails
        # at the connection instead (URLError), not with a PermissionError.
        import urllib.error

        sandbox = MirageSandbox(
            timeout=_TIMEOUT,
            allowed_hosts=["127.0.0.1"],
            block_private_egress=False,
            confine=False,
        )
        fetch = sandbox.bound_functions["http_fetch"]
        with self.assertRaises(urllib.error.URLError):
            await fetch(url="http://127.0.0.1:1/")  # port 1: connection refused

    def test_confinement_unavailable_points_to_wsl_on_windows(self):
        from unittest import mock

        with mock.patch("platform.system", return_value="Windows"):
            ok, reason = confinement_available()
        self.assertFalse(ok)
        self.assertIn("WSL2", reason)

    def test_native_windows_hint_silent_off_windows(self):
        from unittest import mock

        from synalinks.src.utils import confinement_utils

        confinement_utils.WINDOWS_HINTED = False
        with mock.patch("platform.system", return_value="Linux"):
            self.assertIsNone(confinement_utils.maybe_warn_native_windows())

    def test_native_windows_hint_warns_once(self):
        import os
        from unittest import mock

        from synalinks.src.utils import confinement_utils

        confinement_utils.WINDOWS_HINTED = False
        with mock.patch("platform.system", return_value="Windows"):
            with mock.patch.dict("os.environ", {}, clear=False):
                os.environ.pop("SYNALINKS_NO_WINDOWS_HINT", None)
                with self.assertWarns(RuntimeWarning):
                    msg = confinement_utils.maybe_warn_native_windows()
                self.assertIn("WSL2", msg)
                # second call is a no-op (already hinted this process)
                self.assertIsNone(confinement_utils.maybe_warn_native_windows())

    def test_native_windows_hint_suppressed_by_env_var(self):
        from unittest import mock

        from synalinks.src.utils import confinement_utils

        confinement_utils.WINDOWS_HINTED = False
        with mock.patch("platform.system", return_value="Windows"):
            with mock.patch.dict(
                "os.environ", {"SYNALINKS_NO_WINDOWS_HINT": "1"}, clear=False
            ):
                self.assertIsNone(confinement_utils.maybe_warn_native_windows())

    async def test_allowed_hosts_binds_egress_tool(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, allowed_hosts=["api.example.com"])
        self.assertIn("http_fetch", sandbox.bound_functions)

    async def test_granted_capabilities_unconfined(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        caps = sandbox.granted_capabilities()
        self.assertFalse(caps["confined"])
        self.assertFalse(caps["seccomp"])
        self.assertEqual(caps["network"], {"mode": "host"})
        self.assertEqual(caps["tools"], [])
        # Unconfined, the default 'local' runtime is host-reaching by design and
        # is reported as such rather than hidden.
        self.assertEqual(caps["host_runtimes"], ["local"])

    async def test_non_root_mounts_default_to_read_only(self):
        from mirage import RAMResource

        sandbox = MirageSandbox(
            timeout=_TIMEOUT,
            confine=False,
            resources={"/": RAMResource(), "/data": RAMResource()},
        )
        caps = sandbox.granted_capabilities()
        # root scratch stays writable+exec; an external mount is read-only.
        self.assertEqual(caps["mounts"]["/"], "EXEC")
        self.assertEqual(caps["mounts"]["/data"], "READ")

    async def test_explicit_mount_mode_tuple_is_honored(self):
        from mirage import MountMode
        from mirage import RAMResource

        sandbox = MirageSandbox(
            timeout=_TIMEOUT,
            confine=False,
            resources={"/": RAMResource(), "/rw": (RAMResource(), MountMode.WRITE)},
        )
        self.assertEqual(sandbox.granted_capabilities()["mounts"]["/rw"], "WRITE")

    @unittest.skipUnless(_NAMESPACES_OK, _NS_REASON)
    async def test_unvouched_runtime_stripped_under_confine(self):
        # A runtime the sandbox cannot vouch for could execute code on the host
        # outside the confinement prologue, so it is dropped (with a warning)
        # when confinement is active. Requires real confinement: where it is
        # unavailable, confine=True falls back to unconfined and the entry is
        # left intact, so this assertion only holds on a confine-capable host.
        with self.assertWarns(RuntimeWarning):
            sandbox = MirageSandbox(
                timeout=_TIMEOUT,
                confine=True,
                # The graceful strip-with-a-warning path; fail-closed (the
                # default) raises instead, covered by the next test.
                require_confinement=False,
                workspace_kwargs={"runtimes": ["vfs", "not-a-vouched-runtime"]},
            )
        self.assertEqual(sandbox.granted_capabilities()["host_runtimes"], [])
        # The vouched entry survives; only the unknown one is removed.
        self.assertEqual(sandbox.workspace_kwargs["runtimes"], ["vfs"])
        sandbox.close()

    @unittest.skipUnless(_NAMESPACES_OK, _NS_REASON)
    async def test_unvouched_runtime_rejected_under_require_confinement(self):
        with self.assertRaises(ValueError):
            MirageSandbox(
                timeout=_TIMEOUT,
                confine=True,
                require_confinement=True,
                workspace_kwargs={"runtimes": ["not-a-vouched-runtime"]},
            )

    @unittest.skipUnless(_NAMESPACES_OK, _NS_REASON)
    async def test_local_runtime_allowed_under_confinement_when_patched(self):
        # 'local' reaches the host, but the run-python patch prepends the
        # confinement prologue to everything it runs, so it is the one
        # host-reaching runtime the sandbox vouches for. This is the default,
        # and it must not trip the guard.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            self.assertTrue(sandbox.run_python_patched)
            self.assertEqual(sandbox.granted_capabilities()["host_runtimes"], [])
        finally:
            sandbox.close()

    @unittest.skipUnless(_NAMESPACES_OK, _NS_REASON)
    async def test_local_runtime_rejected_when_patch_unavailable(self):
        # Fail-closed: if the confinement prologue cannot be installed, 'local'
        # is an unguarded hole and require_confinement must refuse to build.
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        with mock.patch.object(
            mirage_sandbox, "local_runtime_supported", return_value=False
        ):
            with self.assertRaises(ValueError):
                MirageSandbox(timeout=_TIMEOUT, confine=True, require_confinement=True)

    def test_seccomp_filter_builds_for_known_arch(self):
        from unittest import mock

        from synalinks.src.utils import confinement_utils

        with mock.patch("platform.machine", return_value="x86_64"):
            blob = confinement_utils.build_seccomp_filter()
        self.assertIsInstance(blob, str)
        import base64

        raw = base64.b64decode(blob)
        # a valid classic-BPF program: whole 8-byte sock_filter instructions
        self.assertEqual(len(raw) % 8, 0)
        # unknown arch -> no filter (confinement still applies without one)
        with mock.patch("platform.machine", return_value="riscv128"):
            self.assertIsNone(confinement_utils.build_seccomp_filter())

    # -- filesystem tools ---------------------------------------------------

    async def test_write_then_read_file(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        written = await sandbox.write_file("/proj/main.py", "print('hi')\nX = 2\n")
        self.assertEqual(written["written"], "/proj/main.py")
        read = await sandbox.read_file("/proj/main.py")
        self.assertEqual(read["content"], "print('hi')\nX = 2\n")
        self.assertEqual(read["total_lines"], 2)

    async def test_read_missing_file(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        read = await sandbox.read_file("/nope.txt")
        self.assertIn("error", read)

    async def test_read_file_pagination(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/lines.txt", "a\nb\nc\nd\n")
        page = await sandbox.read_file("/lines.txt", offset=2, limit=2)
        self.assertEqual(page["content"], "b\nc\n")
        self.assertEqual(page["start_line"], 2)
        self.assertEqual(page["end_line"], 3)
        self.assertTrue(page["truncated"])

    async def test_list_files_glob(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/src/a.py", "1")
        await sandbox.write_file("/src/sub/b.py", "2")
        await sandbox.write_file("/notes.md", "3")
        listing = await sandbox.list_files("**/*.py")
        self.assertEqual(listing["files"], ["/src/a.py", "/src/sub/b.py"])
        self.assertEqual(listing["total"], 2)

    async def test_list_files_glob_no_matches(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/src/a.py", "1")
        listing = await sandbox.list_files("**/*.md")
        self.assertEqual(listing["files"], [])
        self.assertEqual(listing["total"], 0)

    async def test_list_files_glob_invalid_pattern(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/src/a.py", "1")
        listing = await sandbox.list_files("[")
        self.assertIsInstance(listing, dict)
        if "error" not in listing:
            self.assertEqual(listing["files"], [])
            self.assertEqual(listing["total"], 0)

    async def test_edit_file(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/c.py", "value = 1\n")
        edit = await sandbox.edit_file("/c.py", "value = 1", "value = 99")
        self.assertEqual(edit["replacements"], 1)
        read = await sandbox.read_file("/c.py")
        self.assertEqual(read["content"], "value = 99\n")

    async def test_edit_file_not_unique(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/d.py", "x\nx\n")
        edit = await sandbox.edit_file("/d.py", "x", "y")
        self.assertIn("error", edit)
        edit_all = await sandbox.edit_file("/d.py", "x", "y", replace_all=True)
        self.assertEqual(edit_all["replacements"], 2)

    async def test_search_files(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/s/a.py", "import os\nprint('found')\n")
        await sandbox.write_file("/s/b.py", "x = 1\n")
        result = await sandbox.search_files("print", glob="**/*.py")
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["matches"][0]["path"], "/s/a.py")
        self.assertEqual(result["matches"][0]["line"], 2)

    async def test_search_files_no_matches(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/s/a.py", "import os\n")
        result = await sandbox.search_files("print", glob="**/*.py")
        self.assertEqual(result["total"], 0)
        self.assertEqual(result["matches"], [])

    async def test_search_files_invalid_glob(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/s/a.py", "print('found')\n")
        result = await sandbox.search_files("print", glob="[")
        self.assertIsInstance(result, dict)
        if "error" in result:
            self.assertTrue(result["error"])
        else:
            self.assertEqual(result["total"], 0)
            self.assertEqual(result["matches"], [])

    async def test_run_python_file(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/script.py", "print(3 * 3)\n")
        out = await sandbox.run_python_file("/script.py")
        self.assertTrue(out["ok"])
        self.assertIn("9", out["stdout"])

    async def test_run_python_file_syntax_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/bad_syntax.py", "def broken(:\n    pass\n")
        out = await sandbox.run_python_file("/bad_syntax.py")
        self.assertFalse(out["ok"])
        self.assertTrue("stderr" in out or "error" in out)

    async def test_run_python_file_runtime_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/runtime_error.py", "raise ValueError('boom')\n")
        out = await sandbox.run_python_file("/runtime_error.py")
        self.assertFalse(out["ok"])
        self.assertTrue("stderr" in out or "error" in out)

    async def test_run_python_file_missing(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        out = await sandbox.run_python_file("/missing.py")
        self.assertIn("error", out)

    # -- serialization & branching ------------------------------------------

    async def test_dump_load_round_trip(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code("secret = 1234")
        await sandbox.write_file("/keep.txt", "persisted")
        blob = sandbox.dump()
        restored = MirageSandbox.load(blob)
        result = await restored.run_code("print(secret)")
        self.assertIn("1234", _stdout(result))
        read = await restored.read_file("/keep.txt")
        self.assertEqual(read["content"], "persisted")

    async def test_get_config_from_config_round_trip(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, name="cfg")
        await sandbox.run_code("token = 'abc'")
        config = sandbox.get_config()
        restored = MirageSandbox.from_config(config)
        self.assertEqual(restored.name, "cfg")
        result = await restored.run_code("print(token)")
        self.assertIn("abc", _stdout(result))

    async def test_fork_isolates_filesystem(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/f.txt", "parent")
        (child,) = await sandbox.fork()
        await child.write_file("/f.txt", "child")
        await child.write_file("/only_child.txt", "x")
        parent_read = await sandbox.read_file("/f.txt")
        child_read = await child.read_file("/f.txt")
        self.assertEqual(parent_read["content"], "parent")
        self.assertEqual(child_read["content"], "child")
        parent_only = await sandbox.read_file("/only_child.txt")
        self.assertIn("error", parent_only)

    async def test_fork_copy_repl_inherits_namespace(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.run_code("base = 5")
        (child,) = await sandbox.fork(copy_repl=True)
        result = await child.run_code("print(base * 2)")
        self.assertIn("10", _stdout(result))
        # Child mutations do not leak back to the parent.
        await child.run_code("base = 999")
        parent_result = await sandbox.run_code("print(base)")
        self.assertIn("5", _stdout(parent_result))

    async def test_diff_and_changes(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/created.txt", "new")
        diff = sandbox.diff()
        self.assertEqual(
            diff["written"], [{"path": "/created.txt", "kind": "create", "size": 3}]
        )
        changes = sandbox.changes()
        self.assertEqual(changes["written"], ["/created.txt"])

    async def test_save_writes_workdir_as_zip(self):
        import os
        import tempfile
        import zipfile

        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/models.py", "class A:\n    pass\n")
        await sandbox.write_file("/pkg/util.py", "x = 1\n")
        with tempfile.TemporaryDirectory() as tmp:
            # Suffix is appended when missing.
            result = sandbox.save(os.path.join(tmp, "out"))
            self.assertTrue(result["path"].endswith("out.zip"))
            self.assertTrue(os.path.exists(result["path"]))
            self.assertEqual(result["files"], 2)
            with zipfile.ZipFile(result["path"]) as archive:
                names = set(archive.namelist())
                # Members are virtual paths without the leading slash.
                self.assertEqual(names, {"models.py", "pkg/util.py"})
                self.assertEqual(
                    archive.read("models.py").decode(), "class A:\n    pass\n"
                )

    async def test_save_unsupported_on_base_sandbox(self):
        with self.assertRaises(NotImplementedError):
            Sandbox().save("out.zip")

    async def test_patch_renders_git_style_unified_diff(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/app/main.py", "def f():\n    return 1\n")
        await sandbox.write_file("/old.txt", "delete me\n")
        (child,) = await sandbox.fork()
        await child.write_file("/app/main.py", "def f():\n    return 2\n")
        await child.write_file("/new.txt", "new")
        await child.files.remove("/old.txt")
        patch = child.patch()
        # Modify: a real hunk against the fork-point text.
        self.assertIn("diff --git a/app/main.py b/app/main.py", patch)
        self.assertIn("-    return 1", patch)
        self.assertIn("+    return 2", patch)
        # Create and delete use the /dev/null git convention.
        self.assertIn("new file mode 100644", patch)
        self.assertIn("--- /dev/null", patch)
        self.assertIn("deleted file mode 100644", patch)
        self.assertIn("+++ /dev/null", patch)
        # No trailing newline is flagged the way git does.
        self.assertIn("\\ No newline at end of file", patch)
        # ``paths`` restricts output to the selected file.
        only = child.patch(paths=["/new.txt"])
        self.assertIn("b/new.txt", only)
        self.assertNotIn("main.py", only)

    async def test_merge_applies_child_changes(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/base.txt", "base")
        (child,) = await sandbox.fork()
        await child.write_file("/new.txt", "from_child")
        report = sandbox.merge(child)
        self.assertIn("/new.txt", report["written"])
        self.assertEqual(report["conflicts"], [])
        read = await sandbox.read_file("/new.txt")
        self.assertEqual(read["content"], "from_child")

    async def test_merge_reports_failed_writes(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        (child,) = await sandbox.fork()
        await child.write_file("/ok.txt", "ok")
        await child.write_file("/bad.txt", "bad")
        real_write = sandbox.write

        async def flaky_write(path, data):
            if path == "/bad.txt":
                return "disk full"
            return await real_write(path, data)

        sandbox.write = flaky_write
        report = sandbox.merge(child)
        self.assertEqual(report["written"], ["/ok.txt"])
        self.assertEqual(report["failed"], {"/bad.txt": "disk full"})

    async def test_merge_conflict_refused_without_force(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        await sandbox.write_file("/shared.txt", "original")
        (child,) = await sandbox.fork()
        await child.write_file("/shared.txt", "child_version")
        # Parent diverges on the same path after the fork.
        await sandbox.write_file("/shared.txt", "parent_version")
        report = sandbox.merge(child)
        self.assertIn("/shared.txt", report["conflicts"])
        self.assertIn("/shared.txt", report["skipped"])
        read = await sandbox.read_file("/shared.txt")
        self.assertEqual(read["content"], "parent_version")
        # force applies the child's version.
        forced = sandbox.merge(child, force=True)
        self.assertIn("/shared.txt", forced["written"])
        read = await sandbox.read_file("/shared.txt")
        self.assertEqual(read["content"], "child_version")

    async def test_workdir_seeds_filesystem_host_safe(self):
        import os
        import shutil
        import tempfile

        workdir = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(workdir, "pkg"))
            with open(os.path.join(workdir, "main.py"), "w") as fh:
                fh.write("print('orig')\n")
            with open(os.path.join(workdir, "pkg", "mod.py"), "w") as fh:
                fh.write("VALUE = 1\n")
            sandbox = MirageSandbox(timeout=_TIMEOUT, workdir=workdir)
            listing = await sandbox.list_files("**/*.py")
            self.assertEqual(listing["files"], ["/main.py", "/pkg/mod.py"])
            # Editing in the sandbox never touches the host directory.
            await sandbox.write_file("/main.py", "print('edited')\n")
            with open(os.path.join(workdir, "main.py")) as fh:
                self.assertEqual(fh.read(), "print('orig')\n")
            self.assertEqual(sandbox.changes()["written"], ["/main.py"])
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


@unittest.skipUnless(_NAMESPACES_OK, _NS_REASON)
class MirageSandboxConfineTest(_SandboxTestCase):
    """In-process confinement (FUSE + user-namespace pivot), Linux-only."""

    async def test_confine_extra_binds_imports_host_dir(self):
        # A host directory bound via extra_binds is visible read-only inside the
        # confined sandbox and importable once added to sys.path.
        import os
        import shutil
        import tempfile

        host_dir = tempfile.mkdtemp()
        try:
            with open(os.path.join(host_dir, "hostlibxyz.py"), "w") as fh:
                fh.write("VALUE = 4242\n")
            sandbox = MirageSandbox(
                timeout=_TIMEOUT, confine=True, extra_binds=[host_dir]
            )
            try:
                result = await sandbox.run_code(
                    f"import sys\nsys.path.insert(0, {host_dir!r})\n"
                    "import hostlibxyz\nprint(hostlibxyz.VALUE)\n"
                )
                self.assertIsNone(result.error)
                self.assertIn("4242", _stdout(result))
                # ...and it's read-only (host import-poisoning guard holds).
                ro = await sandbox.run_code(
                    f"try:\n"
                    f"    open({os.path.join(host_dir, 'hostlibxyz.py')!r}, 'a')"
                    f".write('x')\n"
                    f"    print('WRITABLE')\n"
                    f"except OSError:\n"
                    f"    print('READONLY')\n"
                )
                self.assertIn("READONLY", _stdout(ro))
            finally:
                sandbox.close()
        finally:
            shutil.rmtree(host_dir, ignore_errors=True)

    async def test_confine_runs_and_persists_state(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            self.assertTrue(sandbox.confine)
            await sandbox.run_code("x = 41")
            result = await sandbox.run_code("print(x + 1)")
            self.assertIsNone(result.error)
            self.assertIn("42", _stdout(result))
            # third-party / stdlib imports still resolve inside the pivot
            imp = await sandbox.run_code("import json, math\nprint(math.floor(2.5))")
            self.assertIsNone(imp.error)
            self.assertIn("2", _stdout(imp))
        finally:
            sandbox.close()

    async def test_confine_hides_host_filesystem(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            result = await sandbox.run_code(
                "import os\nprint(os.path.exists('/etc/passwd'))"
            )
            self.assertIn("False", _stdout(result))
        finally:
            sandbox.close()

    async def test_confine_pid_namespace_hides_host_processes(self):
        # The PID namespace means the confined /proc shows only namespaced PIDs:
        # the snippet is PID 1 and no host processes are visible (defense in
        # depth so host /proc/<pid>/root and /environ aren't even reachable).
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            result = await sandbox.run_code(
                "import os\n"
                "pids = [int(p) for p in os.listdir('/proc') if p.isdigit()]\n"
                "print('mypid', os.getpid())\n"
                "print('maxpid', max(pids))\n"
                "print('count', len(pids))\n"
            )
            self.assertIsNone(result.error)
            # PID 1 in the new namespace, and only a handful of namespaced PIDs
            # (no sprawling host process table).
            self.assertIn("mypid 1", _stdout(result))
            lines = dict(ln.split(" ", 1) for ln in _stdout(result).strip().splitlines())
            self.assertLessEqual(int(lines["maxpid"]), 50)
        finally:
            sandbox.close()

    async def test_confine_cuts_network(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            result = await sandbox.run_code(
                "import socket\n"
                "try:\n"
                "    socket.create_connection(('1.1.1.1', 80), 2)\n"
                "    print('REACHABLE')\n"
                "except OSError:\n"
                "    print('CUT')\n"
            )
            self.assertIn("CUT", _stdout(result))
        finally:
            sandbox.close()

    async def test_confine_python_shares_virtual_filesystem(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            # a file written via the tool is visible to python at the same path
            await sandbox.write_file("/data.txt", "virtual-content")
            read = await sandbox.run_code("print(open('/data.txt').read().strip())")
            self.assertIn("virtual-content", _stdout(read))
            # a file python writes is visible to the tool
            await sandbox.run_code("open('/out.txt', 'w').write('from-python')")
            tool = await sandbox.read_file("/out.txt")
            self.assertEqual(tool["content"], "from-python")
        finally:
            sandbox.close()

    async def test_confine_host_tool_bridge_still_works(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)

        async def adder(x, y):
            return {"sum": x + y}

        try:
            # The host-tool bridge must also work synchronously under confinement.
            code = "print(adder(x=2, y=3)['sum'])\n"
            result = await sandbox.run_code(code, external_functions={"adder": adder})
            self.assertIsNone(result.error)
            self.assertIn("5", _stdout(result))
        finally:
            sandbox.close()

    async def test_confine_covers_run_bash_python(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            await sandbox.write_file("/v.txt", "virtual")
            await sandbox.write_file(
                "/probe.py",
                "import os\n"
                "print('passwd', os.path.exists('/etc/passwd'))\n"
                "print('vtxt', open('/v.txt').read())\n",
            )
            out = await sandbox.run_bash("python3 /probe.py")
            self.assertTrue(out["ok"], msg=out["stderr"])
            # host filesystem hidden, virtual filesystem visible at the same path
            self.assertIn("passwd False", out["stdout"])
            self.assertIn("vtxt virtual", out["stdout"])
            # a file the confined run_bash python writes is visible to the tools
            await sandbox.write_file(
                "/w.py", "open('/from_bash.txt', 'w').write('written')\n"
            )
            await sandbox.run_bash("python3 /w.py")
            self.assertEqual(
                (await sandbox.read_file("/from_bash.txt"))["content"], "written"
            )
        finally:
            sandbox.close()

    async def test_unconfined_run_bash_python_is_not_patched(self):
        # The _run_python patch is global once a confined sandbox exists, but it
        # must be a no-op for an unconfined sandbox.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        try:
            await sandbox.write_file(
                "/p.py", "import os\nprint(os.path.exists('/etc/passwd'))\n"
            )
            out = await sandbox.run_bash("python3 /p.py")
            self.assertIn("True", out["stdout"])
        finally:
            sandbox.close()

    async def test_confine_runtime_binds_are_read_only(self):
        # The Python runtime (venv / stdlib) is bound read-only, so confined
        # code cannot poison files the host later imports outside the sandbox.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True)
        try:
            import os as _os

            target = _os.path.join(_os.path.dirname(_os.__file__), "os.py")
            result = await sandbox.run_code(
                "import os\n"
                f"try:\n"
                f"    open({target!r}, 'a').write('x')\n"
                f"    print('WRITABLE')\n"
                f"except OSError as exc:\n"
                f"    print('READONLY')\n"
            )
            self.assertIsNone(result.error)
            self.assertIn("READONLY", _stdout(result))
        finally:
            sandbox.close()

    async def test_confine_seccomp_blocks_denied_syscall(self):
        # ``unshare`` is on the denylist; under seccomp it returns EPERM, so
        # os.unshare raises instead of succeeding (the no-op flags=0 call).
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True, seccomp=True)
        try:
            self.assertTrue(sandbox.granted_capabilities()["seccomp"])
            result = await sandbox.run_code(
                "import os\n"
                "try:\n"
                "    os.unshare(0)\n"
                "    print('ALLOWED')\n"
                "except OSError:\n"
                "    print('BLOCKED')\n"
            )
            self.assertIsNone(result.error)
            self.assertIn("BLOCKED", _stdout(result))
        finally:
            sandbox.close()

    async def test_confine_without_seccomp_allows_syscall(self):
        # With seccomp disabled the same call is not blocked by a filter; proves
        # the block above comes from seccomp, not the namespace.
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True, seccomp=False)
        try:
            self.assertFalse(sandbox.granted_capabilities()["seccomp"])
            result = await sandbox.run_code(
                "import os\n"
                "try:\n"
                "    os.unshare(0)\n"
                "    print('ALLOWED')\n"
                "except OSError:\n"
                "    print('BLOCKED')\n"
            )
            self.assertIsNone(result.error)
            self.assertIn("ALLOWED", _stdout(result))
        finally:
            sandbox.close()

    async def test_confine_granted_capabilities(self):
        sandbox = MirageSandbox(
            timeout=_TIMEOUT, confine=True, allowed_hosts=["api.example.com"]
        )
        try:
            caps = sandbox.granted_capabilities()
            self.assertTrue(caps["confined"])
            self.assertTrue(caps["seccomp"])
            self.assertTrue(caps["read_only_runtime"])
            self.assertEqual(caps["network"]["mode"], "allowlist")
            self.assertIn("http_fetch", caps["tools"])
        finally:
            sandbox.close()

    async def test_require_confinement_runs_when_available(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=True, require_confinement=True)
        try:
            self.assertTrue(sandbox.confine)
            result = await sandbox.run_code("print(40 + 2)")
            self.assertIsNone(result.error)
            self.assertIn("42", _stdout(result))
        finally:
            sandbox.close()

    async def test_fork_confine_none_inherits_parent(self):
        # confine=None (the subagent default) inherits the parent's confinement:
        # confined parent -> confined fork; unconfined parent -> unconfined fork.
        confined = MirageSandbox(timeout=_TIMEOUT, confine=True)
        unconfined = MirageSandbox(timeout=_TIMEOUT, confine=False)
        c_child = u_child = None
        try:
            (c_child,) = await confined.fork(confine=None)
            (u_child,) = await unconfined.fork(confine=None)
            self.assertTrue(c_child.confine)
            self.assertFalse(u_child.confine)
        finally:
            for sandbox in (confined, unconfined, c_child, u_child):
                if sandbox is not None:
                    sandbox.close()

    async def test_confined_fork_inherits_parent_security_posture(self):
        # A subagent's confined fork transparently carries the parent's whole
        # security config (egress allowlist, SSRF guard, extra binds, mount
        # modes, fail-closed), so confinement isn't something the subagent opts
        # into; it inherits it.
        parent = MirageSandbox(
            timeout=_TIMEOUT,
            confine=True,
            require_confinement=True,
            allowed_hosts=["api.example.com"],
            extra_binds=["/usr/share"],
        )
        child = None
        try:
            (child,) = await parent.fork(name="sub", confine=True)
            self.assertTrue(child.confine)
            pcaps = parent.granted_capabilities()
            ccaps = child.granted_capabilities()
            for key in ("confined", "seccomp", "read_only_runtime", "network"):
                self.assertEqual(ccaps[key], pcaps[key])
            self.assertIn("http_fetch", ccaps["tools"])  # egress tool inherited
            self.assertTrue(child.require_confinement)  # fail-closed propagates
            self.assertEqual(child.extra_binds, parent.extra_binds)
            self.assertTrue(child.block_private_egress)
            # and it is genuinely confined to ITS OWN fork (host hidden)
            r = await child.run_code("import os\nprint(os.path.exists('/etc/passwd'))")
            self.assertIn("False", _stdout(r))
        finally:
            parent.close()
            if child is not None:
                child.close()

    async def test_fork_confine_isolates_child_to_its_fork(self):
        # An unconfined parent can hand a subagent a confined fork: the child is
        # locked to its own forked filesystem; the parent stays unconfined.
        parent = MirageSandbox(timeout=_TIMEOUT, confine=False)
        child = None
        try:
            await parent.write_file("/shared.txt", "parent-data")
            (child,) = await parent.fork(confine=True, name="sub")
            self.assertTrue(child.confine)
            # child sees the forked copy at the same path, host hidden
            r = await child.run_code(
                "import os\n"
                "print('shared', open('/shared.txt').read().strip())\n"
                "print('passwd', os.path.exists('/etc/passwd'))\n"
            )
            self.assertIn("shared parent-data", _stdout(r))
            self.assertIn("passwd False", _stdout(r))
            # child writes stay in the child fork; the parent never sees them
            await child.run_code("open('/childonly.txt', 'w').write('x')")
            self.assertIn("error", await parent.read_file("/childonly.txt"))
            # the parent itself is unconfined
            rp = await parent.run_code("import os\nprint(os.path.exists('/etc/passwd'))")
            self.assertIn("True", _stdout(rp))
        finally:
            parent.close()
            if child is not None:
                child.close()


class ShellPythonConfinementTest(_SandboxTestCase):
    """`confine_shell_python` wraps one workspace's ``local`` runtime so
    `run_bash`'s ``python3`` confines itself, without patching Mirage's class
    (which every other workspace and Mirage user in the process shares)."""

    async def test_wraps_the_instance_not_the_class(self):
        from mirage.runtime.python.local import LocalRuntime

        from synalinks.src.sandboxes import mirage_sandbox as ms

        self.assertTrue(ms.local_runtime_supported())
        class_run = LocalRuntime.run
        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        other = MirageSandbox(timeout=_TIMEOUT, confine=False)
        self.assertTrue(ms.confine_shell_python(sandbox.workspace))
        runtime = sandbox.workspace._registry.runtime_bindings["python3"]
        self.assertTrue(runtime.run.confines_shell_python)
        # Idempotent, and scoped to this one workspace.
        wrapped = runtime.run
        self.assertTrue(ms.confine_shell_python(sandbox.workspace))
        bindings = sandbox.workspace._registry.runtime_bindings
        self.assertIs(bindings["python3"].run, wrapped)
        self.assertIs(LocalRuntime.run, class_run)
        other_run = other.workspace._registry.runtime_bindings["python3"].run
        self.assertFalse(getattr(other_run, "confines_shell_python", False))

    async def test_wrapped_runtime_passes_code_through_when_inactive(self):
        from synalinks.src.sandboxes import mirage_sandbox as ms

        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        ms.confine_shell_python(sandbox.workspace)
        # No confined shell command is active, so python3 runs unmodified.
        result = await sandbox.run_bash("python3 -c 'print(6 * 7)'")
        self.assertEqual(result["exit_code"], 0, result["stderr"])
        self.assertIn("42", result["stdout"])


class InfraSelfHealTest(_SandboxTestCase):
    """A dead FUSE mount / confinement-bootstrap failure makes every `run`
    repeat the same infra error; an agent would loop on it until timeout. `run`
    must detect this, rebuild the workspace, and retry once instead."""

    async def test_run_heals_workspace_and_retries_on_infra_failure(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        calls = {"exec": 0, "rebuild": 0}

        async def fake_execute(command, *, stdin=None, timeout=None, record=False):
            calls["exec"] += 1
            if calls["exec"] == 1:
                return ("", "confine-error: OSError(107, 'Transport endpoint ...')", 99)
            return ("healed\n", "", 0)

        sandbox.execute = fake_execute
        sandbox.rebuild_workspace = lambda: calls.__setitem__(
            "rebuild", calls["rebuild"] + 1
        )
        try:
            result = await sandbox.run_code("print('x')")
        finally:
            sandbox.close()
        # exactly one rebuild, two execute attempts, and the healed run wins.
        self.assertEqual(calls["rebuild"], 1)
        self.assertEqual(calls["exec"], 2)
        self.assertIsNone(result.error)
        self.assertIn("healed", _stdout(result))

    async def test_run_gives_up_after_one_heal(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        calls = {"exec": 0, "rebuild": 0}

        async def fake_execute(command, *, stdin=None, timeout=None, record=False):
            calls["exec"] += 1
            return ("", "confine-error: still broken", 99)

        sandbox.execute = fake_execute
        sandbox.rebuild_workspace = lambda: calls.__setitem__(
            "rebuild", calls["rebuild"] + 1
        )
        try:
            result = await sandbox.run_code("print('x')")
        finally:
            sandbox.close()
        # one heal, two attempts, then the infra error surfaces (no infinite loop).
        self.assertEqual(calls["rebuild"], 1)
        self.assertEqual(calls["exec"], 2)
        self.assertIsNotNone(result.error)
        self.assertIn("confine-error", result.error.value)

    async def test_run_bash_heals_workspace_and_retries_on_infra_failure(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        calls = {"exec": 0, "rebuild": 0}

        async def fake_execute(command, *, stdin=None, timeout=None, record=False):
            calls["exec"] += 1
            if calls["exec"] == 1:
                return ("", "OSError(107, 'Transport endpoint is not connected')", 99)
            return ("ok\n", "", 0)

        sandbox.execute = fake_execute
        sandbox.rebuild_workspace = lambda: calls.__setitem__(
            "rebuild", calls["rebuild"] + 1
        )
        try:
            result = await sandbox.run_bash("ls /")
        finally:
            sandbox.close()
        self.assertEqual(calls["rebuild"], 1)
        self.assertEqual(calls["exec"], 2)
        self.assertTrue(result["ok"])
        self.assertIn("ok", result["stdout"])

    async def test_run_does_not_heal_on_ordinary_snippet_error(self):
        sandbox = MirageSandbox(timeout=_TIMEOUT)
        calls = {"exec": 0, "rebuild": 0}

        async def fake_execute(command, *, stdin=None, timeout=None, record=False):
            calls["exec"] += 1
            return ("", "Traceback ...\nValueError: boom", 1)

        sandbox.execute = fake_execute
        sandbox.rebuild_workspace = lambda: calls.__setitem__(
            "rebuild", calls["rebuild"] + 1
        )
        try:
            result = await sandbox.run_code("raise ValueError('boom')")
        finally:
            sandbox.close()
        # a genuine snippet error must not trigger a rebuild/retry.
        self.assertEqual(calls["rebuild"], 0)
        self.assertEqual(calls["exec"], 1)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.name, "ValueError")


class SbplProfileTest(testing.TestCase):
    """Unit tests for the generated Seatbelt profile.

    Pure string generation, so these run on every platform. They are the only
    part of the macOS backend that can be checked off a Darwin host, which
    makes them the first line of defence against a profile that silently grants
    more than intended.
    """

    def _build(self, **overrides):
        cfg = {
            "read_paths": ["/usr", "/System"],
            "rw_paths": ["/private/tmp/sbx"],
            "network": False,
        }
        cfg.update(overrides)
        ns = {}
        exec(MACOS_CONFINE_SRC, ns)
        return ns["_build_sbpl"](cfg)

    def test_profile_is_default_deny(self):
        # Everything else in the profile is an exception to this line; without
        # it the profile grants the world.
        profile = self._build()
        self.assertIn("(version 1)", profile)
        self.assertIn("(deny default)", profile)

    def test_read_paths_are_read_only(self):
        profile = self._build()
        self.assertIn('(allow file-read* (subpath "/usr"))', profile)
        # The runtime set must never become writable: that is what stops the
        # snippet rewriting the interpreter it is about to run under.
        self.assertNotIn('(allow file-read* file-write* (subpath "/usr"))', profile)

    def test_rw_paths_are_writable(self):
        profile = self._build()
        self.assertIn(
            '(allow file-read* file-write* (subpath "/private/tmp/sbx"))', profile
        )

    def test_network_denied_by_default(self):
        profile = self._build()
        # `(deny default)` covers everything; nothing re-opens it.
        self.assertNotIn("(allow network*)", profile)

    def test_host_loopback_is_never_granted(self):
        # macOS has no network namespace, so `(local ip)` would be the *host's*
        # 127.0.0.1: every service bound to localhost would be reachable from
        # code that is supposed to have no network at all. Assert both
        # directions, and under both network settings, since the blanket
        # `(allow network*)` is the only intended way to get out.
        for network in (False, True):
            profile = self._build(network=network)
            self.assertNotIn("(local ip)", profile)
            self.assertNotIn("network-inbound", profile)

    def test_network_allowed_when_requested(self):
        profile = self._build(network=True)
        self.assertIn("(allow network*)", profile)

    def test_rpc_socket_dir_is_allowed_even_without_network(self):
        # The host-tool bridge is a unix socket, which Seatbelt treats as
        # network; without this the bridge would break under the denial above.
        profile = self._build(rpc_socket_dir="/private/tmp/sbx")
        self.assertIn('(allow network-outbound (subpath "/private/tmp/sbx"))', profile)

    def test_process_info_is_denied_except_for_self(self):
        # Other processes: denied, since that is the reconnaissance path. Self:
        # allowed back afterwards (last-match-wins), because a process reading
        # its own proc info is reading its own memory, and dyld does exactly
        # that while starting a freshly exec'd image.
        profile = self._build()
        deny = "(deny process-info*)"
        allow = "(allow process-info* (target self))"
        self.assertIn(deny, profile)
        self.assertGreater(profile.index(allow), profile.index(deny))

    def test_mach_lookup_is_never_unrestricted(self):
        # A bare `(allow mach-lookup)` lets the process ask any daemon to work
        # on its behalf, and the daemon's child does not inherit this profile,
        # so the filesystem and network rules would not apply to it. Every
        # grant must name a specific service.
        profile = self._build()
        self.assertNotIn("(allow mach-lookup)\n", profile)
        for line in profile.splitlines():
            if "mach-lookup" in line:
                self.assertIn("global-name", line)

    def test_mach_lookup_allows_only_startup_services(self):
        # Each entry is IPC into a privileged daemon and is here on evidence: a
        # child exec'd under the profile runs libSystem startup *inside* it, and
        # a denied bootstrap lookup aborts it rather than degrading.
        profile = self._build()
        self.assertIn(
            '(allow mach-lookup (global-name "com.apple.system.notification_center"))',
            profile,
        )
        # Process-spawning brokers are the escape route the allowlist exists to
        # close, so they must never appear whatever else is added.
        self.assertNotIn("launchservicesd", profile)
        self.assertNotIn("com.apple.launchd", profile)

    def test_exec_is_limited_to_the_read_only_runtime_set(self):
        # Host tools stay reachable, as they are on Linux, and a child inherits
        # the profile. What must not be executable is anything the snippet can
        # write: the writable area is not in the exec set.
        profile = self._build(rw_paths=["/private/tmp/sbx"])
        self.assertIn('(allow process-exec (subpath "/usr"))', profile)
        self.assertNotIn('(allow process-exec (subpath "/private/tmp/sbx"))', profile)

    def test_writable_area_is_not_executable(self):
        # Write-xor-execute: a snippet must not be able to drop a dylib into its
        # own scratch dir and `dlopen` it. SBPL is last-match-wins, so the deny
        # has to come *after* the grant it carves into.
        profile = self._build(rw_paths=["/private/tmp/sbx"])
        deny = '(deny file-map-executable (subpath "/private/tmp/sbx"))'
        allow = '(allow file-read* file-write* (subpath "/private/tmp/sbx"))'
        self.assertIn(deny, profile)
        self.assertGreater(profile.index(deny), profile.index(allow))

    def test_runtime_set_is_mappable_executable(self):
        # The other half of write-xor-execute, and not implied by `file-read*`:
        # `file-map-executable` is its own operation, so an exec'd binary and
        # every dylib dyld then loads need it. Without this the child starts
        # and dies in dyld, silently, which is not a failure mode anyone enjoys
        # diagnosing from a CI log.
        profile = self._build()
        self.assertIn('(allow file-map-executable (subpath "/usr"))', profile)
        self.assertIn('(allow file-map-executable (subpath "/System"))', profile)

    def test_symlinked_ancestors_are_traversable(self):
        # Reaching an allowed path means resolving every symlink on the way to
        # it, which `(deny default)` also denies. macOS puts the sandbox's own
        # dirs under `/var/folders/...`, where `/var` is a symlink, so without
        # this the `file-write*` grant below can never match and writes into the
        # sandbox's own area fail while the profile looks correct.
        profile = self._build(rw_paths=["/private/var/folders/ab/cd/T/sbx"])
        for path in ("/var", "/private/var/folders/ab/cd/T", "/private/var/folders"):
            self.assertIn('(allow file-read-metadata (literal "%s"))' % path, profile)

    def test_traversal_grant_is_metadata_only(self):
        # The traversal grants must not become read access: `file-read*` on an
        # ancestor would hand the snippet the contents of everything under it.
        profile = self._build(rw_paths=["/private/var/folders/ab/cd/T/sbx"])
        for line in profile.splitlines():
            if "/private/var/folders/ab/cd" in line and "sbx" not in line:
                self.assertIn("file-read-metadata", line)
        self.assertNotIn('(allow file-read* (subpath "/"))', profile)
        self.assertNotIn('(allow file-read* (subpath "/var"))', profile)

    def test_root_directory_is_readable_not_just_traversable(self):
        # dyld4's CacheFinder locates the shared-cache cryptex by *reading* the
        # root directory (macOS 26), and a denied read there halts a freshly
        # exec'd child in `ignition_halt`: SIGABRT, empty stderr, before dyld
        # can load a single dylib. Metadata on `/` is not enough. The grant is
        # a `literal`, so it exposes the well-known top-level names and nothing
        # of any file's contents.
        profile = self._build()
        self.assertIn('(allow file-read-data (literal "/"))', profile)
        self.assertNotIn('(allow file-read-data (subpath "/"))', profile)

    def test_working_directory_is_traversable(self):
        # `getcwd` is a path operation, so a cwd outside the granted set fails
        # EPERM, and CPython calls it on the first import after confinement
        # (`python3 -c` puts "" at the head of `sys.path`), catching only
        # `FileNotFoundError`. Without this the failure arrives as a
        # `PermissionError` from the import machinery, nowhere near a file.
        profile = self._build()
        cwd = os.getcwd()
        self.assertIn('(allow file-read-metadata (literal "%s"))' % cwd, profile)
        # Metadata, not contents: `listdir` stays denied, which the import
        # machinery handles as an empty directory and moves on.
        self.assertNotIn('(allow file-read* (subpath "%s"))' % cwd, profile)

    def test_rpc_socket_literal_is_allowed_when_known(self):
        # `subpath` filters on `network-outbound` are far less attested than
        # `literal`, and the host knows the socket path by the time the profile
        # is rendered, so the exact path is granted as well.
        profile = self._build(
            rpc_socket_dir="/private/tmp/sbx", sock="/private/tmp/sbx/rpc_1.sock"
        )
        self.assertIn(
            '(allow network-outbound (literal "/private/tmp/sbx/rpc_1.sock"))', profile
        )

    def test_process_table_sysctls_are_denied(self):
        # `kern.proc.*` / `kern.procargs*` hand over every host process and its
        # arguments; the deny must follow the blanket sysctl-read grant, since
        # SBPL is last-match-wins.
        profile = self._build()
        deny = '(deny sysctl-read (sysctl-name-prefix "kern.proc"))'
        self.assertIn(deny, profile)
        self.assertGreater(profile.index(deny), profile.index("(allow sysctl-read)"))

    def test_signals_reach_only_self_and_children(self):
        # Children, so `Popen.kill()` and subprocess timeouts work; nothing
        # else, so the snippet cannot signal host processes.
        profile = self._build()
        signals = [line for line in profile.splitlines() if "(allow signal" in line]
        self.assertEqual(
            signals, ["(allow signal (target self))", "(allow signal (target children))"]
        )

    def test_bin_sh_shell_selector_is_readable(self):
        # macOS `/bin/sh` reads this symlink to pick the real shell; denied, it
        # still runs but writes an error to every shelled-out command's stderr.
        # The grant is the one link, not the directory holding it.
        profile = self._build()
        self.assertIn('(allow file-read* (literal "/private/var/select/sh"))', profile)
        self.assertNotIn('(subpath "/private/var/select")', profile)

    def test_paths_with_quotes_are_escaped(self):
        # SBPL is s-expression syntax: an unescaped quote in a path would end
        # the string early and could terminate the rule, so a crafted directory
        # name must not be able to edit the profile.
        profile = self._build(rw_paths=['/tmp/a"b'])
        self.assertIn(r'"/tmp/a\"b"', profile)
        self.assertNotIn('"/tmp/a"b"', profile)


class MacosProcessPrologueTest(testing.TestCase):
    """The pre-profile half of the macOS bootstrap, which is plain POSIX.

    Runs on every platform, like `SbplProfileTest`: it is the part of the
    Seatbelt path that decides *where the process stands* when the profile
    lands, and standing in the wrong place is what a profile cannot fix.
    """

    def _prepare(self, cfg):
        ns = {}
        exec(MACOS_CONFINE_SRC, ns)
        cwd = os.getcwd()
        path = list(sys.path)
        env = os.environ.get("TMPDIR")
        tempdir = tempfile.tempdir

        def _restore():
            os.chdir(cwd)
            sys.path[:] = path
            tempfile.tempdir = tempdir
            if env is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = env

        self.addCleanup(_restore)
        ns["_prepare_macos_process"](cfg)

    def test_tempdir_points_into_the_sandbox(self):
        # The host temp dir is deliberately not granted, so a snippet reaching
        # for scratch space has to land inside the sandbox's own area.
        scratch = self.get_temp_dir()
        self._prepare({"tmpdir": scratch})
        self.assertEqual(os.environ["TMPDIR"], scratch)
        self.assertEqual(tempfile.gettempdir(), scratch)

    def test_working_directory_moves_into_the_sandbox(self):
        # `getcwd` is a path operation: a cwd outside the granted set fails
        # EPERM under the profile, and metadata on it is not enough. Being
        # somewhere the profile grants outright is.
        scratch = self.get_temp_dir()
        self._prepare({"tmpdir": scratch})
        self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(scratch))

    def test_working_directory_prefers_the_mount(self):
        # The process stands in the FUSE mount, the directory `pivot_root` makes
        # "/" on Linux, so a relative path lands on the virtual filesystem the
        # file tools see rather than in the scratch dir.
        mount = self.get_temp_dir()
        scratch = tempfile.mkdtemp()
        self.addCleanup(os.rmdir, scratch)
        self._prepare({"tmpdir": scratch, "cwd": mount})
        self.assertEqual(os.path.realpath(os.getcwd()), os.path.realpath(mount))
        # TMPDIR still points at the scratch dir, not the mount.
        self.assertEqual(tempfile.gettempdir(), scratch)

    def test_empty_sys_path_entry_is_dropped(self):
        # CPython resolves "" through `getcwd` on the first import after the
        # profile applies, catching only `FileNotFoundError`, so an EPERM there
        # kills an import that a granted `sys.path` entry would have served.
        sys.path.insert(0, "")
        self._prepare({"tmpdir": self.get_temp_dir()})
        self.assertNotIn("", sys.path)
        # ...and only that entry: the runtime set has to survive intact.
        self.assertIn(os.path.dirname(os.__file__), sys.path)

    def test_nproc_rlimit_is_a_budget_over_running_processes(self):
        # RLIMIT_NPROC counts every process of the user on macOS, so the cap is
        # set to what is running plus `max_processes`, not to the bare number
        # (which would stop the sandbox forking at all). Checked in a child
        # interpreter: the limit applies to the process that sets it.
        import subprocess

        script = (
            "import resource, sys\n"
            "ns = {}\n"
            "exec(sys.stdin.read(), ns)\n"
            "ns['_prepare_macos_process']({'rlimits': {'nproc': 8}})\n"
            "print(resource.getrlimit(resource.RLIMIT_NPROC)[0])\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", script],
            input=MACOS_CONFINE_SRC,
            capture_output=True,
            text=True,
            check=True,
        )
        limit = int(out.stdout.strip())
        # More than the bare budget (the user already runs processes), and
        # the budget still fits under it.
        self.assertGreater(limit, 8)
        import resource

        _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        self.assertLessEqual(limit, hard)

    def test_prologue_without_a_scratch_dir_is_harmless(self):
        # `tmpdir` is absent for an unconfined run and on the `run_bash` path
        # before a scratch dir exists; the prologue must not throw.
        here = os.getcwd()
        self._prepare({})
        self.assertEqual(os.getcwd(), here)


@unittest.skipUnless(_SEATBELT_OK, _SB_REASON)
class MacosSeatbeltConfineTest(_SandboxTestCase):
    """Behavioural checks for the macOS backend, on a real Darwin host.

    These assert the boundary actually holds, rather than that the profile
    merely renders. Seatbelt gives no PID isolation and no syscall filter, so
    there is deliberately no counterpart here to the namespace suite's PID and
    seccomp tests.
    """

    @staticmethod
    def _diag(execution):
        """Failure message carrying the sandbox's own stdout and stderr.

        A profile that denies something it should not usually surfaces as a
        *downstream* symptom (a `NameError` for state that never persisted, a
        missing result), while the bootstrap's warning naming the denied path
        goes to stderr, so the message has to carry it.
        """
        return "error=%s\nstdout=%s\nstderr=%s" % (
            execution.error,
            _stdout(execution),
            "".join(execution.logs.stderr),
        )

    def sandbox(self, **kwargs):
        # Seatbelt is the floor under the microVM: force it, so it stays
        # covered on a Mac where the microVM is what a sandbox would pick.
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        with mock.patch.object(
            mirage_sandbox, "confinement_backend", return_value=("seatbelt", "forced")
        ):
            sandbox = MirageSandbox(timeout=_TIMEOUT, **kwargs)
        self.addCleanup(sandbox.close)
        return sandbox

    async def test_capabilities_report_seatbelt(self):
        caps = self.sandbox().granted_capabilities()
        self.assertTrue(caps["confined"])
        self.assertEqual(caps["backend"], "seatbelt")
        self.assertFalse(caps["seccomp"])
        self.assertEqual(caps["network"], {"mode": "cut"})

    async def test_confined_run_keeps_state(self):
        # The boundary is worthless if it also breaks execution: the snippet
        # must still import, run and keep state across calls. Assert on the
        # *first* run too: it writes the dill state into the sandbox's own dir,
        # so a profile that cannot reach that dir fails here.
        sandbox = self.sandbox()
        first = await sandbox.run_code("import json\nkeep = json.dumps({'a': 1})")
        self.assertIsNone(first.error, msg=self._diag(first))
        second = await sandbox.run_code("print(keep)")
        self.assertIsNone(second.error, msg=self._diag(second))
        self.assertIn('{"a": 1}', _stdout(second))

    async def test_confined_run_captures_result(self):
        # The last-expression value travels back through a file in the sandbox
        # dir, so this independently checks that writes there land.
        execution = await self.sandbox().run_code("6 * 7")
        self.assertIsNone(execution.error, msg=self._diag(execution))
        self.assertEqual(execution.text, "42", msg=self._diag(execution))

    async def test_relative_writes_land_on_the_virtual_filesystem(self):
        # The process stands in the FUSE mount, so the snippet's own `open()`
        # and the file tools see one filesystem, as they do on Linux.
        sandbox = self.sandbox()
        if not sandbox.fuse_mountpoint:
            self.skipTest("needs macFUSE: without it Seatbelt confines, unshared")
        execution = await sandbox.run_code("open('note.txt', 'w').write('virt')")
        self.assertIsNone(execution.error, msg=self._diag(execution))
        read = await sandbox.read_file("/note.txt")
        self.assertEqual(read["content"], "virt")

    async def _attempt(self, sandbox, attempt):
        """Run ``attempt`` and return what it printed: its marker or the error."""
        execution = await sandbox.run_code(
            "try:\n"
            + "".join("    " + line + "\n" for line in attempt.splitlines())
            + "except Exception as exc:\n"
            "    print(type(exc).__name__)\n"
        )
        self.assertIsNone(execution.error, msg=self._diag(execution))
        return _stdout(execution)

    async def test_confine_blocks_write_outside_sandbox(self):
        target = os.path.expanduser("~/synalinks_seatbelt_escape.txt")
        out = await self._attempt(
            self.sandbox(), f"open({target!r}, 'w').write('x')\nprint('WROTE')"
        )
        self.assertNotIn("WROTE", out)
        self.assertIn("PermissionError", out)
        self.assertFalse(os.path.exists(target))

    async def test_confine_blocks_writes_to_the_seed_workdir(self):
        # `workdir` is a seed, copied into the virtual filesystem host-side, and
        # the agent's writes must never reach the real directory. Linux gets
        # that from the pivot; macOS is the one platform where the directory is
        # still *there*, so it is the one platform where this has to be asserted.
        import shutil

        workdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, workdir, True)
        host_file = os.path.join(workdir, "main.py")
        with open(host_file, "w") as fh:
            fh.write("print('orig')\n")
        out = await self._attempt(
            self.sandbox(workdir=workdir),
            f"open({host_file!r}, 'w').write('escaped')\nprint('WROTE')",
        )
        self.assertNotIn("WROTE", out)
        with open(host_file) as fh:
            self.assertEqual(fh.read(), "print('orig')\n")

    async def test_confine_blocks_read_outside_runtime_set(self):
        # Reads are restricted, not only writes. /etc/passwd is readable on any
        # unconfined mac.
        out = await self._attempt(
            self.sandbox(), "open('/etc/passwd').read()\nprint('READ')"
        )
        self.assertNotIn("READ", out)
        self.assertIn("PermissionError", out)

    async def test_confine_cuts_network(self):
        out = await self._attempt(
            self.sandbox(),
            "import socket\n"
            "s = socket.socket()\n"
            "s.settimeout(5)\n"
            "s.connect(('1.1.1.1', 80))\n"
            "print('CONNECTED')",
        )
        self.assertNotIn("CONNECTED", out)

    async def test_subprocesses_run_but_inherit_the_profile(self):
        # Shelling out works as on Linux, and that is safe for one reason only:
        # the child inherits the profile. Assert both halves, since the escape
        # check alone is satisfied just as well by a child that never started.
        escape = os.path.expanduser("~/synalinks_subprocess_escape.txt")
        execution = await self.sandbox().run_code(
            "import subprocess\n"
            "ran = subprocess.run(['/bin/sh', '-c', 'echo alive'],"
            " capture_output=True, text=True)\n"
            "print('RC', ran.returncode, 'OUT', ran.stdout.strip(),"
            " 'ERR', ran.stderr.strip())\n"
            f"subprocess.run(['/bin/sh', '-c', 'echo x > {escape}'],"
            " capture_output=True)\n"
        )
        self.assertIsNone(execution.error, msg=self._diag(execution))
        self.assertIn("RC 0 OUT alive ERR", _stdout(execution), msg=self._diag(execution))
        self.assertTrue(_stdout(execution).strip().endswith("ERR"), self._diag(execution))
        self.assertFalse(os.path.exists(escape))

    async def test_run_bash_python_is_confined(self):
        # `run_bash`'s python3 gets the same prologue as `run_code`.
        result = await self.sandbox().run_bash(
            "python3 -c \"open('/etc/passwd').read()\" 2>&1 | tail -1"
        )
        self.assertIn("PermissionError", result["stdout"])


class MicrovmBridgeTest(testing.TestCase):
    """The microVM backend's host-side pieces that need no VM: they run anywhere.

    The wire codec, errno translation, the helper's Seatbelt profile and the
    image checks are where a mistake silently widens what a guest can reach,
    so they are pinned down here rather than only through a booted VM.
    """

    def test_codec_round_trips_bytes_and_nesting(self):
        from synalinks.src.utils.microvm_utils import codec

        value = {"data": b"\x00\xffbin", "list": [1, "a", None, [b"x"]], "ok": True}
        self.assertEqual(codec["decode"](codec["encode"](value)), value)

    def test_codec_drops_objects_it_cannot_carry(self):
        # A ctypes fuse_file_info (or anything else) must not reach the host
        # as a live object; it crosses as None.
        from synalinks.src.utils.microvm_utils import codec

        self.assertEqual(codec["encode"]([object()]), [None])

    def _bridge_call(self, fs, op, *args):
        import socket

        from synalinks.src.utils.microvm_utils import FsBridge
        from synalinks.src.utils.microvm_utils import codec

        path = os.path.join(tempfile.mkdtemp(), "b.sock")
        bridge = FsBridge(fs, path)
        self.addCleanup(bridge.close)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(path)
            codec["send_frame"](sock, {"op": op, "args": codec["encode"](list(args))})
            return codec["recv_frame"](sock)

    def test_bridge_sends_errno_by_name(self):
        # errno numbers differ between macOS and Linux (ENOTEMPTY is 66 on
        # one, 39 on the other); a number would mean a different error there.
        import errno as errno_mod

        class Fs:
            def rmdir(self, path):
                raise OSError(errno_mod.ENOTEMPTY, "not empty")

        self.assertEqual(self._bridge_call(Fs(), "rmdir", "/d"), {"errno": "ENOTEMPTY"})

    def test_bridge_refuses_operations_outside_the_allowlist(self):
        class Fs:
            def drain_ops(self):  # a real MirageFS method, not a FUSE op
                return ["secret"]

        self.assertEqual(self._bridge_call(Fs(), "drain_ops"), {"errno": "ENOSYS"})

    def test_bridge_returns_bytes_intact(self):
        class Fs:
            def read(self, path, size, offset, fh):
                return b"\x00\x01payload"

        from synalinks.src.utils.microvm_utils import codec

        reply = self._bridge_call(Fs(), "read", "/f", 9, 0, 1)
        self.assertEqual(codec["decode"](reply["result"]), b"\x00\x01payload")

    def _profile(self, **overrides):
        from synalinks.src.utils.microvm_utils import helper_profile

        vm = {
            "helper": "/Users/u/Library/Caches/synalinks/microvm/synalinks-krun-1",
            "rootfs": "/Users/u/Library/Caches/synalinks/microvm/rootfs-1",
            "libdir": "/opt/homebrew/lib",
            "network": False,
        }
        vm.update(overrides)
        return helper_profile(vm, "/private/var/folders/x/T/mirage_sandbox_1")

    def test_helper_profile_is_default_deny(self):
        self.assertIn("(deny default)", self._profile())

    def test_helper_profile_writes_only_the_host_dir(self):
        profile = self._profile()
        writes = [line for line in profile.splitlines() if "file-write" in line]
        self.assertEqual(
            writes,
            [
                "(allow file-read* file-write* (subpath "
                '"/private/var/folders/x/T/mirage_sandbox_1"))'
            ],
        )
        # ...where nothing may be mapped executable, after the grant.
        deny = (
            "(deny file-map-executable "
            '(subpath "/private/var/folders/x/T/mirage_sandbox_1"))'
        )
        self.assertGreater(profile.index(deny), profile.index(writes[0]))

    def test_helper_profile_keeps_homebrew_state_out(self):
        # Homebrew's `var` holds databases and service state: only libkrun's
        # libraries are granted, not the whole prefix.
        profile = self._profile()
        self.assertNotIn('(subpath "/opt/homebrew")', profile)
        self.assertNotIn("/opt/homebrew/var", profile)
        self.assertIn('(subpath "/opt/homebrew/Cellar")', profile)

    def test_helper_profile_network_only_when_requested(self):
        self.assertNotIn("(allow network*)", self._profile())
        self.assertIn("(allow network*)", self._profile(network=True))

    def test_downloads_are_verified(self):
        from synalinks.src.utils.microvm_utils import verified
        from synalinks.src.utils.microvm_utils import verified_sha256

        good = __import__("hashlib").sha256(b"blob").hexdigest()
        self.assertEqual(verified_sha256(b"blob", good), b"blob")
        with self.assertRaises(ValueError):
            verified_sha256(b"tampered", good)
        with self.assertRaises(ValueError):
            verified(b"blob", "md5:" + good)

    def test_layer_whiteouts_are_applied(self):
        import io
        import tarfile

        from synalinks.src.utils.microvm_utils import extract_layer

        def layer(entries):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                for name, data in entries:
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            return buf.getvalue()

        root = tempfile.mkdtemp()
        extract_layer(layer([("etc/keep", b"1"), ("etc/gone", b"2")]), root)
        extract_layer(layer([("etc/.wh.gone", b"")]), root)
        self.assertEqual(sorted(os.listdir(os.path.join(root, "etc"))), ["keep"])


@unittest.skipUnless(_MICROVM_OK, _VM_REASON)
class MicrovmConfineTest(_SandboxTestCase):
    """The macOS microVM backend, on a Mac that can run it.

    The Linux backend's own suite also runs against it (the guest confines
    with that code), so these cover what is specific to the VM boundary.
    """

    def sandbox(self, **kwargs):
        sandbox = MirageSandbox(timeout=_TIMEOUT, **kwargs)
        self.addCleanup(sandbox.close)
        return sandbox

    async def test_capabilities_report_microvm(self):
        caps = self.sandbox().granted_capabilities()
        self.assertEqual(caps["backend"], "microvm")
        self.assertTrue(caps["seccomp"])
        self.assertEqual(caps["network"], {"mode": "cut"})

    async def test_code_runs_in_a_linux_guest(self):
        execution = await self.sandbox().run_code("import platform\nplatform.system()")
        self.assertIsNone(execution.error)
        self.assertEqual(execution.text, "'Linux'")

    async def test_host_environment_does_not_reach_the_guest(self):
        os.environ["SYNALINKS_TEST_SECRET"] = "leak"
        self.addCleanup(os.environ.pop, "SYNALINKS_TEST_SECRET", None)
        execution = await self.sandbox().run_code(
            "import os\n'SYNALINKS_TEST_SECRET' in os.environ"
        )
        self.assertEqual(execution.text, "False")

    async def test_host_filesystem_is_absent(self):
        home = os.path.expanduser("~")
        execution = await self.sandbox().run_code(f"import os\nos.path.exists({home!r})")
        self.assertEqual(execution.text, "False")

    async def test_snippet_and_file_tools_share_one_filesystem(self):
        sandbox = self.sandbox()
        await sandbox.run_code("open('/note.txt', 'w').write('from-guest')")
        self.assertEqual((await sandbox.read_file("/note.txt"))["content"], "from-guest")
        await sandbox.write_file("/host.txt", "from-host")
        execution = await sandbox.run_code("open('/host.txt').read()")
        self.assertEqual(execution.text, "'from-host'")

    async def test_host_tools_are_reachable_over_vsock(self):
        async def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        execution = await self.sandbox(external_functions={"add": add}).run_code(
            "add(2, 3)"
        )
        self.assertIsNone(execution.error)
        self.assertEqual(execution.text, "5")

    async def test_network_is_cut_unless_requested(self):
        probe = (
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 80), timeout=5).close()\n"
            "    outcome = 'CONNECTED'\n"
            "except OSError as exc:\n"
            "    outcome = 'blocked'\n"
            "outcome"
        )
        execution = await self.sandbox().run_code(probe)
        self.assertEqual(execution.text, "'blocked'")

    async def test_timeout_kills_the_vm(self):
        sandbox = self.sandbox()
        sandbox.timeout = 3
        execution = await sandbox.run_code("import time\ntime.sleep(60)")
        self.assertIsNotNone(execution.error)
        sandbox.timeout = _TIMEOUT
        after = await sandbox.run_code("1 + 1")
        self.assertEqual(after.text, "2")


class MountLeakTest(_SandboxTestCase):
    """A sandbox's FUSE mount never outlives it, whatever Mirage does.

    Mirage leaks a mount in two timing-dependent ways under load: its macOS
    unmount ignores a failed ``diskutil``, and a mount that goes live after
    its readiness wait timed out is dropped from its bookkeeping. macFUSE has
    only 64 devices, so each leak counts. Both are simulated here.
    """

    def _mounted_sandbox(self):
        from unittest import mock

        from synalinks.src.sandboxes import mirage_sandbox

        # A host FUSE mount backs Seatbelt on macOS and namespaces on Linux.
        backend = ("seatbelt", "forced") if sys.platform == "darwin" else None
        patcher = (
            mock.patch.object(
                mirage_sandbox, "confinement_backend", return_value=backend
            )
            if backend
            else mock.MagicMock()
        )
        with patcher:
            sandbox = MirageSandbox(timeout=_TIMEOUT, require_confinement=False)
        if not sandbox.fuse_mountpoint:
            sandbox.close()
            self.skipTest("no FUSE on this host")
        return sandbox

    def test_failed_unmount_is_finished_on_close(self):
        from unittest import mock

        sandbox = self._mounted_sandbox()
        mountpoint = sandbox.workspace.synalinks_mountpoint
        self.assertTrue(os.path.ismount(mountpoint))
        # Mirage's unmount "runs" but leaves the mount in place.
        with mock.patch("mirage.workspace.fuse.FuseManager.unmount"):
            sandbox.close()
        self.assertFalse(os.path.ismount(mountpoint))
        self.assertFalse(os.path.exists(mountpoint))

    def test_mount_that_misses_its_readiness_wait_is_not_orphaned(self):
        from synalinks.src.sandboxes.mirage_sandbox import ensure_fuse_mounted

        sandbox = MirageSandbox(timeout=_TIMEOUT, confine=False)
        self.addCleanup(sandbox.close)
        ws = sandbox.workspace
        real_add = ws.add_fuse_mount
        live = {}

        def add_then_time_out(*args, **kwargs):
            # The mount comes up, but Mirage reports a timeout: its handle on
            # the mount is lost, exactly as when the wait expires under load.
            path = real_add(*args, **kwargs)
            live["mounted"] = os.path.ismount(path)
            raise TimeoutError("FUSE mount did not become ready")

        ws.add_fuse_mount = add_then_time_out
        try:
            ok = ensure_fuse_mounted(ws)
        except Exception:
            self.skipTest("no FUSE on this host")
        if not live.get("mounted"):
            self.skipTest("no FUSE on this host")
        self.assertFalse(ok)
        mountpoint = ws.synalinks_mountpoint
        self.assertFalse(os.path.ismount(mountpoint))
        self.assertFalse(os.path.exists(mountpoint))
