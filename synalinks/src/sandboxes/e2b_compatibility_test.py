# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""The sandbox matches E2B's ``AsyncSandbox`` interface, checked against E2B's
own SDK: every public method and property exists with the same parameters
(names, kinds, defaults), the same sync / async nature, and every model has
E2B's fields in E2B's order. Our own methods and extra keyword-only arguments
come on top. Skipped when ``e2b-code-interpreter`` is not installed.
"""

import dataclasses
import inspect
import unittest

from synalinks.src import testing
from synalinks.src.sandboxes import sandbox as ours
from synalinks.src.sandboxes.mirage_sandbox import MirageCommands
from synalinks.src.sandboxes.mirage_sandbox import MirageFilesystem
from synalinks.src.sandboxes.mirage_sandbox import MirageSandbox

try:
    import e2b
    import e2b_code_interpreter
    from e2b.sandbox.commands.main import ProcessInfo
    from e2b.sandbox.filesystem.filesystem import WriteEntry
    from e2b.sandbox.filesystem.watch_handle import FilesystemEvent
    from e2b.sandbox.filesystem.watch_handle import FilesystemEventType
    from e2b.sandbox.sandbox_api import SandboxInfo
    from e2b.sandbox.sandbox_api import SandboxMetrics
    from e2b.sandbox.sandbox_api import SandboxQuery
    from e2b.sandbox.sandbox_api import SnapshotInfo
    from e2b.sandbox_async.commands.command import Commands
    from e2b.sandbox_async.commands.command_handle import AsyncCommandHandle
    from e2b.sandbox_async.commands.pty import Pty
    from e2b.sandbox_async.filesystem.filesystem import Filesystem
    from e2b.sandbox_async.filesystem.watch_handle import AsyncWatchHandle
    from e2b.sandbox_async.git import Git
    from e2b.sandbox_async.paginator import AsyncSandboxPaginator

    E2B = True
except ImportError:  # pragma: no cover - e2b is a dev-only dependency
    E2B = False


def interface_pairs():
    """``(E2B class, our class)`` for every class whose methods must match."""
    return [
        (e2b_code_interpreter.AsyncSandbox, MirageSandbox),
        (Filesystem, MirageFilesystem),
        (Commands, MirageCommands),
        (AsyncCommandHandle, ours.CommandHandle),
        (AsyncWatchHandle, ours.WatchHandle),
        (AsyncSandboxPaginator, ours.Paginator),
        (Git, ours.Git),
        (Pty, ours.Pty),
    ]


def model_pairs():
    """``(E2B model, our model)`` for every data type whose fields must match."""
    return [
        (e2b_code_interpreter.Execution, ours.Execution),
        (e2b_code_interpreter.Result, ours.Result),
        (e2b_code_interpreter.Logs, ours.Logs),
        (e2b_code_interpreter.ExecutionError, ours.ExecutionError),
        (e2b_code_interpreter.Context, ours.Context),
        (e2b_code_interpreter.OutputMessage, ours.OutputMessage),
        (e2b.CommandResult, ours.CommandResult),
        (e2b.EntryInfo, ours.EntryInfo),
        (e2b.WriteInfo, ours.WriteInfo),
        (ProcessInfo, ours.ProcessInfo),
        (SandboxInfo, ours.SandboxInfo),
        (SandboxMetrics, ours.SandboxMetrics),
        (SnapshotInfo, ours.SnapshotInfo),
        (SandboxQuery, ours.SandboxQuery),
        (FilesystemEvent, ours.FilesystemEvent),
        (WriteEntry, ours.WriteEntry),
    ]


def raw_member(cls, name):
    """The attribute as defined on the class (descriptors unresolved)."""
    for klass in cls.__mro__:
        if name in vars(klass):
            return vars(klass)[name]
    return None


def function_of(raw):
    """The function behind a method, classmethod, property or variant."""
    if isinstance(raw, (classmethod, staticmethod)):
        return raw.__func__
    if isinstance(raw, property):
        return raw.fget
    return getattr(raw, "method", None) or raw


def signature_problems(e2b_fn, our_fn):
    """How ``our_fn``'s parameters differ from ``e2b_fn``'s (empty if they match).

    Ours may add parameters only after E2B's, as keyword-only arguments or
    within ``**kwargs``.
    """
    theirs = list(inspect.signature(e2b_fn).parameters.values())
    mine = list(inspect.signature(our_fn).parameters.values())
    problems = []
    for index, param in enumerate(theirs):
        if index >= len(mine):
            problems.append(f"missing parameter {param.name!r}")
            continue
        other = mine[index]
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in mine):
                problems.append("missing **kwargs")
            continue
        if other.name != param.name:
            problems.append(f"parameter {index}: {other.name!r} != {param.name!r}")
        elif other.kind is not param.kind:
            problems.append(f"{param.name!r}: kind {other.kind} != {param.kind}")
        elif other.default != param.default:
            problems.append(
                f"{param.name!r}: default {other.default!r} != {param.default!r}"
            )
    for extra in mine[len(theirs) :]:
        if extra.kind not in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.VAR_KEYWORD,
        ):
            problems.append(f"extra positional parameter {extra.name!r}")
    return problems


@unittest.skipUnless(E2B, "e2b-code-interpreter is not installed")
class E2BInterfaceTest(testing.TestCase):
    def test_every_method_and_property_matches(self):
        problems = []
        for theirs, mine in interface_pairs():
            for name in dir(theirs):
                if name.startswith("_"):
                    continue
                label = f"{mine.__name__}.{name}"
                their_raw, my_raw = raw_member(theirs, name), raw_member(mine, name)
                if my_raw is None and not hasattr(mine, name):
                    problems.append(f"{label}: missing")
                    continue
                if isinstance(their_raw, property):
                    if not isinstance(my_raw, property):
                        problems.append(f"{label}: should be a property")
                    continue
                if not callable(function_of(their_raw)):
                    continue  # class attribute: present is enough
                their_fn, my_fn = function_of(their_raw), function_of(my_raw)
                if not callable(my_fn):
                    problems.append(f"{label}: not callable")
                    continue
                if inspect.iscoroutinefunction(their_fn) != inspect.iscoroutinefunction(
                    my_fn
                ):
                    problems.append(f"{label}: async differs")
                if isinstance(their_raw, classmethod) != isinstance(my_raw, classmethod):
                    problems.append(f"{label}: classmethod differs")
                problems += [f"{label}: {p}" for p in signature_problems(their_fn, my_fn)]
        self.assertEqual(problems, [], "\n".join(problems))

    def test_every_model_has_e2b_fields_in_order(self):
        problems = []
        for theirs, mine in model_pairs():
            if dataclasses.is_dataclass(theirs):
                their_fields = [f.name for f in dataclasses.fields(theirs)]
            else:
                their_fields = list(getattr(theirs, "__annotations__", {}))
            if dataclasses.is_dataclass(mine):
                my_fields = [f.name for f in dataclasses.fields(mine)]
            else:
                my_fields = list(getattr(mine, "__annotations__", {}))
            if my_fields[: len(their_fields)] != their_fields:
                problems.append(f"{mine.__name__}: {my_fields} != {their_fields}")
            for name in dir(theirs):
                if not name.startswith("_") and not hasattr(mine, name):
                    problems.append(f"{mine.__name__}.{name}: missing")
        self.assertEqual(problems, [], "\n".join(problems))

    def test_enums_have_e2b_members(self):
        for theirs, mine in (
            (e2b.FileType, ours.FileType),
            (FilesystemEventType, ours.FilesystemEventType),
        ):
            self.assertEqual(
                {m.name: m.value for m in theirs}, {m.name: m.value for m in mine}
            )

    def test_every_exception_exists_with_e2b_hierarchy(self):
        e2b_errors = {
            name: cls
            for name, cls in vars(e2b).items()
            if isinstance(cls, type) and issubclass(cls, BaseException)
        }
        problems = []
        for name, cls in e2b_errors.items():
            mine = getattr(ours, name, None)
            if mine is None:
                problems.append(f"{name}: missing")
                continue
            for base_name, base in e2b_errors.items():
                if issubclass(cls, base) and not issubclass(
                    mine, getattr(ours, base_name)
                ):
                    problems.append(f"{name}: should subclass {base_name}")
        self.assertEqual(problems, [], "\n".join(problems))


class E2BBehaviourTest(testing.TestCase):
    """The E2B features behave as E2B's do, on a real sandbox (no E2B needed)."""

    async def sandbox(self, **kwargs):
        sandbox = await MirageSandbox.create(**kwargs)
        self.addAsyncCleanup(sandbox.kill)
        return sandbox

    # -- run_code --------------------------------------------------------

    async def test_run_code_callbacks_receive_lines_results_and_errors(self):
        sandbox = await self.sandbox()
        lines, results, errors = [], [], []

        async def on_result(result):  # async callbacks are awaited
            results.append(result.text)

        await sandbox.run_code(
            "print('a')\nprint('b')\n6 * 7",
            on_stdout=lambda message: lines.append((message.line, message.error)),
            on_result=on_result,
        )
        self.assertEqual(lines, [("a\n", False), ("b\n", False)])
        self.assertEqual(results, ["42"])
        await sandbox.run_code("1 / 0", on_error=errors.append)
        self.assertEqual(errors[0].name, "ZeroDivisionError")

    async def test_run_code_envs_and_sandbox_envs(self):
        sandbox = await self.sandbox(envs={"BASE": "1"})
        execution = await sandbox.run_code(
            "import os\n(os.environ.get('BASE'), os.environ.get('EXTRA'))",
            envs={"EXTRA": "2"},
        )
        self.assertEqual(execution.text, "('1', '2')")

    async def test_run_code_timeout_overrides_the_sandbox_timeout(self):
        sandbox = await self.sandbox()
        execution = await sandbox.run_code("import time\ntime.sleep(30)", timeout=1)
        self.assertIsNotNone(execution.error)

    async def test_run_code_rejects_other_languages(self):
        sandbox = await self.sandbox()
        with self.assertRaises(ours.InvalidArgumentException):
            await sandbox.run_code("console.log(1)", language="javascript")

    # -- code contexts ---------------------------------------------------

    async def test_code_contexts_have_their_own_namespace(self):
        sandbox = await self.sandbox()
        context = await sandbox.create_code_context(cwd="/")
        await sandbox.run_code("x = 'default'")
        await sandbox.run_code("x = 'context'", context=context)
        self.assertEqual((await sandbox.run_code("x")).text, "'default'")
        self.assertEqual((await sandbox.run_code("x", context=context)).text, "'context'")
        ids = [c.id for c in await sandbox.list_code_contexts()]
        self.assertEqual(ids, ["default", context.id])
        await sandbox.restart_code_context(context)
        restarted = await sandbox.run_code("x", context=context)
        self.assertEqual(restarted.error.name, "NameError")
        await sandbox.remove_code_context(context)
        with self.assertRaises(ours.NotFoundException):
            await sandbox.run_code("1", context=context)

    async def test_code_context_cwd(self):
        sandbox = await self.sandbox()
        if sandbox.virtual_root() is None:
            self.skipTest("the snippet does not reach the virtual filesystem here")
        await sandbox.files.make_dir("/work")
        context = await sandbox.create_code_context(cwd="/work")
        await sandbox.run_code("open('note.txt', 'w').write('hi')", context=context)
        self.assertEqual(await sandbox.files.read("/work/note.txt"), "hi")

    # -- files -----------------------------------------------------------

    async def test_files_read_formats_and_write_sources(self):
        import io

        sandbox = await self.sandbox()
        await sandbox.files.write("/a.bin", io.BytesIO(b"\x00\x01data"))
        self.assertEqual(
            await sandbox.files.read("/a.bin", format="bytes"), b"\x00\x01data"
        )
        chunks = [c async for c in await sandbox.files.read("/a.bin", format="stream")]
        self.assertEqual(b"".join(chunks), b"\x00\x01data")
        infos = await sandbox.files.write_files(
            [{"path": "/x.txt", "data": "x"}, {"path": "/y.txt", "data": "y"}]
        )
        self.assertEqual([i.path for i in infos], ["/x.txt", "/y.txt"])
        with self.assertRaises(ours.FileNotFoundException):
            await sandbox.files.read("/missing.txt")

    async def test_watch_dir_reports_create_write_remove(self):
        import asyncio

        sandbox = await self.sandbox()
        await sandbox.files.make_dir("/watched")
        events = []
        handle = await sandbox.files.watch_dir(
            "/watched", lambda event: events.append((event.name, event.type))
        )
        await sandbox.files.write("/watched/f.txt", "1")
        await asyncio.sleep(0.4)
        await sandbox.files.write("/watched/f.txt", "12")
        await asyncio.sleep(0.4)
        await sandbox.files.remove("/watched/f.txt")
        await asyncio.sleep(0.4)
        await handle.stop()
        kinds = [kind for name, kind in events if name == "f.txt"]
        self.assertEqual(
            kinds,
            [
                ours.FilesystemEventType.CREATE,
                ours.FilesystemEventType.WRITE,
                ours.FilesystemEventType.REMOVE,
            ],
        )

    # -- commands --------------------------------------------------------

    async def test_command_envs_and_cwd_do_not_leak(self):
        sandbox = await self.sandbox()
        await sandbox.files.make_dir("/d")
        lines = []
        result = await sandbox.commands.run(
            "echo $FOO; pwd", envs={"FOO": "bar"}, cwd="/d", on_stdout=lines.append
        )
        self.assertEqual(result.stdout, "bar\n/d\n")
        self.assertEqual(lines, ["bar\n", "/d\n"])
        after = await sandbox.commands.run("echo [$FOO]; pwd")
        self.assertEqual(after.stdout, "[]\n/\n")

    async def test_background_command_with_stdin(self):
        sandbox = await self.sandbox()
        handle = await sandbox.commands.run("cat", background=True, stdin=True)
        self.assertEqual([p.pid for p in await sandbox.commands.list()], [handle.pid])
        await sandbox.commands.send_stdin(handle.pid, "hello ")
        await handle.send_stdin(b"world")
        await handle.close_stdin()
        result = await handle.wait()
        self.assertEqual((result.stdout, handle.exit_code), ("hello world", 0))
        self.assertEqual(await sandbox.commands.list(), [])

    async def test_background_command_kill_and_failure(self):
        sandbox = await self.sandbox()
        handle = await sandbox.commands.run("cat", background=True, stdin=True)
        self.assertTrue(await sandbox.commands.kill(handle.pid))
        with self.assertRaises(ours.CommandExitException):
            await handle.wait()
        failing = await sandbox.commands.run("exit 3", background=True)
        with self.assertRaises(ours.CommandExitException) as caught:
            await failing.wait()
        self.assertEqual(caught.exception.exit_code, 3)

    # -- sandbox lifecycle -----------------------------------------------

    async def test_class_level_calls_by_sandbox_id(self):
        sandbox = await self.sandbox(metadata={"team": "a"})
        info = await MirageSandbox.get_info(sandbox.sandbox_id)
        self.assertEqual(
            (info.sandbox_id, info.metadata), (sandbox.sandbox_id, {"team": "a"})
        )
        listed = await MirageSandbox.list(
            query=ours.SandboxQuery(metadata={"team": "a"})
        ).next_items()
        self.assertEqual([i.sandbox_id for i in listed], [sandbox.sandbox_id])
        self.assertTrue(await MirageSandbox.pause(sandbox.sandbox_id))
        self.assertEqual((await sandbox.get_info()).state, ours.SandboxState.PAUSED)
        self.assertIs(await MirageSandbox.connect(sandbox.sandbox_id), sandbox)
        self.assertEqual((await sandbox.get_info()).state, ours.SandboxState.RUNNING)
        self.assertTrue(await MirageSandbox.kill(sandbox.sandbox_id))
        self.assertFalse(await sandbox.is_running())
        with self.assertRaises(ours.SandboxNotFoundException):
            await MirageSandbox.get_info(sandbox.sandbox_id)

    async def test_set_timeout_kills_the_sandbox(self):
        import asyncio

        sandbox = await self.sandbox()
        await sandbox.set_timeout(1)
        await asyncio.sleep(1.5)
        self.assertFalse(await sandbox.is_running())

    async def test_snapshot_starts_a_sandbox_with_the_state(self):
        sandbox = await self.sandbox()
        await sandbox.run_code("kept = 41")
        await sandbox.files.write("/f.txt", "file")
        snapshot = await sandbox.create_snapshot(name="base")
        listed = await sandbox.list_snapshots().next_items()
        self.assertEqual([s.snapshot_id for s in listed], [snapshot.snapshot_id])
        restored = await self.sandbox(template=snapshot.snapshot_id)
        self.assertEqual((await restored.run_code("kept + 1")).text, "42")
        self.assertEqual(await restored.files.read("/f.txt"), "file")
        self.assertTrue(await MirageSandbox.delete_snapshot(snapshot.snapshot_id))
        with self.assertRaises(ours.TemplateException):
            await MirageSandbox.create(template=snapshot.snapshot_id)

    async def test_fork_count_and_class_level_fork(self):
        sandbox = await self.sandbox()
        forks = await sandbox.fork(count=2)
        for child in forks:
            self.addAsyncCleanup(child.kill)
        self.assertEqual(len({c.sandbox_id for c in forks}), 2)
        (child,) = await MirageSandbox.fork(sandbox.sandbox_id)
        self.addAsyncCleanup(child.kill)
        self.assertIsInstance(child, MirageSandbox)

    async def test_create_maps_network_options(self):
        sandbox = await self.sandbox(network={"allow_out": ["api.example.com"]})
        self.assertEqual(
            sandbox.granted_capabilities()["network"]["allowed_hosts"],
            ["api.example.com"],
        )
        with self.assertRaises(ours.InvalidArgumentException):
            await MirageSandbox.create(network={"deny_out": ["1.1.1.1"]})

    async def test_cloud_only_features_raise_not_supported(self):
        sandbox = await self.sandbox()
        for call in (
            lambda: sandbox.git.clone("https://example.com/r.git"),
            lambda: sandbox.pty.kill(1),
            lambda: sandbox.update_network({}),
        ):
            with self.assertRaises(ours.NotSupportedException):
                await call()
        with self.assertRaises(ours.NotSupportedException):
            sandbox.get_host(8080)
        with self.assertRaises(ours.NotSupportedException):
            await MirageSandbox.create(mcp={"server": {}})
        self.assertIsNone(await sandbox.get_mcp_token())
        (metrics,) = await sandbox.get_metrics()
        self.assertGreater(metrics.cpu_count, 0)
