# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import shutil
import tempfile
from pathlib import Path

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
        # Monty passes the return value through verbatim (an int stays an
        # int — there is no `{"result": ...}` wrapping).
        result = await sandbox.run(
            "import asyncio\n"
            "async def main():\n"
            "    return await triple(x=4)\n"
            "print(asyncio.run(main()))",
            external_functions={"triple": triple},
        )
        self.assertEqual(result.stdout.strip(), "12")

    async def test_history_records_each_run_with_outcome(self):
        sandbox = MontySandbox()
        await sandbox.run("print('one')")
        await sandbox.run("1 / 0")
        history = sandbox.history()
        self.assertEqual([e["code"] for e in history], ["print('one')", "1 / 0"])
        self.assertTrue(history[0]["ok"])
        self.assertIn("one", history[0]["stdout"])
        self.assertFalse(history[1]["ok"])
        self.assertIn("ZeroDivisionError", history[1]["error"])

    async def test_history_is_a_defensive_copy(self):
        sandbox = MontySandbox()
        await sandbox.run("x = 1")
        sandbox.history().append({"code": "tamper"})
        self.assertEqual(len(sandbox.history()), 1)

    async def test_reset_clears_history(self):
        sandbox = MontySandbox()
        await sandbox.run("x = 1")
        sandbox.reset()
        self.assertEqual(sandbox.history(), [])

    async def test_history_round_trips_through_config(self):
        sandbox = MontySandbox()
        await sandbox.run("x = 1")
        await sandbox.run("print(x)")
        restored = MontySandbox.from_config(sandbox.get_config())
        self.assertEqual(restored.history(), sandbox.history())


class MontySandboxOverlayTest(testing.TestCase):
    def _workdir(self):
        # Resolve symlinks so the path matches what MontySandbox stores
        # internally (it calls Path(workdir).resolve()). On macOS the temp
        # dir lives under /var -> /private/var, so an unresolved path would
        # not equal sandbox.workdir.
        tmp = str(Path(tempfile.mkdtemp()).resolve())
        (Path(tmp) / "config.json").write_text('{"debug": false}')
        return tmp

    async def test_in_memory_filesystem_without_workdir(self):
        sandbox = MontySandbox()
        self.assertIsNone(sandbox.workdir)
        # Reading a file that was never written fails...
        result = await sandbox.run(
            "from pathlib import Path\nprint(Path('/config.json').read_text())"
        )
        self.assertFalse(result.ok)
        # ...but the in-memory filesystem is writable.
        result = await sandbox.run(
            "from pathlib import Path\n"
            "Path('/scratch.txt').write_text('hi')\n"
            "print(Path('/scratch.txt').read_text())"
        )
        self.assertTrue(result.ok, result.error)
        self.assertIn("hi", result.stdout)

    async def test_pathlib_reads_workdir_through_overlay(self):
        sandbox = MontySandbox(workdir=self._workdir())
        self.assertIsNotNone(sandbox.workdir)
        result = await sandbox.run(
            "from pathlib import Path\nprint(Path('/config.json').read_text())"
        )
        self.assertTrue(result.ok, result.error)
        self.assertIn("debug", result.stdout)

    async def test_pathlib_write_stays_in_overlay(self):
        workdir = self._workdir()
        sandbox = MontySandbox(workdir=workdir)
        result = await sandbox.run(
            "from pathlib import Path\n"
            "Path('/PLAN.md').write_text('1. plan')\n"
            "print(Path('/PLAN.md').read_text())"
        )
        self.assertTrue(result.ok, result.error)
        self.assertIn("1. plan", result.stdout)
        # Host directory is untouched...
        self.assertFalse((Path(workdir) / "PLAN.md").exists())
        # ...but the change is inspectable host-side.
        self.assertIn("/PLAN.md", sandbox.changes()["written"])

    async def test_os_has_no_filesystem_escape(self):
        sandbox = MontySandbox(workdir=self._workdir())
        # os exposes no filesystem functions even with an overlay mounted.
        for snippet in (
            "import os; os.listdir('/')",
            "import os; os.system('echo hi')",
            "print(open('/config.json').read())",
        ):
            result = await sandbox.run(snippet)
            self.assertFalse(result.ok, f"expected failure for: {snippet}")

    async def test_os_getenv_routes_to_isolated_environ(self):
        sandbox = MontySandbox(environ={"TOKEN": "abc123"})
        result = await sandbox.run("import os\nprint(os.getenv('TOKEN'))")
        self.assertTrue(result.ok, result.error)
        self.assertIn("abc123", result.stdout)

    async def test_overlay_persists_across_runs(self):
        sandbox = MontySandbox(workdir=self._workdir())
        await sandbox.run("from pathlib import Path\nPath('/a.txt').write_text('hi')")
        result = await sandbox.run(
            "from pathlib import Path\nprint(Path('/a.txt').read_text())"
        )
        self.assertTrue(result.ok, result.error)
        self.assertIn("hi", result.stdout)

    async def test_config_round_trips_overlay(self):
        sandbox = MontySandbox(workdir=self._workdir())
        await sandbox.run("from pathlib import Path\nPath('/PLAN.md').write_text('keep')")
        restored = MontySandbox.from_config(sandbox.get_config())
        self.assertEqual(restored.workdir, sandbox.workdir)
        result = await restored.run(
            "from pathlib import Path\nprint(Path('/PLAN.md').read_text())"
        )
        self.assertTrue(result.ok, result.error)
        self.assertIn("keep", result.stdout)

    async def test_config_restores_filesystem_without_workdir(self):
        workdir = self._workdir()  # contains config.json
        sandbox = MontySandbox(workdir=workdir)
        await sandbox.write_file("/PLAN.md", "note")  # overlay write
        config = sandbox.get_config()

        # The workdir is gone (temp cleaned up, restored on another machine).
        shutil.rmtree(workdir)

        restored = MontySandbox.from_config(config)
        # Both the base file and the overlay write are restored from config.
        self.assertEqual(
            (await restored.read_file("/config.json"))["content"], '{"debug": false}'
        )
        self.assertEqual((await restored.read_file("/PLAN.md"))["content"], "note")
        # Sandboxed code can still read the snapshotted base via pathlib.
        result = await restored.run(
            "from pathlib import Path\nprint(Path('/config.json').read_text())"
        )
        self.assertTrue(result.ok, result.error)
        self.assertIn("debug", result.stdout)

    async def test_reset_restores_workdir_files_and_drops_overlay(self):
        workdir = self._workdir()  # contains config.json == '{"debug": false}'
        sandbox = MontySandbox(workdir=workdir)
        # Modify a base file and add a new one, both in the overlay only.
        await sandbox.write_file("/config.json", "MODIFIED")
        await sandbox.write_file("/scratch.txt", "tmp")

        sandbox.reset()

        # Base file is back to its on-disk content, overlay write is gone,
        # and the journal is cleared — but the workdir mount is kept.
        self.assertEqual(
            (await sandbox.read_file("/config.json"))["content"], '{"debug": false}'
        )
        self.assertIn("error", await sandbox.read_file("/scratch.txt"))
        self.assertEqual(sandbox.journal(), [])
        self.assertEqual(sandbox.workdir, workdir)


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
        with self.assertRaises(NotImplementedError):
            base.get_config()
        with self.assertRaises(NotImplementedError):
            Sandbox.from_config({})

    def test_base_class_provides_run_history(self):
        base = Sandbox()
        self.assertEqual(base.history(), [])
        # `_record_run` is the shared hook every backend routes results through.
        result = ExecutionResult(stdout="hi\n", error=None)
        returned = base._record_run("print('hi')", result)
        self.assertIs(returned, result)  # returns the result unchanged
        (entry,) = base.history()
        self.assertEqual(entry["code"], "print('hi')")
        self.assertTrue(entry["ok"])
        self.assertEqual(entry["stdout"], "hi\n")
        self.assertNotIn("result", entry)  # raw last-expression value not stored
        base.clear_history()
        self.assertEqual(base.history(), [])

    def test_base_class_provides_bound_functions(self):
        def tool():
            return 1

        base = Sandbox(external_functions={"tool": tool})
        self.assertEqual(set(base.bound_functions), {"tool"})
        base.bind_functions({"other": tool})
        self.assertEqual(set(base.bound_functions), {"tool", "other"})
        # property hands back a defensive copy
        base.bound_functions["tamper"] = tool
        self.assertEqual(set(base.bound_functions), {"tool", "other"})

    async def test_base_file_methods_default_to_no_filesystem(self):
        # The file tool methods are part of the base contract; with no
        # backend filesystem they return a clear error rather than crash.
        base = Sandbox()
        self.assertIn("error", await base.list_files())
        self.assertIn("error", await base.read_file("/x"))
        self.assertIn("error", await base.write_file("/x", "y"))
        self.assertIn("error", await base.edit_file("/x", "a", "b"))
        self.assertIn("error", await base.search_files("pattern"))
        self.assertIn("error", await base.run_python_file("/x.py"))


class MontySandboxFilesystemTest(testing.TestCase):
    def _workdir(self):
        tmp = tempfile.mkdtemp()
        (Path(tmp) / "config.json").write_text('{"debug": false}')
        sub = Path(tmp) / "src"
        sub.mkdir()
        (sub / "main.py").write_text("print('hi')\n", newline="")
        return tmp

    def test_reads_fall_through_to_base(self):
        fs = MontySandbox(workdir=self._workdir())
        self.assertEqual(fs.path_read_text("/config.json"), '{"debug": false}')
        self.assertTrue(fs.path_is_file("/src/main.py"))
        self.assertTrue(fs.path_is_dir("/src"))

    def test_write_lands_in_overlay_not_host(self):
        workdir = self._workdir()
        fs = MontySandbox(workdir=workdir)
        fs.path_write_text("/PLAN.md", "1. do the thing")
        # Visible to the sandbox...
        self.assertEqual(fs.path_read_text("/PLAN.md"), "1. do the thing")
        self.assertTrue(fs.path_exists("/PLAN.md"))
        # ...but the host directory is untouched.
        self.assertFalse((Path(workdir) / "PLAN.md").exists())

    def test_overlay_shadows_base(self):
        workdir = self._workdir()
        fs = MontySandbox(workdir=workdir)
        fs.path_write_text("/config.json", "OVERRIDDEN")
        self.assertEqual(fs.path_read_text("/config.json"), "OVERRIDDEN")
        # Host file unchanged.
        self.assertEqual((Path(workdir) / "config.json").read_text(), '{"debug": false}')

    def test_unlink_tombstones_base_file(self):
        workdir = self._workdir()
        fs = MontySandbox(workdir=workdir)
        fs.path_unlink("/config.json")
        self.assertFalse(fs.path_exists("/config.json"))
        with self.assertRaises(FileNotFoundError):
            fs.path_read_text("/config.json")
        # Still on disk.
        self.assertTrue((Path(workdir) / "config.json").exists())

    def test_iterdir_merges_base_and_overlay(self):
        fs = MontySandbox(workdir=self._workdir())
        fs.path_write_text("/PLAN.md", "x")
        names = {p.name for p in fs.path_iterdir("/")}
        self.assertEqual(names, {"config.json", "src", "PLAN.md"})

    def test_iterdir_hides_tombstones(self):
        fs = MontySandbox(workdir=self._workdir())
        fs.path_unlink("/config.json")
        names = {p.name for p in fs.path_iterdir("/")}
        self.assertNotIn("config.json", names)

    def test_relative_and_absolute_keys_collapse(self):
        fs = MontySandbox()
        fs.path_write_text("notes.txt", "a")
        self.assertEqual(fs.path_read_text("/notes.txt"), "a")
        self.assertEqual(fs.path_read_text("./notes.txt"), "a")

    def test_dotdot_cannot_escape_root(self):
        workdir = self._workdir()
        fs = MontySandbox(workdir=workdir)
        # Escaping the root just normalizes back inside; the host file
        # does not exist there, so the read fails rather than leaking.
        with self.assertRaises(FileNotFoundError):
            fs.path_read_text("/../../../etc/passwd")

    def test_symlink_escape_refused(self):
        workdir = self._workdir()
        secret = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        secret.write("top secret")
        secret.close()
        link = Path(workdir) / "link.txt"
        try:
            os.symlink(secret.name, link)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unsupported on this platform")
        fs = MontySandbox(workdir=workdir)
        with self.assertRaises(FileNotFoundError):
            fs.path_read_text("/link.txt")

    def test_pure_in_memory_has_no_base(self):
        fs = MontySandbox()
        self.assertIsNone(fs.workdir)
        with self.assertRaises(FileNotFoundError):
            fs.path_read_text("/anything")
        fs.path_write_text("/x", "y")
        self.assertEqual(fs.path_read_text("/x"), "y")

    def test_environ_is_isolated(self):
        fs = MontySandbox(environ={"API_KEY": "sandbox-only"})
        self.assertEqual(fs.getenv("API_KEY"), "sandbox-only")
        self.assertEqual(fs.get_environ(), {"API_KEY": "sandbox-only"})
        self.assertIsNone(fs.getenv("HOME"))  # host env not visible

    def test_changes_reports_written_and_deleted(self):
        fs = MontySandbox(workdir=self._workdir())
        fs.path_write_text("/PLAN.md", "x")
        fs.path_unlink("/config.json")
        changes = fs.changes()
        self.assertIn("/PLAN.md", changes["written"])
        self.assertIn("/config.json", changes["deleted"])

    def test_state_round_trips(self):
        fs = MontySandbox(workdir=self._workdir())
        fs.path_write_text("/PLAN.md", "remember this")
        fs.path_unlink("/config.json")
        state = fs.get_state()

        restored = MontySandbox(workdir=fs.workdir)
        restored.set_state(state)
        self.assertEqual(restored.path_read_text("/PLAN.md"), "remember this")
        self.assertFalse(restored.path_exists("/config.json"))

    def test_get_state_default_excludes_base(self):
        fs = MontySandbox(workdir=self._workdir())
        self.assertNotIn("base", fs.get_state())
        self.assertIn("base", fs.get_state(snapshot_base=True))

    def test_base_snapshot_restores_without_workdir(self):
        fs = MontySandbox(workdir=self._workdir())
        fs.path_write_text("/PLAN.md", "note")  # overlay write
        state = fs.get_state(snapshot_base=True)

        # Restore into a pure in-memory filesystem — no workdir at all.
        restored = MontySandbox()
        restored.set_state(state)
        self.assertIsNone(restored.workdir)
        # Base files survive without the original workdir on disk...
        self.assertEqual(restored.path_read_text("/config.json"), '{"debug": false}')
        self.assertEqual(restored.path_read_text("/src/main.py"), "print('hi')\n")
        self.assertTrue(restored.path_is_dir("/src"))
        # ...the overlay write survives too...
        self.assertEqual(restored.path_read_text("/PLAN.md"), "note")
        # ...and the base is NOT mistaken for an overlay change.
        self.assertEqual(restored.changes()["written"], ["/PLAN.md"])
        self.assertEqual(restored.rglob("*.py"), ["/src/main.py"])

    def test_base_snapshot_preserves_tombstones(self):
        fs = MontySandbox(workdir=self._workdir())
        fs.path_unlink("/config.json")  # tombstone a base file
        restored = MontySandbox()
        restored.set_state(fs.get_state(snapshot_base=True))
        # The base file was captured but stays deleted via the tombstone.
        self.assertFalse(restored.path_exists("/config.json"))
        self.assertEqual(restored.path_read_text("/src/main.py"), "print('hi')\n")

    def test_rename_moves_content(self):
        fs = MontySandbox()
        fs.path_write_text("/a.txt", "data")
        fs.path_rename("/a.txt", "/b.txt")
        self.assertFalse(fs.path_exists("/a.txt"))
        self.assertEqual(fs.path_read_text("/b.txt"), "data")

    def test_journal_records_each_action_in_order(self):
        fs = MontySandbox()
        fs.path_write_text("/a.txt", "data")
        fs.path_mkdir("/sub")
        fs.path_rename("/a.txt", "/b.txt")
        fs.path_unlink("/b.txt")
        actions = [(e["action"], e["path"]) for e in fs.journal()]
        self.assertEqual(
            actions,
            [
                ("write", "/a.txt"),
                ("mkdir", "/sub"),
                ("rename", "/b.txt"),
                ("delete", "/b.txt"),
            ],
        )

    def test_journal_distinguishes_create_and_modify(self):
        fs = MontySandbox(workdir=self._workdir())
        fs.path_write_text("/PLAN.md", "v1")  # new file -> create
        fs.path_write_text("/PLAN.md", "v2")  # rewrite -> modify
        fs.path_write_text("/config.json", "x")  # shadows base -> modify
        writes = [e for e in fs.journal() if e["action"] == "write"]
        self.assertEqual([e["kind"] for e in writes], ["create", "modify", "modify"])
        self.assertEqual(writes[0]["size"], len("v1"))

    def test_journal_records_rename_origin(self):
        fs = MontySandbox()
        fs.path_write_text("/a.txt", "data")
        fs.path_rename("/a.txt", "/b.txt")
        rename = fs.journal()[-1]
        self.assertEqual(rename["action"], "rename")
        self.assertEqual(rename["path"], "/b.txt")
        self.assertEqual(rename["src"], "/a.txt")
        self.assertEqual(rename["size"], len("data"))

    def test_journal_is_a_defensive_copy(self):
        fs = MontySandbox()
        fs.path_write_text("/a.txt", "data")
        fs.journal().append({"action": "tamper", "path": "/x"})
        self.assertEqual(len(fs.journal()), 1)

    def test_journal_round_trips_through_state(self):
        fs = MontySandbox()
        fs.path_write_text("/a.txt", "data")
        fs.path_unlink("/a.txt")
        restored = MontySandbox()
        restored.set_state(fs.get_state())
        self.assertEqual(restored.journal(), fs.journal())

    def test_glob_matches_within_a_single_segment(self):
        fs = MontySandbox(workdir=self._workdir())
        self.assertEqual(fs.glob("*.json"), ["/config.json"])
        self.assertEqual(fs.glob("*.py"), [])  # no top-level .py
        self.assertEqual(fs.glob("src/*.py"), ["/src/main.py"])

    def test_rglob_and_doublestar_are_recursive(self):
        fs = MontySandbox(workdir=self._workdir())
        self.assertEqual(fs.rglob("*.py"), ["/src/main.py"])
        self.assertEqual(fs.glob("**/*.py"), ["/src/main.py"])

    def test_glob_reflects_overlay_writes_and_tombstones(self):
        fs = MontySandbox(workdir=self._workdir())
        fs.path_write_text("/src/new.py", "x")
        fs.path_unlink("/src/main.py")
        self.assertEqual(fs.rglob("*.py"), ["/src/new.py"])

    def test_glob_question_mark_and_character_class(self):
        fs = MontySandbox()
        for name in ("a1.txt", "a2.txt", "ab.txt"):
            fs.path_write_text("/" + name, "")
        self.assertEqual(fs.glob("a?.txt"), ["/a1.txt", "/a2.txt", "/ab.txt"])
        self.assertEqual(fs.glob("a[0-9].txt"), ["/a1.txt", "/a2.txt"])

    def test_glob_with_root(self):
        fs = MontySandbox(workdir=self._workdir())
        self.assertEqual(fs.glob("*.py", root="/src"), ["/src/main.py"])


class MontySandboxWorkdirStressTest(testing.TestCase):
    """Drive the overlay through real sandboxed Python end to end.

    Mounts a populated host workdir, then mutates it from inside the
    sandbox via ``pathlib`` (edit / create-nested / delete / rename /
    bulk-write) and asserts three things hold together: the sandbox sees
    a consistent merged view, the host directory is **byte-for-byte
    untouched**, and the host-side journal / changes reflect what code did.
    """

    def _make_workdir(self):
        tmp = tempfile.mkdtemp()
        (Path(tmp) / "README.md").write_text("project readme")
        (Path(tmp) / "config.json").write_text('{"version": 1}')
        src = Path(tmp) / "src"
        src.mkdir()
        (src / "main.py").write_text("print('main')\n", newline="")
        (src / "util.py").write_text("X = 1\n", newline="")
        docs = Path(tmp) / "docs"
        docs.mkdir()
        (docs / "intro.md").write_text("# Intro")
        return tmp

    def _snapshot(self, root):
        """Map every host file under ``root`` to its bytes, for diffing."""
        root = Path(root)
        return {
            str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*"))
            if p.is_file()
        }

    async def test_full_edit_session_leaves_host_byte_for_byte_untouched(self):
        workdir = self._make_workdir()
        before = self._snapshot(workdir)
        sb = MontySandbox(workdir=workdir, timeout=20)

        # 1. read base files through the overlay, then patch one in place.
        r = await sb.run("""
from pathlib import Path
assert Path('/README.md').read_text() == 'project readme'
assert Path('/src/main.py').read_text().startswith('print')
Path('/src/main.py').write_text('PATCHED')
""")
        self.assertTrue(r.ok, r.error)
        # 2. create new files, including a freshly mkdir'd nested package.
        r = await sb.run("""
from pathlib import Path
Path('/src/new_module.py').write_text('Y=2')
Path('/pkg/sub').mkdir(parents=True)
Path('/pkg/sub/deep.py').write_text('Z=3')
""")
        self.assertTrue(r.ok, r.error)
        # 3. delete a base file, 4. rename another.
        r = await sb.run("from pathlib import Path\nPath('/config.json').unlink()")
        self.assertTrue(r.ok, r.error)
        r = await sb.run(
            "from pathlib import Path\n"
            "Path('/docs/intro.md').rename('/docs/introduction.md')"
        )
        self.assertTrue(r.ok, r.error)

        # The sandbox's own merged view is internally consistent.
        r = await sb.run("""
from pathlib import Path
checks = [
    Path('/src/main.py').read_text() == 'PATCHED',
    Path('/src/new_module.py').read_text() == 'Y=2',
    Path('/pkg/sub/deep.py').read_text() == 'Z=3',
    not Path('/config.json').exists(),
    Path('/docs/introduction.md').read_text() == '# Intro',
    not Path('/docs/intro.md').exists(),
    Path('/src/util.py').read_text() == 'X = 1\\n',
]
print(all(checks))
print(checks)
""")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.stdout.strip().split("\n")[0], "True", r.stdout)

        # Host-side: effective content matches, deletions are tombstoned.
        self.assertEqual(sb.read_overlay("/src/main.py"), b"PATCHED")
        self.assertEqual(sb.read_overlay("/pkg/sub/deep.py"), b"Z=3")
        self.assertIsNone(sb.read_overlay("/config.json"))
        self.assertEqual(sb.read_overlay("/docs/introduction.md"), b"# Intro")
        self.assertIsNone(sb.read_overlay("/docs/intro.md"))

        changes = sb.changes()
        for path in (
            "/src/main.py",
            "/src/new_module.py",
            "/pkg/sub/deep.py",
            "/docs/introduction.md",
        ):
            self.assertIn(path, changes["written"])
        for path in ("/config.json", "/docs/intro.md"):
            self.assertIn(path, changes["deleted"])

        # The guarantee that matters: not one host byte changed.
        self.assertEqual(self._snapshot(workdir), before)

    async def test_journal_captures_the_session_in_order(self):
        sb = MontySandbox(workdir=self._make_workdir(), timeout=20)
        await sb.run("""
from pathlib import Path
Path('/notes.txt').write_text('a')
Path('/notes.txt').write_text('ab')
Path('/d').mkdir()
Path('/notes.txt').rename('/d/notes.txt')
Path('/d/notes.txt').unlink()
""")
        journal = sb.journal()
        self.assertEqual(
            [(e["action"], e["path"]) for e in journal],
            [
                ("write", "/notes.txt"),
                ("write", "/notes.txt"),
                ("mkdir", "/d"),
                ("rename", "/d/notes.txt"),
                ("delete", "/d/notes.txt"),
            ],
        )
        writes = [e for e in journal if e["action"] == "write"]
        self.assertEqual([e["kind"] for e in writes], ["create", "modify"])
        self.assertEqual(journal[3]["src"], "/notes.txt")

    async def test_bulk_generate_and_readback(self):
        workdir = self._make_workdir()
        sb = MontySandbox(workdir=workdir, timeout=20)
        r = await sb.run("""
from pathlib import Path
Path('/out').mkdir()
for i in range(100):
    Path(f'/out/f{i}.txt').write_text(str(i))
total = sum(int(Path(f'/out/f{i}.txt').read_text()) for i in range(100))
print(total)
print(len(list(Path('/out').iterdir())))
""")
        self.assertTrue(r.ok, r.error)
        lines = r.stdout.strip().split("\n")
        self.assertEqual(lines[0], str(sum(range(100))))  # 4950, read back intact
        self.assertEqual(lines[1], "100")
        # None of the 100 generated files leaked to the host.
        self.assertFalse((Path(workdir) / "out").exists())
        written = sb.changes()["written"]
        self.assertEqual(len([w for w in written if w.startswith("/out/")]), 100)

    async def test_overlay_round_trips_and_continues_editing(self):
        workdir = self._make_workdir()
        before = self._snapshot(workdir)
        sb = MontySandbox(workdir=workdir, timeout=20)
        await sb.run("""
from pathlib import Path
Path('/PLAN.md').write_text('phase 1')
Path('/config.json').unlink()
""")
        # Serialize mid-session and resume on a fresh instance.
        restored = MontySandbox.from_config(sb.get_config())
        r = await restored.run("""
from pathlib import Path
print(Path('/PLAN.md').read_text())
print(Path('/config.json').exists())
print(Path('/README.md').read_text())
""")
        self.assertTrue(r.ok, r.error)
        lines = r.stdout.strip().split("\n")
        self.assertEqual(lines[0], "phase 1")  # overlay write survived
        self.assertEqual(lines[1], "False")  # tombstone survived
        self.assertEqual(lines[2], "project readme")  # base still read-through
        # Keep editing the restored sandbox.
        await restored.run(
            "from pathlib import Path\nPath('/PLAN.md').write_text('phase 2')"
        )
        self.assertEqual(restored.read_overlay("/PLAN.md"), b"phase 2")
        # Host untouched across the whole round-trip.
        self.assertEqual(self._snapshot(workdir), before)

    async def test_delete_base_file_then_recreate(self):
        workdir = self._make_workdir()
        sb = MontySandbox(workdir=workdir, timeout=20)
        await sb.run("from pathlib import Path\nPath('/config.json').unlink()")
        r = await sb.run("from pathlib import Path\nprint(Path('/config.json').exists())")
        self.assertEqual(r.stdout.strip(), "False")
        await sb.run("""
from pathlib import Path
Path('/config.json').write_text('{"version": 2}')
""")
        r = await sb.run(
            "from pathlib import Path\nprint(Path('/config.json').read_text())"
        )
        self.assertEqual(r.stdout.strip(), '{"version": 2}')
        # Host original intact despite the delete + recreate.
        self.assertEqual((Path(workdir) / "config.json").read_text(), '{"version": 1}')
        actions = [(e["action"], e.get("kind")) for e in sb.journal()]
        self.assertEqual(actions, [("delete", None), ("write", "create")])

    async def test_glob_helpers_search_the_overlay_from_inside(self):
        # Monty's Path has no glob/rglob; the sandbox exposes them as async
        # globals over the merged overlay view (base + writes - deletes).
        workdir = self._make_workdir()
        sb = MontySandbox(workdir=workdir, timeout=20)
        r = await sb.run("""
import asyncio
from pathlib import Path
Path('/src/extra.py').write_text('E=1')
async def main():
    return await rglob('*.py')
print(sorted(asyncio.run(main())))
""")
        self.assertTrue(r.ok, r.error)
        # base src/main.py + src/util.py, plus the just-written extra.py
        self.assertEqual(
            r.stdout.strip(),
            "['/src/extra.py', '/src/main.py', '/src/util.py']",
        )
        # Non-recursive glob scoped to a subdirectory via root=.
        r = await sb.run("""
import asyncio
async def main():
    return await glob('*.md', root='/docs')
print(asyncio.run(main()))
""")
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.stdout.strip(), "['/docs/intro.md']")


# A small snippet that awaits one external call and prints its return value.
_AWAIT_CALL = (
    "import asyncio\n"
    "async def main():\n"
    "    return await {call}\n"
    "print(asyncio.run(main()))\n"
)


class MontySandboxBoundFunctionsTest(testing.TestCase):
    async def test_bound_function_available_on_every_run(self):
        async def add(a, b):
            return a + b

        sb = MontySandbox(external_functions={"add": add})
        self.assertEqual(set(sb.bound_functions), {"add"})
        for _ in range(2):  # no per-call external_functions, yet still callable
            r = await sb.run(_AWAIT_CALL.format(call="add(a=2, b=3)"))
            self.assertTrue(r.ok, r.error)
            self.assertEqual(r.stdout.strip(), "5")

    async def test_bind_functions_adds_later(self):
        sb = MontySandbox()

        async def greet(name):
            return f"hi {name}"

        sb.bind_functions({"greet": greet})
        r = await sb.run(_AWAIT_CALL.format(call="greet(name='bob')"))
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.stdout.strip(), "hi bob")

    async def test_per_call_external_functions_override_bound(self):
        async def const_one():
            return 1

        async def const_two():
            return 2

        sb = MontySandbox(external_functions={"val": const_one})
        r = await sb.run(
            _AWAIT_CALL.format(call="val()"),
            external_functions={"val": const_two},
        )
        self.assertEqual(r.stdout.strip(), "2")  # per-call wins
        # The override did not mutate the bound set.
        r = await sb.run(_AWAIT_CALL.format(call="val()"))
        self.assertEqual(r.stdout.strip(), "1")

    async def test_bound_functions_survive_reset(self):
        async def add(a, b):
            return a + b

        sb = MontySandbox(external_functions={"add": add})
        await sb.run("x = 1")
        sb.reset()
        self.assertEqual(set(sb.bound_functions), {"add"})
        r = await sb.run(_AWAIT_CALL.format(call="add(a=10, b=1)"))
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.stdout.strip(), "11")

    async def test_bound_functions_property_is_a_defensive_copy(self):
        async def add(a, b):
            return a + b

        sb = MontySandbox(external_functions={"add": add})
        sb.bound_functions["tamper"] = lambda: None
        self.assertEqual(set(sb.bound_functions), {"add"})

    async def test_bound_functions_are_not_serialized(self):
        async def add(a, b):
            return a + b

        sb = MontySandbox(external_functions={"add": add})
        config = sb.get_config()
        self.assertNotIn("external_functions", config)
        # Callables can't survive a config round-trip; re-bind afterwards.
        restored = MontySandbox.from_config(config)
        self.assertEqual(restored.bound_functions, {})

    async def test_bound_function_overrides_filesystem_glob_helper(self):
        # A user-supplied `glob` takes precedence over the built-in overlay
        # helper (per-call / bound functions sit above filesystem helpers).
        async def glob(pattern, root="/"):
            return ["custom"]

        sb = MontySandbox(workdir=tempfile.mkdtemp(), external_functions={"glob": glob})
        r = await sb.run(_AWAIT_CALL.format(call="glob('*')"))
        self.assertTrue(r.ok, r.error)
        self.assertEqual(r.stdout.strip(), "['custom']")


class MontySandboxToolMethodsTest(testing.TestCase):
    """Dict-returning methods a caller can hand to ``synalinks.Tool``.

    The sandbox does not wrap them itself — these tests exercise the
    methods directly and confirm they stay ``Tool``-compatible.
    """

    def _workdir(self):
        tmp = tempfile.mkdtemp()
        (Path(tmp) / "main.py").write_text("print('hi')\n", newline="")
        return tmp

    async def test_run_python_code_runs_and_reports(self):
        sb = MontySandbox()
        out = await sb.run_python_code("x = 40\nprint(x + 2)")
        self.assertTrue(out["ok"])
        self.assertEqual(out["stdout"].strip(), "42")

    async def test_run_python_code_reports_errors(self):
        sb = MontySandbox()
        out = await sb.run_python_code("1 / 0")
        self.assertFalse(out["ok"])
        self.assertIn("ZeroDivisionError", out["error"])

    async def test_run_python_file_executes_overlay_script(self):
        sb = MontySandbox()
        await sb.write_file("/script.py", "print(sum(range(10)))\n")
        out = await sb.run_python_file("/script.py")
        self.assertTrue(out["ok"], out["error"])
        self.assertEqual(out["stdout"].strip(), "45")

    async def test_run_python_file_shares_state_and_errors_on_missing(self):
        sb = MontySandbox()
        # State persists: a script can build on a prior run's namespace.
        await sb.run_python_code("BASE = 100")
        await sb.write_file("/s.py", "print(BASE + 1)\n")
        self.assertEqual((await sb.run_python_file("/s.py"))["stdout"].strip(), "101")
        self.assertIn("error", await sb.run_python_file("/missing.py"))

    async def test_file_methods_round_trip_via_overlay(self):
        workdir = self._workdir()
        sb = MontySandbox(workdir=workdir)
        self.assertEqual((await sb.list_files(pattern="**/*.py"))["files"], ["/main.py"])
        self.assertEqual(
            await sb.write_file("/PLAN.md", "step 1"),
            {"written": "/PLAN.md", "bytes": 6},
        )
        self.assertEqual((await sb.read_file("/PLAN.md"))["content"], "step 1")
        # The host workdir is never touched by overlay writes.
        self.assertFalse((Path(workdir) / "PLAN.md").exists())

    async def test_read_missing_file_returns_error(self):
        sb = MontySandbox(workdir=self._workdir())
        self.assertIn("error", await sb.read_file("/nope.txt"))

    async def test_file_methods_on_in_memory_filesystem(self):
        # With no workdir the sandbox is its own empty in-memory filesystem,
        # so the file methods operate (rather than erroring) — only per-file
        # conditions (missing file) still error.
        sb = MontySandbox()
        self.assertEqual((await sb.list_files())["files"], [])
        self.assertEqual((await sb.search_files("p"))["matches"], [])
        self.assertIn("error", await sb.read_file("/x"))  # absent file
        self.assertEqual(await sb.write_file("/x", "y"), {"written": "/x", "bytes": 1})
        self.assertEqual((await sb.read_file("/x"))["content"], "y")
        self.assertEqual(
            await sb.edit_file("/x", "y", "z"), {"path": "/x", "replacements": 1}
        )

    async def test_read_file_pagination(self):
        workdir = tempfile.mkdtemp()
        (Path(workdir) / "f.txt").write_text("l1\nl2\nl3\nl4\n", newline="")
        sb = MontySandbox(workdir=workdir)
        # 1-based: start at line 2, take 2 lines -> lines 2 and 3.
        page = await sb.read_file("/f.txt", offset=2, limit=2)
        self.assertEqual(page["content"], "l2\nl3\n")
        self.assertEqual((page["start_line"], page["end_line"]), (2, 3))
        self.assertEqual(page["total_lines"], 4)
        self.assertTrue(page["truncated"])
        # The default reads the whole file from line 1, not truncated.
        full = await sb.read_file("/f.txt")
        self.assertEqual((full["start_line"], full["end_line"]), (1, 4))
        self.assertFalse(full["truncated"])

    async def test_list_files_pagination_and_filters_dirs(self):
        workdir = tempfile.mkdtemp()
        sub = Path(workdir) / "pkg"
        sub.mkdir()
        (Path(workdir) / "a.py").write_text("a")
        (sub / "b.py").write_text("b")
        sb = MontySandbox(workdir=workdir)
        full = await sb.list_files(pattern="**/*.py")
        # "pkg" (a directory) is excluded; only files are listed.
        self.assertEqual(full["files"], ["/a.py", "/pkg/b.py"])
        self.assertEqual(full["total"], 2)
        self.assertFalse(full["truncated"])
        first = await sb.list_files(pattern="**/*.py", limit=1)
        self.assertEqual(first["files"], ["/a.py"])
        self.assertTrue(first["truncated"])

    async def test_search_files_glob_plus_grep(self):
        workdir = tempfile.mkdtemp()
        (Path(workdir) / "a.py").write_text("x = 1\nTODO: fix\n", newline="")
        (Path(workdir) / "notes.txt").write_text("TODO: docs\n", newline="")
        sb = MontySandbox(workdir=workdir)
        # grep restricted to .py files by the glob
        res = await sb.search_files(pattern="TODO", glob="**/*.py")
        self.assertEqual(res["total"], 1)
        self.assertEqual(
            res["matches"][0], {"path": "/a.py", "line": 2, "text": "TODO: fix"}
        )
        # widening the glob picks up the .txt match too
        self.assertEqual((await sb.search_files(pattern="TODO"))["total"], 2)

    async def test_search_files_bad_regex_returns_error(self):
        sb = MontySandbox(workdir=tempfile.mkdtemp())
        self.assertIn("error", await sb.search_files(pattern="([unclosed"))

    async def test_search_files_pagination(self):
        workdir = tempfile.mkdtemp()
        (Path(workdir) / "f.txt").write_text("hit\nhit\nhit\n", newline="")
        sb = MontySandbox(workdir=workdir)
        # 1-based: the 2nd match is on line 2.
        page = await sb.search_files(pattern="hit", offset=2, limit=1)
        self.assertEqual([m["line"] for m in page["matches"]], [2])
        self.assertEqual(page["total"], 3)
        self.assertTrue(page["truncated"])

    async def test_edit_file_unique_replace(self):
        workdir = tempfile.mkdtemp()
        (Path(workdir) / "f.py").write_text("x = 1\ny = 2\n", newline="")
        sb = MontySandbox(workdir=workdir)
        self.assertEqual(
            await sb.edit_file("/f.py", old="x = 1", new="x = 99"),
            {"path": "/f.py", "replacements": 1},
        )
        self.assertEqual((await sb.read_file("/f.py"))["content"], "x = 99\ny = 2\n")
        # The host file is untouched (edit is overlay-only).
        self.assertEqual((Path(workdir) / "f.py").read_text(), "x = 1\ny = 2\n")

    async def test_edit_file_rejects_ambiguous_and_missing(self):
        workdir = tempfile.mkdtemp()
        (Path(workdir) / "f.txt").write_text("ab ab ab")
        sb = MontySandbox(workdir=workdir)
        self.assertIn("not unique", (await sb.edit_file("/f.txt", "ab", "X"))["error"])
        self.assertEqual(
            await sb.edit_file("/f.txt", "ab", "X", replace_all=True),
            {"path": "/f.txt", "replacements": 3},
        )
        self.assertIn("not found", (await sb.edit_file("/f.txt", "zzz", "X"))["error"])
        self.assertIn("error", await sb.edit_file("/missing.txt", "a", "b"))

    async def test_methods_are_tool_compatible(self):
        # The caller (not the sandbox) wraps a method as a Tool. This locks
        # the contract that the docstrings/type hints stay Tool-valid.
        from synalinks.src.modules.core.tool import Tool

        sb = MontySandbox(workdir=self._workdir())
        execute = Tool(sb.run_python_code)
        self.assertEqual(execute.name, "run_python_code")
        self.assertEqual(execute.get_tool_schema()["required"], ["code"])
        out = await execute(code="print('via tool')")
        self.assertIn("via tool", out.get_json()["stdout"])
        # Every file method wraps cleanly too (valid docstrings + hints).
        for method in (
            sb.list_files,
            sb.read_file,
            sb.write_file,
            sb.edit_file,
            sb.search_files,
            sb.run_python_file,
        ):
            self.assertFalse(Tool(method).name.startswith("_"))


class MontySandboxBranchingTest(testing.TestCase):
    def _workdir(self):
        tmp = str(Path(tempfile.mkdtemp()).resolve())
        (Path(tmp) / "config.json").write_text('{"debug": false}')
        return tmp

    async def test_fork_sees_parent_files_in_memory(self):
        # A fork sees the parent's effective tree (workdir + overlay) but
        # carries no host workdir coupling of its own.
        workdir = self._workdir()  # contains config.json
        parent = MontySandbox(workdir=workdir)
        await parent.write_file("/notes.txt", "hello")

        child = parent.fork()
        self.assertIsNone(child.workdir)  # self-contained in-memory base
        self.assertEqual(
            (await child.read_file("/config.json"))["content"], '{"debug": false}'
        )
        self.assertEqual((await child.read_file("/notes.txt"))["content"], "hello")

    async def test_fork_writes_are_isolated_from_parent(self):
        parent = MontySandbox()
        await parent.write_file("/shared.txt", "v1")

        child = parent.fork()
        await child.write_file("/shared.txt", "v2")
        await child.write_file("/only_child.txt", "new")

        # Parent is untouched by the child's writes.
        self.assertEqual((await parent.read_file("/shared.txt"))["content"], "v1")
        self.assertIn("error", await parent.read_file("/only_child.txt"))
        # Child sees its own writes.
        self.assertEqual((await child.read_file("/shared.txt"))["content"], "v2")

    async def test_parent_writes_after_fork_do_not_leak_into_child(self):
        parent = MontySandbox()
        await parent.write_file("/a.txt", "a1")
        child = parent.fork()
        await parent.write_file("/a.txt", "a2")  # parent edits after the fork
        await parent.write_file("/b.txt", "b")
        # The child was branched at the fork point and does not see later edits.
        self.assertEqual((await child.read_file("/a.txt"))["content"], "a1")
        self.assertIn("error", await child.read_file("/b.txt"))

    async def test_diff_reports_only_child_changes(self):
        workdir = self._workdir()  # config.json in base
        parent = MontySandbox(workdir=workdir)
        await parent.write_file("/existing.txt", "base")

        child = parent.fork()
        await child.write_file("/existing.txt", "edited")  # modify (in base)
        await child.write_file("/fresh.txt", "brand new")  # create
        await child.run("import pathlib; pathlib.Path('/config.json').unlink()")  # delete

        diff = child.diff()
        kinds = {w["path"]: w["kind"] for w in diff["written"]}
        self.assertEqual(kinds["/existing.txt"], "modify")
        self.assertEqual(kinds["/fresh.txt"], "create")
        self.assertEqual(diff["deleted"], ["/config.json"])
        # A fresh fork has an empty diff.
        self.assertEqual(parent.fork().diff(), {"written": [], "deleted": []})

    async def test_merge_applies_child_changes_to_parent(self):
        parent = MontySandbox()
        await parent.write_file("/keep.txt", "keep")

        child = parent.fork()
        await child.write_file("/new.txt", "from child")
        await child.write_file("/keep.txt", "edited by child")

        report = parent.merge(child)
        self.assertEqual(report["conflicts"], [])
        self.assertCountEqual(report["written"], ["/keep.txt", "/new.txt"])
        self.assertEqual((await parent.read_file("/new.txt"))["content"], "from child")
        self.assertEqual(
            (await parent.read_file("/keep.txt"))["content"], "edited by child"
        )

    async def test_merge_propagates_deletions(self):
        parent = MontySandbox()
        await parent.write_file("/doomed.txt", "bye")
        child = parent.fork()
        await child.run("import pathlib; pathlib.Path('/doomed.txt').unlink()")

        report = parent.merge(child)
        self.assertEqual(report["deleted"], ["/doomed.txt"])
        self.assertIn("error", await parent.read_file("/doomed.txt"))

    async def test_merge_refuses_conflicts_unless_forced(self):
        # Two siblings forked from the same point both edit the same file.
        parent = MontySandbox()
        await parent.write_file("/f.txt", "base")
        a = parent.fork()
        b = parent.fork()
        await a.write_file("/f.txt", "from A")
        await b.write_file("/f.txt", "from B")

        first = parent.merge(a)
        self.assertEqual(first["conflicts"], [])  # parent unchanged when A merges
        self.assertEqual(first["skipped"], [])

        # B now conflicts (parent moved to "from A"): refused by default.
        second = parent.merge(b)
        self.assertEqual(second["conflicts"], ["/f.txt"])
        self.assertEqual(second["skipped"], ["/f.txt"])
        self.assertEqual(second["written"], [])
        self.assertEqual((await parent.read_file("/f.txt"))["content"], "from A")

        # force=True applies the conflicting change (last writer wins).
        forced = parent.merge(b, force=True)
        self.assertEqual(forced["conflicts"], ["/f.txt"])
        self.assertEqual(forced["skipped"], [])
        self.assertEqual(forced["written"], ["/f.txt"])
        self.assertEqual((await parent.read_file("/f.txt"))["content"], "from B")

    async def test_merge_applies_non_conflicting_changes_when_one_conflicts(self):
        # A conflict on one path must not block the rest of the merge.
        parent = MontySandbox()
        await parent.write_file("/shared.txt", "base")
        child = parent.fork()
        await child.write_file("/shared.txt", "child edit")
        await child.write_file("/fresh.txt", "new")
        await parent.write_file("/shared.txt", "parent edit")  # diverge

        report = parent.merge(child)
        self.assertEqual(report["skipped"], ["/shared.txt"])
        self.assertEqual(report["written"], ["/fresh.txt"])
        self.assertEqual(
            (await parent.read_file("/shared.txt"))["content"], "parent edit"
        )
        self.assertEqual((await parent.read_file("/fresh.txt"))["content"], "new")

    async def test_merge_paths_restricts_to_subset(self):
        parent = MontySandbox()
        child = parent.fork()
        await child.write_file("/wanted.txt", "yes")
        await child.write_file("/skipped.txt", "no")

        report = parent.merge(child, paths=["/wanted.txt"])
        self.assertEqual(report["written"], ["/wanted.txt"])
        self.assertEqual((await parent.read_file("/wanted.txt"))["content"], "yes")
        self.assertIn("error", await parent.read_file("/skipped.txt"))

    async def test_parallel_forks_run_independently(self):
        import asyncio

        parent = MontySandbox()
        await parent.write_file("/seed.txt", "seed")

        async def work(branch, value):
            await branch.write_file("/out.txt", value)
            return (await branch.read_file("/out.txt"))["content"]

        forks = [parent.fork(name=f"f{i}") for i in range(5)]
        results = await asyncio.gather(
            *(work(fk, f"value-{i}") for i, fk in enumerate(forks))
        )
        # Each fork kept its own value; none clobbered another or the parent.
        self.assertEqual(results, [f"value-{i}" for i in range(5)])
        self.assertIn("error", await parent.read_file("/out.txt"))

    async def test_fork_copy_repl_inherits_namespace(self):
        parent = MontySandbox()
        await parent.run("shared = 41")
        child = parent.fork(copy_repl=True)
        result = await child.run("print(shared + 1)")
        self.assertIn("42", result.stdout)
        # Without copy_repl (the default), the namespace is clean.
        fresh = parent.fork()
        self.assertFalse((await fresh.run("print(shared)")).ok)

    async def test_merge_repl_adopts_namespace(self):
        # merge(repl=True) folds the child's whole namespace (vars, functions,
        # imports) back into the parent, alongside the parent's own bindings.
        parent = MontySandbox()
        await parent.run("import math")
        await parent.run("x = 41")
        child = parent.fork(copy_repl=True)
        await child.run("import json")
        await child.run("def g(n):\n    return n * n")
        await child.run("z = json.dumps({'k': x})")
        await child.write_file("/out.txt", "child file")

        report = parent.merge(child, repl=True)
        self.assertTrue(report["repl_adopted"])
        self.assertIn("/out.txt", report["written"])  # filesystem merged too
        # Parent now has the child's import, function and variable, plus its own.
        result = await parent.run("print(g(x), z, int(math.sqrt(16)))")
        self.assertTrue(result.ok, result.error)
        self.assertEqual(result.stdout.strip(), '1681 {"k": 41} 4')

    async def test_merge_without_repl_leaves_namespace_untouched(self):
        parent = MontySandbox()
        await parent.run("x = 1")
        child = parent.fork(copy_repl=True)
        await child.run("x = 999")
        report = parent.merge(child)  # repl defaults to False
        self.assertFalse(report["repl_adopted"])
        self.assertEqual((await parent.run("print(x)")).stdout.strip(), "1")
