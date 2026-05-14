# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import shutil
import tempfile
from pathlib import Path

from synalinks.src import testing
from synalinks.src.backend import ChatMessages
from synalinks.src.modules.agents.deep_agent import DeepAgent
from synalinks.src.modules.agents.deep_agent import PathTraversalError
from synalinks.src.modules.agents.deep_agent import _build_tools
from synalinks.src.modules.agents.deep_agent import _resolve_inside_workdir
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Program


async def stamp_now(label: str):
    """Return the input label unchanged.

    Args:
        label (str): Arbitrary text to echo back.
    """
    return {"label": label}


class DeepAgentResolveTest(testing.TestCase):
    """Pure-path tests for the traversal-defense helper."""

    def _workdir(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        return Path(tmpdir)

    def test_relative_path_inside_workdir(self):
        wd = self._workdir()
        resolved = _resolve_inside_workdir(wd, "foo.txt")
        self.assertTrue(str(resolved).startswith(str(wd.resolve())))

    def test_nested_path_inside_workdir(self):
        wd = self._workdir()
        resolved = _resolve_inside_workdir(wd, "subdir/file.txt")
        self.assertTrue(str(resolved).startswith(str(wd.resolve())))

    def test_dot_dot_escape_rejected(self):
        wd = self._workdir()
        with self.assertRaises(PathTraversalError):
            _resolve_inside_workdir(wd, "../escape.txt")

    def test_deep_dot_dot_escape_rejected(self):
        wd = self._workdir()
        with self.assertRaises(PathTraversalError):
            _resolve_inside_workdir(wd, "subdir/../../escape.txt")

    def test_absolute_path_outside_rejected(self):
        wd = self._workdir()
        with self.assertRaises(PathTraversalError):
            _resolve_inside_workdir(wd, "/etc/passwd")

    def test_symlink_to_outside_rejected(self):
        wd = self._workdir()
        # Create an outside file, then a symlink to it from inside the workdir.
        outside_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, outside_dir, ignore_errors=True)
        target = Path(outside_dir) / "secret"
        target.write_text("secret")
        link = wd / "shortcut"
        os.symlink(str(target), str(link))

        # Resolving "shortcut" follows the symlink → outside workdir → reject.
        with self.assertRaises(PathTraversalError):
            _resolve_inside_workdir(wd, "shortcut")


class DeepAgentToolsTest(testing.TestCase):
    """Tool-level tests against a real temp workdir."""

    def _workdir(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        return Path(tmpdir)

    def _tools(
        self,
        workdir,
        allow_write=True,
        allow_bash=True,
        timeout=5.0,
        max_output_chars=10_000,
        max_search_results=100,
    ):
        return _build_tools(
            workdir,
            allow_write=allow_write,
            allow_bash=allow_bash,
            timeout=timeout,
            max_output_chars=max_output_chars,
            max_search_results=max_search_results,
        )

    async def test_read_file_basic(self):
        wd = self._workdir()
        (wd / "hello.txt").write_text("hello world")
        read_file, *_ = self._tools(wd)

        result = await read_file(path="hello.txt", offset=0, limit=1000)
        # Single line, prefixed with 1-based line number + tab.
        self.assertEqual(result["content"], "1\thello world")
        self.assertEqual(result["lines_returned"], 1)
        self.assertEqual(result["total_lines"], 1)
        self.assertFalse(result["has_more"])

    async def test_read_file_multiple_lines(self):
        wd = self._workdir()
        (wd / "multi.txt").write_text("alpha\nbeta\ngamma\n")
        read_file, *_ = self._tools(wd)

        result = await read_file(path="multi.txt", offset=0, limit=10)
        self.assertEqual(result["content"], "1\talpha\n2\tbeta\n3\tgamma")
        self.assertEqual(result["lines_returned"], 3)
        self.assertEqual(result["total_lines"], 3)
        self.assertFalse(result["has_more"])

    async def test_read_file_offset_skips_lines(self):
        wd = self._workdir()
        (wd / "page.txt").write_text("a\nb\nc\nd\ne\n")
        read_file, *_ = self._tools(wd)

        result = await read_file(path="page.txt", offset=2, limit=2)
        self.assertEqual(result["content"], "3\tc\n4\td")
        self.assertEqual(result["lines_returned"], 2)
        self.assertEqual(result["total_lines"], 5)
        self.assertTrue(result["has_more"])

    async def test_read_file_limit_paginates_within_file(self):
        wd = self._workdir()
        (wd / "long.txt").write_text("\n".join(str(i) for i in range(100)) + "\n")
        read_file, *_ = self._tools(wd)

        # First page: lines 1-10.
        page1 = await read_file(path="long.txt", offset=0, limit=10)
        self.assertEqual(page1["lines_returned"], 10)
        self.assertEqual(page1["total_lines"], 100)
        self.assertTrue(page1["has_more"])
        self.assertTrue(page1["content"].startswith("1\t0"))

        # Last page: offset=95, should return lines 96-100 (5 lines).
        page_last = await read_file(path="long.txt", offset=95, limit=10)
        self.assertEqual(page_last["lines_returned"], 5)
        self.assertFalse(page_last["has_more"])

    async def test_read_file_line_truncated_to_max_output_chars(self):
        wd = self._workdir()
        # A single 1000-char line.
        (wd / "wide.txt").write_text("X" * 1000)
        read_file, *_ = self._tools(wd, max_output_chars=10)

        result = await read_file(path="wide.txt", offset=0, limit=1)
        # Prefix + truncated 10-char content.
        self.assertEqual(result["content"], "1\t" + "X" * 10)
        self.assertEqual(result["lines_returned"], 1)
        self.assertEqual(result["total_lines"], 1)

    async def test_read_file_missing(self):
        wd = self._workdir()
        read_file, *_ = self._tools(wd)
        result = await read_file(path="does_not_exist.txt", offset=0, limit=100)
        self.assertIn("error", result)

    async def test_read_file_rejects_traversal(self):
        wd = self._workdir()
        read_file, *_ = self._tools(wd)
        result = await read_file(path="../etc/passwd", offset=0, limit=100)
        self.assertIn("error", result)
        self.assertIn("outside", result["error"])

    async def test_list_directory_basic(self):
        wd = self._workdir()
        (wd / "a.txt").write_text("a")
        (wd / "b.txt").write_text("bb")
        (wd / "sub").mkdir()
        _, list_directory, *_ = self._tools(wd)

        result = await list_directory(path=".")
        self.assertEqual(result["entry_count"], 3)
        names = {e["name"] for e in result["entries"]}
        self.assertEqual(names, {"a.txt", "b.txt", "sub"})
        sub_entry = next(e for e in result["entries"] if e["name"] == "sub")
        self.assertEqual(sub_entry["type"], "dir")

    async def test_search_files_glob_only(self):
        wd = self._workdir()
        (wd / "a.py").write_text("print(1)")
        (wd / "b.py").write_text("print(2)")
        (wd / "c.md").write_text("# title")
        (wd / "sub").mkdir()
        (wd / "sub" / "d.py").write_text("print(3)")

        _, _, search_files, *_ = self._tools(wd)
        result = await search_files(file_pattern="**/*.py", content_pattern="")

        self.assertEqual(result["match_count"], 3)
        self.assertEqual(
            set(result["files"]), {"a.py", "b.py", os.path.join("sub", "d.py")}
        )
        self.assertFalse(result["truncated"])

    async def test_search_files_glob_excludes_other_extensions(self):
        wd = self._workdir()
        (wd / "a.py").write_text("x")
        (wd / "b.md").write_text("y")

        _, _, search_files, *_ = self._tools(wd)
        result = await search_files(file_pattern="*.py", content_pattern="")

        self.assertEqual(result["files"], ["a.py"])

    async def test_search_files_grep_mode(self):
        wd = self._workdir()
        (wd / "x.py").write_text("def foo():\n    pass\n\ndef bar():\n    foo()\n")
        (wd / "y.py").write_text("def baz():\n    pass\n")

        _, _, search_files, *_ = self._tools(wd)
        result = await search_files(file_pattern="*.py", content_pattern=r"def \w+")

        # 3 def lines across the two files.
        self.assertEqual(result["match_count"], 3)
        # Each match carries path + line number + line.
        for m in result["matches"]:
            self.assertIn("path", m)
            self.assertIn("line_number", m)
            self.assertIn("line", m)

    async def test_search_files_grep_with_no_match(self):
        wd = self._workdir()
        (wd / "x.py").write_text("hello world")

        _, _, search_files, *_ = self._tools(wd)
        result = await search_files(file_pattern="*.py", content_pattern="nonexistent")

        self.assertEqual(result["match_count"], 0)
        self.assertEqual(result["matches"], [])

    async def test_search_files_caps_at_max_search_results(self):
        wd = self._workdir()
        for i in range(20):
            (wd / f"f{i}.txt").write_text("match")

        _, _, search_files, *_ = self._tools(wd, max_search_results=5)
        result = await search_files(file_pattern="*.txt", content_pattern="")

        self.assertEqual(result["match_count"], 5)
        self.assertTrue(result["truncated"])

    async def test_search_files_grep_caps_at_max_search_results(self):
        wd = self._workdir()
        # 10 lines all matching, expect cap at 3.
        (wd / "big.txt").write_text("\n".join("hit" for _ in range(10)))

        _, _, search_files, *_ = self._tools(wd, max_search_results=3)
        result = await search_files(file_pattern="*.txt", content_pattern="hit")

        self.assertEqual(result["match_count"], 3)
        self.assertTrue(result["truncated"])

    async def test_search_files_skips_oversized_files(self):
        # Skipping the actual 1MB+ creation; just verify a 1MB+ file is filtered.
        wd = self._workdir()
        big_path = wd / "huge.txt"
        # Write 1MB + 100 bytes to trip the _SEARCH_MAX_FILE_BYTES guard.
        big_path.write_text("match\n" * 175_000)
        # And a small file that should still be searched.
        (wd / "small.txt").write_text("match")

        _, _, search_files, *_ = self._tools(wd)
        result = await search_files(file_pattern="*.txt", content_pattern="match")

        # Matches come only from small.txt — the big one is skipped.
        paths_with_matches = {m["path"] for m in result["matches"]}
        self.assertIn("small.txt", paths_with_matches)
        self.assertNotIn("huge.txt", paths_with_matches)

    async def test_search_files_invalid_regex(self):
        wd = self._workdir()
        (wd / "x.py").write_text("hi")

        _, _, search_files, *_ = self._tools(wd)
        result = await search_files(file_pattern="*.py", content_pattern="[unclosed")

        self.assertIn("error", result)
        self.assertIn("regex", result["error"])

    async def test_search_files_skips_symlink_escape(self):
        wd = self._workdir()
        outside = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        (Path(outside) / "secret.txt").write_text("secret")
        os.symlink(os.path.join(outside, "secret.txt"), str(wd / "secret.txt"))

        _, _, search_files, *_ = self._tools(wd)
        # Symlink resolves outside workdir → filtered out, no match.
        result = await search_files(
            file_pattern="*.txt", content_pattern="secret"
        )
        self.assertEqual(result["match_count"], 0)

    async def test_list_directory_rejects_traversal(self):
        wd = self._workdir()
        _, list_directory, *_ = self._tools(wd)
        result = await list_directory(path="..")
        self.assertIn("error", result)

    async def test_write_file_creates_and_overwrites(self):
        wd = self._workdir()
        _, _, _, write_file, _, _ = self._tools(wd)

        r1 = await write_file(path="new.txt", content="hello")
        self.assertTrue(r1["created"])
        self.assertEqual((wd / "new.txt").read_text(), "hello")

        r2 = await write_file(path="new.txt", content="bye")
        self.assertFalse(r2["created"])
        self.assertEqual((wd / "new.txt").read_text(), "bye")

    async def test_write_file_creates_parent_dirs(self):
        wd = self._workdir()
        _, _, _, write_file, _, _ = self._tools(wd)

        r = await write_file(path="deep/nested/path.txt", content="x")
        self.assertTrue(r["created"])
        self.assertTrue((wd / "deep" / "nested" / "path.txt").exists())

    async def test_write_file_rejects_traversal(self):
        wd = self._workdir()
        _, _, _, write_file, _, _ = self._tools(wd)
        result = await write_file(path="../escape.txt", content="x")
        self.assertIn("error", result)

    async def test_edit_file_single_replacement(self):
        wd = self._workdir()
        (wd / "src.txt").write_text("alpha BETA gamma")
        _, _, _, _, edit_file, _ = self._tools(wd)

        r = await edit_file(path="src.txt", old_string="BETA", new_string="DELTA")
        self.assertNotIn("error", r)
        self.assertEqual((wd / "src.txt").read_text(), "alpha DELTA gamma")

    async def test_edit_file_rejects_zero_matches(self):
        wd = self._workdir()
        (wd / "src.txt").write_text("alpha BETA gamma")
        _, _, _, _, edit_file, _ = self._tools(wd)

        r = await edit_file(path="src.txt", old_string="MISSING", new_string="x")
        self.assertIn("error", r)
        self.assertIn("not found", r["error"])

    async def test_edit_file_rejects_multiple_matches(self):
        wd = self._workdir()
        (wd / "src.txt").write_text("foo foo")
        _, _, _, _, edit_file, _ = self._tools(wd)

        r = await edit_file(path="src.txt", old_string="foo", new_string="bar")
        self.assertIn("error", r)
        self.assertIn("2 times", r["error"])

    async def test_run_bash_basic(self):
        wd = self._workdir()
        (wd / "hi.txt").write_text("hi")
        *_, run_bash = self._tools(wd)

        r = await run_bash(command="ls")
        self.assertEqual(r["returncode"], 0)
        self.assertIn("hi.txt", r["stdout"])

    async def test_run_bash_cwd_is_workdir(self):
        wd = self._workdir()
        *_, run_bash = self._tools(wd)

        r = await run_bash(command="pwd")
        # readlink resolves on macOS for /private/var → /var symlink etc.
        self.assertEqual(r["stdout"].strip(), str(wd.resolve()))

    async def test_run_bash_captures_nonzero(self):
        wd = self._workdir()
        *_, run_bash = self._tools(wd)

        r = await run_bash(command="false")
        self.assertNotEqual(r["returncode"], 0)

    async def test_run_bash_timeout(self):
        wd = self._workdir()
        *_, run_bash = self._tools(wd, timeout=0.2)

        r = await run_bash(command="sleep 5")
        self.assertIn("error", r)
        self.assertIn("timed out", r["error"])

    async def test_run_bash_truncates_stdout(self):
        wd = self._workdir()
        *_, run_bash = self._tools(wd, max_output_chars=20)

        # Produce ~100 chars of stdout, expect truncation.
        r = await run_bash(command="python3 -c 'print(\"X\" * 100)'")
        self.assertTrue(r["stdout_truncated"])
        self.assertLessEqual(len(r["stdout"]), 20)


class DeepAgentInstantiationTest(testing.TestCase):
    """End-to-end instantiation tests — wires DeepAgent but doesn't run LM."""

    def _workdir(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        return tmpdir

    async def test_agent_instantiation(self):
        wd = self._workdir()
        lm = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=ChatMessages)
        outputs = await DeepAgent(
            workdir=wd,
            language_model=lm,
            name="da",
        )(inputs)

        program = Program(inputs=inputs, outputs=outputs, name="da_program")
        self.assertIsNotNone(program)

    async def test_agent_default_tool_set(self):
        wd = self._workdir()
        lm = LanguageModel(model="ollama/mistral")

        agent = DeepAgent(workdir=wd, language_model=lm, name="all_tools")
        tool_names = set(agent.agent.tools.keys())
        self.assertEqual(
            tool_names,
            {
                "read_file",
                "list_directory",
                "search_files",
                "write_file",
                "edit_file",
                "run_bash",
            },
        )

    async def test_agent_read_only_drops_write_tools(self):
        wd = self._workdir()
        lm = LanguageModel(model="ollama/mistral")

        agent = DeepAgent(
            workdir=wd,
            language_model=lm,
            allow_write=False,
            name="ro",
        )
        tool_names = set(agent.agent.tools.keys())
        self.assertNotIn("write_file", tool_names)
        self.assertNotIn("edit_file", tool_names)
        # Read-only mode still has read/search/list.
        self.assertIn("read_file", tool_names)
        self.assertIn("search_files", tool_names)
        self.assertIn("list_directory", tool_names)

    async def test_agent_no_bash_drops_run_bash(self):
        wd = self._workdir()
        lm = LanguageModel(model="ollama/mistral")

        agent = DeepAgent(
            workdir=wd,
            language_model=lm,
            allow_bash=False,
            name="no_bash",
        )
        tool_names = set(agent.agent.tools.keys())
        self.assertNotIn("run_bash", tool_names)

    async def test_agent_appends_user_tools(self):
        wd = self._workdir()
        lm = LanguageModel(model="ollama/mistral")

        agent = DeepAgent(
            workdir=wd,
            language_model=lm,
            tools=[Tool(stamp_now)],
            name="with_extra",
        )
        self.assertIn("stamp_now", agent.agent.tools)

    async def test_agent_tool_name_collision_raises(self):
        wd = self._workdir()
        lm = LanguageModel(model="ollama/mistral")

        async def run_bash(command: str):
            """Shadows the built-in.

            Args:
                command (str): unused.
            """
            return {"oops": True}

        with self.assertRaises(ValueError):
            DeepAgent(workdir=wd, language_model=lm, tools=[run_bash])

    async def test_agent_rejects_missing_workdir(self):
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            DeepAgent(workdir="/this/path/does/not/exist/anywhere", language_model=lm)

    async def test_agent_rejects_empty_workdir(self):
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            DeepAgent(workdir="", language_model=lm)

    async def test_agent_rejects_non_directory_workdir(self):
        tmp = tempfile.NamedTemporaryFile(delete=False)
        self.addCleanup(os.unlink, tmp.name)
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            DeepAgent(workdir=tmp.name, language_model=lm)

    async def test_agent_rejects_invalid_timeout(self):
        wd = self._workdir()
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            DeepAgent(workdir=wd, language_model=lm, timeout=0)
        with self.assertRaises(ValueError):
            DeepAgent(workdir=wd, language_model=lm, timeout=-1)

    async def test_agent_rejects_invalid_max_output_chars(self):
        wd = self._workdir()
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            DeepAgent(workdir=wd, language_model=lm, max_output_chars=0)

    async def test_agent_rejects_invalid_max_search_results(self):
        wd = self._workdir()
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            DeepAgent(workdir=wd, language_model=lm, max_search_results=0)
        with self.assertRaises(ValueError):
            DeepAgent(workdir=wd, language_model=lm, max_search_results=-1)
