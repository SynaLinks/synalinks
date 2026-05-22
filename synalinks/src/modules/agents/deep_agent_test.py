# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
import os
import shutil
import tempfile
from pathlib import Path

from synalinks.src import ops
from synalinks.src import testing
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import DataModel
from synalinks.src.modules.agents.deep_agent import DeepAgent
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


class DeepAgentSandboxTest(testing.TestCase):
    """The agent's tools are its MontySandbox methods (overlay-backed)."""

    def _workdir(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        return tmpdir

    async def test_tools_read_workdir_and_writes_stay_host_safe(self):
        wd = self._workdir()
        (Path(wd) / "hello.txt").write_text("hi from disk")
        lm = LanguageModel(model="ollama/mistral")
        agent = DeepAgent(workdir=wd, language_model=lm, name="fs")

        # read_file falls through to the real workdir...
        read = await agent.sandbox.read_file("/hello.txt")
        self.assertEqual(read["content"], "hi from disk")
        # ...write_file lands in the overlay, never on the host disk.
        await agent.sandbox.write_file("/PLAN.md", "step 1")
        self.assertEqual((await agent.sandbox.read_file("/PLAN.md"))["content"], "step 1")
        self.assertFalse((Path(wd) / "PLAN.md").exists())
        self.assertIn("/PLAN.md", agent.sandbox.changes()["written"])

    async def test_search_files_greps_the_workdir(self):
        wd = self._workdir()
        (Path(wd) / "a.py").write_text("x = 1\nTODO: fix\n")
        lm = LanguageModel(model="ollama/mistral")
        agent = DeepAgent(workdir=wd, language_model=lm, name="grep")

        res = await agent.sandbox.search_files(pattern="TODO", glob="**/*.py")
        self.assertEqual(res["total"], 1)
        self.assertEqual(res["matches"][0]["path"], "/a.py")
        self.assertEqual(res["matches"][0]["line"], 2)

    async def test_run_python_file_runs_a_script_built_in_the_overlay(self):
        wd = self._workdir()
        lm = LanguageModel(model="ollama/mistral")
        agent = DeepAgent(workdir=wd, language_model=lm, name="runner")

        # The build-then-run workflow: write a self-contained script to the
        # overlay, then run it with run_python_file.
        await agent.sandbox.write_file("/script.py", "print(sum(range(10)))\n")
        result = await agent.sandbox.run_python_file("/script.py")
        self.assertTrue(result["ok"], result["error"])
        self.assertEqual(result["stdout"].strip(), "45")
        # The script file never touched the host disk.
        self.assertFalse((Path(wd) / "script.py").exists())


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
                "list_files",
                "search_files",
                "write_file",
                "edit_file",
                "run_python_code",
                "run_python_file",
            },
        )

    async def test_agent_without_workdir_uses_in_memory_filesystem(self):
        lm = LanguageModel(model="ollama/mistral")
        agent = DeepAgent(language_model=lm, name="in_memory")
        self.assertIsNone(agent.workdir)
        self.assertIsNone(agent.sandbox.workdir)
        # The in-memory filesystem is writable and readable.
        await agent.sandbox.write_file("/scratch.txt", "hi")
        self.assertEqual((await agent.sandbox.read_file("/scratch.txt"))["content"], "hi")
        # The full tool set is always available (nothing to gate).
        self.assertEqual(
            set(agent.agent.tools.keys()),
            {
                "read_file",
                "list_files",
                "search_files",
                "write_file",
                "edit_file",
                "run_python_code",
                "run_python_file",
            },
        )

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

        async def read_file(path: str):
            """Shadows the built-in.

            Args:
                path (str): unused.
            """
            return {"oops": True}

        with self.assertRaises(ValueError):
            DeepAgent(workdir=wd, language_model=lm, tools=[read_file])

    async def test_agent_rejects_nonexistent_workdir(self):
        lm = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            DeepAgent(workdir="/this/path/does/not/exist/anywhere", language_model=lm)

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


class DeepAgentInputsSummaryTest(testing.TestCase):
    """Data inputs are summarized for the LM; full values land in the overlay."""

    def _agent(self, workdir=None):
        return DeepAgent(
            workdir=workdir,
            language_model=LanguageModel(model="ollama/mistral"),
            name="d",
        )

    async def test_data_input_summarized_and_written_to_overlay(self):
        agent = self._agent()

        class Doc(DataModel):
            title: str
            body: str

        long_body = "x" * 500
        summary = await agent._materialize_inputs(Doc(title="t", body=long_body))

        # Full, untruncated inputs land in inputs.json in the overlay.
        written = await agent.sandbox.read_file("inputs.json")
        self.assertEqual(
            json.loads(written["content"]), {"title": "t", "body": long_body}
        )

        # The LM is handed only an InputsSummary naming that file, with the long
        # field previewed/truncated rather than dumped in full.
        sj = summary.get_json()
        self.assertEqual(sj["inputs_file"], "inputs.json")
        body_field = next(f for f in sj["fields"] if f["name"] == "body")
        self.assertTrue(body_field["truncated"])
        self.assertLess(len(body_field["preview"]), len(long_body))

    async def test_pure_chatmessages_passes_through_untouched(self):
        agent = self._agent()
        cm = ChatMessages(messages=[])
        out = await agent._materialize_inputs(cm)
        self.assertIs(out, cm)
        # No inputs file written for a pure conversation.
        self.assertEqual(agent.sandbox.changes()["written"], [])

    async def test_messages_plus_data_is_summarized_not_passed_through(self):
        # A model carrying `messages` AND a data field is NOT a pure
        # conversation (is_strictly_chat_messages is False), so it's summarized.
        agent = self._agent()

        class Query(DataModel):
            query: str

        mixed = await ops.concat(ChatMessages(messages=[]), Query(query="hi"))
        out = await agent._materialize_inputs(mixed)
        self.assertNotEqual(out, mixed)
        self.assertEqual(out.get_json()["inputs_file"], "inputs.json")

    async def test_inputs_path_avoids_workdir_collision(self):
        wd = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, wd, ignore_errors=True)
        (Path(wd) / "inputs.json").write_text('{"keep": "me"}')
        agent = self._agent(workdir=wd)

        class Doc(DataModel):
            q: str

        summary = await agent._materialize_inputs(Doc(q="hi"))
        # Picked a non-colliding name; the mounted workdir file is untouched.
        self.assertEqual(summary.get_json()["inputs_file"], "inputs_1.json")
        self.assertEqual((Path(wd) / "inputs.json").read_text(), '{"keep": "me"}')
