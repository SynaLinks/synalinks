# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

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


def _lm_response(*, content=None, tool_calls=None):
    """Build a litellm-shaped response for the native function-calling path.

    ``tool_calls`` is a list of ``{"name": ..., "arguments": {...}}``; each is
    wrapped into the OpenAI nested ``{id, type, function}`` envelope with the
    arguments JSON-encoded, matching what the LM layer parses at runtime.
    """
    message = {"content": content}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]),
                },
            }
            for i, tc in enumerate(tool_calls)
        ]
    return {"choices": [{"message": message}]}


class DeepAgentSandboxTest(testing.TestCase):
    """The agent's tools are its MirageSandbox methods (filesystem-backed)."""

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
        tool_names = set(agent.tools.keys())
        self.assertEqual(
            tool_names,
            {
                "read_file",
                "list_files",
                "search_files",
                "write_file",
                "edit_file",
                "run_bash",
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
            set(agent.tools.keys()),
            {
                "read_file",
                "list_files",
                "search_files",
                "write_file",
                "edit_file",
                "run_bash",
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
        self.assertIn("stamp_now", agent.tools)

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


class DeepAgentSubagentTest(testing.TestCase):
    """Subagent delegation: parallel forks, parent-reviewed merges."""

    def _lm(self):
        return LanguageModel(model="ollama/mistral")

    def _tool_names(self, agent):
        return set(agent.tools.keys())

    async def test_subagents_off_by_default(self):
        agent = DeepAgent(language_model=self._lm(), name="off")
        self.assertFalse(agent._subagents_enabled)
        names = self._tool_names(agent)
        self.assertNotIn("spawn_subagents", names)
        self.assertNotIn("merge_subagent", names)
        self.assertNotIn("discard_subagent", names)

    async def test_subagents_enabled_exposes_tools_and_guidance(self):
        agent = DeepAgent(language_model=self._lm(), max_subagent_depth=1, name="on")
        self.assertTrue(agent._subagents_enabled)
        names = self._tool_names(agent)
        self.assertIn("spawn_subagents", names)
        self.assertIn("merge_subagent", names)
        self.assertIn("discard_subagent", names)
        self.assertIn("spawn_subagents", agent.instructions)

    async def test_subagent_at_max_depth_cannot_spawn(self):
        # A subagent (depth 1) under max depth 1 does not get the spawn tools.
        sub = DeepAgent(
            language_model=self._lm(),
            max_subagent_depth=1,
            _subagent_depth=1,
            name="leaf",
        )
        self.assertFalse(sub._subagents_enabled)
        self.assertNotIn("spawn_subagents", self._tool_names(sub))

    async def test_negative_depth_rejected(self):
        with self.assertRaises(ValueError):
            DeepAgent(language_model=self._lm(), max_subagent_depth=-1)

    async def test_sandbox_param_is_used_as_is(self):
        from synalinks.src.sandboxes.mirage_sandbox import MirageSandbox

        sb = MirageSandbox()
        await sb.write_file("/seed.txt", "seed")
        agent = DeepAgent(language_model=self._lm(), sandbox=sb, name="reuse")
        self.assertIs(agent.sandbox, sb)
        self.assertEqual((await agent.sandbox.read_file("/seed.txt"))["content"], "seed")

    async def test_get_config_round_trips_max_subagent_depth(self):
        agent = DeepAgent(language_model=self._lm(), max_subagent_depth=2, name="cfg")
        config = agent.get_config()
        self.assertEqual(config["max_subagent_depth"], 2)
        restored = DeepAgent.from_config(config)
        self.assertEqual(restored.max_subagent_depth, 2)
        self.assertTrue(restored._subagents_enabled)

    async def test_merge_subagent_applies_fork_changes(self):
        agent = DeepAgent(language_model=self._lm(), max_subagent_depth=1, name="m")
        await agent.sandbox.write_file("/keep.txt", "base")
        # Stand in for a finished subagent: a fork that changed some files.
        fork = agent.sandbox.fork()
        await fork.write_file("/new.txt", "child")
        await fork.write_file("/keep.txt", "edited by child")
        agent._subagents["subagent_0"] = fork

        report = await agent.merge_subagent("subagent_0")
        self.assertCountEqual(report["written"], ["/keep.txt", "/new.txt"])
        self.assertEqual(report["conflicts"], [])
        self.assertEqual((await agent.sandbox.read_file("/new.txt"))["content"], "child")
        self.assertEqual(
            (await agent.sandbox.read_file("/keep.txt"))["content"], "edited by child"
        )

    async def test_merge_subagent_paths_subset(self):
        agent = DeepAgent(language_model=self._lm(), max_subagent_depth=1, name="ms")
        fork = agent.sandbox.fork()
        await fork.write_file("/wanted.txt", "yes")
        await fork.write_file("/skipped.txt", "no")
        agent._subagents["subagent_0"] = fork

        report = await agent.merge_subagent("subagent_0", paths=["/wanted.txt"])
        self.assertEqual(report["written"], ["/wanted.txt"])
        self.assertIn("error", await agent.sandbox.read_file("/skipped.txt"))

    async def test_merge_and_discard_unknown_handle(self):
        agent = DeepAgent(language_model=self._lm(), max_subagent_depth=1, name="u")
        self.assertIn("error", await agent.merge_subagent("nope"))
        self.assertIn("error", await agent.discard_subagent("nope"))

    async def test_discard_subagent_drops_branch(self):
        agent = DeepAgent(language_model=self._lm(), max_subagent_depth=1, name="d")
        agent._subagents["subagent_0"] = agent.sandbox.fork()
        self.assertEqual(
            await agent.discard_subagent("subagent_0"), {"discarded": "subagent_0"}
        )
        self.assertNotIn("subagent_0", agent._subagents)

    async def test_spawn_subagents_empty(self):
        agent = DeepAgent(language_model=self._lm(), max_subagent_depth=1, name="e")
        self.assertIn("error", await agent.spawn_subagents([]))

    @patch("litellm.acompletion")
    async def test_spawn_runs_parallel_isolated_subagents(self, mock_completion):
        # Every LM call returns a plain answer (no tool calls), so each subagent
        # finalizes immediately — deterministic regardless of parallel ordering.
        def no_tools(*args, **kwargs):
            return _lm_response(content="nothing to do")

        mock_completion.side_effect = no_tools

        agent = DeepAgent(language_model=self._lm(), max_subagent_depth=1, name="par")
        await agent.sandbox.write_file("/parent.txt", "owned by parent")

        result = await agent.spawn_subagents(["explore A", "explore B"])
        subs = result["subagents"]
        self.assertEqual(len(subs), 2)
        self.assertEqual({s["handle"] for s in subs}, {"subagent_0", "subagent_1"})
        for s in subs:
            self.assertEqual(s["diff"], {"written": [], "deleted": []})
            self.assertIsInstance(s["result"], str)
        # Forks are registered for later merge; parent filesystem is untouched.
        self.assertEqual(set(agent._subagents), {"subagent_0", "subagent_1"})
        self.assertEqual(agent.sandbox.changes()["written"], ["/parent.txt"])

    @patch("litellm.acompletion")
    async def test_subagent_handles_reset_each_call(self, mock_completion):
        # Handles from one turn must not leak into the next.
        mock_completion.side_effect = lambda *a, **k: _lm_response(content="ok")
        agent = DeepAgent(language_model=self._lm(), max_subagent_depth=1, name="reset")
        await agent.spawn_subagents(["a"])
        self.assertEqual(set(agent._subagents), {"subagent_0"})
        # A new turn clears the registry and restarts handle numbering.
        await agent.call(ChatMessages(messages=[]))
        self.assertEqual(agent._subagents, {})

    @patch("litellm.acompletion")
    async def test_spawn_then_merge_a_writing_subagent(self, mock_completion):
        # The subagent issues a native write_file tool call until a tool result
        # is in its trajectory, then finalizes. Content-aware (not a fixed-length
        # list) so it is robust to however many LM turns the loop takes.
        def fake(*args, **kwargs):
            text = json.dumps(kwargs.get("messages", []))
            if "tool_call_id" in text:  # the write already executed
                return _lm_response(content="wrote /sub.txt")
            return _lm_response(
                content="creating the file",
                tool_calls=[
                    {
                        "name": "write_file",
                        "arguments": {"path": "/sub.txt", "content": "from sub"},
                    }
                ],
            )

        mock_completion.side_effect = fake

        agent = DeepAgent(language_model=self._lm(), max_subagent_depth=1, name="w")
        result = await agent.spawn_subagents(["create /sub.txt"])
        sub = result["subagents"][0]
        # The subagent's pending change shows up in its diff...
        self.assertIn("/sub.txt", [w["path"] for w in sub["diff"]["written"]])
        # ...but NOT in the parent until merged.
        self.assertIn("error", await agent.sandbox.read_file("/sub.txt"))

        report = await agent.merge_subagent(sub["handle"])
        self.assertIn("/sub.txt", report["written"])
        self.assertEqual(
            (await agent.sandbox.read_file("/sub.txt"))["content"], "from sub"
        )
