# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from synalinks.src import rewards
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import Rating10
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.rewards.deep_agent_as_judge import DeepAgentAsJudge
from synalinks.src.rewards.deep_agent_as_judge import DeepAgentAsJudgeProgram
from synalinks.src.sandboxes.mirage_sandbox import MirageSandbox
from synalinks.src.saving.object_registration import register_synalinks_serializable


def _lm_response(*, content=None, tool_calls=None):
    """Build a litellm-shaped response for the native function-calling path."""
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


@register_synalinks_serializable()
async def word_count(text: str):
    """Count the words of a text.

    Args:
        text (str): The text to count words in.
    """
    return {"count": len(text.split())}


class Task(DataModel):
    task: str = Field(description="The coding task")


class Solution(DataModel):
    code: str = Field(description="The Python code solving the task")


class DeepAgentAsJudgeTest(testing.TestCase):
    def _workdir(self):
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        return tmpdir

    @patch("litellm.acompletion")
    async def test_deep_agent_as_judge(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = DeepAgentAsJudge(language_model=language_model, max_iterations=3)

        y_true = Solution(code="print(152648 + 485)")
        y_pred = Task(task="Print the sum of 152648 and 485.") + Solution(
            code="print(153133)"
        )

        mock_completion.side_effect = [
            # Turn 1: the judge runs the predicted code from the inputs file.
            # The shell feeds the file to python: a spawned python3 cannot open
            # the sandbox filesystem itself on every platform (no FUSE on macOS).
            _lm_response(
                content="Running the prediction.",
                tool_calls=[
                    {
                        "name": "run_bash",
                        "arguments": {
                            "command": (
                                "cat /inputs.json | python3 -c \"import json, sys; "
                                "exec(json.load(sys.stdin)['code'])\""
                            )
                        },
                    }
                ],
            ),
            # Turn 2: no tool calls, the loop stops.
            _lm_response(content="It prints 153133, same as the gold code."),
            # Final grading turn.
            _lm_response(content='{"critique": "Verified by running it.", "reward": 20}'),
        ]

        score = await reward(y_true=y_true, y_pred=y_pred)
        self.assertEqual(score, 1.0)
        self.assertEqual(mock_completion.call_count, 3)
        # The sandbox actually executed the prediction before grading.
        grading_messages = mock_completion.call_args.kwargs["messages"]
        tool_messages = [m for m in grading_messages if m.get("role") == "tool"]
        self.assertTrue(
            any("153133" in json.dumps(m.get("content")) for m in tool_messages),
            f"run_bash output not found in tool messages: {tool_messages}",
        )
        # The prompt carries a summary naming the inputs file, not the values.
        first_call_messages = mock_completion.call_args_list[0].kwargs["messages"]
        self.assertIn("inputs.json", json.dumps(first_call_messages))

    @patch("litellm.acompletion")
    async def test_deep_agent_as_judge_runs_workdir_tests(self, mock_completion):
        # The workdir ships the tests; the judge drops the predicted code next
        # to them and runs them.
        workdir = self._workdir()
        Path(workdir, "test_solution.py").write_text(
            "assert add(2, 3) == 5\nprint('PASS')\n"
        )
        language_model = LanguageModel(model="ollama/mistral")
        reward = DeepAgentAsJudge(
            language_model=language_model, workdir=workdir, max_iterations=3
        )

        y_true = Solution(code="def add(a, b):\n    return a + b\n")
        y_pred = Solution(code="def add(a, b):\n    return b + a\n")

        mock_completion.side_effect = [
            # Turn 1: write the prediction as `solution.py`, then run the tests
            # against it. The shell moves the files: a spawned python3 cannot
            # open the sandbox filesystem itself on every platform.
            _lm_response(
                content="Running the tests against the prediction.",
                tool_calls=[
                    {
                        "name": "run_bash",
                        "arguments": {
                            "command": (
                                "cat /inputs.json | python3 -c \"import json, sys; "
                                "print(json.load(sys.stdin)['code'])\" > /solution.py"
                                " && cat /solution.py /test_solution.py | python3 -c "
                                "\"import sys; exec(sys.stdin.read())\""
                            )
                        },
                    }
                ],
            ),
            # Turn 2: no tool calls, the loop stops.
            _lm_response(content="The tests pass."),
            # Final grading turn.
            _lm_response(content='{"critique": "Tests pass.", "reward": 20}'),
        ]

        score = await reward(y_true=y_true, y_pred=y_pred)
        self.assertEqual(score, 1.0)
        grading_messages = mock_completion.call_args.kwargs["messages"]
        tool_messages = [m for m in grading_messages if m.get("role") == "tool"]
        self.assertTrue(
            any("PASS" in json.dumps(m.get("content")) for m in tool_messages),
            f"test output not found in tool messages: {tool_messages}",
        )
        # The run stayed in the sandbox: the real workdir has no solution file.
        self.assertFalse(Path(workdir, "solution.py").exists())

    @patch("litellm.acompletion")
    async def test_deep_agent_as_judge_empty_prediction(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = DeepAgentAsJudge(language_model=language_model)

        score = await reward(y_true=Solution(code="print(1)"), y_pred=None)
        self.assertEqual(score, 0.0)
        mock_completion.assert_not_called()

    @patch("litellm.acompletion")
    async def test_deep_agent_as_judge_without_gold_reference(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = DeepAgentAsJudge(language_model=language_model, max_iterations=3)

        mock_completion.side_effect = [
            # Turn 1: the judge reads what it was given.
            _lm_response(
                content="Reading the inputs.",
                tool_calls=[{"name": "read_file", "arguments": {"path": "inputs.json"}}],
            ),
            # Turn 2: no tool calls, the loop stops.
            _lm_response(content="Only a prediction, no reference."),
            # Final grading turn.
            _lm_response(content='{"critique": "Plausible.", "reward": 20}'),
        ]

        score = await reward(y_true=None, y_pred=Solution(code="print(1)"))
        self.assertEqual(score, 1.0)
        # Without a reference the inputs file only holds the prediction.
        grading_messages = mock_completion.call_args.kwargs["messages"]
        tool_messages = [m for m in grading_messages if m.get("role") == "tool"]
        self.assertTrue(
            any("print(1)" in json.dumps(m.get("content")) for m in tool_messages),
            f"inputs file not found in tool messages: {tool_messages}",
        )
        self.assertNotIn("gold_", json.dumps(tool_messages))

    @patch("litellm.acompletion")
    async def test_deep_agent_as_judge_with_rating10_score_type(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = DeepAgentAsJudge(language_model=language_model, score_type=Rating10)

        # The judge asks the LM for a 1..10 integer...
        reward_schema = reward.program.agent.final_generator.schema["properties"][
            "reward"
        ]
        self.assertEqual(reward_schema["type"], "integer")
        self.assertEqual(reward_schema["enum"], list(range(1, 11)))

        mock_completion.side_effect = [
            _lm_response(content="No verification needed."),
            _lm_response(content='{"critique": "Close enough.", "reward": 7}'),
        ]
        score = await reward(
            y_true=Solution(code="print(1)"), y_pred=Solution(code="print(1.0)")
        )
        # ...and the reward is normalized: (7 - 1) / (10 - 1)
        self.assertAlmostEqual(score, 6 / 9)

    async def test_deep_agent_as_judge_symbolic_output_schema(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = DeepAgentAsJudge(language_model=language_model, score_type=Rating10)

        outputs = await reward.program(
            [
                Solution(code="print(1)").to_symbolic_data_model(),
                Solution(code="print(1)").to_symbolic_data_model(),
            ]
        )
        self.assertTrue(is_symbolic_data_model(outputs))
        # Whatever the judge's scale, the program advertises a 0..1 reward.
        reward_schema = outputs.get_schema()["properties"]["reward"]
        self.assertEqual(reward_schema["type"], "number")
        self.assertEqual(reward_schema["minimum"], 0.0)
        self.assertEqual(reward_schema["maximum"], 1.0)

    @patch("litellm.acompletion")
    async def test_deep_agent_as_judge_resets_sandbox_between_samples(
        self, mock_completion
    ):
        workdir = self._workdir()
        Path(workdir, "seed.txt").write_text("from the workdir")
        language_model = LanguageModel(model="ollama/mistral")
        reward = DeepAgentAsJudge(language_model=language_model, workdir=workdir)
        self.assertTrue(reward.program.reset_sandbox)
        sandbox = reward.program.agent.sandbox

        mock_completion.side_effect = lambda *args, **kwargs: _lm_response(
            content='{"critique": "ok", "reward": 20}'
        )
        # Scratch left by a previous evaluation...
        await sandbox.write_file("/scratch.txt", "left over")
        await reward(y_true=Solution(code="a"), y_pred=Solution(code="b"))
        # ...is gone, while the workdir seed is back.
        self.assertFalse((await sandbox.read_file("/scratch.txt")).get("ok", False))
        self.assertEqual(
            (await sandbox.read_file("/seed.txt"))["content"], "from the workdir"
        )

        reward = DeepAgentAsJudge(
            language_model=language_model, workdir=workdir, reset_sandbox=False
        )
        sandbox = reward.program.agent.sandbox
        await sandbox.write_file("/scratch.txt", "kept")
        await reward(y_true=Solution(code="a"), y_pred=Solution(code="b"))
        self.assertEqual((await sandbox.read_file("/scratch.txt"))["content"], "kept")

    @patch("litellm.acompletion")
    async def test_deep_agent_as_judge_supplied_sandbox(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        sandbox = MirageSandbox()
        await sandbox.write_file("/fixture.txt", "owned by the caller")
        reward = DeepAgentAsJudge(language_model=language_model, sandbox=sandbox)
        # A caller-owned sandbox is graded in as is, never wiped.
        self.assertIs(reward.program.agent.sandbox, sandbox)
        self.assertFalse(reward.program.reset_sandbox)

        mock_completion.side_effect = [
            _lm_response(content="No verification needed."),
            _lm_response(content='{"critique": "ok", "reward": 20}'),
        ]
        await reward(y_true=Solution(code="a"), y_pred=Solution(code="b"))
        self.assertEqual(
            (await sandbox.read_file("/fixture.txt"))["content"], "owned by the caller"
        )

    def test_deep_agent_as_judge_extra_tools(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = DeepAgentAsJudge(language_model=language_model, tools=[Tool(word_count)])
        # User tools sit alongside the built-in file and shell tools.
        self.assertIn("word_count", reward.program.agent.tools)
        self.assertIn("run_bash", reward.program.agent.tools)

        program = DeepAgentAsJudgeProgram.from_config(reward.program.get_config())
        self.assertIn("word_count", program.agent.tools)

    def test_deep_agent_as_judge_default_instructions(self):
        workdir = self._workdir()
        language_model = LanguageModel(model="ollama/mistral")
        reward = DeepAgentAsJudge(
            language_model=language_model, workdir=workdir, score_type=Rating10
        )
        instructions = reward.program.agent.instructions
        self.assertIn(f"Workdir: {Path(workdir).resolve()}", instructions)
        self.assertIn("run_bash", instructions)
        self.assertIn("impartial judge", instructions)
        self.assertIn("`inputs_file`", instructions)
        self.assertIn("an integer between 1 and 10", instructions)
        self.assertEqual(reward.program.agent.final_instructions, instructions)

    def test_deep_agent_as_judge_custom_instructions(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = DeepAgentAsJudge(
            language_model=language_model,
            instructions="Grade the prediction.",
            final_instructions="Grade it now.",
        )
        self.assertEqual(reward.program.agent.instructions, "Grade the prediction.")
        self.assertEqual(reward.program.agent.final_instructions, "Grade it now.")

    def test_deep_agent_as_judge_config_round_trip(self):
        workdir = self._workdir()
        primary = LanguageModel(model="ollama/mistral")
        cheap = LanguageModel(model="ollama/llama3")
        reward = DeepAgentAsJudge(
            language_model=primary,
            sub_language_model=cheap,
            score_type=Rating10,
            max_iterations=2,
            timeout=5,
            workdir=workdir,
            max_subagent_depth=1,
        )
        config = reward.program.get_config()
        self.assertEqual(config["score_type"], "Rating10")
        self.assertEqual(config["workdir"], workdir)
        program = DeepAgentAsJudgeProgram.from_config(config)
        self.assertIs(program.score_type, Rating10)
        self.assertEqual(program.max_iterations, 2)
        self.assertEqual(program.agent.timeout, 5.0)
        self.assertEqual(program.agent.max_subagent_depth, 1)
        self.assertEqual(program.agent.sub_language_model.model, "ollama_chat/llama3")
        self.assertTrue(program.reset_sandbox)

        reward = DeepAgentAsJudge.from_config(reward.get_config())
        self.assertIsInstance(reward, DeepAgentAsJudge)
        self.assertIs(reward.program.score_type, Rating10)
        self.assertEqual(reward.program.agent.max_subagent_depth, 1)
        self.assertIn("spawn_subagents", reward.program.agent.tools)

    def test_deep_agent_as_judge_registry_round_trip(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = DeepAgentAsJudge(
            language_model=language_model,
            score_type=Rating10,
            in_mask=["code"],
            name="my_judge",
        )

        restored = rewards.deserialize(rewards.serialize(reward))
        self.assertIsInstance(restored, DeepAgentAsJudge)
        self.assertEqual(restored.name, "my_judge")
        self.assertEqual(restored.in_mask, ["code"])
        self.assertIs(restored.program.score_type, Rating10)
        self.assertIs(rewards.ALL_OBJECTS_DICT["deepagentasjudge"], DeepAgentAsJudge)
        self.assertIs(rewards.ALL_OBJECTS_DICT["deep_agent_as_judge"], DeepAgentAsJudge)
