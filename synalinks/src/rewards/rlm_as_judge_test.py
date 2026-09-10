# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import rewards
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import Rating10
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.rewards.rlm_as_judge import RLMAsJudge
from synalinks.src.rewards.rlm_as_judge import RLMAsJudgeProgram
from synalinks.src.sandboxes.mirage_sandbox import MirageSandbox
from synalinks.src.saving.object_registration import register_synalinks_serializable


def _exec_tool_call(code, call_id="call_1"):
    """A litellm response where the LM calls `run_python_code` with `code`."""
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "run_python_code",
                                "arguments": json.dumps({"code": code}),
                            },
                        }
                    ],
                }
            }
        ]
    }


@register_synalinks_serializable()
async def square(x: int) -> int:
    """Square an integer.

    Args:
        x (int): the integer to square.
    """
    return x * x


class Query(DataModel):
    query: str = Field(description="The user query")


class Answer(DataModel):
    answer: str = Field(description="The correct answer")


class RLMAsJudgeTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_rlm_as_judge(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RLMAsJudge(language_model=language_model, max_iterations=3)

        y_true = Answer(answer="153133")
        y_pred = Query(query="How much is 152648 + 485?") + Answer(answer="153133")

        mock_completion.side_effect = [
            # Turn 1: the judge recomputes in the sandbox and compares to gold.
            _exec_tool_call(
                "print(int(inputs['answer']) == 152648 + 485, inputs['gold_answer'])"
            ),
            # Turn 2: it submits the verdict, which ends the run.
            _exec_tool_call(
                "submit(result={'critique': 'Recomputed, matches gold.', 'reward': 20})",
                "call_2",
            ),
        ]

        score = await reward(y_true=y_true, y_pred=y_pred)
        self.assertEqual(score, 1.0)
        # `submit` short-circuits the final grading turn: two LM calls only.
        self.assertEqual(mock_completion.call_count, 2)
        # The sandbox saw the gold field and the prediction.
        second_call_messages = mock_completion.call_args.kwargs["messages"]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        self.assertTrue(
            any("True 153133" in json.dumps(m.get("content")) for m in tool_messages),
            f"sandbox observation not found in tool messages: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_rlm_as_judge_empty_prediction(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RLMAsJudge(language_model=language_model)

        score = await reward(y_true=Answer(answer="Paris"), y_pred=None)
        self.assertEqual(score, 0.0)
        mock_completion.assert_not_called()

    @patch("litellm.acompletion")
    async def test_rlm_as_judge_without_gold_reference(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RLMAsJudge(language_model=language_model, max_iterations=3)

        mock_completion.side_effect = [
            # Turn 1: the judge lists what it was given.
            _exec_tool_call("print(sorted(inputs))"),
            # Turn 2: it submits the verdict.
            _exec_tool_call(
                "submit(result={'critique': 'Plausible.', 'reward': 20})", "call_2"
            ),
        ]

        score = await reward(y_true=None, y_pred=Answer(answer="Paris"))
        self.assertEqual(score, 1.0)
        # Without a reference the sandbox only holds the prediction: no `gold_` key.
        second_call_messages = mock_completion.call_args.kwargs["messages"]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        self.assertTrue(
            any("['answer']" in json.dumps(m.get("content")) for m in tool_messages),
            f"sandbox observation not found in tool messages: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_rlm_as_judge_sandbox_tool(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RLMAsJudge(language_model=language_model, tools=[Tool(square)])
        # User tools live inside the sandbox, not as native tool calls.
        self.assertIn("square", reward.program.agent.tools)

        mock_completion.side_effect = [
            _exec_tool_call(
                "submit(result={'critique': 'square(4) is ' + str(square(x=4)), "
                "'reward': 20})"
            ),
        ]

        score = await reward(y_true=Answer(answer="16"), y_pred=Answer(answer="16"))
        self.assertEqual(score, 1.0)

    @patch("litellm.acompletion")
    async def test_rlm_as_judge_with_rating10_score_type(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RLMAsJudge(language_model=language_model, score_type=Rating10)

        # The judge asks for a number bounded by the 1..10 scale...
        reward_schema = reward.program.agent.schema["properties"]["reward"]
        self.assertEqual(reward_schema["type"], "number")
        self.assertEqual(reward_schema["minimum"], 1)
        self.assertEqual(reward_schema["maximum"], 10)
        self.assertNotIn("enum", reward_schema)

        mock_completion.side_effect = [
            _exec_tool_call("submit(result={'critique': 'Close enough.', 'reward': 7})"),
        ]
        score = await reward(y_true=Answer(answer="Paris"), y_pred=Answer(answer="paris"))
        # ...and the reward is normalized: (7 - 1) / (10 - 1)
        self.assertAlmostEqual(score, 6 / 9)

    @patch("litellm.acompletion")
    async def test_rlm_as_judge_accepts_computed_float_reward(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RLMAsJudge(language_model=language_model, max_iterations=2)

        mock_completion.side_effect = [
            # A pass rate scaled onto the 1..20 range is a float: `submit`
            # must accept it rather than reject it against an integer enum.
            _exec_tool_call(
                "passed, total = 7, 8\n"
                "submit(result={'critique': '7 of 8 checks pass.', "
                "'reward': 1 + 19 * passed / total})"
            ),
        ]
        score = await reward(y_true=Answer(answer="a"), y_pred=Answer(answer="b"))
        self.assertAlmostEqual(score, 7 / 8)
        self.assertEqual(mock_completion.call_count, 1)

    @patch("litellm.acompletion")
    async def test_rlm_as_judge_rejects_out_of_range_reward(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RLMAsJudge(language_model=language_model, max_iterations=2)

        mock_completion.side_effect = [
            # Out of bounds: the validation error comes back as an observation...
            _exec_tool_call("submit(result={'critique': 'Too high.', 'reward': 25})"),
            # ...and the judge retries within the scale.
            _exec_tool_call(
                "submit(result={'critique': 'Corrected.', 'reward': 20})", "call_2"
            ),
        ]
        score = await reward(y_true=Answer(answer="a"), y_pred=Answer(answer="a"))
        self.assertEqual(score, 1.0)
        self.assertEqual(mock_completion.call_count, 2)

    @patch("litellm.acompletion")
    async def test_rlm_as_judge_falls_back_to_final_grading(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RLMAsJudge(language_model=language_model, max_iterations=1)

        mock_completion.side_effect = [
            # The only code turn never calls `submit`...
            _exec_tool_call("print(inputs['answer'])"),
            # ...so the final grading step formats the trajectory.
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"critique": "Out of turns.", "reward": 1}'
                        }
                    }
                ]
            },
        ]
        score = await reward(y_true=Answer(answer="a"), y_pred=Answer(answer="b"))
        self.assertEqual(score, 0.0)
        self.assertEqual(mock_completion.call_count, 2)

    async def test_rlm_as_judge_symbolic_output_schema(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RLMAsJudge(language_model=language_model, score_type=Rating10)

        outputs = await reward.program(
            [
                Answer(answer="Paris").to_symbolic_data_model(),
                Answer(answer="Paris").to_symbolic_data_model(),
            ]
        )
        self.assertTrue(is_symbolic_data_model(outputs))
        # Whatever the judge's scale, the program advertises a 0..1 reward.
        reward_schema = outputs.get_schema()["properties"]["reward"]
        self.assertEqual(reward_schema["type"], "number")
        self.assertEqual(reward_schema["minimum"], 0.0)
        self.assertEqual(reward_schema["maximum"], 1.0)

    def test_rlm_as_judge_default_instructions_follow_mode_and_score_type(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RLMAsJudge(language_model=language_model, score_type=Rating10)
        instructions = reward.program.agent.instructions
        self.assertIn("llm_query", instructions)
        self.assertIn("impartial judge", instructions)
        self.assertIn("`gold_`", instructions)
        self.assertIn("a number (integer or float) between 1 and 10", instructions)
        self.assertIn(MirageSandbox.description, instructions)

        reward = RLMAsJudge(language_model=language_model, recursive=False)
        instructions = reward.program.agent.instructions
        self.assertNotIn("llm_query", instructions)
        self.assertIn("impartial judge", instructions)
        self.assertIn("a number (integer or float) between 1 and 20", instructions)

    def test_rlm_as_judge_custom_instructions(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RLMAsJudge(
            language_model=language_model,
            instructions="Grade the prediction.",
            final_instructions="Grade it now.",
        )
        instructions = reward.program.agent.instructions
        # Custom instructions replace the judging text; the agent still
        # appends how its sandbox works.
        self.assertTrue(instructions.startswith("Grade the prediction."))
        self.assertNotIn("impartial judge", instructions)
        self.assertIn(MirageSandbox.description, instructions)
        self.assertEqual(reward.program.agent.final_instructions, "Grade it now.")

    def test_rlm_as_judge_supplied_sandbox(self):
        language_model = LanguageModel(model="ollama/mistral")
        sandbox = MirageSandbox()
        reward = RLMAsJudge(language_model=language_model, sandbox=sandbox)
        self.assertIs(reward.program.agent.sandbox, sandbox)
        self.assertIs(reward.program.sandbox_type, MirageSandbox)
        self.assertIsNotNone(reward.program.get_config()["sandbox"])

    def test_rlm_as_judge_config_round_trip(self):
        primary = LanguageModel(model="ollama/mistral")
        cheap = LanguageModel(model="ollama/llama3")
        reward = RLMAsJudge(
            language_model=primary,
            sub_language_model=cheap,
            tools=[Tool(square)],
            score_type=Rating10,
            max_iterations=2,
            max_llm_calls=7,
            recursive=True,
        )
        config = reward.program.get_config()
        self.assertEqual(config["score_type"], "Rating10")
        self.assertEqual(config["max_llm_calls"], 7)
        program = RLMAsJudgeProgram.from_config(config)
        self.assertIs(program.score_type, Rating10)
        self.assertEqual(program.max_iterations, 2)
        self.assertEqual(program.agent.max_llm_calls, 7)
        self.assertEqual(program.agent.sub_language_model.model, "ollama_chat/llama3")
        self.assertIs(program.sandbox_type, MirageSandbox)
        self.assertIn("square", program.agent.tools)

        reward = RLMAsJudge.from_config(reward.get_config())
        self.assertIsInstance(reward, RLMAsJudge)
        self.assertIs(reward.program.score_type, Rating10)
        self.assertEqual(reward.program.agent.max_llm_calls, 7)
        self.assertIs(reward.program.agent.sandbox_type, MirageSandbox)
        self.assertIn("square", reward.program.agent.tools)

    def test_rlm_as_judge_registry_round_trip(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = RLMAsJudge(
            language_model=language_model,
            score_type=Rating10,
            in_mask=["answer"],
            name="my_judge",
        )

        restored = rewards.deserialize(rewards.serialize(reward))
        self.assertIsInstance(restored, RLMAsJudge)
        self.assertEqual(restored.name, "my_judge")
        self.assertEqual(restored.in_mask, ["answer"])
        self.assertIs(restored.program.score_type, Rating10)
        self.assertIs(rewards.ALL_OBJECTS_DICT["rlmasjudge"], RLMAsJudge)
        self.assertIs(rewards.ALL_OBJECTS_DICT["rlm_as_judge"], RLMAsJudge)
