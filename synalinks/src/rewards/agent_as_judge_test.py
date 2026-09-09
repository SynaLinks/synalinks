# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

import numpy as np

from synalinks.src import rewards
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import Rating10
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.optimizers.random_few_shot import RandomFewShot
from synalinks.src.programs import Program
from synalinks.src.rewards.agent_as_judge import AgentAsJudge
from synalinks.src.rewards.agent_as_judge import AgentAsJudgeProgram
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
async def calculate(expression: str):
    """Calculate the result of a mathematical expression.

    Args:
        expression (str): The mathematical expression to calculate, such as
            '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
            parentheses, and spaces.
    """
    if not all(char in "0123456789+-*/(). " for char in expression):
        return {"result": None, "log": "Error: invalid characters in expression"}
    return {
        "result": round(float(eval(expression, {"__builtins__": None}, {})), 2),
        "log": "Successfully executed",
    }


class Query(DataModel):
    query: str = Field(description="The user query")


class Answer(DataModel):
    answer: str = Field(description="The correct answer")


class AnswerWithThinking(DataModel):
    thinking: str = Field(description="The step by step thinking process")
    answer: str = Field(description="The correct answer")


class AgentAsJudgeTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_agent_as_judge(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = AgentAsJudge(
            language_model=language_model,
            tools=[Tool(calculate)],
            max_iterations=3,
        )

        y_true = Answer(answer="153133")
        y_pred = Query(query="How much is 152648 + 485?") + Answer(answer="153133")

        mock_completion.side_effect = [
            # Turn 1: the judge verifies the prediction with its tool.
            _lm_response(
                content="Checking the arithmetic.",
                tool_calls=[
                    {"name": "calculate", "arguments": {"expression": "152648 + 485"}}
                ],
            ),
            # Turn 2: no tool calls, the loop stops.
            _lm_response(content="The tool confirms 153133."),
            # Final grading turn.
            _lm_response(
                content='{"critique": "Verified with the calculator.", "reward": 20}'
            ),
        ]

        score = await reward(y_true=y_true, y_pred=y_pred)
        self.assertEqual(score, 1.0)
        self.assertEqual(mock_completion.call_count, 3)
        # The tool result was fed back to the judge before grading.
        grading_messages = mock_completion.call_args.kwargs["messages"]
        tool_messages = [m for m in grading_messages if m.get("role") == "tool"]
        self.assertTrue(
            any("153133" in json.dumps(m.get("content")) for m in tool_messages),
            f"calculate result not found in tool messages: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_agent_as_judge_empty_prediction(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = AgentAsJudge(language_model=language_model, tools=[Tool(calculate)])

        score = await reward(y_true=Answer(answer="Paris"), y_pred=None)
        self.assertEqual(score, 0.0)
        mock_completion.assert_not_called()

    @patch("litellm.acompletion")
    async def test_agent_as_judge_without_gold_reference(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = AgentAsJudge(language_model=language_model, tools=[Tool(calculate)])

        mock_completion.side_effect = [
            _lm_response(content="Nothing to verify."),
            _lm_response(content='{"critique": "Plausible.", "reward": 20}'),
        ]

        score = await reward(y_true=None, y_pred=Answer(answer="Paris"))
        self.assertEqual(score, 1.0)
        # Without a reference the judge only sees the prediction: no `gold_` key.
        prompt = json.dumps(mock_completion.call_args_list[0].kwargs["messages"])
        self.assertNotIn("gold_", prompt)
        self.assertIn("Paris", prompt)

    @patch("litellm.acompletion")
    async def test_agent_as_judge_masked_fields(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = AgentAsJudge(
            language_model=language_model,
            tools=[Tool(calculate)],
            in_mask=["answer"],
        )

        y_true = AnswerWithThinking(thinking="The gold reasoning", answer="Paris")
        y_pred = AnswerWithThinking(thinking="The predicted reasoning", answer="Paris")

        mock_completion.side_effect = [
            _lm_response(content="Nothing to verify."),
            _lm_response(content='{"critique": "Matches gold.", "reward": 20}'),
        ]

        score = await reward(y_true=y_true, y_pred=y_pred)
        self.assertEqual(score, 1.0)
        # Only the kept field reaches the judge, on both sides of the pair.
        prompt = json.dumps(mock_completion.call_args_list[0].kwargs["messages"])
        self.assertIn("gold_answer", prompt)
        self.assertNotIn("reasoning", prompt)

    @patch("litellm.acompletion")
    async def test_agent_as_judge_no_verdict(self, mock_completion):
        # The provider returns no content: the agent yields no verdict, which
        # must score 0.0 without raising inside the training loop.
        language_model = LanguageModel(model="ollama/mistral")
        reward = AgentAsJudge(language_model=language_model, tools=[Tool(calculate)])

        mock_completion.return_value = _lm_response(content=None)

        score = await reward(y_true=Answer(answer="a"), y_pred=Answer(answer="b"))
        self.assertEqual(score, 0.0)

    @patch("litellm.acompletion")
    async def test_agent_as_judge_with_rating10_score_type(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        reward = AgentAsJudge(
            language_model=language_model,
            tools=[Tool(calculate)],
            score_type=Rating10,
        )

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
        score = await reward(y_true=Answer(answer="Paris"), y_pred=Answer(answer="paris"))
        # ...and the reward is normalized: (7 - 1) / (10 - 1)
        self.assertAlmostEqual(score, 6 / 9)

    async def test_agent_as_judge_symbolic_output_schema(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = AgentAsJudge(
            language_model=language_model,
            tools=[Tool(calculate)],
            score_type=Rating10,
        )

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
        self.assertNotIn("enum", reward_schema)

    async def test_agent_as_judge_malformed_inputs(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = AgentAsJudge(language_model=language_model, tools=[Tool(calculate)])

        with self.assertRaisesRegex(ValueError, "list or tuple"):
            await reward.program(Answer(answer="Paris"))
        with self.assertRaisesRegex(ValueError, "length of 2"):
            await reward.program([Answer(answer="Paris")])

    def test_agent_as_judge_requires_tools(self):
        language_model = LanguageModel(model="ollama/mistral")
        with self.assertRaisesRegex(ValueError, "tools"):
            AgentAsJudge(language_model=language_model)

    def test_agent_as_judge_default_instructions_follow_score_type(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = AgentAsJudge(
            language_model=language_model,
            tools=[Tool(calculate)],
            score_type=Rating10,
        )
        instructions = reward.program.agent.instructions
        self.assertIn("Use the available tools", instructions)
        self.assertIn("an integer between 1 and 10", instructions)
        self.assertEqual(reward.program.agent.final_instructions, instructions)

    def test_agent_as_judge_config_round_trip(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = AgentAsJudge(
            language_model=language_model,
            tools=[Tool(calculate)],
            score_type=Rating10,
            max_iterations=2,
        )
        config = reward.program.get_config()
        self.assertEqual(config["score_type"], "Rating10")
        program = AgentAsJudgeProgram.from_config(config)
        self.assertIs(program.score_type, Rating10)
        self.assertEqual(program.max_iterations, 2)
        self.assertEqual(list(program.agent.tools), ["calculate"])

        reward = AgentAsJudge.from_config(reward.get_config())
        self.assertIsInstance(reward, AgentAsJudge)
        self.assertIs(reward.program.score_type, Rating10)
        self.assertEqual(list(reward.program.agent.tools), ["calculate"])

    def test_agent_as_judge_registry_round_trip(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = AgentAsJudge(
            language_model=language_model,
            tools=[Tool(calculate)],
            score_type=Rating10,
            in_mask=["answer"],
            name="my_judge",
        )

        restored = rewards.deserialize(rewards.serialize(reward))
        self.assertIsInstance(restored, AgentAsJudge)
        self.assertEqual(restored.name, "my_judge")
        self.assertEqual(restored.in_mask, ["answer"])
        self.assertIs(restored.program.score_type, Rating10)
        self.assertEqual(list(restored.program.agent.tools), ["calculate"])
        self.assertIs(rewards.ALL_OBJECTS_DICT["agentasjudge"], AgentAsJudge)
        self.assertIs(rewards.ALL_OBJECTS_DICT["agent_as_judge"], AgentAsJudge)

    @patch("litellm.acompletion")
    async def test_agent_as_judge_fit(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")

        x0 = Input(data_model=Query)
        x1 = await Generator(data_model=Answer, language_model=language_model)(x0)
        program = Program(inputs=x0, outputs=x1, name="math_qa")
        program.compile(
            reward=AgentAsJudge(
                language_model=language_model,
                tools=[Tool(calculate)],
                score_type=Rating10,
            ),
            optimizer=RandomFewShot(),
        )

        # The trainer interleaves program and judge calls, so the mock answers
        # by turn kind: tool turns for the judge, a grade, or the program's answer.
        async def completion(**kwargs):
            messages = kwargs["messages"]
            if kwargs.get("tools"):
                if any(m.get("role") == "tool" for m in messages):
                    return _lm_response(content="Verified.")
                return _lm_response(
                    content="Checking.",
                    tool_calls=[
                        {"name": "calculate", "arguments": {"expression": "12 * 12"}}
                    ],
                )
            if "impartial judge" in json.dumps(messages):
                return _lm_response(content='{"critique": "Right.", "reward": 7}')
            return _lm_response(content='{"answer": "144"}')

        mock_completion.side_effect = completion

        x_train = np.array(
            [Query(query="What is 12 * 12?"), Query(query="Square 12.")],
            dtype="object",
        )
        y_train = np.array([Answer(answer="144"), Answer(answer="144")], dtype="object")

        history = await program.fit(x=x_train, y=y_train, epochs=1, verbose=0)
        # Every sample was graded 7 on the 1..10 scale: (7 - 1) / (10 - 1)
        self.assertAlmostEqual(history.history["reward"][-1], 6 / 9)
