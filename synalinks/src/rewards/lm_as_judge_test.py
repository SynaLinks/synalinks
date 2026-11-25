# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

import numpy as np

from synalinks.src import ops
from synalinks.src import rewards
from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import FineScore
from synalinks.src.backend import Rating10
from synalinks.src.modules import Generator
from synalinks.src.modules import Input
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Program
from synalinks.src.rewards.lm_as_judge import LMAsJudge
from synalinks.src.rewards.lm_as_judge import LMAsJudgeProgram


class LMAsJudgeTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_lm_as_judge(self, mock_completion):
        class Query(DataModel):
            query: str = Field(description="The user query")

        class AnswerWithThinking(DataModel):
            thinking: str = Field(description="The step by step thinking process")
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="ollama/mistral")

        reward = LMAsJudge(language_model=language_model)

        inputs = Query(query="What is the French capital?")

        y_true = AnswerWithThinking(
            thinking="The French capital is Paris",
            answer="Paris",
        )

        y_pred = AnswerWithThinking(
            thinking="The French capital is well known",
            answer="Paris",
        )

        y_pred = inputs + y_pred

        expected_string = (
            """{"critique": "The answer is correct so we can attribute a high reward", """
            """"reward": 20}"""
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        score = await reward(y_true=y_true, y_pred=y_pred)
        self.assertEqual(score, 1.0)

    @patch("litellm.acompletion")
    async def test_lm_as_judge_empty_prediction(self, mock_completion):
        # Regression: an empty prediction (e.g. the LM returned no content)
        # must score 0.0 without calling the judge LM and without crashing
        # the `ProgramAsJudge` wrapper with
        # `AttributeError: 'float' object has no attribute 'get'`.
        class AnswerWithThinking(DataModel):
            thinking: str = Field(description="The step by step thinking process")
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="ollama/mistral")
        reward = LMAsJudge(language_model=language_model)

        y_true = AnswerWithThinking(
            thinking="The French capital is Paris",
            answer="Paris",
        )

        score = await reward(y_true=y_true, y_pred=None)
        self.assertEqual(score, 0.0)
        mock_completion.assert_not_called()

    @patch("litellm.acompletion")
    async def test_lm_as_judge_with_rating10_score_type(self, mock_completion):
        class Answer(DataModel):
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="ollama/mistral")
        reward = LMAsJudge(language_model=language_model, score_type=Rating10)

        # The judge asks the LM for a 1..10 integer...
        reward_schema = reward.program.critique.generator.schema["properties"]["reward"]
        self.assertEqual(reward_schema["type"], "integer")
        self.assertEqual(reward_schema["enum"], list(range(1, 11)))

        mock_completion.return_value = {
            "choices": [
                {"message": {"content": '{"critique": "Close enough.", "reward": 7}'}}
            ]
        }
        score = await reward(y_true=Answer(answer="Paris"), y_pred=Answer(answer="paris"))
        # ...and the reward is normalized: (7 - 1) / (10 - 1)
        self.assertAlmostEqual(score, 6 / 9)

    @patch("litellm.acompletion")
    async def test_lm_as_judge_score_type_bounds(self, mock_completion):
        class Answer(DataModel):
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="ollama/mistral")
        reward = LMAsJudge(language_model=language_model, score_type="Rating")

        # Exactly one LM call per evaluation, including the very first one:
        # building the judge program must trace symbolically, not call the LM.
        mock_completion.side_effect = [
            {"choices": [{"message": {"content": '{"critique": "bad", "reward": 1}'}}]},
            {"choices": [{"message": {"content": '{"critique": "top", "reward": 5}'}}]},
        ]
        low = await reward(y_true=Answer(answer="a"), y_pred=Answer(answer="b"))
        high = await reward(y_true=Answer(answer="a"), y_pred=Answer(answer="a"))
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 1.0)
        self.assertEqual(mock_completion.call_count, 2)

    def test_lm_as_judge_default_instructions_follow_score_type(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = LMAsJudge(language_model=language_model, score_type=Rating10)
        instructions = reward.program.critique.instructions
        self.assertIn("an integer between 1 and 10", instructions)
        self.assertIn("10 very good", instructions)
        reward = LMAsJudge(language_model=language_model)
        self.assertIs(reward.program.score_type, FineScore)
        self.assertIn("a float between 0.0 and 1.0", reward.program.critique.instructions)

    def test_lm_as_judge_score_type_config_round_trip(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = LMAsJudge(language_model=language_model, score_type=Rating10)
        config = reward.program.get_config()
        self.assertEqual(config["score_type"], "Rating10")
        program = LMAsJudgeProgram.from_config(config)
        self.assertIs(program.score_type, Rating10)
        self.assertIs(program.critique.score_type, Rating10)

    @patch("litellm.acompletion")
    async def test_lm_as_judge_without_gold_reference(self, mock_completion):
        class Answer(DataModel):
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="ollama/mistral")
        reward = LMAsJudge(language_model=language_model)

        mock_completion.return_value = {
            "choices": [
                {"message": {"content": '{"critique": "Plausible.", "reward": 20}'}}
            ]
        }
        score = await reward(y_true=None, y_pred=Answer(answer="Paris"))
        self.assertEqual(score, 1.0)
        # Without a reference the judge only sees the prediction: no `gold_` key.
        prompt = json.dumps(mock_completion.call_args.kwargs["messages"])
        self.assertNotIn("gold_", prompt)
        self.assertIn("Paris", prompt)

    async def test_lm_as_judge_malformed_inputs(self):
        class Answer(DataModel):
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="ollama/mistral")
        reward = LMAsJudge(language_model=language_model)

        with self.assertRaisesRegex(ValueError, "list or tuple"):
            await reward.program(Answer(answer="Paris"))
        with self.assertRaisesRegex(ValueError, r"\[y_true, y_pred\]"):
            await reward.program([Answer(answer="Paris")])

    def test_lm_as_judge_registry_round_trip(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = LMAsJudge(
            language_model=language_model,
            score_type=Rating10,
            in_mask=["answer"],
            name="my_judge",
        )

        restored = rewards.deserialize(rewards.serialize(reward))
        self.assertIsInstance(restored, LMAsJudge)
        self.assertEqual(restored.name, "my_judge")
        self.assertEqual(restored.in_mask, ["answer"])
        self.assertIs(restored.program.score_type, Rating10)
        self.assertIsInstance(rewards.get("lmasjudge"), LMAsJudge)
        self.assertIsInstance(rewards.get("lm_as_judge"), LMAsJudge)

    @patch("litellm.acompletion")
    async def test_lm_as_judge_labels_input_output_and_gold(self, mock_completion):
        # A chat program with `return_inputs=True` echoes its input in its
        # output: the judge sees the input once (`input_`), then the output
        # (`output_`), then the gold reference (`gold_`), in one user turn.
        x = ChatMessages(
            messages=[ChatMessage(role="user", content="What is the French capital?")]
        ).to_json_data_model()
        answer = ChatMessage(role="assistant", content="The capital of France is Paris.")
        y_pred = await ops.concat(x, answer.to_json_data_model())
        y_true = ChatMessage(role="assistant", content="Paris.")
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"critique": "Ok", "reward": 1.0}'}}]
        }

        reward = LMAsJudge(language_model=LanguageModel(model="ollama/mistral"))
        score = await reward(y_true=y_true, y_pred=y_pred, x=x)
        self.assertEqual(score, 1.0)
        messages = mock_completion.call_args.kwargs["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(
            messages[1]["content"],
            "{'input_messages': [{'role': 'user', 'content': "
            "'What is the French capital?'}], 'output_role': 'assistant', "
            "'output_content': 'The capital of France is Paris.', "
            "'gold_role': 'assistant', 'gold_content': 'Paris.'}",
        )

        # Without the program's inputs nor a reference, only the output.
        await reward(y_true=None, y_pred=y_pred)
        content = mock_completion.call_args.kwargs["messages"][1]["content"]
        self.assertTrue(content.startswith("{'output_messages': "))
        self.assertNotIn("input_", content)
        self.assertNotIn("gold_", content)

    @patch("litellm.acompletion")
    async def test_lm_as_judge_receives_program_inputs_from_evaluate(
        self, mock_completion
    ):
        class Query(DataModel):
            query: str = Field(description="The user query")

        class Answer(DataModel):
            answer: str = Field(description="The correct answer")

        language_model = LanguageModel(model="ollama/mistral")
        x0 = Input(data_model=Query)
        x1 = await Generator(data_model=Answer, language_model=language_model)(x0)
        program = Program(inputs=x0, outputs=x1)
        program.compile(reward=LMAsJudge(language_model=language_model))
        mock_completion.side_effect = [
            {"choices": [{"message": {"content": '{"answer": "Paris"}'}}]},
            {"choices": [{"message": {"content": '{"critique": "Ok", "reward": 1.0}'}}]},
        ]

        await program.evaluate(
            x=np.array([Query(query="What is the French capital?")], dtype="object"),
            y=np.array([Answer(answer="Paris")], dtype="object"),
            batch_size=1,
        )
        judge_prompt = mock_completion.call_args.kwargs["messages"][1]["content"]
        self.assertEqual(
            judge_prompt,
            "{'input_query': 'What is the French capital?', "
            "'output_answer': 'Paris', 'gold_answer': 'Paris'}",
        )
