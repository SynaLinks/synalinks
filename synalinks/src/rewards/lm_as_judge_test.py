# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import Rating10
from synalinks.src.modules.language_models import LanguageModel
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
            """"reward": 1.0}"""
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
        self.assertIn("a float between 0.0 and 1.0", reward.program.critique.instructions)

    def test_lm_as_judge_score_type_config_round_trip(self):
        language_model = LanguageModel(model="ollama/mistral")
        reward = LMAsJudge(language_model=language_model, score_type=Rating10)
        config = reward.program.get_config()
        self.assertEqual(config["score_type"], "Rating10")
        program = LMAsJudgeProgram.from_config(config)
        self.assertIs(program.score_type, Rating10)
        self.assertIs(program.critique.score_type, Rating10)
