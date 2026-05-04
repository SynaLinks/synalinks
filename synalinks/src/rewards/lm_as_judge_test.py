# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.rewards.lm_as_judge import LMAsJudge


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
