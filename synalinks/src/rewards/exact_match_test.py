# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.rewards.exact_match import ExactMatch


class ExactMatchTest(testing.TestCase):
    async def test_base_function(self):
        class Answer(DataModel):
            answer: str

        y_true = Answer(answer="Paris")
        y_pred = Answer(answer="Toulouse")
        exact_match = ExactMatch()
        reward = await exact_match(y_true, y_pred)
        self.assertEqual(reward, 0.0)

        y_true = Answer(answer="Paris")
        y_pred = Answer(answer="Paris")
        exact_match = ExactMatch()
        reward = await exact_match(y_true, y_pred)
        self.assertEqual(reward, 1.0)

    async def test_base_function_masked(self):
        class Answer(DataModel):
            answer: str

        class AnswerWithText(DataModel):
            text: str
            answer: str

        y_true = Answer(answer="Paris")
        y_pred = AnswerWithText(text="The french capital is Toulouse", answer="Toulouse")
        exact_match = ExactMatch(in_mask=["answer"])
        reward = await exact_match(y_true, y_pred)
        self.assertEqual(reward, 0.0)

        exact_match = ExactMatch(out_mask=["text"])
        reward = await exact_match(y_true, y_pred)
        self.assertEqual(reward, 0.0)

        y_true = Answer(answer="Paris")
        y_pred = AnswerWithText(text="The french capital is Paris", answer="Paris")
        exact_match = ExactMatch(in_mask=["answer"])
        reward = await exact_match(y_true, y_pred)
        self.assertEqual(reward, 1.0)

        exact_match = ExactMatch(out_mask=["text"])
        reward = await exact_match(y_true, y_pred)
        self.assertEqual(reward, 1.0)
