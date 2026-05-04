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

    async def test_base_function_pattern_masked(self):
        class Answer(DataModel):
            answer: str

        class AnswerWithThinking(DataModel):
            thinking: str
            answer: str

        y_true = Answer(answer="Paris")
        y_pred = AnswerWithThinking(
            thinking="The french capital is Paris",
            answer="Paris",
        )
        exact_match = ExactMatch(out_mask_pattern=r"^thinking$")
        reward = await exact_match(y_true, y_pred)
        self.assertEqual(reward, 1.0)

        exact_match = ExactMatch(in_mask_pattern=r"^answer$")
        reward = await exact_match(y_true, y_pred)
        self.assertEqual(reward, 1.0)

    def test_pattern_in_get_config(self):
        cfg = ExactMatch(
            in_mask_pattern=r"^answer$",
            out_mask_pattern=r"^thinking$",
        ).get_config()
        self.assertEqual(cfg["in_mask_pattern"], r"^answer$")
        self.assertEqual(cfg["out_mask_pattern"], r"^thinking$")
        clone = ExactMatch.from_config(cfg)
        self.assertEqual(clone.in_mask_pattern, r"^answer$")
        self.assertEqual(clone.out_mask_pattern, r"^thinking$")
