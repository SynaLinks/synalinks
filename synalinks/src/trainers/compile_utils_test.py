# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.rewards.exact_match import ExactMatch
from synalinks.src.trainers.compile_utils import CompileReward


class CompileRewardMultiOutputTest(testing.TestCase):
    """Cover the multi-output (nested) path of ``CompileReward.call``."""

    async def test_multi_output_dict_call_awaits_each_reward(self):
        class Answer(DataModel):
            answer: str

        compile_reward = CompileReward(
            reward={"a": ExactMatch(), "b": ExactMatch()},
            output_names=["a", "b"],
        )
        y_true = {"a": Answer(answer="x"), "b": Answer(answer="y")}
        y_pred = {"a": Answer(answer="x"), "b": Answer(answer="y")}

        result = await compile_reward(y_true, y_pred)
        # Both outputs match → 1.0 + 1.0 = 2.0.
        self.assertEqual(float(result), 2.0)

    async def test_multi_output_partial_match(self):
        class Answer(DataModel):
            answer: str

        compile_reward = CompileReward(
            reward={"a": ExactMatch(), "b": ExactMatch()},
            output_names=["a", "b"],
        )
        y_true = {"a": Answer(answer="x"), "b": Answer(answer="y")}
        y_pred = {"a": Answer(answer="x"), "b": Answer(answer="WRONG")}

        result = await compile_reward(y_true, y_pred)
        self.assertEqual(float(result), 1.0)
