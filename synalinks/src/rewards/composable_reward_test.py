# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.rewards.composable_reward import ComposableReward
from synalinks.src.rewards.exact_match import ExactMatch
from synalinks.src.rewards.exact_match import exact_match
from synalinks.src.rewards.reward import Reward
from synalinks.src.rewards.reward_wrappers import RewardFunctionWrapper
from synalinks.src.saving import serialization_lib


class Answer(DataModel):
    answer: str = Field(description="The answer")
    detail: str = Field(description="Additional detail", default="")


@synalinks_export("synalinks.tests.ConstantReward")
class ConstantReward(Reward):
    def __init__(self, value=1.0, name="constant_reward", **kwargs):
        super().__init__(name=name, **kwargs)
        self.value = value

    async def call(self, y_true, y_pred):
        return self.value

    def get_config(self):
        config = super().get_config()
        config["value"] = self.value
        return config


class ComposableRewardTest(testing.TestCase):
    async def test_composable_reward_weighted_mean(self):
        reward = ComposableReward(
            rewards=[ConstantReward(1.0), ConstantReward(0.0)],
            weights=[3, 1],
        )
        score = await reward(
            y_true=Answer(answer="Paris"),
            y_pred=Answer(answer="Paris"),
        )
        self.assertEqual(score, 0.75)

    async def test_composable_reward_sum_reduction(self):
        reward = ComposableReward(
            rewards=[ConstantReward(1.0), ConstantReward(0.5)],
            weights=[2, 3],
            reduction="sum",
        )
        score = await reward(
            y_true=Answer(answer="Paris"),
            y_pred=Answer(answer="Paris"),
        )
        self.assertEqual(score, 3.5)

    async def test_composable_reward_max_reduction(self):
        reward = ComposableReward(
            rewards=[ConstantReward(1.0), ConstantReward(0.5)],
            weights=[2, 3],
            reduction="max",
        )
        score = await reward(
            y_true=Answer(answer="Paris"),
            y_pred=Answer(answer="Paris"),
        )
        self.assertEqual(score, 2.0)

    async def test_composable_reward_min_reduction(self):
        reward = ComposableReward(
            rewards=[ConstantReward(1.0), ConstantReward(0.5)],
            weights=[2, 3],
            reduction="min",
        )
        score = await reward(
            y_true=Answer(answer="Paris"),
            y_pred=Answer(answer="Paris"),
        )
        self.assertEqual(score, 1.5)

    async def test_composable_reward_none_reduction(self):
        reward = ComposableReward(
            rewards=[ConstantReward(1.0), ConstantReward(0.5)],
            weights=[2, 3],
            reduction="none",
        )
        scores = await reward(
            y_true=Answer(answer="Paris"),
            y_pred=Answer(answer="Paris"),
        )
        # One weighted value per child reward, in order.
        self.assertEqual(scores, [2.0, 1.5])

    async def test_composable_reward_wraps_bare_functions(self):
        reward = ComposableReward(rewards=[exact_match, ConstantReward(0.0)])
        self.assertIsInstance(reward.rewards[0], RewardFunctionWrapper)
        score = await reward(
            y_true=Answer(answer="Paris"),
            y_pred=Answer(answer="Paris"),
        )
        self.assertEqual(score, 0.5)

    async def test_composable_reward_keeps_child_masks(self):
        reward = ComposableReward(
            rewards=[
                ExactMatch(in_mask=["answer"]),
                ExactMatch(in_mask=["detail"]),
            ],
            weights=[1, 1],
        )
        score = await reward(
            y_true=Answer(answer="Paris", detail="capital"),
            y_pred=Answer(answer="Paris", detail="city"),
        )
        self.assertEqual(score, 0.5)

    def test_composable_reward_validation(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            ComposableReward(rewards=[])
        with self.assertRaisesRegex(ValueError, "same length"):
            ComposableReward(rewards=[ConstantReward()], weights=[1, 2])
        with self.assertRaisesRegex(ValueError, "positive"):
            ComposableReward(rewards=[ConstantReward()], weights=[0])
        with self.assertRaisesRegex(ValueError, "Invalid value"):
            ComposableReward(rewards=[ConstantReward()], reduction="invalid")

    def test_composable_reward_config_round_trip(self):
        reward = ComposableReward(
            rewards=[ConstantReward(0.25), ExactMatch()],
            weights=[1, 3],
            reduction="sum",
        )
        restored = serialization_lib.deserialize_synalinks_object(
            serialization_lib.serialize_synalinks_object(reward)
        )
        self.assertIsInstance(restored, ComposableReward)
        self.assertEqual(restored.weights, [1.0, 3.0])
        self.assertEqual(restored.reduction, "sum")
        self.assertEqual(len(restored.rewards), 2)
