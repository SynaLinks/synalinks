# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.rewards.reward import Reward
from synalinks.src.rewards.reward import apply_masks
from synalinks.src.rewards.reward_wrappers import RewardFunctionWrapper
from synalinks.src.saving import serialization_lib


@synalinks_export(
    [
        "synalinks.ComposableReward",
        "synalinks.rewards.ComposableReward",
    ]
)
class ComposableReward(Reward):
    """Compose multiple rewards into one weighted reward.

    `ComposableReward` evaluates each child reward and combines them according to
    `reduction`. The child rewards keep their own masks, while this reward owns
    the reduction across child rewards.

    Args:
        rewards (list): List of `Reward` instances or async reward functions.
        weights (list): Optional positive weights, one per reward. Defaults to
            equal weights.
        name (str): Optional name for the reward instance.
        reduction (str): Optional. How child rewards are combined. One of
            `"mean"`, `"sum"`, `"min"`, `"max"`, `"none"` or `None`. Defaults
            to `"mean"`.
        in_mask (list): Optional outer fields to keep before computing rewards.
        out_mask (list): Optional outer fields to remove before computing rewards.
        in_mask_pattern (str): Optional regex of outer fields to keep.
        out_mask_pattern (str): Optional regex of outer fields to remove.
    """

    def __init__(
        self,
        rewards,
        weights=None,
        name="composable_reward",
        reduction="mean",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            name=name,
            reduction=reduction,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )
        if not rewards:
            raise ValueError("`rewards` is required and must not be empty.")
        self.rewards = [self.standardize_reward(reward) for reward in rewards]
        if weights is None:
            weights = [1.0] * len(self.rewards)
        if len(weights) != len(self.rewards):
            raise ValueError(
                "`weights` must have the same length as `rewards`. "
                f"Received {len(weights)} weights for {len(self.rewards)} rewards."
            )
        self.weights = [float(weight) for weight in weights]
        if any(weight <= 0 for weight in self.weights):
            raise ValueError("All `weights` values must be positive.")

    def standardize_reward(self, reward):
        if isinstance(reward, Reward):
            return reward
        return RewardFunctionWrapper(reward)

    async def __call__(self, y_true, y_pred):
        y_true, y_pred = apply_masks(
            y_true,
            y_pred,
            in_mask=self.in_mask,
            in_mask_pattern=self.in_mask_pattern,
            out_mask=self.out_mask,
            out_mask_pattern=self.out_mask_pattern,
        )
        return await self.call(y_true, y_pred)

    async def call(self, y_true, y_pred):
        weighted_values = []
        for reward, weight in zip(self.rewards, self.weights):
            values = await self.call_reward(reward, y_true, y_pred)
            weighted_values.append(
                tree.map_structure(lambda value: float(value) * weight, values)
            )
        if self.reduction in (None, "none"):
            return weighted_values
        if self.reduction == "min":
            return tree.map_structure(lambda *values: min(values), *weighted_values)
        if self.reduction == "max":
            return tree.map_structure(lambda *values: max(values), *weighted_values)

        total = weighted_values[0]
        for values in weighted_values[1:]:
            total = tree.map_structure(lambda a, b: a + b, total, values)
        if self.reduction == "mean":
            weight_sum = sum(self.weights)
            return tree.map_structure(lambda value: value / weight_sum, total)
        return total

    async def call_reward(self, reward, y_true, y_pred):
        y_true, y_pred = apply_masks(
            y_true,
            y_pred,
            in_mask=reward.in_mask,
            in_mask_pattern=reward.in_mask_pattern,
            out_mask=reward.out_mask,
            out_mask_pattern=reward.out_mask_pattern,
        )
        return await reward.call(y_true, y_pred)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rewards": serialization_lib.serialize_synalinks_object(
                    self.rewards
                ),
                "weights": self.weights,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["rewards"] = serialization_lib.deserialize_synalinks_object(
            config["rewards"]
        )
        return cls(**config)
