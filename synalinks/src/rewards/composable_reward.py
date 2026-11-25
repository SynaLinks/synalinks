# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.metrics.metric import apply_masks
from synalinks.src.metrics.metric import mask_data_models
from synalinks.src.rewards.reward import Reward
from synalinks.src.rewards.reward_wrappers import RewardFunctionWrapper
from synalinks.src.saving import serialization_lib
from synalinks.src.utils.python_utils import accepts_argument


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
        weights (list): Optional positive weights, one per reward, used by the
            `"mean"` and `"sum"` reductions. Defaults to equal weights.
        name (str): Optional name for the reward instance.
        reduction (str): Optional. How child rewards are combined. One of
            `"mean"` (weighted mean), `"sum"` (weighted sum), `"min"`, `"max"`,
            `"none"` or `None` (both combine like `"mean"`, so each sample
            keeps a single reward). Defaults to `"mean"`.
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

    async def __call__(self, y_true, y_pred, x=None):
        masks = {
            "in_mask": self.in_mask,
            "in_mask_pattern": self.in_mask_pattern,
            "out_mask": self.out_mask,
            "out_mask_pattern": self.out_mask_pattern,
        }
        y_true, y_pred = apply_masks(y_true, y_pred, **masks)
        return await self.call(y_true, y_pred, x=mask_data_models(x, **masks))

    async def call(self, y_true, y_pred, x=None):
        values = [await self.call_reward(r, y_true, y_pred, x) for r in self.rewards]
        if self.reduction == "min":
            return tree.map_structure(lambda *v: min(map(float, v)), *values)
        if self.reduction == "max":
            return tree.map_structure(lambda *v: max(map(float, v)), *values)
        total = tree.map_structure(
            lambda *v: sum(float(x) * w for x, w in zip(v, self.weights)), *values
        )
        if self.reduction == "sum":
            return total
        weight_sum = sum(self.weights)
        return tree.map_structure(lambda value: value / weight_sum, total)

    async def call_reward(self, reward, y_true, y_pred, x=None):
        masks = {
            "in_mask": reward.in_mask,
            "in_mask_pattern": reward.in_mask_pattern,
            "out_mask": reward.out_mask,
            "out_mask_pattern": reward.out_mask_pattern,
        }
        y_true, y_pred = apply_masks(y_true, y_pred, **masks)
        if accepts_argument(reward.call, "x"):
            return await reward.call(y_true, y_pred, x=mask_data_models(x, **masks))
        return await reward.call(y_true, y_pred)

    async def close(self):
        """Close each reward this one combines. Idempotent."""
        for reward in self.rewards:
            await reward.close()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rewards": serialization_lib.serialize_synalinks_object(self.rewards),
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
