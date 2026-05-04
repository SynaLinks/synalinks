# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.rewards.reward_wrappers import RewardFunctionWrapper


@synalinks_export("synalinks.rewards.exact_match")
async def exact_match(y_true, y_pred):
    """
    Computes the exact match between `y_true` and `y_pred`.

    If their values are equal, it returns a reward of 1.0; otherwise, it returns 0.0.

    Args:
        y_true (JsonDataModel): The ground truth JSON data_model.
        y_pred (JsonDataModel): The predicted JSON data_model.

    Returns:
        (float): The reward value, which is 1.0 if the values match exactly,
            and 0.0 otherwise.
    """
    reward = 0.0
    if y_pred is not None:
        if y_pred.get_json() == y_true.get_json():
            reward = 1.0
    return reward


@synalinks_export(
    [
        "synalinks.ExactMatch",
        "synalinks.rewards.ExactMatch",
    ]
)
class ExactMatch(RewardFunctionWrapper):
    """Computes the exact match between `y_true` and `y_pred`.

    Example:

    ```python
    program.compile(
        reward=synalinks.rewards.ExactMatch(),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```

    Args:
        name (str): Optional. string name of the reward instance.
        in_mask (list): Optional. list of keys to keep to compute the reward.
        out_mask (list): Optional. list of keys to remove to compute the reward.
        in_mask_pattern (str): Optional. Regex pattern; fields whose names match
            are kept (combined with ``in_mask`` via OR).
        out_mask_pattern (str): Optional. Regex pattern; fields whose names match
            are dropped (combined with ``out_mask`` via OR).
    """

    def __init__(
        self,
        name="exact_match",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        super().__init__(
            fn=exact_match,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )

    def get_config(self):
        return {
            "name": self.name,
            "in_mask": self.in_mask,
            "out_mask": self.out_mask,
            "in_mask_pattern": self.in_mask_pattern,
            "out_mask_pattern": self.out_mask_pattern,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
