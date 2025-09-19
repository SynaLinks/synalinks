# Modified from: keras/src/losses/loss.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import ops
from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common import numpy as np
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.naming import auto_name


@synalinks_export(["synalinks.Reward", "synalinks.rewards.Reward"])
class Reward(SynalinksSaveable):
    """Reward base class.

    This is the class to subclass in order to create new custom rewards.

    Args:
        name: Optional name for the reward instance.

    To be implemented by subclasses:

    * `call()`: Contains the logic for eval calculation using `y_true`,
        `y_pred`.
    """

    def __init__(
        self,
        name=None,
        reduction="mean",
        in_mask=None,
        out_mask=None,
    ):
        self.name = name or auto_name(self.__class__.__name__)
        self.reduction = standardize_reduction(reduction)
        self.in_mask = in_mask
        self.out_mask = out_mask

    async def __call__(self, y_true, y_pred):
        with ops.name_scope(self.name):
            y_pred = tree.map_structure(
                lambda x: ops.convert_to_json_data_model(x), y_pred
            )
            y_true = tree.map_structure(
                lambda x: ops.convert_to_json_data_model(x), y_true
            )

            if self.in_mask and y_pred:
                y_pred = tree.map_structure(
                    lambda x: x.in_mask(mask=self.in_mask), y_pred
                )
            if self.in_mask and y_true:
                y_true = tree.map_structure(
                    lambda x: x.in_mask(mask=self.in_mask), y_true
                )
            if self.out_mask and y_pred:
                y_pred = tree.map_structure(
                    lambda x: x.out_mask(mask=self.out_mask), y_pred
                )
            if self.out_mask and y_true:
                y_true = tree.map_structure(
                    lambda x: x.out_mask(mask=self.out_mask), y_true
                )

            rewards = await self.call(y_true, y_pred)
            return reduce_values(
                rewards,
                reduction=self.reduction,
            )

    async def call(self, y_true, y_pred):
        raise NotImplementedError

    def get_config(self):
        return {
            "name": self.name,
            "reduction": self.reduction,
            "in_mask": self.in_mask,
            "out_mask": self.out_mask,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _obj_type(self):
        return "Reward"


def standardize_reduction(reduction):
    allowed = {
        "sum",
        None,
        "none",
        "mean",
    }
    if reduction not in allowed:
        raise ValueError(
            "Invalid value for argument `reduction`. "
            f"Expected one of {allowed}. Received: "
            f"reduction={reduction}"
        )
    return reduction


def squeeze_or_expand_to_same_rank(x1, x2, expand_rank_1=True):
    """Squeeze/expand last dim if ranks differ from expected by exactly 1."""
    x1_rank = len(x1.shape)
    x2_rank = len(x2.shape)
    if x1_rank == x2_rank:
        return x1, x2
    if x1_rank == x2_rank + 1:
        if x1.shape[-1] == 1:
            if x2_rank == 1 and expand_rank_1:
                x2 = np.expand_dims(x2, axis=-1)
            else:
                x1 = np.squeeze(x1, axis=-1)
    if x2_rank == x1_rank + 1:
        if x2.shape[-1] == 1:
            if x1_rank == 1 and expand_rank_1:
                x1 = np.expand_dims(x1, axis=-1)
            else:
                x2 = np.squeeze(x2, axis=-1)
    return x1, x2


def reduce_values(values, reduction="mean"):
    if reduction is None or reduction == "none" or not hasattr(values, "__len__"):
        return float(values)
    reward = np.sum(values)
    if reduction == "mean":
        divisor = np.prod(np.convert_to_tensor(np.shape(values)))
        reward = np.divide_no_nan(reward, divisor)
    return float(reward)
