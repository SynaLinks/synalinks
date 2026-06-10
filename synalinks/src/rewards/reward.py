# Modified from: keras/src/losses/loss.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

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
        name (str): Optional name for the reward instance.
        reduction (str): Optional. One of ``"mean"``, ``"sum"``, ``"min"``,
            ``"max"``, ``"none"`` or ``None``. Applied by ``__call__`` when
            invoked on a batch directly (standalone evaluation) and
            propagated through ``compile`` to control how the
            trainer/optimizer reduce per-sample rewards into the scalar
            shown in progress logs and used for candidate scoring. Use
            ``"min"`` to score by the worst sample (robust/pessimistic) or
            ``"max"`` for the best (optimistic / best-of-N). ``"none"``/
            ``None`` falls back to ``"mean"`` for those scalar consumers
            (per-sample values are always preserved for the optimizer's RL
            bookkeeping).
        in_mask (list): Optional. List of exact field names to keep before
            computing the reward.
        out_mask (list): Optional. List of exact field names to drop before
            computing the reward.
        in_mask_pattern (str): Optional. Regex pattern; fields whose names
            match are kept (combined with ``in_mask`` via OR).
        out_mask_pattern (str): Optional. Regex pattern; fields whose names
            match are dropped (combined with ``out_mask`` via OR).

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
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        self.name = name or auto_name(self.__class__.__name__)
        self.reduction = standardize_reduction(reduction)
        self.in_mask = in_mask
        self.out_mask = out_mask
        self.in_mask_pattern = in_mask_pattern
        self.out_mask_pattern = out_mask_pattern

    async def __call__(self, y_true, y_pred):
        with ops.name_scope(self.name):
            y_true, y_pred = apply_masks(
                y_true,
                y_pred,
                in_mask=self.in_mask,
                in_mask_pattern=self.in_mask_pattern,
                out_mask=self.out_mask,
                out_mask_pattern=self.out_mask_pattern,
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
            "in_mask_pattern": self.in_mask_pattern,
            "out_mask_pattern": self.out_mask_pattern,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _obj_type(self):
        return "Reward"


def apply_masks(
    y_true,
    y_pred,
    in_mask=None,
    in_mask_pattern=None,
    out_mask=None,
    out_mask_pattern=None,
):
    """Convert ``y_true``/``y_pred`` to ``JsonDataModel`` and apply field masks.

    Works on both a single sample (a leaf ``JsonDataModel``) and a list/tree
    of samples — ``tree.map_structure`` handles either uniformly.
    """
    y_pred = tree.map_structure(lambda x: ops.convert_to_json_data_model(x), y_pred)
    y_true = tree.map_structure(lambda x: ops.convert_to_json_data_model(x), y_true)

    if (in_mask or in_mask_pattern) and y_pred:
        y_pred = tree.map_structure(
            lambda x: (
                x.in_mask(mask=in_mask, pattern=in_mask_pattern) if x is not None else x
            ),
            y_pred,
        )
    if (in_mask or in_mask_pattern) and y_true:
        y_true = tree.map_structure(
            lambda x: (
                x.in_mask(mask=in_mask, pattern=in_mask_pattern) if x is not None else x
            ),
            y_true,
        )
    if (out_mask or out_mask_pattern) and y_pred:
        y_pred = tree.map_structure(
            lambda x: (
                x.out_mask(mask=out_mask, pattern=out_mask_pattern)
                if x is not None
                else x
            ),
            y_pred,
        )
    if (out_mask or out_mask_pattern) and y_true:
        y_true = tree.map_structure(
            lambda x: (
                x.out_mask(mask=out_mask, pattern=out_mask_pattern)
                if x is not None
                else x
            ),
            y_true,
        )
    return y_true, y_pred


def standardize_reduction(reduction):
    allowed = {
        "sum",
        None,
        "none",
        "mean",
        "min",
        "max",
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
    # An empty batch carries no signal to reduce, so every reduction collapses
    # to a scalar 0.0. This keeps downstream scalar consumers (trackers, tuner
    # objectives) from ever receiving an empty list (the ``"none"`` branch) or
    # hitting ``min``/``max``'s "zero-size array" error.
    if hasattr(values, "__len__") and len(values) == 0:
        return 0.0
    if reduction is None or reduction == "none":
        # Preserve the per-sample structure: scalars stay scalars, lists/
        # arrays are returned unreduced.
        return values if hasattr(values, "__len__") else float(values)
    if not hasattr(values, "__len__"):
        return float(values)
    values = np.convert_to_tensor(values)
    if reduction == "min":
        return float(np.min(values))
    if reduction == "max":
        return float(np.max(values))
    reward = np.sum(values)
    if reduction == "mean":
        divisor = np.prod(np.convert_to_tensor(np.shape(values)))
        reward = np.divide_no_nan(reward, divisor)
    return float(reward)


def reduce_rewards(rewards, reduction="mean"):
    """Reduce a per-sample reward list to a scalar for trackers/scoring.

    ``reduction="none"`` / ``None`` falls back to ``"mean"`` since callers
    here always need a scalar (progress logs, candidate scoring).
    """
    if not rewards:
        return 0.0
    rewards = np.convert_to_tensor(rewards)
    if reduction == "sum":
        return float(np.sum(rewards))
    if reduction == "min":
        return float(np.min(rewards))
    if reduction == "max":
        return float(np.max(rewards))
    return float(np.mean(rewards))
