# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.rewards.reward import Reward
from synalinks.src.rewards.reward import apply_masks
from synalinks.src.rewards.reward import reduce_values
from synalinks.src.saving import serialization_lib


@synalinks_export(["synalinks.BatchReward", "synalinks.rewards.BatchReward"])
class BatchReward(Reward):
    """Batched reward base class.

    Subclasses receive the entire batch at once and must return one reward
    per sample. Use this when the reward needs cross-sample context (e.g.
    group-relative scores, batch normalization, paired comparisons).

    To be implemented by subclasses:

    * ``call(y_true, y_pred)``: ``y_true`` and ``y_pred`` are lists of
      length ``batch_size``. MUST return a ``list[float]`` of the same
      length, one reward per sample.

    Args:
        name (str): Optional name for the reward instance.
        reduction (str): Optional. One of ``"mean"``, ``"sum"``, ``"min"``,
            ``"max"``, ``"none"`` or ``None``. Applied by ``__call__`` when
            called on a batch directly. The trainer consumes the unreduced
            per-sample list via ``compute_batch``, but propagates this
            value to control the scalar shown in progress logs and used
            for candidate scoring (``"none"``/``None`` falls back to
            ``"mean"`` for those).
        in_mask (list): Optional. List of exact field names to keep before
            computing the reward.
        out_mask (list): Optional. List of exact field names to drop before
            computing the reward.
        in_mask_pattern (str): Optional. Regex pattern; fields whose names
            match are kept (combined with ``in_mask``).
        out_mask_pattern (str): Optional. Regex pattern; fields whose names
            match are dropped (combined with ``out_mask``).
    """

    async def __call__(self, y_true, y_pred):
        rewards = await self.compute_batch(y_true, y_pred)
        return reduce_values(rewards, reduction=self.reduction)

    async def compute_batch(self, y_true, y_pred):
        """Apply masks and return the per-sample reward list (unreduced).

        This is what the trainer calls — it expects the raw ``list[float]``
        of length ``batch_size`` so it can treat each entry as that
        sample's reward.
        """
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
            return _validate_batch_rewards(rewards, y_pred, type(self).__name__)

    async def call(self, y_true, y_pred):
        raise NotImplementedError

    def _obj_type(self):
        return "BatchReward"


@synalinks_export("synalinks.rewards.BatchRewardFunctionWrapper")
class BatchRewardFunctionWrapper(BatchReward):
    """Wrap a stateless batched function into a ``BatchReward``.

    The wrapped function receives the full batch and must return a
    ``list[float]`` of length ``batch_size``.

    Example:

    ```python
    async def my_batch_reward(y_true, y_pred):
        # y_true, y_pred: list[JsonDataModel] of length batch_size
        return [1.0 if t.get_json() == p.get_json() else 0.0
                for t, p in zip(y_true, y_pred)]

    program.compile(
        reward=synalinks.rewards.BatchRewardFunctionWrapper(fn=my_batch_reward),
        optimizer=synalinks.optimizers.RandomFewShot(),
    )
    ```

    Args:
        fn (callable): Async batched reward function with signature
            ``fn(y_true, y_pred, **kwargs) -> list[float]``.
        name (str): Optional. string name of the reward instance.
        reduction (str): Optional. One of ``"mean"``, ``"sum"``, ``"min"``,
            ``"max"``, ``"none"`` or ``None``. Used by standalone
            ``__call__`` and propagated through ``compile`` to set the
            scalar reduction used by the trainer's progress log and the
            optimizer's candidate scoring (``"none"``/``None`` falls back
            to ``"mean"`` there).
        in_mask (list): Optional.
        out_mask (list): Optional.
        in_mask_pattern (str): Optional.
        out_mask_pattern (str): Optional.
        **kwargs (keyword arguments): Extra keyword arguments forwarded
            to ``fn``.
    """

    def __init__(
        self,
        fn,
        reduction="mean",
        name=None,
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            reduction=reduction,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )
        self.fn = fn
        self._fn_kwargs = kwargs

    async def call(self, y_true, y_pred):
        return await self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = super().get_config()
        config["fn"] = serialization_lib.serialize_synalinks_object(self.fn)
        config["fn_kwargs"] = serialization_lib.serialize_synalinks_object(
            self._fn_kwargs
        )
        return config

    @classmethod
    def from_config(cls, config):
        if "fn" in config:
            config = serialization_lib.deserialize_synalinks_object(config)
        fn_kwargs = config.pop("fn_kwargs", None) or {}
        return cls(**config, **fn_kwargs)

    def __repr__(self):
        return f"<BatchRewardFunctionWrapper({self.fn}, kwargs={self._fn_kwargs})>"


def _validate_batch_rewards(rewards, y_pred, cls_name):
    if isinstance(rewards, (list, tuple)):
        pass
    elif hasattr(rewards, "__iter__") and not isinstance(rewards, (str, bytes, dict)):
        # Accept numpy arrays / generators / other iterables of floats.
        rewards = list(rewards)
    else:
        raise TypeError(
            f"`{cls_name}.call` must return a list of per-sample floats. "
            f"Got {type(rewards).__name__}."
        )
    if hasattr(y_pred, "__len__") and len(rewards) != len(y_pred):
        raise ValueError(
            f"`{cls_name}.call` returned {len(rewards)} rewards but the "
            f"batch has {len(y_pred)} samples — they must match."
        )
    return [float(r) for r in rewards]
