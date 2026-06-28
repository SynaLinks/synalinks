# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.metrics.metric import Metric


@synalinks_export(["synalinks.BatchMetric", "synalinks.metrics.BatchMetric"])
class BatchMetric(Metric):
    """Batched metric base class (the metric counterpart of `BatchReward`).

    A regular `Metric` is updated one sample at a time by the trainer. A
    `BatchMetric` instead receives the **entire batch at once**, so it can
    compute quantities that only make sense across a group of samples — e.g.
    ``pass@k`` over the ``k`` samples drawn for a single problem.

    The trainer detects `BatchMetric` instances and routes the whole
    ``(y_true, y_pred)`` batch to them (mirroring how `BatchReward` is fed via
    `has_batch_rewards` / `compute_batch`); non-batch metrics in the same
    `compile(metrics=[...])` list keep their per-sample updates.

    To be implemented by subclasses:

    * ``update_state(y_true, y_pred)``: ``y_true`` and ``y_pred`` are lists of
      length ``batch_size`` (one batch). Accumulate state from the group.
    * ``result()``: return the current scalar (or dict) value.

    Note:
        With the "whole batch = one problem's k samples" convention, set
        ``batch_size = k`` so each batch handed to the metric is the ``k``
        samples of a single problem.
    """

    async def update_state(self, y_true, y_pred):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError

    def _obj_type(self):
        return "BatchMetric"
