# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.metrics.metric import Metric


@synalinks_export(["synalinks.BatchMetric", "synalinks.metrics.BatchMetric"])
class BatchMetric(Metric):
    """Batched metric base class (the metric counterpart of `BatchReward`).

    A regular `Metric` is updated one sample at a time by the trainer. A
    `BatchMetric` instead receives the **entire batch at once**, so it can
    compute quantities that only make sense across a group of samples, e.g.
    ``pass@k`` over the ``k`` samples drawn for a single problem.

    The trainer detects `BatchMetric` instances and routes the whole
    ``(y_true, y_pred)`` batch to them (mirroring how `BatchReward` is fed via
    `has_batch_rewards` / `compute_batch`); non-batch metrics in the same
    `compile(metrics=[...])` list keep their per-sample updates.

    To be implemented by subclasses:

    * ``update_state(y_true, y_pred)``: ``y_true`` and ``y_pred`` are lists of
      length ``batch_size`` (one batch). Accumulate state from the group.
      Declare an ``x`` argument to also get the batch's inputs (a list)
      during ``fit`` / ``evaluate``.
    * ``result()``: return the current scalar (or dict) value.

    Note:
        With the "whole batch = one problem's k samples" convention, set
        ``batch_size = k`` so each batch handed to the metric is the ``k``
        samples of a single problem.

    Example (with the concrete `PassAtK` subclass):

    ```python
    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        metrics=[
            synalinks.metrics.PassAtK(
                k=5,
                reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
            ),
        ],
    )
    # batch_size = k samples of one problem
    await program.evaluate(x=x, y=y, batch_size=5)
    ```
    """

    async def update_state(self, y_true, y_pred):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError
