# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Sampling-based agent metrics (`pass@k` and friends).

These are **batched** metrics: each batch handed to the metric is treated as
the ``k`` samples drawn for a *single* problem (so the caller sets
``batch_size = k``). Each sample is scored for correctness against its target
with a pluggable reward (default `ExactMatch`); the per-problem ``(n, c)``
pair — ``n`` samples, ``c`` correct — is reduced through the standard unbiased
estimators, and the per-problem values are averaged over the dataset.

Because a regular `Metric` only ever sees one sample at a time, these subclass
`BatchMetric`, which the trainer feeds the whole batch at once (the metric
counterpart of `BatchReward`).

Class hierarchy:

    SampledRewardMetric        (base; scores the batch, counts (n, c))
    ├── PassAtK                 1 - C(n-c, k) / C(n, k)   — optimistic
    ├── PassHatK                C(c, k)   / C(n, k)        — consistency
    └── GapK                    PassAtK - PassHatK         — reliability gap

The ``pass@k`` / ``pass^k`` / ``gap-k`` triplet separates capability from
reliability: ``pass@k`` is optimistic (solved in *at least one* of k tries),
``pass^k`` is consistent (solved in *all* k tries), and ``gap-k`` is the
flakiness between them.

References:
    - ``pass@k`` unbiased estimator: Chen et al., "Evaluating Large Language
      Models Trained on Code" (HumanEval), arXiv:2107.03374.
    - ``pass^k`` (all k trials succeed) and the resulting reliability gap:
      Yao et al., "tau-bench: A Benchmark for Tool-Agent-User Interaction in
      Real-World Domains", arXiv:2406.12045.

    ``gap-k`` has no separate source: it is the derived difference
    ``pass@k - pass^k`` (the reliability gap surfaced by tau-bench), so the two
    references above cover all three metrics.
"""

import math

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.metrics.batch_metric import BatchMetric
from synalinks.src.saving import serialization_lib


class TotalWithCount(DataModel):
    total: float = 0.0
    count: int = 0


class SampledRewardMetric(BatchMetric):
    """Base for sampling-based agent metrics (`pass@k` and friends).

    Scores each sample in the batch against its target with `reward`, reduces
    the per-problem ``(n, c)`` pair via `_estimate`, and accumulates the
    running mean over the dataset.

    Args:
        k (int): The number of samples ``k`` in ``@k``. Clamped to the number
            of samples actually present in the batch.
        reward (Reward): The per-sample correctness signal, callable as
            ``await reward(y_true, y_pred) -> float``. Defaults to
            `synalinks.rewards.ExactMatch`. Configure its masks (e.g.
            ``ExactMatch(in_mask=["answer"])``) to compare only the relevant
            fields of each sample.
        pass_threshold (float): A sample counts as correct when its reward is
            ``>= pass_threshold``. Defaults to ``1.0`` (binary rewards).
        name (str): Optional. Name of the metric instance.
    """

    direction = "up"

    def __init__(
        self,
        k=1,
        reward=None,
        pass_threshold=1.0,
        name=None,
    ):
        super().__init__(name=name)
        if reward is None:
            # Lazy import: `rewards` pulls in `modules`, which would create a
            # circular import if done at module load time.
            from synalinks.src import rewards

            reward = rewards.ExactMatch()
        self.k = int(k)
        self.reward = reward
        self.pass_threshold = float(pass_threshold)
        self.total_with_count = self.add_variable(
            data_model=TotalWithCount, name="total_with_count"
        )

    async def _count(self, y_true, y_pred):
        """Return ``(n, c)``: total samples in the batch and number passing."""
        if not isinstance(y_pred, (list, tuple)):
            y_pred = [y_pred]
            y_true = [y_true]
        n = len(y_pred)
        c = 0
        for y_t, y_p in zip(y_true, y_pred):
            reward = await self.reward(y_t, y_p)
            if reward is not None and float(reward) >= self.pass_threshold:
                c += 1
        return n, c

    def _estimate(self, n, c):
        raise NotImplementedError

    async def update_state(self, y_true, y_pred):
        n, c = await self._count(y_true, y_pred)
        value = self._estimate(n, c)
        total = self.total_with_count.get("total")
        count = self.total_with_count.get("count")
        self.total_with_count.update(
            {"total": float(total + value), "count": int(count + 1)}
        )

    def reset_state(self):
        self.total_with_count.assign(TotalWithCount())

    def result(self):
        count = self.total_with_count.get("count")
        if not count:
            return 0.0
        return float(self.total_with_count.get("total") / count)

    def get_config(self):
        return {
            "name": self.name,
            "k": self.k,
            "reward": serialization_lib.serialize_synalinks_object(self.reward),
            "pass_threshold": self.pass_threshold,
        }

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["reward"] = serialization_lib.deserialize_synalinks_object(
            config.pop("reward")
        )
        return cls(**config)


@synalinks_export(
    [
        "synalinks.metrics.PassAtK",
        "synalinks.PassAtK",
    ]
)
class PassAtK(SampledRewardMetric):
    """``pass@k`` — fraction of problems solved in *at least one* of k samples.

    A batched metric: each batch is the ``k`` samples of one problem (set
    ``batch_size = k``). Uses the unbiased HumanEval estimator over the ``n``
    samples in the batch, of which ``c`` are correct::

        pass@k = 1 - C(n - c, k) / C(n, k)

    (``= 1.0`` when ``n - c < k``). Averaged over the dataset.

    Example:

    ```python
    import synalinks

    class Question(synalinks.DataModel):
        question: str

    class Answer(synalinks.DataModel):
        answer: str

    K = 5

    # Sampling (temperature > 0) so the K rollouts of one prompt differ;
    # with greedy decoding all K are identical and pass@k collapses to pass@1.
    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=synalinks.LanguageModel(model="openai/gpt-4o-mini"),
        temperature=0.8,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs)

    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        metrics=[synalinks.metrics.PassAtK(k=K)],
    )

    # repeat == batch_size == K  ->  each batch is one problem's K samples.
    dataset = synalinks.HuggingFaceDataset(
        hf_dataset_name="openai/gsm8k",
        hf_config_name="main",
        split="test",
        input_data_model=Question,
        output_data_model=Answer,
        input_template='{"question": {{ question | tojson }}}',
        output_template='{"answer": {{ answer.split("####")[-1].strip() | tojson }}}',
        batch_size=K,
        repeat=K,
        limit=20,
    )

    metrics = await program.evaluate(x=dataset())
    print(metrics["pass_at_k"])
    ```

    Args:
        k (int): The number of samples ``k``.
        reward (Reward): Per-sample correctness signal (default `ExactMatch`).
        pass_threshold (float): Reward threshold for a sample to count as
            correct. Defaults to ``1.0``.
        name (str): Optional. Name of the metric instance.
    """

    def __init__(self, k=1, reward=None, pass_threshold=1.0, name="pass_at_k"):
        super().__init__(k=k, reward=reward, pass_threshold=pass_threshold, name=name)

    def _estimate(self, n, c):
        k = min(self.k, n)
        if k <= 0:
            return 0.0
        if n - c < k:
            return 1.0
        return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


@synalinks_export(
    [
        "synalinks.metrics.PassHatK",
        "synalinks.PassHatK",
    ]
)
class PassHatK(SampledRewardMetric):
    """``pass^k`` — fraction of problems solved in *all* k samples (consistency).

    A batched metric (each batch is one problem's ``k`` samples). Unbiased
    estimator of the probability that ``k`` samples drawn (without replacement)
    from the ``n`` batch samples are *all* correct::

        pass^k = C(c, k) / C(n, k)

    (``= 0.0`` when ``c < k``). Averaged over the dataset. Always ``<=`` the
    corresponding `PassAtK`; the difference is reported by `GapK`.

    Example:

    ```python
    import synalinks

    class Question(synalinks.DataModel):
        question: str

    class Answer(synalinks.DataModel):
        answer: str

    K = 5

    # Sampling (temperature > 0) so the K rollouts of one prompt differ.
    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=synalinks.LanguageModel(model="openai/gpt-4o-mini"),
        temperature=0.8,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs)

    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        metrics=[synalinks.metrics.PassHatK(k=K)],
    )

    # repeat == batch_size == K  ->  each batch is one problem's K samples.
    dataset = synalinks.HuggingFaceDataset(
        hf_dataset_name="openai/gsm8k",
        hf_config_name="main",
        split="test",
        input_data_model=Question,
        output_data_model=Answer,
        input_template='{"question": {{ question | tojson }}}',
        output_template='{"answer": {{ answer.split("####")[-1].strip() | tojson }}}',
        batch_size=K,
        repeat=K,
        limit=20,
    )

    metrics = await program.evaluate(x=dataset())
    print(metrics["pass_hat_k"])  # consistency: solved in ALL K samples
    ```

    Args:
        k (int): The number of samples ``k``.
        reward (Reward): Per-sample correctness signal (default `ExactMatch`).
        pass_threshold (float): Reward threshold for a sample to count as
            correct. Defaults to ``1.0``.
        name (str): Optional. Name of the metric instance.
    """

    def __init__(self, k=1, reward=None, pass_threshold=1.0, name="pass_hat_k"):
        super().__init__(k=k, reward=reward, pass_threshold=pass_threshold, name=name)

    def _estimate(self, n, c):
        k = min(self.k, n)
        if k <= 0 or c < k:
            return 0.0
        return math.comb(c, k) / math.comb(n, k)


@synalinks_export(
    [
        "synalinks.metrics.GapK",
        "synalinks.GapK",
    ]
)
class GapK(SampledRewardMetric):
    """``gap-k`` — the reliability gap ``pass@k - pass^k``.

    A batched metric (each batch is one problem's ``k`` samples). The flakiness
    between optimistic and consistent performance: ``0`` means a problem is
    either always solved or never solved across the k samples; larger values
    mean the agent's success is sample-dependent. Lower is more reliable
    (``direction = "down"``). Averaged over the dataset.

    Example:

    ```python
    import synalinks

    class Question(synalinks.DataModel):
        question: str

    class Answer(synalinks.DataModel):
        answer: str

    K = 5

    # Sampling (temperature > 0) so the K rollouts of one prompt differ.
    inputs = synalinks.Input(data_model=Question)
    outputs = await synalinks.Generator(
        data_model=Answer,
        language_model=synalinks.LanguageModel(model="openai/gpt-4o-mini"),
        temperature=0.8,
    )(inputs)
    program = synalinks.Program(inputs=inputs, outputs=outputs)

    # Report all three together: the optimistic bound, the consistency floor,
    # and the gap between them.
    program.compile(
        reward=synalinks.rewards.ExactMatch(in_mask=["answer"]),
        metrics=[
            synalinks.metrics.PassAtK(k=K),
            synalinks.metrics.PassHatK(k=K),
            synalinks.metrics.GapK(k=K),
        ],
    )

    # repeat == batch_size == K  ->  each batch is one problem's K samples.
    dataset = synalinks.HuggingFaceDataset(
        hf_dataset_name="openai/gsm8k",
        hf_config_name="main",
        split="test",
        input_data_model=Question,
        output_data_model=Answer,
        input_template='{"question": {{ question | tojson }}}',
        output_template='{"answer": {{ answer.split("####")[-1].strip() | tojson }}}',
        batch_size=K,
        repeat=K,
        limit=20,
    )

    metrics = await program.evaluate(x=dataset())
    print(metrics["gap_k"])  # flakiness = pass@k - pass^k
    ```

    Args:
        k (int): The number of samples ``k``.
        reward (Reward): Per-sample correctness signal (default `ExactMatch`).
        pass_threshold (float): Reward threshold for a sample to count as
            correct. Defaults to ``1.0``.
        name (str): Optional. Name of the metric instance.
    """

    direction = "down"

    def __init__(self, k=1, reward=None, pass_threshold=1.0, name="gap_k"):
        super().__init__(k=k, reward=reward, pass_threshold=pass_threshold, name=name)

    def _estimate(self, n, c):
        k = min(self.k, n)
        if k <= 0:
            return 0.0
        if n - c < k:
            pass_at_k = 1.0
        else:
            pass_at_k = 1.0 - (math.comb(n - c, k) / math.comb(n, k))
        pass_hat_k = 0.0 if c < k else math.comb(c, k) / math.comb(n, k)
        return pass_at_k - pass_hat_k
