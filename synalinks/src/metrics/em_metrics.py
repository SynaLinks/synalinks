# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Operational metrics for `EmbeddingModel` runtime counters.

All metrics read counters that the EM populates on each provider call.
Routing across `inference`, `reward`, and `optimizer` phases follows the
active `op_scope` (a contextvar the trainer sets per phase).

Class hierarchy:

    EmbeddingModelOperationalMetric                  (base, "inference")
    ├── EmbeddingModelRewardsOperationalMetric             ("reward")
    └── EmbeddingModelOptimizersOperationalMetric          ("optimizer")
"""

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.op_scope import read_phase_wall_clock_s
from synalinks.src.metrics.metric import Metric

_TRACKED_SUFFIXES = (
    "calls",
    "prompt_tokens",
    "tokens",
    "vectors",
    "elapsed_s",
    "cost",
    "cached_tokens",
    "failed_calls",
    "fallback_activations",
)


def _collect_embedding_models(program):
    """Walk a program's module tree and return every unique EmbeddingModel
    (including those reached via the `fallback` chain), preserving order.
    """
    from synalinks.src.modules.embedding_models import EmbeddingModel

    ems = []
    seen = set()

    def _add_chain(em):
        while em is not None and id(em) not in seen:
            if isinstance(em, EmbeddingModel):
                seen.add(id(em))
                ems.append(em)
                em = getattr(em, "fallback", None)
            else:
                break

    modules = []
    if hasattr(program, "_flatten_modules"):
        modules = program._flatten_modules(include_self=True, recursive=True)
    for module in modules:
        _add_chain(getattr(module, "embedding_model", None))
        if isinstance(module, EmbeddingModel):
            _add_chain(module)
    return ems


# ---------------------------------------------------------------------------
# Inference-phase base + metrics
# ---------------------------------------------------------------------------


@synalinks_export(
    [
        "synalinks.metrics.EmbeddingModelOperationalMetric",
        "synalinks.EmbeddingModelOperationalMetric",
    ]
)
class EmbeddingModelOperationalMetric(Metric):
    """Base class for `EmbeddingModel` runtime-counter metrics.

    Subclasses set `_phase` to one of ``"inference"``, ``"reward"``, or
    ``"optimizer"`` to read the corresponding counter set on each bound
    embedding model. Counters are populated by the EM based on the active
    ``op_scope`` (contextvar) the trainer sets for each phase.

    Binds itself automatically to every `EmbeddingModel` reachable from
    the program (and their `.fallback` chains) on `program.compile()`.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingModelOperationalMetric(),
        ],
    )
    ```
    """

    _phase = "inference"

    def __init__(self, name=None):
        super().__init__(name=name)
        self._embedding_models = []
        self._baselines = {suffix: 0 for suffix in _TRACKED_SUFFIXES}
        self._wall_baseline = 0.0

    @property
    def embedding_models(self):
        return list(self._embedding_models)

    def bind_program(self, program):
        self._embedding_models = _collect_embedding_models(program)
        self._snapshot()

    def _attr(self, suffix):
        return f"{self._phase}_cumulated_{suffix}"

    def _read(self, suffix):
        attr = self._attr(suffix)
        return sum(getattr(em, attr, 0) for em in self._embedding_models)

    def _snapshot(self):
        for suffix in _TRACKED_SUFFIXES:
            self._baselines[suffix] = self._read(suffix)
        self._wall_baseline = read_phase_wall_clock_s(self._phase)

    def _delta(self, suffix):
        return self._read(suffix) - self._baselines.get(suffix, 0)

    def _wall_clock_delta(self):
        """Wall-clock seconds the trainer spent in this metric's phase since
        the last snapshot: the throughput denominator (concurrency-safe,
        unlike summed `elapsed_s`).
        """
        return read_phase_wall_clock_s(self._phase) - self._wall_baseline

    def reset_state(self):
        self._snapshot()

    async def update_state(self, *args, **kwargs):
        return

    def result(self):
        raise NotImplementedError

    def get_config(self):
        return {"name": self.name}


@synalinks_export("synalinks.metrics.EmbeddingTokens")
class EmbeddingTokens(EmbeddingModelOperationalMetric):
    """Cumulated tokens consumed by embedding calls during this run.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingTokens(),
        ],
    )
    ```
    """

    def __init__(self, name="embedding_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("tokens"))


@synalinks_export("synalinks.metrics.EmbeddingVectors")
class EmbeddingVectors(EmbeddingModelOperationalMetric):
    """Cumulated vectors produced by embedding calls during this run.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingVectors(),
        ],
    )
    ```
    """

    def __init__(self, name="embedding_vectors"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("vectors"))


@synalinks_export("synalinks.metrics.EmbeddingCost")
class EmbeddingCost(EmbeddingModelOperationalMetric):
    """Cumulated embedding-provider cost (USD) for this run.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingCost(),
        ],
    )
    ```
    """

    def __init__(self, name="embedding_cost"):
        super().__init__(name=name)

    def result(self):
        return float(self._delta("cost"))


@synalinks_export("synalinks.metrics.EmbeddingThroughput")
class EmbeddingThroughput(EmbeddingModelOperationalMetric):
    """Embedding calls per second (RPS) over this run.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingThroughput(),
        ],
    )
    ```
    """

    def __init__(self, name="embedding_throughput"):
        super().__init__(name=name)

    def result(self):
        wall = self._wall_clock_delta()
        if wall <= 0.0:
            return 0.0
        return self._delta("calls") / wall


@synalinks_export("synalinks.metrics.AvgEmbeddingLatency")
class AvgEmbeddingLatency(EmbeddingModelOperationalMetric):
    """Average wall-clock latency in seconds per embedding call over this run.

    Computed as ``elapsed_s / calls``. Because ``elapsed_s`` accumulates each
    call's own duration, this reports the mean per-call latency regardless of
    concurrency -- unlike `EmbeddingThroughput`, which divides by the phase's
    wall-clock span. The two coincide (latency = 1 / throughput) only when
    calls run serially.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgEmbeddingLatency(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_embedding_latency"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("elapsed_s") / calls


@synalinks_export("synalinks.metrics.EmbeddingTokensPerSecond")
class EmbeddingTokensPerSecond(EmbeddingModelOperationalMetric):
    """Embedded tokens per second over this run.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingTokensPerSecond(),
        ],
    )
    ```
    """

    def __init__(self, name="embedding_tokens_per_second"):
        super().__init__(name=name)

    def result(self):
        wall = self._wall_clock_delta()
        if wall <= 0.0:
            return 0.0
        return self._delta("tokens") / wall


@synalinks_export("synalinks.metrics.EmbeddingVectorsPerSecond")
class EmbeddingVectorsPerSecond(EmbeddingModelOperationalMetric):
    """Vectors produced per second over this run.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingVectorsPerSecond(),
        ],
    )
    ```
    """

    def __init__(self, name="embedding_vectors_per_second"):
        super().__init__(name=name)

    def result(self):
        wall = self._wall_clock_delta()
        if wall <= 0.0:
            return 0.0
        return self._delta("vectors") / wall


@synalinks_export("synalinks.metrics.AvgEmbeddingTokensPerCall")
class AvgEmbeddingTokensPerCall(EmbeddingModelOperationalMetric):
    """Average tokens per embedding call over this run.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgEmbeddingTokensPerCall(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_embedding_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("tokens") / calls


@synalinks_export("synalinks.metrics.AvgEmbeddingVectorsPerCall")
class AvgEmbeddingVectorsPerCall(EmbeddingModelOperationalMetric):
    """Average vectors (i.e. batch size) per embedding call over this run.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgEmbeddingVectorsPerCall(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_embedding_vectors_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("vectors") / calls


@synalinks_export("synalinks.metrics.AvgEmbeddingCostPerCall")
class AvgEmbeddingCostPerCall(EmbeddingModelOperationalMetric):
    """Average embedding-provider cost per call over this run.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgEmbeddingCostPerCall(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_embedding_cost_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("cost") / calls


@synalinks_export("synalinks.metrics.AvgEmbeddingCachedTokensPerCall")
class AvgEmbeddingCachedTokensPerCall(EmbeddingModelOperationalMetric):
    """Average cached prompt tokens per embedding call over this run.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgEmbeddingCachedTokensPerCall(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_embedding_cached_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("cached_tokens") / calls


@synalinks_export("synalinks.metrics.EmbeddingCachedTokens")
class EmbeddingCachedTokens(EmbeddingModelOperationalMetric):
    """Prompt tokens served from cache during embedding inference.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingCachedTokens(),
        ],
    )
    ```
    """

    def __init__(self, name="embedding_cached_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("cached_tokens"))


@synalinks_export("synalinks.metrics.EmbeddingCacheHitRate")
class EmbeddingCacheHitRate(EmbeddingModelOperationalMetric):
    """Cache hit rate for embedding inputs: cached_tokens / prompt_tokens.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingCacheHitRate(),
        ],
    )
    ```
    """

    def __init__(self, name="embedding_cache_hit_rate"):
        super().__init__(name=name)

    def result(self):
        prompt = self._delta("prompt_tokens")
        if prompt <= 0:
            return 0.0
        return self._delta("cached_tokens") / prompt


@synalinks_export("synalinks.metrics.EmbeddingFailedCalls")
class EmbeddingFailedCalls(EmbeddingModelOperationalMetric):
    """Embedding calls that exhausted all retries and failed this run.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingFailedCalls(),
        ],
    )
    ```
    """

    def __init__(self, name="embedding_failed_calls"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("failed_calls"))


@synalinks_export("synalinks.metrics.EmbeddingFallbackActivations")
class EmbeddingFallbackActivations(EmbeddingModelOperationalMetric):
    """Times a failed embedding call triggered its `fallback` chain.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingFallbackActivations(),
        ],
    )
    ```
    """

    def __init__(self, name="embedding_fallback_activations"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("fallback_activations"))


@synalinks_export("synalinks.metrics.EmbeddingErrorRate")
class EmbeddingErrorRate(EmbeddingModelOperationalMetric):
    """Fraction of embedding calls that failed: failed / (succeeded + failed).

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingErrorRate(),
        ],
    )
    ```
    """

    def __init__(self, name="embedding_error_rate"):
        super().__init__(name=name)

    def result(self):
        failed = self._delta("failed_calls")
        total = self._delta("calls") + failed
        if total <= 0:
            return 0.0
        return failed / total


# ---------------------------------------------------------------------------
# Reward-phase metrics
# ---------------------------------------------------------------------------


@synalinks_export(
    [
        "synalinks.metrics.EmbeddingModelRewardsOperationalMetric",
        "synalinks.EmbeddingModelRewardsOperationalMetric",
    ]
)
class EmbeddingModelRewardsOperationalMetric(EmbeddingModelOperationalMetric):
    """Base for embedding metrics scoped to the reward-computation phase.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingModelRewardsOperationalMetric(),
        ],
    )
    ```
    """

    _phase = "reward"


@synalinks_export("synalinks.metrics.RewardEmbeddingTokens")
class RewardEmbeddingTokens(EmbeddingModelRewardsOperationalMetric):
    """Tokens consumed by embedding calls during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.RewardEmbeddingTokens(),
        ],
    )
    ```
    """

    def __init__(self, name="reward_embedding_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("tokens"))


@synalinks_export("synalinks.metrics.RewardEmbeddingVectors")
class RewardEmbeddingVectors(EmbeddingModelRewardsOperationalMetric):
    """Vectors produced by embedding calls during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.RewardEmbeddingVectors(),
        ],
    )
    ```
    """

    def __init__(self, name="reward_embedding_vectors"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("vectors"))


@synalinks_export("synalinks.metrics.RewardEmbeddingCost")
class RewardEmbeddingCost(EmbeddingModelRewardsOperationalMetric):
    """Provider cost of embedding calls during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.RewardEmbeddingCost(),
        ],
    )
    ```
    """

    def __init__(self, name="reward_embedding_cost"):
        super().__init__(name=name)

    def result(self):
        return float(self._delta("cost"))


@synalinks_export("synalinks.metrics.RewardEmbeddingThroughput")
class RewardEmbeddingThroughput(EmbeddingModelRewardsOperationalMetric):
    """Embedding calls per second during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.RewardEmbeddingThroughput(),
        ],
    )
    ```
    """

    def __init__(self, name="reward_embedding_throughput"):
        super().__init__(name=name)

    def result(self):
        wall = self._wall_clock_delta()
        if wall <= 0.0:
            return 0.0
        return self._delta("calls") / wall


@synalinks_export("synalinks.metrics.AvgRewardEmbeddingLatency")
class AvgRewardEmbeddingLatency(EmbeddingModelRewardsOperationalMetric):
    """Average latency (s) per embedding call during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgRewardEmbeddingLatency(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_reward_embedding_latency"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("elapsed_s") / calls


@synalinks_export("synalinks.metrics.RewardEmbeddingTokensPerSecond")
class RewardEmbeddingTokensPerSecond(EmbeddingModelRewardsOperationalMetric):
    """Embedded tokens per second during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.RewardEmbeddingTokensPerSecond(),
        ],
    )
    ```
    """

    def __init__(self, name="reward_embedding_tokens_per_second"):
        super().__init__(name=name)

    def result(self):
        wall = self._wall_clock_delta()
        if wall <= 0.0:
            return 0.0
        return self._delta("tokens") / wall


@synalinks_export("synalinks.metrics.RewardEmbeddingVectorsPerSecond")
class RewardEmbeddingVectorsPerSecond(EmbeddingModelRewardsOperationalMetric):
    """Vectors produced per second during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.RewardEmbeddingVectorsPerSecond(),
        ],
    )
    ```
    """

    def __init__(self, name="reward_embedding_vectors_per_second"):
        super().__init__(name=name)

    def result(self):
        wall = self._wall_clock_delta()
        if wall <= 0.0:
            return 0.0
        return self._delta("vectors") / wall


@synalinks_export("synalinks.metrics.AvgRewardEmbeddingTokensPerCall")
class AvgRewardEmbeddingTokensPerCall(EmbeddingModelRewardsOperationalMetric):
    """Average tokens per embedding call during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgRewardEmbeddingTokensPerCall(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_reward_embedding_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("tokens") / calls


@synalinks_export("synalinks.metrics.AvgRewardEmbeddingVectorsPerCall")
class AvgRewardEmbeddingVectorsPerCall(EmbeddingModelRewardsOperationalMetric):
    """Average batch size of embedding calls during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgRewardEmbeddingVectorsPerCall(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_reward_embedding_vectors_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("vectors") / calls


@synalinks_export("synalinks.metrics.AvgRewardEmbeddingCostPerCall")
class AvgRewardEmbeddingCostPerCall(EmbeddingModelRewardsOperationalMetric):
    """Average embedding-call cost during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgRewardEmbeddingCostPerCall(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_reward_embedding_cost_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("cost") / calls


@synalinks_export("synalinks.metrics.AvgRewardEmbeddingCachedTokensPerCall")
class AvgRewardEmbeddingCachedTokensPerCall(EmbeddingModelRewardsOperationalMetric):
    """Average cached prompt tokens per embedding call during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgRewardEmbeddingCachedTokensPerCall(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_reward_embedding_cached_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("cached_tokens") / calls


@synalinks_export("synalinks.metrics.RewardEmbeddingCachedTokens")
class RewardEmbeddingCachedTokens(EmbeddingModelRewardsOperationalMetric):
    """Prompt tokens served from cache during reward-phase embeddings.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.RewardEmbeddingCachedTokens(),
        ],
    )
    ```
    """

    def __init__(self, name="reward_embedding_cached_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("cached_tokens"))


@synalinks_export("synalinks.metrics.RewardEmbeddingCacheHitRate")
class RewardEmbeddingCacheHitRate(EmbeddingModelRewardsOperationalMetric):
    """Cache hit rate for reward-phase embedding inputs.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.RewardEmbeddingCacheHitRate(),
        ],
    )
    ```
    """

    def __init__(self, name="reward_embedding_cache_hit_rate"):
        super().__init__(name=name)

    def result(self):
        prompt = self._delta("prompt_tokens")
        if prompt <= 0:
            return 0.0
        return self._delta("cached_tokens") / prompt


@synalinks_export("synalinks.metrics.RewardEmbeddingFailedCalls")
class RewardEmbeddingFailedCalls(EmbeddingModelRewardsOperationalMetric):
    """Embedding calls that failed during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.RewardEmbeddingFailedCalls(),
        ],
    )
    ```
    """

    def __init__(self, name="reward_embedding_failed_calls"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("failed_calls"))


@synalinks_export("synalinks.metrics.RewardEmbeddingFallbackActivations")
class RewardEmbeddingFallbackActivations(EmbeddingModelRewardsOperationalMetric):
    """Embedding fallback activations during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.RewardEmbeddingFallbackActivations(),
        ],
    )
    ```
    """

    def __init__(self, name="reward_embedding_fallback_activations"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("fallback_activations"))


@synalinks_export("synalinks.metrics.RewardEmbeddingErrorRate")
class RewardEmbeddingErrorRate(EmbeddingModelRewardsOperationalMetric):
    """Fraction of embedding calls that failed during reward computation.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.RewardEmbeddingErrorRate(),
        ],
    )
    ```
    """

    def __init__(self, name="reward_embedding_error_rate"):
        super().__init__(name=name)

    def result(self):
        failed = self._delta("failed_calls")
        total = self._delta("calls") + failed
        if total <= 0:
            return 0.0
        return failed / total


# ---------------------------------------------------------------------------
# Optimizer-phase metrics
# ---------------------------------------------------------------------------


@synalinks_export(
    [
        "synalinks.metrics.EmbeddingModelOptimizersOperationalMetric",
        "synalinks.EmbeddingModelOptimizersOperationalMetric",
    ]
)
class EmbeddingModelOptimizersOperationalMetric(EmbeddingModelOperationalMetric):
    """Base for embedding metrics scoped to the optimizer phase.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.EmbeddingModelOptimizersOperationalMetric(),
        ],
    )
    ```
    """

    _phase = "optimizer"


@synalinks_export("synalinks.metrics.OptimizerEmbeddingTokens")
class OptimizerEmbeddingTokens(EmbeddingModelOptimizersOperationalMetric):
    """Tokens consumed by embedding calls during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.OptimizerEmbeddingTokens(),
        ],
    )
    ```
    """

    def __init__(self, name="optimizer_embedding_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("tokens"))


@synalinks_export("synalinks.metrics.OptimizerEmbeddingVectors")
class OptimizerEmbeddingVectors(EmbeddingModelOptimizersOperationalMetric):
    """Vectors produced by embedding calls during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.OptimizerEmbeddingVectors(),
        ],
    )
    ```
    """

    def __init__(self, name="optimizer_embedding_vectors"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("vectors"))


@synalinks_export("synalinks.metrics.OptimizerEmbeddingCost")
class OptimizerEmbeddingCost(EmbeddingModelOptimizersOperationalMetric):
    """Provider cost of embedding calls during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.OptimizerEmbeddingCost(),
        ],
    )
    ```
    """

    def __init__(self, name="optimizer_embedding_cost"):
        super().__init__(name=name)

    def result(self):
        return float(self._delta("cost"))


@synalinks_export("synalinks.metrics.OptimizerEmbeddingThroughput")
class OptimizerEmbeddingThroughput(EmbeddingModelOptimizersOperationalMetric):
    """Embedding calls per second during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.OptimizerEmbeddingThroughput(),
        ],
    )
    ```
    """

    def __init__(self, name="optimizer_embedding_throughput"):
        super().__init__(name=name)

    def result(self):
        wall = self._wall_clock_delta()
        if wall <= 0.0:
            return 0.0
        return self._delta("calls") / wall


@synalinks_export("synalinks.metrics.AvgOptimizerEmbeddingLatency")
class AvgOptimizerEmbeddingLatency(EmbeddingModelOptimizersOperationalMetric):
    """Average latency (s) per embedding call during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgOptimizerEmbeddingLatency(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_optimizer_embedding_latency"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("elapsed_s") / calls


@synalinks_export("synalinks.metrics.OptimizerEmbeddingTokensPerSecond")
class OptimizerEmbeddingTokensPerSecond(EmbeddingModelOptimizersOperationalMetric):
    """Embedded tokens per second during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.OptimizerEmbeddingTokensPerSecond(),
        ],
    )
    ```
    """

    def __init__(self, name="optimizer_embedding_tokens_per_second"):
        super().__init__(name=name)

    def result(self):
        wall = self._wall_clock_delta()
        if wall <= 0.0:
            return 0.0
        return self._delta("tokens") / wall


@synalinks_export("synalinks.metrics.OptimizerEmbeddingVectorsPerSecond")
class OptimizerEmbeddingVectorsPerSecond(EmbeddingModelOptimizersOperationalMetric):
    """Vectors produced per second during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.OptimizerEmbeddingVectorsPerSecond(),
        ],
    )
    ```
    """

    def __init__(self, name="optimizer_embedding_vectors_per_second"):
        super().__init__(name=name)

    def result(self):
        wall = self._wall_clock_delta()
        if wall <= 0.0:
            return 0.0
        return self._delta("vectors") / wall


@synalinks_export("synalinks.metrics.AvgOptimizerEmbeddingTokensPerCall")
class AvgOptimizerEmbeddingTokensPerCall(EmbeddingModelOptimizersOperationalMetric):
    """Average tokens per embedding call during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgOptimizerEmbeddingTokensPerCall(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_optimizer_embedding_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("tokens") / calls


@synalinks_export("synalinks.metrics.AvgOptimizerEmbeddingVectorsPerCall")
class AvgOptimizerEmbeddingVectorsPerCall(EmbeddingModelOptimizersOperationalMetric):
    """Average batch size of embedding calls during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgOptimizerEmbeddingVectorsPerCall(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_optimizer_embedding_vectors_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("vectors") / calls


@synalinks_export("synalinks.metrics.AvgOptimizerEmbeddingCostPerCall")
class AvgOptimizerEmbeddingCostPerCall(EmbeddingModelOptimizersOperationalMetric):
    """Average embedding-call cost during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgOptimizerEmbeddingCostPerCall(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_optimizer_embedding_cost_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("cost") / calls


@synalinks_export("synalinks.metrics.AvgOptimizerEmbeddingCachedTokensPerCall")
class AvgOptimizerEmbeddingCachedTokensPerCall(
    EmbeddingModelOptimizersOperationalMetric
):
    """Average cached prompt tokens per embedding call during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.AvgOptimizerEmbeddingCachedTokensPerCall(),
        ],
    )
    ```
    """

    def __init__(self, name="avg_optimizer_embedding_cached_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("cached_tokens") / calls


@synalinks_export("synalinks.metrics.OptimizerEmbeddingCachedTokens")
class OptimizerEmbeddingCachedTokens(EmbeddingModelOptimizersOperationalMetric):
    """Prompt tokens served from cache during optimizer-phase embeddings.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.OptimizerEmbeddingCachedTokens(),
        ],
    )
    ```
    """

    def __init__(self, name="optimizer_embedding_cached_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("cached_tokens"))


@synalinks_export("synalinks.metrics.OptimizerEmbeddingCacheHitRate")
class OptimizerEmbeddingCacheHitRate(EmbeddingModelOptimizersOperationalMetric):
    """Cache hit rate for optimizer-phase embedding inputs.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.OptimizerEmbeddingCacheHitRate(),
        ],
    )
    ```
    """

    def __init__(self, name="optimizer_embedding_cache_hit_rate"):
        super().__init__(name=name)

    def result(self):
        prompt = self._delta("prompt_tokens")
        if prompt <= 0:
            return 0.0
        return self._delta("cached_tokens") / prompt


@synalinks_export("synalinks.metrics.OptimizerEmbeddingFailedCalls")
class OptimizerEmbeddingFailedCalls(EmbeddingModelOptimizersOperationalMetric):
    """Embedding calls that failed during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.OptimizerEmbeddingFailedCalls(),
        ],
    )
    ```
    """

    def __init__(self, name="optimizer_embedding_failed_calls"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("failed_calls"))


@synalinks_export("synalinks.metrics.OptimizerEmbeddingFallbackActivations")
class OptimizerEmbeddingFallbackActivations(EmbeddingModelOptimizersOperationalMetric):
    """Embedding fallback activations during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.OptimizerEmbeddingFallbackActivations(),
        ],
    )
    ```
    """

    def __init__(self, name="optimizer_embedding_fallback_activations"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("fallback_activations"))


@synalinks_export("synalinks.metrics.OptimizerEmbeddingErrorRate")
class OptimizerEmbeddingErrorRate(EmbeddingModelOptimizersOperationalMetric):
    """Fraction of embedding calls that failed during the optimizer step.

    Example:

    ```python
    program.compile(
        metrics=[
            synalinks.metrics.OptimizerEmbeddingErrorRate(),
        ],
    )
    ```
    """

    def __init__(self, name="optimizer_embedding_error_rate"):
        super().__init__(name=name)

    def result(self):
        failed = self._delta("failed_calls")
        total = self._delta("calls") + failed
        if total <= 0:
            return 0.0
        return failed / total
