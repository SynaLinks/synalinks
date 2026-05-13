# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Operational metrics for `EmbeddingModel` runtime counters.

All metrics read counters that the EM populates on each provider call.
Routing across `inference`, `reward`, and `optimizer` phases follows the
`synalinks_op_scope` global flag set by the trainer.

Class hierarchy:

    EmbeddingModelOperationalMetric                  (base, "inference")
    ├── EmbeddingModelRewardsOperationalMetric             ("reward")
    └── EmbeddingModelOptimizersOperationalMetric          ("optimizer")
"""

from synalinks.src.api_export import synalinks_export
from synalinks.src.metrics.metric import Metric

_TRACKED_SUFFIXES = (
    "calls",
    "prompt_tokens",
    "tokens",
    "vectors",
    "elapsed_s",
    "cost",
    "cached_tokens",
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
    embedding model. Counters are populated by the EM based on the
    ``synalinks_op_scope`` global flag set by the trainer.

    Binds itself automatically to every `EmbeddingModel` reachable from
    the program (and their `.fallback` chains) on `program.compile()`.
    """

    _phase = "inference"

    def __init__(self, name=None):
        super().__init__(name=name)
        self._embedding_models = []
        self._baselines = {suffix: 0 for suffix in _TRACKED_SUFFIXES}

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

    def _delta(self, suffix):
        return self._read(suffix) - self._baselines.get(suffix, 0)

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
    """Cumulated tokens consumed by embedding calls during this run."""

    def __init__(self, name="embedding_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("tokens"))


@synalinks_export("synalinks.metrics.EmbeddingVectors")
class EmbeddingVectors(EmbeddingModelOperationalMetric):
    """Cumulated vectors produced by embedding calls during this run."""

    def __init__(self, name="embedding_vectors"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("vectors"))


@synalinks_export("synalinks.metrics.EmbeddingCost")
class EmbeddingCost(EmbeddingModelOperationalMetric):
    """Cumulated embedding-provider cost (USD) for this run."""

    def __init__(self, name="embedding_cost"):
        super().__init__(name=name)

    def result(self):
        return float(self._delta("cost"))


@synalinks_export("synalinks.metrics.EmbeddingThroughput")
class EmbeddingThroughput(EmbeddingModelOperationalMetric):
    """Embedding calls per second (RPS) over this run."""

    def __init__(self, name="embedding_throughput"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("calls") / elapsed


@synalinks_export("synalinks.metrics.EmbeddingTokensPerSecond")
class EmbeddingTokensPerSecond(EmbeddingModelOperationalMetric):
    """Embedded tokens per second over this run."""

    def __init__(self, name="embedding_tokens_per_second"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("tokens") / elapsed


@synalinks_export("synalinks.metrics.EmbeddingVectorsPerSecond")
class EmbeddingVectorsPerSecond(EmbeddingModelOperationalMetric):
    """Vectors produced per second over this run."""

    def __init__(self, name="embedding_vectors_per_second"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("vectors") / elapsed


@synalinks_export("synalinks.metrics.AvgEmbeddingTokensPerCall")
class AvgEmbeddingTokensPerCall(EmbeddingModelOperationalMetric):
    """Average tokens per embedding call over this run."""

    def __init__(self, name="avg_embedding_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("tokens") / calls


@synalinks_export("synalinks.metrics.AvgEmbeddingVectorsPerCall")
class AvgEmbeddingVectorsPerCall(EmbeddingModelOperationalMetric):
    """Average vectors (i.e. batch size) per embedding call over this run."""

    def __init__(self, name="avg_embedding_vectors_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("vectors") / calls


@synalinks_export("synalinks.metrics.AvgEmbeddingCostPerCall")
class AvgEmbeddingCostPerCall(EmbeddingModelOperationalMetric):
    """Average embedding-provider cost per call over this run."""

    def __init__(self, name="avg_embedding_cost_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("cost") / calls


@synalinks_export("synalinks.metrics.EmbeddingCachedTokens")
class EmbeddingCachedTokens(EmbeddingModelOperationalMetric):
    """Prompt tokens served from cache during embedding inference."""

    def __init__(self, name="embedding_cached_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("cached_tokens"))


@synalinks_export("synalinks.metrics.EmbeddingCacheHitRate")
class EmbeddingCacheHitRate(EmbeddingModelOperationalMetric):
    """Cache hit rate for embedding inputs: cached_tokens / prompt_tokens."""

    def __init__(self, name="embedding_cache_hit_rate"):
        super().__init__(name=name)

    def result(self):
        prompt = self._delta("prompt_tokens")
        if prompt <= 0:
            return 0.0
        return self._delta("cached_tokens") / prompt


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
    """Base for embedding metrics scoped to the reward-computation phase."""

    _phase = "reward"


@synalinks_export("synalinks.metrics.RewardEmbeddingTokens")
class RewardEmbeddingTokens(EmbeddingModelRewardsOperationalMetric):
    """Tokens consumed by embedding calls during reward computation."""

    def __init__(self, name="reward_embedding_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("tokens"))


@synalinks_export("synalinks.metrics.RewardEmbeddingVectors")
class RewardEmbeddingVectors(EmbeddingModelRewardsOperationalMetric):
    """Vectors produced by embedding calls during reward computation."""

    def __init__(self, name="reward_embedding_vectors"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("vectors"))


@synalinks_export("synalinks.metrics.RewardEmbeddingCost")
class RewardEmbeddingCost(EmbeddingModelRewardsOperationalMetric):
    """Provider cost of embedding calls during reward computation."""

    def __init__(self, name="reward_embedding_cost"):
        super().__init__(name=name)

    def result(self):
        return float(self._delta("cost"))


@synalinks_export("synalinks.metrics.RewardEmbeddingThroughput")
class RewardEmbeddingThroughput(EmbeddingModelRewardsOperationalMetric):
    """Embedding calls per second during reward computation."""

    def __init__(self, name="reward_embedding_throughput"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("calls") / elapsed


@synalinks_export("synalinks.metrics.RewardEmbeddingTokensPerSecond")
class RewardEmbeddingTokensPerSecond(EmbeddingModelRewardsOperationalMetric):
    """Embedded tokens per second during reward computation."""

    def __init__(self, name="reward_embedding_tokens_per_second"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("tokens") / elapsed


@synalinks_export("synalinks.metrics.RewardEmbeddingVectorsPerSecond")
class RewardEmbeddingVectorsPerSecond(EmbeddingModelRewardsOperationalMetric):
    """Vectors produced per second during reward computation."""

    def __init__(self, name="reward_embedding_vectors_per_second"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("vectors") / elapsed


@synalinks_export("synalinks.metrics.AvgRewardEmbeddingTokensPerCall")
class AvgRewardEmbeddingTokensPerCall(EmbeddingModelRewardsOperationalMetric):
    """Average tokens per embedding call during reward computation."""

    def __init__(self, name="avg_reward_embedding_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("tokens") / calls


@synalinks_export("synalinks.metrics.AvgRewardEmbeddingVectorsPerCall")
class AvgRewardEmbeddingVectorsPerCall(EmbeddingModelRewardsOperationalMetric):
    """Average batch size of embedding calls during reward computation."""

    def __init__(self, name="avg_reward_embedding_vectors_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("vectors") / calls


@synalinks_export("synalinks.metrics.AvgRewardEmbeddingCostPerCall")
class AvgRewardEmbeddingCostPerCall(EmbeddingModelRewardsOperationalMetric):
    """Average embedding-call cost during reward computation."""

    def __init__(self, name="avg_reward_embedding_cost_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("cost") / calls


@synalinks_export("synalinks.metrics.RewardEmbeddingCachedTokens")
class RewardEmbeddingCachedTokens(EmbeddingModelRewardsOperationalMetric):
    """Prompt tokens served from cache during reward-phase embeddings."""

    def __init__(self, name="reward_embedding_cached_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("cached_tokens"))


@synalinks_export("synalinks.metrics.RewardEmbeddingCacheHitRate")
class RewardEmbeddingCacheHitRate(EmbeddingModelRewardsOperationalMetric):
    """Cache hit rate for reward-phase embedding inputs."""

    def __init__(self, name="reward_embedding_cache_hit_rate"):
        super().__init__(name=name)

    def result(self):
        prompt = self._delta("prompt_tokens")
        if prompt <= 0:
            return 0.0
        return self._delta("cached_tokens") / prompt


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
    """Base for embedding metrics scoped to the optimizer phase."""

    _phase = "optimizer"


@synalinks_export("synalinks.metrics.OptimizerEmbeddingTokens")
class OptimizerEmbeddingTokens(EmbeddingModelOptimizersOperationalMetric):
    """Tokens consumed by embedding calls during the optimizer step."""

    def __init__(self, name="optimizer_embedding_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("tokens"))


@synalinks_export("synalinks.metrics.OptimizerEmbeddingVectors")
class OptimizerEmbeddingVectors(EmbeddingModelOptimizersOperationalMetric):
    """Vectors produced by embedding calls during the optimizer step."""

    def __init__(self, name="optimizer_embedding_vectors"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("vectors"))


@synalinks_export("synalinks.metrics.OptimizerEmbeddingCost")
class OptimizerEmbeddingCost(EmbeddingModelOptimizersOperationalMetric):
    """Provider cost of embedding calls during the optimizer step."""

    def __init__(self, name="optimizer_embedding_cost"):
        super().__init__(name=name)

    def result(self):
        return float(self._delta("cost"))


@synalinks_export("synalinks.metrics.OptimizerEmbeddingThroughput")
class OptimizerEmbeddingThroughput(EmbeddingModelOptimizersOperationalMetric):
    """Embedding calls per second during the optimizer step."""

    def __init__(self, name="optimizer_embedding_throughput"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("calls") / elapsed


@synalinks_export("synalinks.metrics.OptimizerEmbeddingTokensPerSecond")
class OptimizerEmbeddingTokensPerSecond(EmbeddingModelOptimizersOperationalMetric):
    """Embedded tokens per second during the optimizer step."""

    def __init__(self, name="optimizer_embedding_tokens_per_second"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("tokens") / elapsed


@synalinks_export("synalinks.metrics.OptimizerEmbeddingVectorsPerSecond")
class OptimizerEmbeddingVectorsPerSecond(EmbeddingModelOptimizersOperationalMetric):
    """Vectors produced per second during the optimizer step."""

    def __init__(self, name="optimizer_embedding_vectors_per_second"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("vectors") / elapsed


@synalinks_export("synalinks.metrics.AvgOptimizerEmbeddingTokensPerCall")
class AvgOptimizerEmbeddingTokensPerCall(EmbeddingModelOptimizersOperationalMetric):
    """Average tokens per embedding call during the optimizer step."""

    def __init__(self, name="avg_optimizer_embedding_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("tokens") / calls


@synalinks_export("synalinks.metrics.AvgOptimizerEmbeddingVectorsPerCall")
class AvgOptimizerEmbeddingVectorsPerCall(EmbeddingModelOptimizersOperationalMetric):
    """Average batch size of embedding calls during the optimizer step."""

    def __init__(self, name="avg_optimizer_embedding_vectors_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("vectors") / calls


@synalinks_export("synalinks.metrics.AvgOptimizerEmbeddingCostPerCall")
class AvgOptimizerEmbeddingCostPerCall(EmbeddingModelOptimizersOperationalMetric):
    """Average embedding-call cost during the optimizer step."""

    def __init__(self, name="avg_optimizer_embedding_cost_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("cost") / calls


@synalinks_export("synalinks.metrics.OptimizerEmbeddingCachedTokens")
class OptimizerEmbeddingCachedTokens(EmbeddingModelOptimizersOperationalMetric):
    """Prompt tokens served from cache during optimizer-phase embeddings."""

    def __init__(self, name="optimizer_embedding_cached_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("cached_tokens"))


@synalinks_export("synalinks.metrics.OptimizerEmbeddingCacheHitRate")
class OptimizerEmbeddingCacheHitRate(EmbeddingModelOptimizersOperationalMetric):
    """Cache hit rate for optimizer-phase embedding inputs."""

    def __init__(self, name="optimizer_embedding_cache_hit_rate"):
        super().__init__(name=name)

    def result(self):
        prompt = self._delta("prompt_tokens")
        if prompt <= 0:
            return 0.0
        return self._delta("cached_tokens") / prompt
