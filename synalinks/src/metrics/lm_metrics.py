# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Operational metrics for `LanguageModel` runtime counters.

All metrics in this file read counters that the LM populates on each
provider call. Routing across `inference`, `reward`, and `optimizer`
phases follows the `synalinks_op_scope` global flag set by the trainer.

Class hierarchy:

    LMOperationalMetric         (base, _phase = "inference")
    ├── LMRewardsOperationalMetric     (_phase = "reward")
    └── LMOptimizersOperationalMetric  (_phase = "optimizer")
"""

from synalinks.src.api_export import synalinks_export
from synalinks.src.metrics.metric import Metric

_TRACKED_SUFFIXES = (
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "tokens",
    "elapsed_s",
    "cost",
    "cached_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
)


def _collect_language_models(program):
    """Walk a program's module tree and return every unique LanguageModel
    (including those reached via the `fallback` chain), preserving order.
    """
    from synalinks.src.modules.language_models import LanguageModel

    lms = []
    seen = set()

    def _add_chain(lm):
        while lm is not None and id(lm) not in seen:
            if isinstance(lm, LanguageModel):
                seen.add(id(lm))
                lms.append(lm)
                lm = getattr(lm, "fallback", None)
            else:
                break

    modules = []
    if hasattr(program, "_flatten_modules"):
        modules = program._flatten_modules(include_self=True, recursive=True)
    for module in modules:
        _add_chain(getattr(module, "language_model", None))
        if isinstance(module, LanguageModel):
            _add_chain(module)
    return lms


# ---------------------------------------------------------------------------
# Inference-phase base + metrics
# ---------------------------------------------------------------------------


@synalinks_export(
    [
        "synalinks.metrics.LMOperationalMetric",
        "synalinks.LMOperationalMetric",
    ]
)
class LMOperationalMetric(Metric):
    """Base class for `LanguageModel` runtime-counter metrics.

    Subclasses set `_phase` to one of ``"inference"``, ``"reward"``, or
    ``"optimizer"`` to read from the corresponding counter set on each
    bound LM. Counters are populated by the LM based on the
    ``synalinks_op_scope`` global flag set by the trainer.

    The metric binds itself automatically to every `LanguageModel`
    reachable from the program (and their `.fallback` chains) when
    `program.compile()` is called, and counters are summed across all.
    """

    _phase = "inference"

    def __init__(self, name=None):
        super().__init__(name=name)
        self._language_models = []
        self._baselines = {suffix: 0 for suffix in _TRACKED_SUFFIXES}

    @property
    def language_models(self):
        return list(self._language_models)

    def bind_program(self, program):
        self._language_models = _collect_language_models(program)
        self._snapshot()

    def _attr(self, suffix):
        return f"{self._phase}_cumulated_{suffix}"

    def _read(self, suffix):
        attr = self._attr(suffix)
        return sum(getattr(lm, attr, 0) for lm in self._language_models)

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


@synalinks_export("synalinks.metrics.InputTokens")
class InputTokens(LMOperationalMetric):
    """Cumulated input (prompt) tokens consumed during this run."""

    def __init__(self, name="input_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("prompt_tokens"))


@synalinks_export("synalinks.metrics.OutputTokens")
class OutputTokens(LMOperationalMetric):
    """Cumulated output (completion) tokens generated during this run."""

    def __init__(self, name="output_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("completion_tokens"))


@synalinks_export("synalinks.metrics.TotalTokens")
class TotalTokens(LMOperationalMetric):
    """Cumulated total tokens (input + output) for this run."""

    def __init__(self, name="total_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("tokens"))


@synalinks_export("synalinks.metrics.AvgInputTokensPerCall")
class AvgInputTokensPerCall(LMOperationalMetric):
    """Average input tokens per LM call over this run."""

    def __init__(self, name="avg_input_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("prompt_tokens") / calls


@synalinks_export("synalinks.metrics.AvgOutputTokensPerCall")
class AvgOutputTokensPerCall(LMOperationalMetric):
    """Average output tokens per LM call over this run."""

    def __init__(self, name="avg_output_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("completion_tokens") / calls


@synalinks_export("synalinks.metrics.TokensPerSecond")
class TokensPerSecond(LMOperationalMetric):
    """Throughput in total tokens per second over this run."""

    def __init__(self, name="tokens_per_second"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("tokens") / elapsed


@synalinks_export("synalinks.metrics.Throughput")
class Throughput(LMOperationalMetric):
    """Throughput in LM calls per second (RPS) over this run."""

    def __init__(self, name="throughput"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("calls") / elapsed


@synalinks_export("synalinks.metrics.Cost")
class Cost(LMOperationalMetric):
    """Cumulated provider cost (USD, as reported by litellm) for this run."""

    def __init__(self, name="cost"):
        super().__init__(name=name)

    def result(self):
        return float(self._delta("cost"))


@synalinks_export("synalinks.metrics.AvgCostPerCall")
class AvgCostPerCall(LMOperationalMetric):
    """Average provider cost per LM call over this run."""

    def __init__(self, name="avg_cost_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("cost") / calls


@synalinks_export("synalinks.metrics.CachedTokens")
class CachedTokens(LMOperationalMetric):
    """Prompt tokens served from provider-side prompt cache during this run.

    For Anthropic this is reported as `cache_read_input_tokens`; for OpenAI
    as `cached_tokens`. LiteLLM normalizes both into
    `usage.prompt_tokens_details.cached_tokens`.
    """

    def __init__(self, name="cached_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("cached_tokens"))


@synalinks_export("synalinks.metrics.CacheCreationTokens")
class CacheCreationTokens(LMOperationalMetric):
    """Tokens written to the prompt cache during this run (Anthropic
    `cache_creation_input_tokens`; you pay a higher rate for these).
    """

    def __init__(self, name="cache_creation_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("cache_creation_tokens"))


@synalinks_export("synalinks.metrics.CacheHitRate")
class CacheHitRate(LMOperationalMetric):
    """Fraction of prompt tokens served from cache: cached / prompt_tokens.

    A high value here is one of the biggest cost levers; aim for 0.7+
    on a production workload with stable system prompts.
    """

    def __init__(self, name="cache_hit_rate"):
        super().__init__(name=name)

    def result(self):
        prompt = self._delta("prompt_tokens")
        if prompt <= 0:
            return 0.0
        return self._delta("cached_tokens") / prompt


@synalinks_export("synalinks.metrics.ReasoningTokens")
class ReasoningTokens(LMOperationalMetric):
    """Reasoning/thinking tokens produced during this run (Claude
    extended thinking, OpenAI o-series). Not included in the visible
    completion content but billed as output tokens.
    """

    def __init__(self, name="reasoning_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("reasoning_tokens"))


@synalinks_export("synalinks.metrics.ReasoningTokenShare")
class ReasoningTokenShare(LMOperationalMetric):
    """Fraction of completion tokens spent on reasoning:
    reasoning_tokens / completion_tokens. Signals whether a thinking
    model is actually thinking on the workload.
    """

    def __init__(self, name="reasoning_token_share"):
        super().__init__(name=name)

    def result(self):
        completion = self._delta("completion_tokens")
        if completion <= 0:
            return 0.0
        return self._delta("reasoning_tokens") / completion


# ---------------------------------------------------------------------------
# Reward-phase metrics
# ---------------------------------------------------------------------------


@synalinks_export(
    [
        "synalinks.metrics.LMRewardsOperationalMetric",
        "synalinks.LMRewardsOperationalMetric",
    ]
)
class LMRewardsOperationalMetric(LMOperationalMetric):
    """Base for LM metrics scoped to the reward-computation phase.

    Reads from each bound LM's `reward_cumulated_*` counters, which the
    LM populates while `Trainer.compute_reward` is running.
    """

    _phase = "reward"


@synalinks_export("synalinks.metrics.RewardInputTokens")
class RewardInputTokens(LMRewardsOperationalMetric):
    """Input (prompt) tokens consumed by LM calls during reward computation."""

    def __init__(self, name="reward_input_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("prompt_tokens"))


@synalinks_export("synalinks.metrics.RewardOutputTokens")
class RewardOutputTokens(LMRewardsOperationalMetric):
    """Output tokens generated by LM calls during reward computation."""

    def __init__(self, name="reward_output_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("completion_tokens"))


@synalinks_export("synalinks.metrics.RewardTotalTokens")
class RewardTotalTokens(LMRewardsOperationalMetric):
    """Total tokens consumed by LM calls during reward computation."""

    def __init__(self, name="reward_total_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("tokens"))


@synalinks_export("synalinks.metrics.AvgRewardInputTokensPerCall")
class AvgRewardInputTokensPerCall(LMRewardsOperationalMetric):
    """Average input tokens per LM call during reward computation."""

    def __init__(self, name="avg_reward_input_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("prompt_tokens") / calls


@synalinks_export("synalinks.metrics.AvgRewardOutputTokensPerCall")
class AvgRewardOutputTokensPerCall(LMRewardsOperationalMetric):
    """Average output tokens per LM call during reward computation."""

    def __init__(self, name="avg_reward_output_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("completion_tokens") / calls


@synalinks_export("synalinks.metrics.RewardTokensPerSecond")
class RewardTokensPerSecond(LMRewardsOperationalMetric):
    """Throughput in tokens per second during reward computation."""

    def __init__(self, name="reward_tokens_per_second"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("tokens") / elapsed


@synalinks_export("synalinks.metrics.RewardThroughput")
class RewardThroughput(LMRewardsOperationalMetric):
    """LM calls per second (RPS) during reward computation."""

    def __init__(self, name="reward_throughput"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("calls") / elapsed


@synalinks_export("synalinks.metrics.RewardCost")
class RewardCost(LMRewardsOperationalMetric):
    """Provider cost (USD) of LM calls during reward computation."""

    def __init__(self, name="reward_cost"):
        super().__init__(name=name)

    def result(self):
        return float(self._delta("cost"))


@synalinks_export("synalinks.metrics.AvgRewardCostPerCall")
class AvgRewardCostPerCall(LMRewardsOperationalMetric):
    """Average LM-call cost during reward computation."""

    def __init__(self, name="avg_reward_cost_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("cost") / calls


@synalinks_export("synalinks.metrics.RewardCachedTokens")
class RewardCachedTokens(LMRewardsOperationalMetric):
    """Prompt tokens served from cache during reward computation."""

    def __init__(self, name="reward_cached_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("cached_tokens"))


@synalinks_export("synalinks.metrics.RewardCacheCreationTokens")
class RewardCacheCreationTokens(LMRewardsOperationalMetric):
    """Tokens written to the prompt cache during reward computation."""

    def __init__(self, name="reward_cache_creation_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("cache_creation_tokens"))


@synalinks_export("synalinks.metrics.RewardCacheHitRate")
class RewardCacheHitRate(LMRewardsOperationalMetric):
    """Prompt cache hit rate during reward computation."""

    def __init__(self, name="reward_cache_hit_rate"):
        super().__init__(name=name)

    def result(self):
        prompt = self._delta("prompt_tokens")
        if prompt <= 0:
            return 0.0
        return self._delta("cached_tokens") / prompt


@synalinks_export("synalinks.metrics.RewardReasoningTokens")
class RewardReasoningTokens(LMRewardsOperationalMetric):
    """Reasoning tokens produced during reward computation."""

    def __init__(self, name="reward_reasoning_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("reasoning_tokens"))


@synalinks_export("synalinks.metrics.RewardReasoningTokenShare")
class RewardReasoningTokenShare(LMRewardsOperationalMetric):
    """Reasoning share of completion tokens during reward computation."""

    def __init__(self, name="reward_reasoning_token_share"):
        super().__init__(name=name)

    def result(self):
        completion = self._delta("completion_tokens")
        if completion <= 0:
            return 0.0
        return self._delta("reasoning_tokens") / completion


# ---------------------------------------------------------------------------
# Optimizer-phase metrics
# ---------------------------------------------------------------------------


@synalinks_export(
    [
        "synalinks.metrics.LMOptimizersOperationalMetric",
        "synalinks.LMOptimizersOperationalMetric",
    ]
)
class LMOptimizersOperationalMetric(LMOperationalMetric):
    """Base for LM metrics scoped to the optimizer phase.

    Reads from each bound LM's `optimizer_cumulated_*` counters, which
    the LM populates while `Optimizer.optimize` is running (but not
    while nested reward computation is in progress — those go to the
    rewards bucket).
    """

    _phase = "optimizer"


@synalinks_export("synalinks.metrics.OptimizerInputTokens")
class OptimizerInputTokens(LMOptimizersOperationalMetric):
    """Input (prompt) tokens consumed by LM calls during the optimizer step."""

    def __init__(self, name="optimizer_input_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("prompt_tokens"))


@synalinks_export("synalinks.metrics.OptimizerOutputTokens")
class OptimizerOutputTokens(LMOptimizersOperationalMetric):
    """Output tokens generated by LM calls during the optimizer step."""

    def __init__(self, name="optimizer_output_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("completion_tokens"))


@synalinks_export("synalinks.metrics.OptimizerTotalTokens")
class OptimizerTotalTokens(LMOptimizersOperationalMetric):
    """Total tokens consumed by LM calls during the optimizer step."""

    def __init__(self, name="optimizer_total_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("tokens"))


@synalinks_export("synalinks.metrics.AvgOptimizerInputTokensPerCall")
class AvgOptimizerInputTokensPerCall(LMOptimizersOperationalMetric):
    """Average input tokens per LM call during the optimizer step."""

    def __init__(self, name="avg_optimizer_input_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("prompt_tokens") / calls


@synalinks_export("synalinks.metrics.AvgOptimizerOutputTokensPerCall")
class AvgOptimizerOutputTokensPerCall(LMOptimizersOperationalMetric):
    """Average output tokens per LM call during the optimizer step."""

    def __init__(self, name="avg_optimizer_output_tokens_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("completion_tokens") / calls


@synalinks_export("synalinks.metrics.OptimizerTokensPerSecond")
class OptimizerTokensPerSecond(LMOptimizersOperationalMetric):
    """Throughput in tokens per second during the optimizer step."""

    def __init__(self, name="optimizer_tokens_per_second"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("tokens") / elapsed


@synalinks_export("synalinks.metrics.OptimizerThroughput")
class OptimizerThroughput(LMOptimizersOperationalMetric):
    """LM calls per second (RPS) during the optimizer step."""

    def __init__(self, name="optimizer_throughput"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta("elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta("calls") / elapsed


@synalinks_export("synalinks.metrics.OptimizerCost")
class OptimizerCost(LMOptimizersOperationalMetric):
    """Provider cost (USD) of LM calls during the optimizer step."""

    def __init__(self, name="optimizer_cost"):
        super().__init__(name=name)

    def result(self):
        return float(self._delta("cost"))


@synalinks_export("synalinks.metrics.AvgOptimizerCostPerCall")
class AvgOptimizerCostPerCall(LMOptimizersOperationalMetric):
    """Average LM-call cost during the optimizer step."""

    def __init__(self, name="avg_optimizer_cost_per_call"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta("calls")
        if calls <= 0:
            return 0.0
        return self._delta("cost") / calls


@synalinks_export("synalinks.metrics.OptimizerCachedTokens")
class OptimizerCachedTokens(LMOptimizersOperationalMetric):
    """Prompt tokens served from cache during the optimizer step."""

    def __init__(self, name="optimizer_cached_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("cached_tokens"))


@synalinks_export("synalinks.metrics.OptimizerCacheCreationTokens")
class OptimizerCacheCreationTokens(LMOptimizersOperationalMetric):
    """Tokens written to the prompt cache during the optimizer step."""

    def __init__(self, name="optimizer_cache_creation_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("cache_creation_tokens"))


@synalinks_export("synalinks.metrics.OptimizerCacheHitRate")
class OptimizerCacheHitRate(LMOptimizersOperationalMetric):
    """Prompt cache hit rate during the optimizer step."""

    def __init__(self, name="optimizer_cache_hit_rate"):
        super().__init__(name=name)

    def result(self):
        prompt = self._delta("prompt_tokens")
        if prompt <= 0:
            return 0.0
        return self._delta("cached_tokens") / prompt


@synalinks_export("synalinks.metrics.OptimizerReasoningTokens")
class OptimizerReasoningTokens(LMOptimizersOperationalMetric):
    """Reasoning tokens produced during the optimizer step."""

    def __init__(self, name="optimizer_reasoning_tokens"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta("reasoning_tokens"))


@synalinks_export("synalinks.metrics.OptimizerReasoningTokenShare")
class OptimizerReasoningTokenShare(LMOptimizersOperationalMetric):
    """Reasoning share of completion tokens during the optimizer step."""

    def __init__(self, name="optimizer_reasoning_token_share"):
        super().__init__(name=name)

    def result(self):
        completion = self._delta("completion_tokens")
        if completion <= 0:
            return 0.0
        return self._delta("reasoning_tokens") / completion
