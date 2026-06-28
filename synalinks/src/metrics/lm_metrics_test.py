# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import asyncio
from unittest import mock

from synalinks.src import testing
from synalinks.src.backend.common.global_state import clear_session
from synalinks.src.backend.common.op_scope import _add_phase_wall_clock_s
from synalinks.src.backend.common.op_scope import current_op_scope
from synalinks.src.backend.common.op_scope import op_scope
from synalinks.src.backend.common.op_scope import read_phase_wall_clock_s
from synalinks.src.metrics.lm_metrics import AvgCacheCreationTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgCachedTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgCostPerCall
from synalinks.src.metrics.lm_metrics import AvgInputTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgLatency
from synalinks.src.metrics.lm_metrics import AvgOptimizerCacheCreationTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgOptimizerCachedTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgOptimizerCostPerCall
from synalinks.src.metrics.lm_metrics import AvgOptimizerInputTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgOptimizerLatency
from synalinks.src.metrics.lm_metrics import AvgOptimizerOutputTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgOptimizerReasoningTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgOptimizerTotalTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgOutputTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgReasoningTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardCacheCreationTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardCachedTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardCostPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardInputTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardLatency
from synalinks.src.metrics.lm_metrics import AvgRewardOutputTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardReasoningTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgRewardTotalTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgTimeToFirstToken
from synalinks.src.metrics.lm_metrics import AvgTimeToLastToken
from synalinks.src.metrics.lm_metrics import AvgTotalTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgTrajectoryTimeToFirstToken
from synalinks.src.metrics.lm_metrics import CacheCreationTokens
from synalinks.src.metrics.lm_metrics import CachedTokens
from synalinks.src.metrics.lm_metrics import CacheHitRate
from synalinks.src.metrics.lm_metrics import Cost
from synalinks.src.metrics.lm_metrics import ErrorRate
from synalinks.src.metrics.lm_metrics import FailedCalls
from synalinks.src.metrics.lm_metrics import FallbackActivations
from synalinks.src.metrics.lm_metrics import InputTokens
from synalinks.src.metrics.lm_metrics import OptimizerCacheCreationTokens
from synalinks.src.metrics.lm_metrics import OptimizerCachedTokens
from synalinks.src.metrics.lm_metrics import OptimizerCacheHitRate
from synalinks.src.metrics.lm_metrics import OptimizerCost
from synalinks.src.metrics.lm_metrics import OptimizerErrorRate
from synalinks.src.metrics.lm_metrics import OptimizerFailedCalls
from synalinks.src.metrics.lm_metrics import OptimizerFallbackActivations
from synalinks.src.metrics.lm_metrics import OptimizerInputTokens
from synalinks.src.metrics.lm_metrics import OptimizerReasoningTokens
from synalinks.src.metrics.lm_metrics import OptimizerReasoningTokenShare
from synalinks.src.metrics.lm_metrics import OptimizerThroughput
from synalinks.src.metrics.lm_metrics import OptimizerTokensPerSecond
from synalinks.src.metrics.lm_metrics import OptimizerTotalTokens
from synalinks.src.metrics.lm_metrics import OutputTokens
from synalinks.src.metrics.lm_metrics import ReasoningTokens
from synalinks.src.metrics.lm_metrics import ReasoningTokenShare
from synalinks.src.metrics.lm_metrics import RewardCacheCreationTokens
from synalinks.src.metrics.lm_metrics import RewardCachedTokens
from synalinks.src.metrics.lm_metrics import RewardCacheHitRate
from synalinks.src.metrics.lm_metrics import RewardCost
from synalinks.src.metrics.lm_metrics import RewardErrorRate
from synalinks.src.metrics.lm_metrics import RewardFailedCalls
from synalinks.src.metrics.lm_metrics import RewardFallbackActivations
from synalinks.src.metrics.lm_metrics import RewardInputTokens
from synalinks.src.metrics.lm_metrics import RewardReasoningTokens
from synalinks.src.metrics.lm_metrics import RewardReasoningTokenShare
from synalinks.src.metrics.lm_metrics import RewardThroughput
from synalinks.src.metrics.lm_metrics import RewardTokensPerSecond
from synalinks.src.metrics.lm_metrics import RewardTotalTokens
from synalinks.src.metrics.lm_metrics import Throughput
from synalinks.src.metrics.lm_metrics import TokensPerSecond
from synalinks.src.metrics.lm_metrics import TotalTokens
from synalinks.src.metrics.lm_metrics import _collect_language_models
from synalinks.src.modules.language_models import LanguageModel

_LM_SUFFIXES = (
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "tokens",
    "elapsed_s",
    "cost",
)


def _stub_lm(fallback=None):
    lm = LanguageModel(model="ollama/mistral", fallback=fallback)
    for phase in ("inference", "reward", "optimizer"):
        for suffix in _LM_SUFFIXES:
            zero = 0.0 if suffix in ("elapsed_s", "cost") else 0
            setattr(lm, f"{phase}_cumulated_{suffix}", zero)
    return lm


def _record(lm, prompt, completion, elapsed, cost, phase="inference"):
    p = f"{phase}_cumulated_"
    setattr(lm, p + "calls", getattr(lm, p + "calls") + 1)
    setattr(lm, p + "prompt_tokens", getattr(lm, p + "prompt_tokens") + prompt)
    setattr(
        lm, p + "completion_tokens", getattr(lm, p + "completion_tokens") + completion
    )
    setattr(lm, p + "tokens", getattr(lm, p + "tokens") + prompt + completion)
    setattr(lm, p + "elapsed_s", getattr(lm, p + "elapsed_s") + elapsed)
    setattr(lm, p + "cost", getattr(lm, p + "cost") + cost)


class _FakeModule:
    """Minimal stand-in for a Module with a `.language_model` attribute."""

    def __init__(self, lm=None, submodules=()):
        self.language_model = lm
        self._modules = list(submodules)


class _FakeProgram:
    def __init__(self, modules):
        self._modules = list(modules)

    def _flatten_modules(self, include_self=True, recursive=True):
        out = [self] if include_self else []
        stack = list(self._modules)
        while stack:
            m = stack.pop(0)
            out.append(m)
            if recursive:
                stack = list(m._modules) + stack
        return out


def _bind(metric, lms):
    program = _FakeProgram(modules=[_FakeModule(lm=lm) for lm in lms])
    metric.bind_program(program)
    return metric


class OperationalMetricsResultTest(testing.TestCase):
    def test_input_output_total_tokens(self):
        lm = _stub_lm()
        m_in = _bind(InputTokens(), [lm])
        m_out = _bind(OutputTokens(), [lm])
        m_tot = _bind(TotalTokens(), [lm])
        _record(lm, prompt=100, completion=20, elapsed=0.5, cost=0.001)
        _record(lm, prompt=200, completion=50, elapsed=1.0, cost=0.002)
        self.assertEqual(m_in.result(), 300)
        self.assertEqual(m_out.result(), 70)
        self.assertEqual(m_tot.result(), 370)

    def test_averages_per_call(self):
        lm = _stub_lm()
        m_avg_in = _bind(AvgInputTokensPerCall(), [lm])
        m_avg_out = _bind(AvgOutputTokensPerCall(), [lm])
        _record(lm, prompt=100, completion=10, elapsed=0.5, cost=0.001)
        _record(lm, prompt=300, completion=30, elapsed=0.5, cost=0.001)
        self.assertEqual(m_avg_in.result(), 200.0)
        self.assertEqual(m_avg_out.result(), 20.0)

    def test_throughput_and_tokens_per_second(self):
        lm = _stub_lm()
        m_tps = _bind(TokensPerSecond(), [lm])
        m_rps = _bind(Throughput(), [lm])
        # Two calls of 2s each (summed elapsed = 4s) that ran concurrently in
        # a 2s wall-clock window. Throughput divides by wall-clock, so it
        # reflects real RPS (2/2) rather than the deflated 2/4 the old
        # summed-elapsed denominator would give.
        _add_phase_wall_clock_s("inference", 2.0)
        _record(lm, prompt=400, completion=100, elapsed=2.0, cost=0.0)
        _record(lm, prompt=400, completion=100, elapsed=2.0, cost=0.0)
        self.assertEqual(m_tps.result(), 1000 / 2.0)
        self.assertEqual(m_rps.result(), 2 / 2.0)

    def test_avg_latency(self):
        lm = _stub_lm()
        m_lat = _bind(AvgLatency(), [lm])
        _record(lm, prompt=400, completion=100, elapsed=2.0, cost=0.0)
        _record(lm, prompt=400, completion=100, elapsed=3.0, cost=0.0)
        # Mean per-call latency = summed call time / calls = 5.0 / 2
        # (unchanged by the wall-clock throughput switch).
        self.assertEqual(m_lat.result(), 5.0 / 2)

    def test_cost_metrics(self):
        lm = _stub_lm()
        m_cost = _bind(Cost(), [lm])
        m_avg = _bind(AvgCostPerCall(), [lm])
        _record(lm, prompt=10, completion=5, elapsed=0.1, cost=0.003)
        _record(lm, prompt=10, completion=5, elapsed=0.1, cost=0.005)
        self.assertAlmostEqual(m_cost.result(), 0.008)
        self.assertAlmostEqual(m_avg.result(), 0.004)

    def test_total_tokens_average(self):
        lm = _stub_lm()
        m = _bind(AvgTotalTokensPerCall(), [lm])
        _record(lm, prompt=100, completion=20, elapsed=0.1, cost=0.0)
        _record(lm, prompt=300, completion=80, elapsed=0.1, cost=0.0)
        # total tokens = 120 + 380 = 500 over 2 calls.
        self.assertEqual(m.result(), 250.0)

    def test_extra_token_averages(self):
        lm = _stub_lm_with_extras()
        m_cached = _bind(AvgCachedTokensPerCall(), [lm])
        m_cachecreate = _bind(AvgCacheCreationTokensPerCall(), [lm])
        m_reasoning = _bind(AvgReasoningTokensPerCall(), [lm])
        _record(lm, prompt=100, completion=20, elapsed=0.1, cost=0.0)
        _record(lm, prompt=100, completion=20, elapsed=0.1, cost=0.0)
        _bump(lm, "cached_tokens", 60, phase="inference")
        _bump(lm, "cache_creation_tokens", 40, phase="inference")
        _bump(lm, "reasoning_tokens", 30, phase="inference")
        self.assertEqual(m_cached.result(), 30.0)  # 60 / 2
        self.assertEqual(m_cachecreate.result(), 20.0)  # 40 / 2
        self.assertEqual(m_reasoning.result(), 15.0)  # 30 / 2

    def test_failure_metrics(self):
        lm = _stub_lm()
        m_failed = _bind(FailedCalls(), [lm])
        m_fallback = _bind(FallbackActivations(), [lm])
        m_error = _bind(ErrorRate(), [lm])
        # 3 successful calls; 1 failed call that fell back.
        _record(lm, prompt=10, completion=5, elapsed=0.1, cost=0.0)
        _record(lm, prompt=10, completion=5, elapsed=0.1, cost=0.0)
        _record(lm, prompt=10, completion=5, elapsed=0.1, cost=0.0)
        _bump(lm, "failed_calls", 1, phase="inference")
        _bump(lm, "fallback_activations", 1, phase="inference")
        self.assertEqual(m_failed.result(), 1)
        self.assertEqual(m_fallback.result(), 1)
        self.assertEqual(m_error.result(), 1 / 4)  # failed / (succeeded + failed)

    def test_reset_state_snapshots_baseline(self):
        lm = _stub_lm()
        m = _bind(TotalTokens(), [lm])
        _record(lm, prompt=100, completion=50, elapsed=1.0, cost=0.002)
        self.assertEqual(m.result(), 150)
        m.reset_state()
        self.assertEqual(m.result(), 0)
        _record(lm, prompt=10, completion=5, elapsed=0.1, cost=0.0)
        self.assertEqual(m.result(), 15)

    def test_zero_division_safety(self):
        lm = _stub_lm()
        m_avg_in = _bind(AvgInputTokensPerCall(), [lm])
        m_tps = _bind(TokensPerSecond(), [lm])
        m_rps = _bind(Throughput(), [lm])
        m_lat = _bind(AvgLatency(), [lm])
        m_avg_total = _bind(AvgTotalTokensPerCall(), [lm])
        m_error = _bind(ErrorRate(), [lm])
        self.assertEqual(m_avg_in.result(), 0.0)
        self.assertEqual(m_tps.result(), 0.0)
        self.assertEqual(m_rps.result(), 0.0)
        self.assertEqual(m_lat.result(), 0.0)
        self.assertEqual(m_avg_total.result(), 0.0)
        self.assertEqual(m_error.result(), 0.0)

    def test_result_is_zero_before_bind(self):
        # No bind_program() call → no LMs → metric reports 0 instead of crashing.
        self.assertEqual(TotalTokens().result(), 0)
        self.assertEqual(Cost().result(), 0.0)
        self.assertEqual(Throughput().result(), 0.0)


class StreamingLatencyMetricsTest(testing.TestCase):
    """`AvgTimeToFirstToken` / `AvgTimeToLastToken` average the per-call TTFT /
    TTLT the `StreamingIterator` records onto the inference-phase counters."""

    def _record_stream(self, lm, ttft, ttlt, phase="inference"):
        _bump(lm, "streaming_calls", 1, phase=phase)
        _bump(lm, "streaming_ttft_s", ttft, phase=phase)
        _bump(lm, "streaming_ttlt_s", ttlt, phase=phase)

    def test_zero_before_any_stream(self):
        lm = _stub_lm()
        m_ttft = _bind(AvgTimeToFirstToken(), [lm])
        m_ttlt = _bind(AvgTimeToLastToken(), [lm])
        self.assertEqual(m_ttft.result(), 0.0)
        self.assertEqual(m_ttlt.result(), 0.0)

    def test_averages_over_streamed_calls(self):
        lm = _stub_lm()
        m_ttft = _bind(AvgTimeToFirstToken(), [lm])
        m_ttlt = _bind(AvgTimeToLastToken(), [lm])
        # Two streamed calls: TTFT 0.2s / 0.4s, TTLT 1.0s / 2.0s.
        self._record_stream(lm, ttft=0.2, ttlt=1.0)
        self._record_stream(lm, ttft=0.4, ttlt=2.0)
        self.assertAlmostEqual(m_ttft.result(), 0.3)  # (0.2 + 0.4) / 2
        self.assertAlmostEqual(m_ttlt.result(), 1.5)  # (1.0 + 2.0) / 2

    def test_only_inference_phase_counts(self):
        # Streaming is inference-only; counters recorded under another phase
        # must not bleed into the inference-scoped streaming metrics.
        lm = _stub_lm()
        m_ttft = _bind(AvgTimeToFirstToken(), [lm])
        self._record_stream(lm, ttft=9.0, ttlt=9.0, phase="reward")
        self.assertEqual(m_ttft.result(), 0.0)

    def test_trajectory_ttft_averages_over_trajectory_calls(self):
        # Whole-trajectory TTFT uses its own (trajectory_calls) denominator and
        # ignores streamed calls that ran outside an agent trajectory.
        lm = _stub_lm()
        m_traj = _bind(AvgTrajectoryTimeToFirstToken(), [lm])
        self.assertEqual(m_traj.result(), 0.0)
        # Two agent trajectories: whole-trajectory TTFT 1.5s / 2.5s.
        _bump(lm, "trajectory_calls", 1, phase="inference")
        _bump(lm, "trajectory_ttft_s", 1.5, phase="inference")
        _bump(lm, "trajectory_calls", 1, phase="inference")
        _bump(lm, "trajectory_ttft_s", 2.5, phase="inference")
        self.assertAlmostEqual(m_traj.result(), 2.0)  # (1.5 + 2.5) / 2


class OperationalMetricsAutoBindTest(testing.TestCase):
    def test_collect_walks_modules_and_fallback(self):
        leaf_lm = _stub_lm()
        primary_lm = _stub_lm(fallback=leaf_lm)
        other_lm = _stub_lm()
        program = _FakeProgram(
            modules=[
                _FakeModule(lm=primary_lm),
                _FakeModule(lm=None, submodules=[_FakeModule(lm=other_lm)]),
            ]
        )
        lms = _collect_language_models(program)
        ids = [id(lm) for lm in lms]
        self.assertIn(id(primary_lm), ids)
        self.assertIn(id(leaf_lm), ids)
        self.assertIn(id(other_lm), ids)
        self.assertEqual(len(set(ids)), 3)

    def test_collect_dedupes_shared_lm(self):
        shared_lm = _stub_lm()
        program = _FakeProgram(
            modules=[_FakeModule(lm=shared_lm), _FakeModule(lm=shared_lm)]
        )
        lms = _collect_language_models(program)
        self.assertEqual(len(lms), 1)
        self.assertIs(lms[0], shared_lm)

    def test_bind_program_aggregates_across_modules(self):
        lm_a = _stub_lm()
        lm_b = _stub_lm()
        m = _bind(TotalTokens(), [lm_a, lm_b])
        _record(lm_a, prompt=10, completion=5, elapsed=0.1, cost=0.0)
        _record(lm_b, prompt=20, completion=5, elapsed=0.1, cost=0.0)
        self.assertEqual(m.result(), 40)

    def test_bind_program_resets_baseline(self):
        # Counters that already moved before bind_program() are baselined out.
        lm = _stub_lm()
        _record(lm, prompt=500, completion=100, elapsed=3.0, cost=0.01)
        m = _bind(TotalTokens(), [lm])
        self.assertEqual(m.result(), 0)
        _record(lm, prompt=10, completion=5, elapsed=0.1, cost=0.0)
        self.assertEqual(m.result(), 15)


class LMPhaseRoutingTest(testing.TestCase):
    """Pins the per-phase counter routing: recording into one phase must
    not bleed into another phase's metric.
    """

    def test_phases_are_independent(self):
        lm = _stub_lm()
        m_inf = _bind(TotalTokens(), [lm])
        m_rew = _bind(RewardTotalTokens(), [lm])
        m_opt = _bind(OptimizerTotalTokens(), [lm])

        _record(lm, prompt=100, completion=20, elapsed=0.0, cost=0.001, phase="inference")
        _record(lm, prompt=50, completion=10, elapsed=0.0, cost=0.0005, phase="reward")
        _record(lm, prompt=200, completion=40, elapsed=0.0, cost=0.002, phase="optimizer")

        self.assertEqual(m_inf.result(), 120)
        self.assertEqual(m_rew.result(), 60)
        self.assertEqual(m_opt.result(), 240)

    def test_reward_input_tokens(self):
        lm = _stub_lm()
        m = _bind(RewardInputTokens(), [lm])
        _record(lm, prompt=300, completion=10, elapsed=0.0, cost=0.0, phase="reward")
        _record(lm, prompt=999, completion=999, elapsed=0.0, cost=0.0, phase="inference")
        self.assertEqual(m.result(), 300)

    def test_optimizer_input_tokens(self):
        lm = _stub_lm()
        m = _bind(OptimizerInputTokens(), [lm])
        _record(lm, prompt=400, completion=10, elapsed=0.0, cost=0.0, phase="optimizer")
        _record(lm, prompt=999, completion=999, elapsed=0.0, cost=0.0, phase="reward")
        self.assertEqual(m.result(), 400)

    def test_reward_cost(self):
        lm = _stub_lm()
        m = _bind(RewardCost(), [lm])
        _record(lm, prompt=10, completion=0, elapsed=0.0, cost=0.003, phase="reward")
        _record(lm, prompt=10, completion=0, elapsed=0.0, cost=0.999, phase="inference")
        self.assertAlmostEqual(m.result(), 0.003)

    def test_optimizer_cost(self):
        lm = _stub_lm()
        m = _bind(OptimizerCost(), [lm])
        _record(lm, prompt=10, completion=0, elapsed=0.0, cost=0.005, phase="optimizer")
        _record(lm, prompt=10, completion=0, elapsed=0.0, cost=0.999, phase="reward")
        self.assertAlmostEqual(m.result(), 0.005)


class OpScopeTest(testing.TestCase):
    """`op_scope` marks a region as inference/reward/optimizer so the LM/EM
    attribute calls to a phase and throughput can use the phase's wall-clock.
    It is contextvars-backed (propagates into concurrent child tasks, unlike a
    threading.local) and accrues per-phase wall-clock with a nesting stack.
    Pin the contract so a refactor can't silently break attribution.
    """

    def test_default_scope_is_none(self):
        self.assertIsNone(current_op_scope())

    def test_context_manager_sets_and_restores(self):
        self.assertIsNone(current_op_scope())
        with op_scope("reward"):
            self.assertEqual(current_op_scope(), "reward")
            with op_scope("optimizer"):
                self.assertEqual(current_op_scope(), "optimizer")
            self.assertEqual(current_op_scope(), "reward")
        self.assertIsNone(current_op_scope())

    def test_propagates_into_concurrent_child_tasks(self):
        # The reason for contextvars over threading.local: a scope entered
        # before asyncio.gather() is inherited by every child coroutine (this
        # is exactly the trainer's compute_reward / predict_on_batch pattern),
        # so concurrent LM calls are all attributed to the right phase.
        async def _run():
            async def observe():
                await asyncio.sleep(0)  # force interleaving
                return current_op_scope()

            with op_scope("reward"):
                in_reward = await asyncio.gather(*[observe() for _ in range(5)])
            with op_scope("optimizer"):
                in_optimizer = await asyncio.gather(*[observe() for _ in range(5)])
            return in_reward, in_optimizer

        in_reward, in_optimizer = asyncio.run(_run())
        self.assertEqual(in_reward, ["reward"] * 5)
        self.assertEqual(in_optimizer, ["optimizer"] * 5)

    def test_nested_wall_clock_is_self_time(self):
        # Optimizer wraps reward; each phase should be credited only its own
        # time, so the nested reward span is not double-counted on optimizer.
        clear_session()  # reset the (thread-local) wall-clock accumulators
        # perf_counter() is called once per enter and once per exit:
        # enter(opt)=0, enter(rew)=2, exit(rew)=5, exit(opt)=9
        ticks = [0.0, 2.0, 5.0, 9.0]
        with mock.patch(
            "synalinks.src.backend.common.op_scope.time.perf_counter",
            side_effect=ticks,
        ):
            with op_scope("optimizer"):
                with op_scope("reward"):
                    pass
        self.assertEqual(read_phase_wall_clock_s("reward"), 3.0)  # 5 - 2
        self.assertEqual(read_phase_wall_clock_s("optimizer"), 6.0)  # (2-0)+(9-5)


# The reward- and optimizer-phase metric classes mirror the inference set
# one-for-one; their `result()` methods are otherwise uncovered.


def _bump(lm, suffix, n, phase):
    attr = f"{phase}_cumulated_{suffix}"
    setattr(lm, attr, getattr(lm, attr, 0) + n)


# `_stub_lm` only resets the suffixes in `_LM_SUFFIXES`; the cache/reasoning
# counters live on real LMs but aren't pre-populated by the stub. Zero them
# explicitly so `_bump` has a clean baseline.
def _stub_lm_with_extras():
    lm = _stub_lm()
    for phase in ("inference", "reward", "optimizer"):
        for suffix in ("cached_tokens", "cache_creation_tokens", "reasoning_tokens"):
            setattr(lm, f"{phase}_cumulated_{suffix}", 0)
    return lm


class LMRatesAndCachePerPhaseTest(testing.TestCase):
    """`result()` paths for rate / per-call / cache-hit / reasoning-share
    metrics. The inference set is covered by tests above; this class extends
    the contract to the reward and optimizer phases — same shape, different
    counter prefix — and adds cached-tokens / cache-creation / reasoning
    coverage for inference."""

    PHASES = (
        (
            "inference",
            TokensPerSecond,
            Throughput,
            AvgLatency,
            AvgInputTokensPerCall,
            AvgOutputTokensPerCall,
            AvgCostPerCall,
            CachedTokens,
            CacheCreationTokens,
            CacheHitRate,
            ReasoningTokens,
            ReasoningTokenShare,
        ),
        (
            "reward",
            RewardTokensPerSecond,
            RewardThroughput,
            AvgRewardLatency,
            AvgRewardInputTokensPerCall,
            AvgRewardOutputTokensPerCall,
            AvgRewardCostPerCall,
            RewardCachedTokens,
            RewardCacheCreationTokens,
            RewardCacheHitRate,
            RewardReasoningTokens,
            RewardReasoningTokenShare,
        ),
        (
            "optimizer",
            OptimizerTokensPerSecond,
            OptimizerThroughput,
            AvgOptimizerLatency,
            AvgOptimizerInputTokensPerCall,
            AvgOptimizerOutputTokensPerCall,
            AvgOptimizerCostPerCall,
            OptimizerCachedTokens,
            OptimizerCacheCreationTokens,
            OptimizerCacheHitRate,
            OptimizerReasoningTokens,
            OptimizerReasoningTokenShare,
        ),
    )

    def test_all_rate_and_cache_metrics(self):
        for (
            phase,
            Tps,
            Rps,
            Lat,
            AvgIn,
            AvgOut,
            AvgCost,
            Cached,
            CacheCreate,
            HitRate,
            Reasoning,
            RShare,
        ) in self.PHASES:
            with self.subTest(phase=phase):
                lm = _stub_lm_with_extras()
                metrics = {
                    "tps": _bind(Tps(), [lm]),
                    "rps": _bind(Rps(), [lm]),
                    "lat": _bind(Lat(), [lm]),
                    "avg_in": _bind(AvgIn(), [lm]),
                    "avg_out": _bind(AvgOut(), [lm]),
                    "avg_cost": _bind(AvgCost(), [lm]),
                    "cached": _bind(Cached(), [lm]),
                    "cache_create": _bind(CacheCreate(), [lm]),
                    "hit_rate": _bind(HitRate(), [lm]),
                    "reasoning": _bind(Reasoning(), [lm]),
                    "r_share": _bind(RShare(), [lm]),
                }
                # Zero-state: every divisor-guarded metric returns 0(.0).
                ratio_keys = (
                    "tps",
                    "rps",
                    "lat",
                    "avg_in",
                    "avg_out",
                    "avg_cost",
                    "hit_rate",
                    "r_share",
                )
                for key in ratio_keys:
                    self.assertEqual(metrics[key].result(), 0.0, msg=f"{phase}.{key}")
                for key in ("cached", "cache_create", "reasoning"):
                    self.assertEqual(metrics[key].result(), 0, msg=f"{phase}.{key}")

                # Populate: 2 calls, 400 prompt + 100 completion, 4s, $0.004
                # cost, 50 cached, 10 cache-creation, 40 reasoning.
                _record(
                    lm, prompt=200, completion=50, elapsed=2.0, cost=0.002, phase=phase
                )
                _record(
                    lm, prompt=200, completion=50, elapsed=2.0, cost=0.002, phase=phase
                )
                _bump(lm, "cached_tokens", 50, phase=phase)
                _bump(lm, "cache_creation_tokens", 10, phase=phase)
                _bump(lm, "reasoning_tokens", 40, phase=phase)
                _add_phase_wall_clock_s(phase, 4.0)  # throughput denominator

                self.assertEqual(metrics["tps"].result(), 500 / 4.0)
                self.assertEqual(metrics["rps"].result(), 2 / 4.0)
                self.assertEqual(metrics["lat"].result(), 4.0 / 2)
                self.assertEqual(metrics["avg_in"].result(), 200.0)
                self.assertEqual(metrics["avg_out"].result(), 50.0)
                self.assertAlmostEqual(metrics["avg_cost"].result(), 0.002)
                self.assertEqual(metrics["cached"].result(), 50)
                self.assertEqual(metrics["cache_create"].result(), 10)
                self.assertEqual(metrics["hit_rate"].result(), 50 / 400)
                self.assertEqual(metrics["reasoning"].result(), 40)
                self.assertEqual(metrics["r_share"].result(), 40 / 100)


class LMExtraAveragesPerPhaseTest(testing.TestCase):
    """Extra per-call token averages (total / cached / cache-creation /
    reasoning) for the reward and optimizer phases. The inference variants
    are covered by dedicated tests above."""

    PHASES = (
        (
            "reward",
            AvgRewardTotalTokensPerCall,
            AvgRewardCachedTokensPerCall,
            AvgRewardCacheCreationTokensPerCall,
            AvgRewardReasoningTokensPerCall,
        ),
        (
            "optimizer",
            AvgOptimizerTotalTokensPerCall,
            AvgOptimizerCachedTokensPerCall,
            AvgOptimizerCacheCreationTokensPerCall,
            AvgOptimizerReasoningTokensPerCall,
        ),
    )

    def test_extra_averages_per_phase(self):
        for (
            phase,
            AvgTotal,
            AvgCached,
            AvgCacheCreate,
            AvgReasoning,
        ) in self.PHASES:
            with self.subTest(phase=phase):
                lm = _stub_lm_with_extras()
                m_total = _bind(AvgTotal(), [lm])
                m_cached = _bind(AvgCached(), [lm])
                m_cachecreate = _bind(AvgCacheCreate(), [lm])
                m_reasoning = _bind(AvgReasoning(), [lm])
                # Zero-state.
                self.assertEqual(m_total.result(), 0.0)
                self.assertEqual(m_cached.result(), 0.0)
                self.assertEqual(m_cachecreate.result(), 0.0)
                self.assertEqual(m_reasoning.result(), 0.0)
                # 2 calls, 500 total tokens, 60 cached, 40 cache-creation,
                # 30 reasoning.
                _record(lm, prompt=100, completion=20, elapsed=0.1, cost=0.0, phase=phase)
                _record(lm, prompt=300, completion=80, elapsed=0.1, cost=0.0, phase=phase)
                _bump(lm, "cached_tokens", 60, phase=phase)
                _bump(lm, "cache_creation_tokens", 40, phase=phase)
                _bump(lm, "reasoning_tokens", 30, phase=phase)
                self.assertEqual(m_total.result(), 250.0)  # 500 / 2
                self.assertEqual(m_cached.result(), 30.0)  # 60 / 2
                self.assertEqual(m_cachecreate.result(), 20.0)  # 40 / 2
                self.assertEqual(m_reasoning.result(), 15.0)  # 30 / 2


class LMFailureMetricsPerPhaseTest(testing.TestCase):
    """`FailedCalls` / `FallbackActivations` / `ErrorRate` for the reward and
    optimizer phases (inference is covered by a dedicated test above)."""

    PHASES = (
        ("reward", RewardFailedCalls, RewardFallbackActivations, RewardErrorRate),
        (
            "optimizer",
            OptimizerFailedCalls,
            OptimizerFallbackActivations,
            OptimizerErrorRate,
        ),
    )

    def test_failure_metrics_per_phase(self):
        for phase, FailedM, FallbackM, ErrorM in self.PHASES:
            with self.subTest(phase=phase):
                lm = _stub_lm()
                m_failed = _bind(FailedM(), [lm])
                m_fallback = _bind(FallbackM(), [lm])
                m_error = _bind(ErrorM(), [lm])
                # Zero-state: error rate guarded against no calls.
                self.assertEqual(m_failed.result(), 0)
                self.assertEqual(m_fallback.result(), 0)
                self.assertEqual(m_error.result(), 0.0)
                # 1 success, 3 failed (2 of which fell back).
                _record(lm, prompt=10, completion=5, elapsed=0.1, cost=0.0, phase=phase)
                _bump(lm, "failed_calls", 3, phase=phase)
                _bump(lm, "fallback_activations", 2, phase=phase)
                self.assertEqual(m_failed.result(), 3)
                self.assertEqual(m_fallback.result(), 2)
                self.assertEqual(m_error.result(), 3 / 4)  # 3 failed / (1 + 3)
