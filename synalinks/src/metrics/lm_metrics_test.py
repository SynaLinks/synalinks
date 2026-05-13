# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend.common import global_state
from synalinks.src.metrics.lm_metrics import AvgCostPerCall
from synalinks.src.metrics.lm_metrics import AvgInputTokensPerCall
from synalinks.src.metrics.lm_metrics import AvgOutputTokensPerCall
from synalinks.src.metrics.lm_metrics import Cost
from synalinks.src.metrics.lm_metrics import InputTokens
from synalinks.src.metrics.lm_metrics import OptimizerCost
from synalinks.src.metrics.lm_metrics import OptimizerInputTokens
from synalinks.src.metrics.lm_metrics import OptimizerTotalTokens
from synalinks.src.metrics.lm_metrics import OutputTokens
from synalinks.src.metrics.lm_metrics import RewardCost
from synalinks.src.metrics.lm_metrics import RewardInputTokens
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
        _record(lm, prompt=400, completion=100, elapsed=2.0, cost=0.0)
        _record(lm, prompt=400, completion=100, elapsed=2.0, cost=0.0)
        self.assertEqual(m_tps.result(), 1000 / 4.0)
        self.assertEqual(m_rps.result(), 2 / 4.0)

    def test_cost_metrics(self):
        lm = _stub_lm()
        m_cost = _bind(Cost(), [lm])
        m_avg = _bind(AvgCostPerCall(), [lm])
        _record(lm, prompt=10, completion=5, elapsed=0.1, cost=0.003)
        _record(lm, prompt=10, completion=5, elapsed=0.1, cost=0.005)
        self.assertAlmostEqual(m_cost.result(), 0.008)
        self.assertAlmostEqual(m_avg.result(), 0.004)

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
        self.assertEqual(m_avg_in.result(), 0.0)
        self.assertEqual(m_tps.result(), 0.0)
        self.assertEqual(m_rps.result(), 0.0)

    def test_result_is_zero_before_bind(self):
        # No bind_program() call → no LMs → metric reports 0 instead of crashing.
        self.assertEqual(TotalTokens().result(), 0)
        self.assertEqual(Cost().result(), 0.0)
        self.assertEqual(Throughput().result(), 0.0)


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
        _record(
            lm, prompt=200, completion=40, elapsed=0.0, cost=0.002, phase="optimizer"
        )

        self.assertEqual(m_inf.result(), 120)
        self.assertEqual(m_rew.result(), 60)
        self.assertEqual(m_opt.result(), 240)

    def test_reward_input_tokens(self):
        lm = _stub_lm()
        m = _bind(RewardInputTokens(), [lm])
        _record(lm, prompt=300, completion=10, elapsed=0.0, cost=0.0, phase="reward")
        _record(
            lm, prompt=999, completion=999, elapsed=0.0, cost=0.0, phase="inference"
        )
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


class OpScopeFlagTest(testing.TestCase):
    """The trainer sets ``global_state["synalinks_op_scope"]`` to one of
    "inference" | "reward" | "optimizer" | None, and the LM/EM route
    counter updates based on the value. Pin the contract here so a future
    refactor doesn't silently break attribution.
    """

    def test_scope_default_is_none(self):
        global_state.set_global_attribute("synalinks_op_scope", None)
        self.assertIsNone(global_state.get_global_attribute("synalinks_op_scope"))

    def test_scope_round_trip(self):
        prev = global_state.get_global_attribute("synalinks_op_scope")
        try:
            for value in ("inference", "reward", "optimizer"):
                global_state.set_global_attribute("synalinks_op_scope", value)
                self.assertEqual(
                    global_state.get_global_attribute("synalinks_op_scope"),
                    value,
                )
        finally:
            global_state.set_global_attribute("synalinks_op_scope", prev)
