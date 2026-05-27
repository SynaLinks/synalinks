# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import backend
from synalinks.src import modules
from synalinks.src import testing
from synalinks.src.metrics.program_metrics import ProgramAvgCostPerInvocation
from synalinks.src.metrics.program_metrics import ProgramCalls
from synalinks.src.metrics.program_metrics import ProgramCallsPerSecond
from synalinks.src.metrics.program_metrics import ProgramCost
from synalinks.src.metrics.program_metrics import ProgramElapsedTime
from synalinks.src.modules.embedding_models import EmbeddingModel
from synalinks.src.modules.language_models import LanguageModel


def _stub_lm(fallback=None):
    lm = LanguageModel(model="ollama/mistral", fallback=fallback)
    lm.inference_cumulated_cost = 0.0
    return lm


def _stub_em(fallback=None):
    em = EmbeddingModel(model="ollama/all-minilm", fallback=fallback)
    em.inference_cumulated_cost = 0.0
    return em


class _FakeModule:
    """Module stand-in carrying `.language_model` / `.embedding_model`."""

    def __init__(self, lm=None, em=None, submodules=()):
        self.language_model = lm
        self.embedding_model = em
        self._modules = list(submodules)


class _FakeProgram:
    """Program stand-in: exposes the same counter surface a real Module has
    after our instrumentation, plus `_flatten_modules` for the LM/EM
    discovery helpers.
    """

    def __init__(self, modules=()):
        self._modules = list(modules)
        # Mirror Module-base counters added in module.py.
        self.cumulated_invocations = 0
        self.cumulated_invocation_elapsed_s = 0.0
        for _phase in ("inference", "reward", "optimizer"):
            setattr(self, f"{_phase}_cumulated_invocations", 0)
            setattr(self, f"{_phase}_cumulated_invocation_elapsed_s", 0.0)

    def _flatten_modules(self, include_self=True, recursive=True):
        out = [self] if include_self else []
        stack = list(self._modules)
        while stack:
            m = stack.pop(0)
            out.append(m)
            if recursive:
                stack = list(m._modules) + stack
        return out


def _record_program(program, calls, elapsed):
    program.inference_cumulated_invocations += calls
    program.inference_cumulated_invocation_elapsed_s += elapsed


def _record_lm_cost(lm, cost):
    lm.inference_cumulated_cost += cost


def _record_em_cost(em, cost):
    em.inference_cumulated_cost += cost


def _bind(metric, program):
    metric.bind_program(program)
    return metric


class ProgramOperationalMetricsResultTest(testing.TestCase):
    def test_program_calls_counts_program_invocations(self):
        program = _FakeProgram()
        metric = _bind(ProgramCalls(), program)
        _record_program(program, calls=4, elapsed=2.0)
        self.assertEqual(metric.result(), 4)

    def test_program_elapsed_time_reads_program_wall_clock(self):
        program = _FakeProgram()
        metric = _bind(ProgramElapsedTime(), program)
        _record_program(program, calls=2, elapsed=3.5)
        self.assertAlmostEqual(metric.result(), 3.5)

    def test_calls_per_second(self):
        program = _FakeProgram()
        metric = _bind(ProgramCallsPerSecond(), program)
        _record_program(program, calls=10, elapsed=2.0)
        self.assertEqual(metric.result(), 5.0)

    def test_calls_per_second_zero_when_no_elapsed(self):
        program = _FakeProgram()
        metric = _bind(ProgramCallsPerSecond(), program)
        self.assertEqual(metric.result(), 0.0)

    def test_program_cost_sums_lm_and_em(self):
        lm = _stub_lm()
        em = _stub_em()
        program = _FakeProgram(modules=[_FakeModule(lm=lm), _FakeModule(em=em)])
        metric = _bind(ProgramCost(), program)
        _record_lm_cost(lm, 0.01)
        _record_em_cost(em, 0.0002)
        self.assertAlmostEqual(metric.result(), 0.0102)

    def test_avg_cost_per_invocation(self):
        lm = _stub_lm()
        em = _stub_em()
        program = _FakeProgram(modules=[_FakeModule(lm=lm), _FakeModule(em=em)])
        metric = _bind(ProgramAvgCostPerInvocation(), program)
        _record_program(program, calls=4, elapsed=1.0)
        _record_lm_cost(lm, 0.02)
        _record_em_cost(em, 0.004)
        self.assertAlmostEqual(metric.result(), 0.024 / 4)

    def test_avg_cost_per_invocation_zero_when_no_calls(self):
        lm = _stub_lm()
        program = _FakeProgram(modules=[_FakeModule(lm=lm)])
        metric = _bind(ProgramAvgCostPerInvocation(), program)
        _record_lm_cost(lm, 0.5)  # cost without invocations: still 0 result
        self.assertEqual(metric.result(), 0.0)

    def test_baseline_excludes_pre_bind_activity(self):
        """Counters that already moved before bind_program() must not leak;
        only post-bind deltas count.
        """
        lm = _stub_lm()
        program = _FakeProgram(modules=[_FakeModule(lm=lm)])
        _record_program(program, calls=10, elapsed=5.0)
        _record_lm_cost(lm, 1.0)
        metric = _bind(ProgramCalls(), program)
        cost = _bind(ProgramCost(), program)
        self.assertEqual(metric.result(), 0)
        self.assertEqual(cost.result(), 0.0)
        _record_program(program, calls=1, elapsed=0.1)
        _record_lm_cost(lm, 0.001)
        self.assertEqual(metric.result(), 1)
        self.assertAlmostEqual(cost.result(), 0.001)

    def test_reset_state_resnapshots_baseline(self):
        program = _FakeProgram()
        metric = _bind(ProgramCalls(), program)
        _record_program(program, calls=3, elapsed=1.0)
        self.assertEqual(metric.result(), 3)
        metric.reset_state()
        self.assertEqual(metric.result(), 0)
        _record_program(program, calls=1, elapsed=0.5)
        self.assertEqual(metric.result(), 1)

    def test_cost_aggregates_across_multiple_lms_and_ems(self):
        lm_a = _stub_lm()
        lm_b = _stub_lm()
        em_a = _stub_em()
        em_b = _stub_em()
        program = _FakeProgram(
            modules=[
                _FakeModule(lm=lm_a),
                _FakeModule(lm=lm_b),
                _FakeModule(em=em_a),
                _FakeModule(em=em_b),
            ]
        )
        metric = _bind(ProgramCost(), program)
        _record_lm_cost(lm_a, 0.001)
        _record_lm_cost(lm_b, 0.002)
        _record_em_cost(em_a, 0.0001)
        _record_em_cost(em_b, 0.0002)
        self.assertAlmostEqual(metric.result(), 0.0033)

    def test_cost_traverses_fallback_chain(self):
        lm_fallback = _stub_lm()
        lm_primary = _stub_lm(fallback=lm_fallback)
        em_fallback = _stub_em()
        em_primary = _stub_em(fallback=em_fallback)
        program = _FakeProgram(
            modules=[_FakeModule(lm=lm_primary), _FakeModule(em=em_primary)]
        )
        metric = _bind(ProgramCost(), program)
        _record_lm_cost(lm_primary, 0.001)
        _record_lm_cost(lm_fallback, 0.002)
        _record_em_cost(em_primary, 0.0001)
        _record_em_cost(em_fallback, 0.0002)
        self.assertAlmostEqual(metric.result(), 0.0033)


class _PassthroughModule(modules.Module):
    """Minimal Module that returns its input. Used to drive `__call__`
    end-to-end so the entry-module instrumentation actually fires.
    """

    async def call(self, x):
        return x

    async def compute_output_spec(self, inputs):
        return backend.SymbolicDataModel(data_model=inputs)


class ProgramOperationalMetricsIntegrationTest(testing.TestCase):
    """Drives a real Module through `__call__` to confirm the counters
    populated by `_maybe_reset_call_context` are what the metrics read.
    """

    async def test_metric_reads_real_invocations(self):
        from synalinks.src.backend.common.op_scope import op_scope

        class Query(backend.DataModel):
            query: str

        module = _PassthroughModule()
        metric = ProgramCalls()
        metric.bind_program(module)

        with op_scope("inference"):
            await module(backend.SymbolicDataModel(data_model=Query))
            await module(backend.SymbolicDataModel(data_model=Query))
            await module(backend.SymbolicDataModel(data_model=Query))

        self.assertEqual(metric.result(), 3)

    async def test_metric_reads_real_elapsed(self):
        from synalinks.src.backend.common.op_scope import op_scope

        class Query(backend.DataModel):
            query: str

        module = _PassthroughModule()
        metric = ProgramElapsedTime()
        metric.bind_program(module)

        with op_scope("inference"):
            await module(backend.SymbolicDataModel(data_model=Query))

        # Real elapsed; just assert positive and finite.
        result = metric.result()
        self.assertGreater(result, 0.0)
        self.assertLess(result, 60.0)
