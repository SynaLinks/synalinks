# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.metrics.em_metrics import AvgEmbeddingCostPerCall
from synalinks.src.metrics.em_metrics import AvgEmbeddingTokensPerCall
from synalinks.src.metrics.em_metrics import AvgEmbeddingVectorsPerCall
from synalinks.src.metrics.em_metrics import EmbeddingCost
from synalinks.src.metrics.em_metrics import EmbeddingThroughput
from synalinks.src.metrics.em_metrics import EmbeddingTokens
from synalinks.src.metrics.em_metrics import EmbeddingVectors
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingCost
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingTokens
from synalinks.src.metrics.em_metrics import OptimizerEmbeddingVectors
from synalinks.src.metrics.em_metrics import RewardEmbeddingCost
from synalinks.src.metrics.em_metrics import RewardEmbeddingTokens
from synalinks.src.metrics.em_metrics import _collect_embedding_models
from synalinks.src.modules.embedding_models import EmbeddingModel

_EM_SUFFIXES = (
    "calls",
    "prompt_tokens",
    "tokens",
    "vectors",
    "elapsed_s",
    "cost",
)


def _stub_em(fallback=None):
    em = EmbeddingModel(model="ollama/mxbai-embed-large", fallback=fallback)
    for phase in ("inference", "reward", "optimizer"):
        for suffix in _EM_SUFFIXES:
            zero = 0.0 if suffix in ("elapsed_s", "cost") else 0
            setattr(em, f"{phase}_cumulated_{suffix}", zero)
    return em


def _record(em, *, prompt=0, vectors=0, elapsed=0.0, cost=0.0, phase="inference"):
    p = f"{phase}_cumulated_"
    setattr(em, p + "calls", getattr(em, p + "calls") + 1)
    setattr(em, p + "prompt_tokens", getattr(em, p + "prompt_tokens") + prompt)
    setattr(em, p + "tokens", getattr(em, p + "tokens") + prompt)
    setattr(em, p + "vectors", getattr(em, p + "vectors") + vectors)
    setattr(em, p + "elapsed_s", getattr(em, p + "elapsed_s") + elapsed)
    setattr(em, p + "cost", getattr(em, p + "cost") + cost)


class _FakeModule:
    def __init__(self, em=None, submodules=()):
        self.embedding_model = em
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


def _bind(metric, ems):
    program = _FakeProgram(modules=[_FakeModule(em=em) for em in ems])
    metric.bind_program(program)
    return metric


class EmbeddingMetricsResultTest(testing.TestCase):
    def test_tokens_vectors_cost(self):
        em = _stub_em()
        m_tok = _bind(EmbeddingTokens(), [em])
        m_vec = _bind(EmbeddingVectors(), [em])
        m_cost = _bind(EmbeddingCost(), [em])
        _record(em, prompt=100, vectors=3, elapsed=0.5, cost=0.0001)
        _record(em, prompt=50, vectors=2, elapsed=0.2, cost=0.00005)
        self.assertEqual(m_tok.result(), 150)
        self.assertEqual(m_vec.result(), 5)
        self.assertAlmostEqual(m_cost.result(), 0.00015)

    def test_averages_per_call(self):
        em = _stub_em()
        m_avg_tok = _bind(AvgEmbeddingTokensPerCall(), [em])
        m_avg_vec = _bind(AvgEmbeddingVectorsPerCall(), [em])
        m_avg_cost = _bind(AvgEmbeddingCostPerCall(), [em])
        _record(em, prompt=100, vectors=2, elapsed=0.5, cost=0.001)
        _record(em, prompt=300, vectors=6, elapsed=0.5, cost=0.003)
        self.assertEqual(m_avg_tok.result(), 200.0)
        self.assertEqual(m_avg_vec.result(), 4.0)
        self.assertAlmostEqual(m_avg_cost.result(), 0.002)

    def test_throughput(self):
        em = _stub_em()
        m_rps = _bind(EmbeddingThroughput(), [em])
        _record(em, prompt=100, vectors=2, elapsed=2.0, cost=0.0)
        _record(em, prompt=100, vectors=2, elapsed=2.0, cost=0.0)
        self.assertEqual(m_rps.result(), 2 / 4.0)

    def test_zero_division_safety(self):
        em = _stub_em()
        m = _bind(AvgEmbeddingTokensPerCall(), [em])
        self.assertEqual(m.result(), 0.0)


class EMAutoBindTest(testing.TestCase):
    def test_collect_walks_modules_and_fallback(self):
        leaf_em = _stub_em()
        primary_em = _stub_em(fallback=leaf_em)
        other_em = _stub_em()
        program = _FakeProgram(
            modules=[
                _FakeModule(em=primary_em),
                _FakeModule(em=None, submodules=[_FakeModule(em=other_em)]),
            ]
        )
        ems = _collect_embedding_models(program)
        ids = [id(em) for em in ems]
        self.assertIn(id(primary_em), ids)
        self.assertIn(id(leaf_em), ids)
        self.assertIn(id(other_em), ids)
        self.assertEqual(len(set(ids)), 3)

    def test_collect_dedupes_shared_em(self):
        shared_em = _stub_em()
        program = _FakeProgram(
            modules=[_FakeModule(em=shared_em), _FakeModule(em=shared_em)]
        )
        ems = _collect_embedding_models(program)
        self.assertEqual(len(ems), 1)
        self.assertIs(ems[0], shared_em)

    def test_bind_program_resets_baseline(self):
        em = _stub_em()
        _record(em, prompt=500, vectors=20, elapsed=3.0, cost=0.01)
        m = _bind(EmbeddingTokens(), [em])
        self.assertEqual(m.result(), 0)
        _record(em, prompt=10, vectors=1, elapsed=0.1, cost=0.0)
        self.assertEqual(m.result(), 10)


class EMPhaseRoutingTest(testing.TestCase):
    def test_phases_are_independent(self):
        em = _stub_em()
        m_inf = _bind(EmbeddingTokens(), [em])
        m_rew = _bind(RewardEmbeddingTokens(), [em])
        m_opt = _bind(OptimizerEmbeddingTokens(), [em])

        _record(em, prompt=100, vectors=5, phase="inference")
        _record(em, prompt=50, vectors=3, phase="reward")
        _record(em, prompt=200, vectors=10, phase="optimizer")

        self.assertEqual(m_inf.result(), 100)
        self.assertEqual(m_rew.result(), 50)
        self.assertEqual(m_opt.result(), 200)

    def test_reward_embedding_cost(self):
        em = _stub_em()
        m = _bind(RewardEmbeddingCost(), [em])
        _record(em, cost=0.0007, phase="reward")
        _record(em, cost=0.999, phase="inference")
        self.assertAlmostEqual(m.result(), 0.0007)

    def test_optimizer_embedding_vectors(self):
        em = _stub_em()
        m = _bind(OptimizerEmbeddingVectors(), [em])
        _record(em, vectors=42, phase="optimizer")
        _record(em, vectors=999, phase="reward")
        self.assertEqual(m.result(), 42)

    def test_optimizer_embedding_cost(self):
        em = _stub_em()
        m = _bind(OptimizerEmbeddingCost(), [em])
        _record(em, cost=0.0009, phase="optimizer")
        _record(em, cost=0.999, phase="inference")
        self.assertAlmostEqual(m.result(), 0.0009)
