# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Program-wide operational metrics.

Reads counters that live directly on the bound program object —
specifically `{phase}_cumulated_invocations` and
`{phase}_cumulated_invocation_elapsed_s`, which
`Module._maybe_reset_call_context` bumps whenever the program is the
entry module of a top-level call. That gives true wall-clock elapsed
and invocation counts at the program boundary, not a sum of nested
LM/EM per-call times.

For cost, which is paid by individual providers and tracked on each
`LanguageModel` / `EmbeddingModel`, the base class also collects the LM
and EM trees reachable from the program (including `.fallback` chains)
so subclasses can sum across them.
"""

from synalinks.src.api_export import synalinks_export
from synalinks.src.metrics.em_metrics import _collect_embedding_models
from synalinks.src.metrics.lm_metrics import _collect_language_models
from synalinks.src.metrics.metric import Metric

_PROGRAM_SUFFIXES = ("invocations", "invocation_elapsed_s")


@synalinks_export(
    [
        "synalinks.metrics.ProgramOperationalMetric",
        "synalinks.ProgramOperationalMetric",
    ]
)
class ProgramOperationalMetric(Metric):
    """Base class for program-wide runtime-counter metrics.

    Subclasses set `_phase` to one of ``"inference"``, ``"reward"``, or
    ``"optimizer"`` to read the corresponding counter set. Counters are
    populated based on the ``synalinks_op_scope`` global flag set by the
    trainer.
    """

    _phase = "inference"
    direction = "down"

    def __init__(self, name=None):
        super().__init__(name=name)
        self._program = None
        self._language_models = []
        self._embedding_models = []
        self._program_baselines = {suffix: 0 for suffix in _PROGRAM_SUFFIXES}
        self._model_cost_baseline = 0.0

    @property
    def program(self):
        return self._program

    @property
    def language_models(self):
        return list(self._language_models)

    @property
    def embedding_models(self):
        return list(self._embedding_models)

    def bind_program(self, program):
        self._program = program
        self._language_models = _collect_language_models(program)
        self._embedding_models = _collect_embedding_models(program)
        self._snapshot()

    def _attr(self, suffix):
        return f"{self._phase}_cumulated_{suffix}"

    def _read_program(self, suffix):
        if self._program is None:
            return 0
        return getattr(self._program, self._attr(suffix), 0)

    def _read_model_cost(self):
        attr = self._attr("cost")
        lm_cost = sum(getattr(lm, attr, 0) for lm in self._language_models)
        em_cost = sum(getattr(em, attr, 0) for em in self._embedding_models)
        return lm_cost + em_cost

    def _snapshot(self):
        for suffix in _PROGRAM_SUFFIXES:
            self._program_baselines[suffix] = self._read_program(suffix)
        self._model_cost_baseline = self._read_model_cost()

    def _delta_program(self, suffix):
        return self._read_program(suffix) - self._program_baselines.get(suffix, 0)

    def _delta_model_cost(self):
        return self._read_model_cost() - self._model_cost_baseline

    def reset_state(self):
        self._snapshot()

    async def update_state(self, *args, **kwargs):
        return

    def result(self):
        raise NotImplementedError

    def get_config(self):
        return {"name": self.name}


@synalinks_export("synalinks.metrics.ProgramCalls")
class ProgramCalls(ProgramOperationalMetric):
    """Number of top-level program invocations during this run.

    One increment per call to `program(...)`, regardless of how many LM
    or EM calls the program makes internally.
    """

    def __init__(self, name="program_calls"):
        super().__init__(name=name)

    def result(self):
        return int(self._delta_program("invocations"))


@synalinks_export("synalinks.metrics.ProgramElapsedTime")
class ProgramElapsedTime(ProgramOperationalMetric):
    """End-to-end wall-clock seconds spent inside `program(...)` during
    this run. Sums per-invocation wall-clock; does not double-count
    nested LM/EM calls.
    """

    def __init__(self, name="program_elapsed_time"):
        super().__init__(name=name)

    def result(self):
        return float(self._delta_program("invocation_elapsed_s"))


@synalinks_export("synalinks.metrics.ProgramCallsPerSecond")
class ProgramCallsPerSecond(ProgramOperationalMetric):
    """Throughput: program invocations per wall-clock second."""

    def __init__(self, name="program_calls_per_second"):
        super().__init__(name=name)

    def result(self):
        elapsed = self._delta_program("invocation_elapsed_s")
        if elapsed <= 0.0:
            return 0.0
        return self._delta_program("invocations") / elapsed


@synalinks_export("synalinks.metrics.ProgramCost")
class ProgramCost(ProgramOperationalMetric):
    """Total provider cost (USD, as reported by litellm) across every LM
    and EM reached from the program for this run. The program object
    itself doesn't pay providers — cost is summed from its bound models.
    """

    def __init__(self, name="program_cost"):
        super().__init__(name=name)

    def result(self):
        return float(self._delta_model_cost())


@synalinks_export("synalinks.metrics.ProgramAvgCostPerInvocation")
class ProgramAvgCostPerInvocation(ProgramOperationalMetric):
    """Average provider cost per program invocation."""

    def __init__(self, name="program_avg_cost_per_invocation"):
        super().__init__(name=name)

    def result(self):
        calls = self._delta_program("invocations")
        if calls <= 0:
            return 0.0
        return self._delta_model_cost() / calls
