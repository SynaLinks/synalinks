# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import backend
from synalinks.src import tree
from synalinks.src.backend.common import numpy as np
from synalinks.src.backend.common.op_scope import read_phase_wall_clock_s
from synalinks.src.metrics.metric import Metric


def _reward_holders(reward, holders, seen):
    """Collect the objects of a reward that may hold a model, recursively."""
    if reward is None or id(reward) in seen:
        return
    seen.add(id(reward))
    # Rewards that call a model directly (e.g. `LMAsJudge`) hold it as an
    # attribute, like modules do.
    holders.append(reward)
    program = getattr(reward, "program", None)
    if hasattr(program, "_flatten_modules"):
        holders.extend(program._flatten_modules(include_self=True, recursive=True))
    for sub_reward in getattr(reward, "rewards", None) or []:
        _reward_holders(sub_reward, holders, seen)


def model_holders(program):
    """Return the modules (and rewards, optimizer) whose models a program runs.

    That is every module of the program, plus the modules of its compiled
    reward: a judge's model (e.g. the one of `RubricsAsJudge`) runs during the
    reward phase, so its counters belong to the program's run too. The same
    holds for the compiled optimizer (e.g. `OMEGA`'s language and embedding
    models, and its mutation/crossover programs) during the optimizer phase.

    Args:
        program (Program): The program to walk.

    Returns:
        (list): The modules, rewards and optimizer, in order, each at most once.
    """
    holders = []
    if hasattr(program, "_flatten_modules"):
        holders.extend(program._flatten_modules(include_self=True, recursive=True))
    compile_reward = getattr(program, "_compile_reward", None)
    if compile_reward is not None:
        seen = set()
        for reward in tree.flatten(getattr(compile_reward, "_user_reward", None)):
            _reward_holders(reward, holders, seen)
    optimizer = getattr(program, "_optimizer", None)
    if optimizer is not None:
        holders.append(optimizer)
        for name in ("mutation_programs", "crossover_programs"):
            for sub_program in (getattr(optimizer, name, None) or {}).values():
                holders.extend(
                    sub_program._flatten_modules(include_self=True, recursive=True)
                )
    return holders


def aggregate_scores(score, average, axis, intermediate_weights=None):
    """Apply the `average` reduction over per-field scores.

    Args:
        score (list | float): The per-field scores.
        average (str): One of `None`, `"micro"`, `"macro"` or `"weighted"`.
        axis (int): The axis to average over for `"micro"`/`"macro"`.
        intermediate_weights (list): The per-field support, for `"weighted"`.

    Returns:
        (float | list): A Python scalar, or one score per field when
            `average=None`.
    """
    score = np.convert_to_tensor(score)
    if average == "weighted":
        weights = np.divide(
            intermediate_weights,
            np.sum(intermediate_weights) + backend.epsilon(),
        )
        score = np.sum(score * weights)
    elif average is not None:  # [micro, macro]
        score = np.mean(score, axis)
    # numpy 1.25+ deprecates float() on a >0-D array even when size == 1.
    score_arr = np.convert_to_numpy(score)
    if score_arr.size == 1:
        return float(score_arr.item())
    return [float(v) for v in score_arr.tolist()]


def collect_models(program, model_types, attrs):
    """Return every unique model of `model_types` that a program runs.

    Walks `model_holders(program)`, reads each attribute in `attrs` (and the
    holder itself) and follows each model's `fallback` chain, preserving
    order.

    Args:
        program (Program): The program to walk.
        model_types (tuple): The model classes to collect.
        attrs (tuple): The holder attributes that may hold a model.

    Returns:
        (list): The models, each at most once.
    """
    models = []
    seen = set()
    for holder in model_holders(program):
        for model in [getattr(holder, attr, None) for attr in attrs] + [holder]:
            while isinstance(model, model_types) and id(model) not in seen:
                seen.add(id(model))
                models.append(model)
                model = getattr(model, "fallback", None)
    return models


def language_models(program):
    """Return every unique `LanguageModel` / `DecisionModel` a program runs.

    Checks `language_model`, `sub_language_model` (the distinct instance that
    agents such as `RecursiveLanguageModelAgent` and `DeepAgent` use for
    recursive sub-queries / subagents) and `decision_model` (same counters as
    an LM). Includes `fallback` chains and the models of the compiled reward
    (e.g. a judge's), which run during the reward phase.

    Args:
        program (Program): The program to walk.

    Returns:
        (list): The models, each at most once.
    """
    from synalinks.src.modules.decision_models import DecisionModel
    from synalinks.src.modules.language_models import LanguageModel

    return collect_models(
        program,
        (LanguageModel, DecisionModel),
        ("language_model", "sub_language_model", "decision_model"),
    )


def embedding_models(program):
    """Return every unique `EmbeddingModel` a program runs (with fallbacks).

    Args:
        program (Program): The program to walk.

    Returns:
        (list): The models, each at most once.
    """
    from synalinks.src.modules.embedding_models import EmbeddingModel

    return collect_models(program, (EmbeddingModel,), ("embedding_model",))


def program_models(program):
    """Return every language, decision and embedding model a program runs.

    Args:
        program (Program): The program to walk.

    Returns:
        (list): The models, each at most once.
    """
    return language_models(program) + embedding_models(program)


class ModelOperationalMetric(Metric):
    """Shared base of the LM and EM runtime-counter metrics.

    Subclasses set `_collect` (a `program -> models` function),
    `_tracked_suffixes` (the counters to snapshot) and `_phase`. The metric
    reads `{_phase}_cumulated_{suffix}` summed across the bound models, as a
    delta since the last snapshot.
    """

    _phase = "inference"
    _tracked_suffixes = ()
    direction = "down"

    def __init__(self, name=None):
        super().__init__(name=name)
        self._models = []
        self._baselines = {suffix: 0 for suffix in self._tracked_suffixes}
        self._wall_baseline = 0.0
        # The bound program's own phase clock (thread default when unbound).
        self._phase_clock = None

    def bind_program(self, program):
        self._models = self._collect(program)
        self._phase_clock = getattr(program, "_phase_clock", None)
        self._snapshot()

    def _attr(self, suffix):
        return f"{self._phase}_cumulated_{suffix}"

    def _read(self, suffix):
        attr = self._attr(suffix)
        return sum(getattr(model, attr, 0) for model in self._models)

    def _snapshot(self):
        for suffix in self._tracked_suffixes:
            self._baselines[suffix] = self._read(suffix)
        self._wall_baseline = read_phase_wall_clock_s(self._phase, self._phase_clock)

    def _delta(self, suffix):
        return self._read(suffix) - self._baselines.get(suffix, 0)

    def _wall_clock_delta(self):
        """Wall-clock seconds the trainer spent in this metric's phase since
        the last snapshot. Used as the throughput denominator so concurrent
        (overlapping) calls don't inflate it the way summed `elapsed_s` does.
        """
        return (
            read_phase_wall_clock_s(self._phase, self._phase_clock) - self._wall_baseline
        )

    def reset_state(self):
        self._snapshot()

    async def update_state(self, *args, **kwargs):
        return

    def result(self):
        raise NotImplementedError

    def get_config(self):
        return {"name": self.name}
