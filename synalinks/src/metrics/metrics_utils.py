# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import backend
from synalinks.src import tree
from synalinks.src.backend.common import numpy as np


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
