# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import tree


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
    """Return the modules (and rewards) whose models a program runs.

    That is every module of the program, plus the modules of its compiled
    reward: a judge's model (e.g. the one of `RubricsAsJudge`) runs during the
    reward phase, so its counters belong to the program's run too.

    Args:
        program (Program): The program to walk.

    Returns:
        (list): The modules and rewards, in order, each at most once.
    """
    holders = []
    if hasattr(program, "_flatten_modules"):
        holders.extend(program._flatten_modules(include_self=True, recursive=True))
    compile_reward = getattr(program, "_compile_reward", None)
    if compile_reward is not None:
        seen = set()
        for reward in tree.flatten(getattr(compile_reward, "_user_reward", None)):
            _reward_holders(reward, holders, seen)
    return holders
