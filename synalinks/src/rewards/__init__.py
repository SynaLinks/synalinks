import inspect

from synalinks.src.api_export import synalinks_export
from synalinks.src.rewards.cosine_similarity import CosineSimilarity
from synalinks.src.rewards.cosine_similarity import cosine_similarity
from synalinks.src.rewards.exact_match import ExactMatch
from synalinks.src.rewards.exact_match import exact_match
from synalinks.src.rewards.reward import Reward
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge
from synalinks.src.rewards.reward_wrappers import RewardFunctionWrapper
from synalinks.src.saving import serialization_lib
from synalinks.src.utils.naming import to_snake_case

# `LMAsJudge` lives in this set but is imported lazily at the bottom of this
# module to avoid a circular import (`lm_as_judge` -> `programs.Program` ->
# `trainers.Trainer` -> `synalinks.src.rewards`).
ALL_OBJECTS = {
    # Base
    Reward,
    RewardFunctionWrapper,
    # Concrete
    ExactMatch,
    CosineSimilarity,
    ProgramAsJudge,
}

ALL_OBJECTS_DICT = {cls.__name__: cls for cls in ALL_OBJECTS}


@synalinks_export("synalinks.rewards.serialize")
def serialize(reward):
    """Serializes reward function or `Reward` instance.

    Args:
        reward: A Keras `Reward` instance or a reward function.

    Returns:
        Reward configuration dictionary.
    """
    return serialization_lib.serialize_synalinks_object(reward)


@synalinks_export("synalinks.rewards.deserialize")
def deserialize(name, custom_objects=None):
    """Deserializes a serialized reward class/function instance.

    Args:
        name: Reward configuration.
        custom_objects: Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Keras `Reward` instance or a reward function.
    """
    return serialization_lib.deserialize_synalinks_object(
        name,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@synalinks_export("synalinks.rewards.get")
def get(identifier):
    """Retrieves a Synalinks reward as a `function`/`Reward` class instance.

    The `identifier` may be the string name of a reward function or `Reward` class.

    >>> reward = rewards.get("exact_match")
    >>> type(reward)
    <class 'function'>
    >>> reward = rewards.get("ExactMatch")
    >>> type(reward)
    <class '...ExactMatch'>

    You can also specify `config` of the reward to this function by passing dict
    containing `class_name` and `config` as an identifier. Also note that the
    `class_name` must map to a `Reward` class

    >>> identifier = {"class_name": "ExactMatch",
    ...               "config": {"in_mask": ["answer"]}}
    >>> reward = rewards.get(identifier)
    >>> type(reward)
    <class '...ExactMatch'>

    Args:
        identifier: A reward identifier. One of None or string name of a reward
            function/class or reward configuration dictionary or a reward function
            or a reward class instance.

    Returns:
        A Synalinks reward as a `function`/ `Reward` class instance.
    """
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier, None)
    else:
        obj = identifier

    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(f"Could not interpret reward identifier: {identifier}")


# Late import to break a cycle: `lm_as_judge` -> `programs.Program` ->
# `trainers.Trainer` -> back to `synalinks.src.rewards`.
from synalinks.src.rewards.lm_as_judge import LMAsJudge  # noqa: E402

ALL_OBJECTS.add(LMAsJudge)
ALL_OBJECTS_DICT[LMAsJudge.__name__] = LMAsJudge
