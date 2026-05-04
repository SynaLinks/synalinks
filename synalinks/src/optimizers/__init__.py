from synalinks.src.api_export import synalinks_export
from synalinks.src.optimizers.evolutionary_optimizer import EvolutionaryOptimizer
from synalinks.src.optimizers.greedy_optimizer import GreedyOptimizer
from synalinks.src.optimizers.omega import OMEGA
from synalinks.src.optimizers.optimizer import Optimizer
from synalinks.src.optimizers.random_few_shot import RandomFewShot
from synalinks.src.saving import serialization_lib

ALL_OBJECTS = {
    # Base
    Optimizer,
    # Concrete
    EvolutionaryOptimizer,
    GreedyOptimizer,
    OMEGA,
    RandomFewShot,
}

ALL_OBJECTS_DICT = {cls.__name__.lower(): cls for cls in ALL_OBJECTS}


@synalinks_export("synalinks.optimizers.serialize")
def serialize(optimizer):
    """Serializes an `Optimizer` instance.

    Args:
        optimizer: A Synalinks `Optimizer` instance.

    Returns:
        Optimizer configuration dictionary.
    """
    return serialization_lib.serialize_synalinks_object(optimizer)


@synalinks_export("synalinks.optimizers.deserialize")
def deserialize(config, custom_objects=None):
    """Deserializes a serialized optimizer instance.

    Args:
        config: Optimizer configuration dictionary.
        custom_objects: Optional dictionary mapping names (strings)
            to custom objects (classes and functions) to be considered
            during deserialization.

    Returns:
        A Synalinks `Optimizer` instance.
    """
    # Make deserialization case-insensitive for built-in optimizers.
    if config["class_name"].lower() in ALL_OBJECTS_DICT:
        config["class_name"] = config["class_name"].lower()
    return serialization_lib.deserialize_synalinks_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@synalinks_export("synalinks.optimizers.get")
def get(identifier):
    """Retrieves a Synalinks `Optimizer` instance.

    The `identifier` may be the string name of an `Optimizer` class. LM/EM
    dependencies (e.g. for `OMEGA` or `EvolutionaryOptimizer`) are resolved
    at call time via `synalinks.set_default_language_model(...)` /
    `synalinks.set_default_embedding_model(...)` when not passed explicitly.

    >>> optimizer = optimizers.get("OMEGA")
    >>> type(optimizer)
    <class '...OMEGA'>
    >>> optimizer = optimizers.get("random_few_shot")
    >>> type(optimizer)
    <class '...RandomFewShot'>

    You can also pass a config dict (`{"class_name": ..., "config": ...}`)
    or an existing `Optimizer` instance (returned unchanged).

    Args:
        identifier: An optimizer identifier. One of `None`, a string class
            name (CamelCase or snake_case), a configuration dictionary, or
            an `Optimizer` instance.

    Returns:
        A Synalinks `Optimizer` instance.
    """
    if identifier is None:
        return None
    if isinstance(identifier, Optimizer):
        return identifier
    if isinstance(identifier, dict):
        return deserialize(identifier)
    if isinstance(identifier, str):
        cls = ALL_OBJECTS_DICT.get(identifier.lower())
        if cls is None:
            raise ValueError(f"Could not interpret optimizer identifier: {identifier}")
        return cls()
    raise ValueError(f"Could not interpret optimizer identifier: {identifier}")
