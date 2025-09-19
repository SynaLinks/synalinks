from synalinks.src.api_export import synalinks_export
from synalinks.src.optimizers.omega_evolve import OMEGAEvolve
from synalinks.src.optimizers.optimizer import Optimizer
from synalinks.src.optimizers.random_few_shot import RandomFewShot
from synalinks.src.saving import serialization_lib

ALL_OBJECTS = {
    Optimizer,
    RandomFewShot,
    OMEGAEvolve,
}

ALL_OBJECTS_DICT = {cls.__name__.lower(): cls for cls in ALL_OBJECTS}


@synalinks_export("synalinks.optimizers.serialize")
def serialize(optimizer):
    """Returns the optimizer configuration as a Python dict.

    Args:
        optimizer: An `Optimizer` instance to serialize.

    Returns:
        Python dict which contains the configuration of the optimizer.
    """
    return serialization_lib.serialize_synalinks_object(optimizer)


@synalinks_export("synalinks.optimizers.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Synalinks optimizer object via its configuration.

    Args:
        config: Optimizer configuration dictionary.
        custom_objects: Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Synalinks Optimizer instance.
    """
    # Make deserialization case-insensitive for built-in optimizers.
    if config["class_name"].lower() in ALL_OBJECTS_DICT:
        config["class_name"] = config["class_name"].lower()

    return serialization_lib.deserialize_synalinks_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )
