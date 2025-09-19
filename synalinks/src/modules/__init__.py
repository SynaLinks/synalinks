from synalinks.src.api_export import synalinks_export
from synalinks.src.modules.core.action import Action
from synalinks.src.modules.core.branch import Branch
from synalinks.src.modules.core.decision import Decision
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.core.generator import default_instructions
from synalinks.src.modules.core.generator import default_prompt_template
from synalinks.src.modules.core.identity import Identity
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.input_module import InputModule
from synalinks.src.modules.core.not_module import Not
from synalinks.src.modules.knowledge.embedding import Embedding
from synalinks.src.modules.knowledge.update_knowledge import UpdateKnowledge
from synalinks.src.modules.module import Module
from synalinks.src.modules.ttc.chain_of_thought import ChainOfThought
from synalinks.src.modules.ttc.self_critique import SelfCritique
from synalinks.src.saving import serialization_lib


@synalinks_export("synalinks.modules.serialize")
def serialize(module):
    """Returns the module configuration as a Python dict.

    Args:
        module: A `synalinks.modules.module` instance to serialize.

    Returns:
        Python dict which contains the configuration of the module.
    """
    return serialization_lib.serialize_synalinks_object(module)


@synalinks_export("synalinks.modules.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Synalinks module object via its configuration.

    Args:
        config: A python dict containing a serialized module configuration.
        custom_objects: Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Synalinks module instance.
    """
    obj = serialization_lib.deserialize_synalinks_object(
        config,
        custom_objects=custom_objects,
    )
    if not isinstance(obj, Module):
        raise ValueError(
            "`synalinks.modules.deserialize` was passed a `config` object that is "
            f"not a `synalinks.modules.Module`. Received: {config}"
        )
    return obj
