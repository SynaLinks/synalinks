from synalinks.src.api_export import synalinks_export
from synalinks.src.modules.embedding_models.embedding_model import EmbeddingModel
from synalinks.src.saving import serialization_lib

ALL_OBJECTS = {
    EmbeddingModel,
}

ALL_OBJECTS_DICT = {cls.__name__.lower(): cls for cls in ALL_OBJECTS}


@synalinks_export("synalinks.embedding_models.serialize")
def serialize(embedding_model):
    """Returns the embedding model configuration as a Python dict.

    Args:
        embedding_model (EmbeddingModel): An `EmbeddingModel` instance to serialize.

    Returns:
        Python dict which contains the configuration of the embedding model.
    """
    return serialization_lib.serialize_synalinks_object(embedding_model)


@synalinks_export("synalinks.embedding_models.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Synalinks embedding model object via its configuration.

    Args:
        config (dict): EmbeddingModel configuration dictionary.
        custom_objects (dict): Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Synalinks EmbeddingModel instance.
    """
    # Make deserialization case-insensitive for built-in embedding model.
    if config["class_name"].lower() in ALL_OBJECTS_DICT:
        config["class_name"] = config["class_name"].lower()

    return serialization_lib.deserialize_synalinks_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@synalinks_export("synalinks.embedding_models.get")
def get(identifier):
    """Retrieves a Synalinks EmbeddingModel instance.

    Args:
        identifier (str | dict | LanguageModel): EmbeddingModel identifier, one of:
            - String: a model name (e.g. `"openai/text-embedding-3-small"`),
              used to construct an `EmbeddingModel(model=identifier)`.
            - Dictionary: configuration dictionary.
            - Synalinks EmbeddingModel instance (it will be returned unchanged).

    Returns:
        A Synalinks EmbeddingModel instance.
    """
    if identifier is None:
        return None
    elif isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = EmbeddingModel(model=identifier)
    else:
        obj = identifier

    if isinstance(obj, EmbeddingModel):
        return obj
    raise ValueError(f"Could not interpret embedding model identifier: {identifier}")
