from synalinks.src.api_export import synalinks_export
from synalinks.src.modules.language_models.language_model import LanguageModel
from synalinks.src.saving import serialization_lib

ALL_OBJECTS = {
    LanguageModel,
}

ALL_OBJECTS_DICT = {cls.__name__.lower(): cls for cls in ALL_OBJECTS}


@synalinks_export("synalinks.language_models.serialize")
def serialize(language_model):
    """Returns the language model configuration as a Python dict.

    Args:
        language_model (LanguageModel): A `LanguageModel` instance to serialize.

    Returns:
        Python dict which contains the configuration of the language model.
    """
    return serialization_lib.serialize_synalinks_object(language_model)


@synalinks_export("synalinks.language_models.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Synalinks language model object via its configuration.

    Args:
        config (dict): LanguageModel configuration dictionary.
        custom_objects (dict): Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Synalinks LanguageModel instance.
    """
    # Make deserialization case-insensitive for built-in language model.
    if config["class_name"].lower() in ALL_OBJECTS_DICT:
        config["class_name"] = config["class_name"].lower()

    return serialization_lib.deserialize_synalinks_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@synalinks_export("synalinks.language_models.get")
def get(identifier):
    """Retrieves a Synalinks LanguageModel instance.

    Resolution order: a concrete instance (or string / config dict) is
    always preferred; ``None`` falls back to the configured default
    (``synalinks.config.default_language_model()``). If no default is
    configured, a ``ValueError`` is raised — this function never
    returns ``None``.

    Args:
        identifier (str | dict | LanguageModel | None): LanguageModel
            identifier, one of:
            - String: a model name (e.g. `"openai/gpt-4o-mini"`), used
              to construct a `LanguageModel(model=identifier)`.
            - Dictionary: configuration dictionary.
            - Synalinks LanguageModel instance (returned unchanged).
            - ``None``: resolved against
              ``synalinks.set_default_language_model(...)``.

    Returns:
        (LanguageModel): A concrete Synalinks LanguageModel instance.
    """
    if identifier is None:
        # Lazy import to avoid a backend → modules cycle. ``None`` falls
        # back to the configured default (which itself may be ``None`` when
        # no default has been registered, in which case ``None`` is what
        # we return — callers that *require* a concrete instance see the
        # error from the actual call site, not from this resolver).
        from synalinks.src.backend.config import default_language_model

        identifier = default_language_model()
        if identifier is None:
            return None

    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = LanguageModel(model=identifier)
    else:
        obj = identifier

    if isinstance(obj, LanguageModel):
        return obj
    raise ValueError(f"Could not interpret language model identifier: {identifier}")
