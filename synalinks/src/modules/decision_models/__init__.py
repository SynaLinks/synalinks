from synalinks.src.api_export import synalinks_export
from synalinks.src.modules.decision_models.decision_model import DecisionModel
from synalinks.src.saving import serialization_lib

ALL_OBJECTS = {
    DecisionModel,
}

ALL_OBJECTS_DICT = {cls.__name__.lower(): cls for cls in ALL_OBJECTS}


@synalinks_export("synalinks.decision_models.serialize")
def serialize(decision_model):
    """Returns the decision model configuration as a Python dict.

    Args:
        decision_model (DecisionModel): A `DecisionModel` instance to serialize.

    Returns:
        Python dict which contains the configuration of the decision model.
    """
    return serialization_lib.serialize_synalinks_object(decision_model)


@synalinks_export("synalinks.decision_models.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Synalinks decision model object via its configuration.

    Args:
        config (dict): DecisionModel configuration dictionary.
        custom_objects (dict): Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Synalinks DecisionModel instance.
    """
    # Make deserialization case-insensitive for built-in decision models.
    if config["class_name"].lower() in ALL_OBJECTS_DICT:
        config["class_name"] = config["class_name"].lower()

    return serialization_lib.deserialize_synalinks_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@synalinks_export("synalinks.decision_models.supported_providers")
def supported_providers():
    """Returns the supported decision model provider prefixes.

    Shortcut for `DecisionModel.supported_providers()`.

    ```python
    import synalinks

    print(synalinks.decision_models.supported_providers())
    ```

    Returns:
        (list): The sorted list of supported provider prefixes.
    """
    return DecisionModel.supported_providers()


@synalinks_export("synalinks.decision_models.get")
def get(identifier):
    """Retrieves a Synalinks DecisionModel instance.

    Resolution order: a concrete instance (or string / config dict) is
    always preferred; ``None`` falls back to the configured default
    (``synalinks.config.default_decision_model()``). If no default is
    configured, a ``ValueError`` is raised; this function never
    returns ``None``.

    Args:
        identifier (str | dict | DecisionModel | None): DecisionModel
            identifier, one of:
            - String: a model name (e.g. `"typesafe/jev-latest"`),
              used to construct a `DecisionModel(model=identifier)`.
            - Dictionary: configuration dictionary.
            - Synalinks DecisionModel instance (returned unchanged).
            - ``None``: resolved against
              ``synalinks.set_default_decision_model(...)``.

    Returns:
        (DecisionModel): A concrete Synalinks DecisionModel instance.
    """
    if identifier is None:
        # Lazy import to avoid a backend → modules cycle. ``None`` falls
        # back to the configured default (which itself may be ``None``
        # when no default has been registered, in which case ``None``
        # is what we return; callers that *require* a concrete
        # instance see the error from the actual call site).
        from synalinks.src.backend.config import default_decision_model

        identifier = default_decision_model()
        if identifier is None:
            return None

    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = DecisionModel(model=identifier)
    else:
        obj = identifier

    if isinstance(obj, DecisionModel):
        return obj
    raise ValueError(f"Could not interpret decision model identifier: {identifier}")
