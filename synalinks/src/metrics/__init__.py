import inspect

from synalinks.src.api_export import synalinks_export
from synalinks.src.metrics.metric import Metric
from synalinks.src.metrics.reduction_metrics import Mean
from synalinks.src.metrics.reduction_metrics import MeanMetricWrapper
from synalinks.src.metrics.reduction_metrics import Sum
from synalinks.src.saving import serialization_lib

ALL_OBJECTS = {
    # Base
    Metric,
    Mean,
    MeanMetricWrapper,
    Sum,
}

ALL_OBJECTS_DICT = {cls.__name__.lower(): cls for cls in ALL_OBJECTS}


@synalinks_export("synalinks.metrics.serialize")
def serialize(metric):
    """Serializes metric function or `Metric` instance.

    Args:
        metric: A Synalinks `Metric` instance or a metric function.

    Returns:
        Metric configuration dictionary.
    """
    return serialization_lib.serialize_synalinks_object(metric)


@synalinks_export("synalinks.metrics.deserialize")
def deserialize(config, custom_objects=None):
    """Deserializes a serialized metric class/function instance.

    Args:
        config: Metric configuration.
        custom_objects: Optional dictionary mapping names (strings)
            to custom objects (classes and functions) to be
            considered during deserialization.

    Returns:
        A Synalinks `Metric` instance or a metric function.
    """
    # Make deserialization case-insensitive for built-in metrics.
    if config["class_name"].lower() in ALL_OBJECTS_DICT:
        config["class_name"] = config["class_name"].lower()
    return serialization_lib.deserialize_synalinks_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@synalinks_export("synalinks.metrics.get")
def get(identifier):
    """Retrieves a Synalinks metric as a `function`/`Metric` class instance.

    The `identifier` may be the string name of a metric function or class.

    >>> metric = metrics.get("categorical_crossentropy")
    >>> type(metric)
    <class 'function'>
    >>> metric = metrics.get("CategoricalCrossentropy")
    >>> type(metric)
    <class '...metrics.CategoricalCrossentropy'>

    You can also specify `config` of the metric to this function by passing dict
    containing `class_name` and `config` as an identifier. Also note that the
    `class_name` must map to a `Metric` class

    >>> identifier = {"class_name": "CategoricalCrossentropy",
    ...               "config": {"from_logits": True}}
    >>> metric = metrics.get(identifier)
    >>> type(metric)
    <class '...metrics.CategoricalCrossentropy'>

    Args:
        identifier: A metric identifier. One of None or string name of a metric
            function/class or metric configuration dictionary or a metric
            function or a metric class instance

    Returns:
        A Synalinks metric as a `function`/ `Metric` class instance.
    """
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        obj = ALL_OBJECTS_DICT.get(identifier.lower(), None)
    else:
        obj = identifier
    if callable(obj):
        if inspect.isclass(obj):
            obj = obj()
        return obj
    else:
        raise ValueError(f"Could not interpret metric identifier: {identifier}")
