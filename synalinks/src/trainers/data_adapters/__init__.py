import types

from synalinks.src.trainers.data_adapters import array_data_adapter
from synalinks.src.trainers.data_adapters import data_adapter
from synalinks.src.trainers.data_adapters.array_data_adapter import ArrayDataAdapter
from synalinks.src.trainers.data_adapters.generator_data_adapter import (
    GeneratorDataAdapter,
)


def get_data_adapter(
    x,
    y=None,
    batch_size=None,
    steps_per_epoch=None,
    shuffle=False,
):
    # Allow passing a custom data adapter.
    if isinstance(x, data_adapter.DataAdapter):
        return x

    if array_data_adapter.can_convert_arrays((x, y)):
        return ArrayDataAdapter(
            x,
            y,
            shuffle=shuffle,
            batch_size=batch_size,
            steps=steps_per_epoch,
        )
    elif isinstance(x, types.GeneratorType):
        if y is not None:
            raise_unsupported_arg("y", "the targets", "PyDataset")
        return GeneratorDataAdapter(x)

    elif hasattr(x, "__iter__") and not isinstance(x, (str, bytes, dict)):
        # Custom iterable (e.g. a user-defined `Dataset` class). If it also
        # exposes `__len__`, pass that through as `num_batches` so synalinks
        # knows the epoch is bounded.
        if y is not None:
            raise_unsupported_arg("y", "the targets", "PyDataset")
        try:
            num_batches = len(x)
        except (TypeError, NotImplementedError):
            num_batches = None
        return GeneratorDataAdapter(iter(x), num_batches=num_batches)

    else:
        raise ValueError(f"Unrecognized data type: x={x} (of type {type(x)})")


def raise_unsupported_arg(arg_name, arg_description, input_type):
    raise ValueError(
        f"When providing `x` as a {input_type}, `{arg_name}` "
        f"should not be passed. Instead, {arg_description} should "
        f"be included as part of the {input_type}."
    )
