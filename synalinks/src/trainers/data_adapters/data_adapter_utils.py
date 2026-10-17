# Modified from: keras/src/trainers/data_adapters/data_adapter_utils.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import numpy as np

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export

NUM_BATCHES_FOR_SPEC = 2


def get_numpy_iterator(iterable):
    """Wraps an iterable so each yielded structure has numpy-array leaves.

    Synalinks data adapters yield batches as `(x,)` or `(x, y)` tuples whose
    leaves are arrays of `DataModel` instances. This helper walks each batch
    with `tree.map_structure` and ensures every leaf is a numpy object array,
    matching the format produced by `ArrayDataAdapter.get_numpy_iterator`.
    Items that are already `np.ndarray` are passed through unchanged.
    """

    def _to_numpy(leaf):
        if isinstance(leaf, np.ndarray):
            return leaf
        return np.asarray(leaf, dtype="object")

    for batch in iterable:
        yield tree.map_structure(_to_numpy, batch)


@synalinks_export("synalinks.utils.unpack_x_y")
def unpack_x_y(data):
    """Unpacks user-provided data tuple.

    This is a convenience utility to be used when overriding
    `Program.train_step`, `Program.test_step`, or `Program.predict_step`.
    This utility makes it easy to support data of the form `(x,)`, or
    `(x, y)`.

    Args:
        data: A tuple of the form `(x,)`, or `(x, y)`.

    Returns:
        The unpacked tuple, with `None`s for `y` if they are
        not provided.
    """
    if isinstance(data, list):
        data = tuple(data)
    if not isinstance(data, tuple):
        return (data, None)
    elif len(data) == 1:
        return (data[0], None)
    elif len(data) == 2:
        return (data[0], data[1])
    error_msg = (
        f"Data is expected to be in format `x`, `(x,)`, or `(x, y)`, found: {data}"
    )
    raise ValueError(error_msg)


def unpack_x_targets(data):
    """`unpack_x_y`, with one `None` target per sample when there are none.

    Rewards, metrics and optimizers take a target for each prediction. Data
    without targets is for a reward that needs none (a judge grading the
    prediction on its own): each prediction gets `None` as its target.

    Args:
        data: A tuple of the form `(x,)`, or `(x, y)`.

    Returns:
        The unpacked `(x, y)`, `y` an object array of `None`s when the data
        has no targets.
    """
    x, y = unpack_x_y(data)
    if y is None and x is not None:
        y = np.array([None] * len(tree.flatten(x)[0]), dtype="object")
    return x, y


@synalinks_export("synalinks.utils.pack_x_y")
def pack_x_y(x, y=None):
    """Packs user-provided data into a tuple.

    This is a convenience utility for packing data into the tuple formats
    that `Program.fit()` uses.

    Args:
        x: `DataModel`s to pass to `Program`.
        y: Ground-truth targets to pass to `Program`.

    Returns:
        Tuple in the format used in `Program.fit()`.
    """
    if y is None:
        return (x,)
    return (x, y)
