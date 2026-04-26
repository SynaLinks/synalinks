# Modified from: keras/src/trainers/data_adapters/array_slicing.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import math

import numpy as np

from synalinks.src import tree

ARRAY_TYPES = (np.ndarray, list, tuple)


def can_slice_array(x):
    return x is None or isinstance(x, ARRAY_TYPES) or hasattr(x, "__array__")


def train_validation_split(arrays, validation_split):
    """Split arrays into train and validation subsets in deterministic order.

    The last part of data will become validation data.

    Args:
        arrays: Data models to split. Allowed inputs are arbitrarily nested
            structures of data models of NumPy arrays.
        validation_split: Float between 0 and 1. The proportion of the dataset
            to include in the validation split. The rest of the dataset will be
            included in the training split.

    Returns:
        `(train_arrays, validation_arrays)`
    """

    flat_arrays = tree.flatten(arrays)
    unsplitable = [type(t) for t in flat_arrays if not can_slice_array(t)]
    if unsplitable:
        raise ValueError(
            "Argument `validation_split` is only supported "
            "for NumPy arrays."
            f"Found incompatible type in the input: {unsplitable}"
        )

    if all(t is None for t in flat_arrays):
        return arrays, arrays

    first_non_none = None
    for t in flat_arrays:
        if t is not None:
            first_non_none = t
            break

    # Assumes all arrays have the same batch shape or are `None`.
    batch_dim = int(first_non_none.shape[0])
    split_at = int(math.floor(batch_dim * (1.0 - validation_split)))

    if split_at == 0 or split_at == batch_dim:
        raise ValueError(
            f"Training data contains {batch_dim} samples, which is not "
            "sufficient to split it into a validation and training set as "
            f"specified by `validation_split={validation_split}`. Either "
            "provide more data, or a different value for the "
            "`validation_split` argument."
        )

    def _split(t, start, end):
        if t is None:
            return t
        return t[start:end]

    train_arrays = tree.map_structure(lambda x: _split(x, start=0, end=split_at), arrays)
    val_arrays = tree.map_structure(
        lambda x: _split(x, start=split_at, end=batch_dim), arrays
    )
    return train_arrays, val_arrays
