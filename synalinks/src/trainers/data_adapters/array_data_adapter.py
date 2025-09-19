# Modified from: keras/src/trainers/data_adapters/array_data_adapter.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import functools
import math

import numpy as np

from synalinks.src import tree
from synalinks.src.trainers.data_adapters import array_slicing
from synalinks.src.trainers.data_adapters import data_adapter_utils
from synalinks.src.trainers.data_adapters.data_adapter import DataAdapter


class ArrayDataAdapter(DataAdapter):
    """Adapter for array-like objects, e.g. NumPy arrays."""

    def __init__(
        self,
        x,
        y=None,
        batch_size=None,
        steps=None,
        shuffle=False,
    ):
        inputs = data_adapter_utils.pack_x_y(x, y)
        num_samples = set(i.shape[0] for i in tree.flatten(inputs)).pop()
        self._num_samples = num_samples
        self._inputs = inputs

        # If batch_size is not passed but steps is, calculate from the input
        # data.  Defaults to `1`.
        if not batch_size:
            batch_size = int(math.ceil(num_samples / steps)) if steps else 1

        self._size = int(math.ceil(num_samples / batch_size))
        self._batch_size = batch_size
        self._partial_batch_size = num_samples % batch_size
        self._shuffle = shuffle

    def get_numpy_iterator(self):
        inputs = np.array(self._inputs, dtype="object")

        def slice_and_convert_to_numpy(sliceable, indices=None):
            x = sliceable[indices]
            x = np.array(x, dtype="object")
            return x

        return self._get_iterator(slice_and_convert_to_numpy, inputs)

    def _get_iterator(self, slice_and_convert_fn, inputs):
        global_permutation = None
        if self._shuffle and self._shuffle != "batch":
            global_permutation = np.random.permutation(self._num_samples)

        for i in range(self._size):
            start = i * self._batch_size
            stop = min((i + 1) * self._batch_size, self._num_samples)
            if self._shuffle == "batch":
                indices = np.random.permutation(stop - start) + start
            elif self._shuffle:
                indices = global_permutation[start:stop]
            else:
                indices = slice(start, stop)

            slice_indices_and_convert_fn = functools.partial(
                slice_and_convert_fn, indices=indices
            )
            x = tree.map_structure(slice_indices_and_convert_fn, inputs[0])
            if inputs.shape[0] > 1:
                y = tree.map_structure(slice_indices_and_convert_fn, inputs[1])
            else:
                y = None
            yield data_adapter_utils.pack_x_y(x, y=y)

    @property
    def num_batches(self):
        return self._size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def has_partial_batch(self):
        return self._partial_batch_size > 0

    @property
    def partial_batch_size(self):
        return self._partial_batch_size or None


def can_convert_arrays(arrays):
    """Check if array like-inputs can be handled by `ArrayDataAdapter`

    Args:
        inputs: Structure of `Tensor`s, NumPy arrays, or tensor-like.

    Returns:
        `True` if `arrays` can be handled by `ArrayDataAdapter`, `False`
        otherwise.
    """

    return all(tree.flatten(tree.map_structure(array_slicing.can_slice_array, arrays)))
