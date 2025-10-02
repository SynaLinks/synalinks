# Modified from: keras/src/backend/numpy/numpy.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

# Because we only use tensors for metrics and rewards, we don't need much
# and because we don't perform any gradient descent
# nor store weights that can benefit from specific dtype
# we don't need dtype inference as we can use floatx for everything

import numpy as np

from synalinks.src.backend import floatx


def standardize_axis_for_numpy(axis):
    """Standardize an axis to a tuple if it is a list."""
    return tuple(axis) if isinstance(axis, list) else axis


def convert_to_numpy(x):
    return np.array(x)


def shape(x):
    return x.shape


def convert_to_tensor(x):
    return np.array(x, dtype=floatx())


def zeros(shape, dtype=None):
    return np.zeros(shape, dtype=floatx())


def add(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return np.add(x1, x2)


def subtract(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return np.subtract(x1, x2)


def multiply(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return np.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    return np.mean(x, axis=axis, keepdims=keepdims).astype(floatx())


def median(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    return np.median(x, axis=axis, keepdims=keepdims).astype(floatx())


def sum(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    return np.sum(x, axis=axis, keepdims=keepdims).astype(floatx())


def divide(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return np.divide(x1, x2)


def prod(x, axis=None, keepdims=False):
    axis = standardize_axis_for_numpy(axis)
    x = convert_to_tensor(x)
    return np.prod(x, axis=axis, keepdims=keepdims, dtype=floatx())


def squeeze(x, axis=None):
    axis = standardize_axis_for_numpy(axis)
    return np.squeeze(x, axis=axis)


def expand_dims(x, axis):
    axis = standardize_axis_for_numpy(axis)
    return np.expand_dims(x, axis)


def divide_no_nan(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    # No need for the double-where trick since we don't calculate gradients in
    # numpy backend.
    return np.where(x2 == 0, np.array(0, dtype=floatx()), np.divide(x1, x2))


def broadcast_to(x, shape):
    return np.broadcast_to(x, shape)


def normalize(x, axis=-1, order=2):
    norm = np.atleast_1d(np.linalg.norm(x, order, axis))
    norm[norm == 0] = 1

    # axis cannot be `None`
    axis = axis or -1
    return x / np.expand_dims(norm, axis)
