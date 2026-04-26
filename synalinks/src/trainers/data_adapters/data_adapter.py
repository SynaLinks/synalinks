# Modified from: keras/src/trainers/data_adapters/data_adapter.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)


class DataAdapter:
    """Base class for input data adapters.

    The purpose of a DataAdapter is to provide a unified interface to
    iterate over input data provided in a variety of formats -- such as
    Python lists, Python Generators, etc.
    """

    def get_numpy_iterator(self):
        """Get a Python iterable for the `DataAdapter`, that yields Numpy object arrays.

        Returns:
            A Python iterator.
        """
        raise NotImplementedError

    @property
    def num_batches(self):
        """Return the size (number of batches) for the dataset created.

        For certain type of the data input, the number of batches is known, eg
        for list data, the size is same as (number_of_element / batch_size).
        Whereas for dataset or python generator, the size is unknown since it
        may or may not have an end state.

        Returns:
            int, the number of batches for the dataset, or None if it is
            unknown.  The caller could use this to control the loop of training,
            show progress bar, or handle unexpected StopIteration error.
        """
        raise NotImplementedError

    @property
    def batch_size(self):
        """Return the batch size of the dataset created.

        For certain type of the data input, the batch size is known, and even
        required, like numpy array. Whereas for dataset, the batch is unknown
        unless we take a peek.

        Returns:
          int, the batch size of the dataset, or None if it is unknown.
        """
        raise NotImplementedError

    @property
    def has_partial_batch(self):
        """Whether the dataset has partial batch at the end."""
        raise NotImplementedError

    @property
    def partial_batch_size(self):
        """The size of the final partial batch for dataset.

        Will return None if has_partial_batch is False or batch_size is None.
        """
        raise NotImplementedError

    def on_epoch_begin(self):
        """A hook called before each epoch."""
        pass

    def on_epoch_end(self):
        """A hook called after each epoch."""
        pass
