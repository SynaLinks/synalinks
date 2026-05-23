# Modified from: keras/src/trainers/data_adapters/generator_data_adapter.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import itertools

from synalinks.src.trainers.data_adapters import data_adapter_utils
from synalinks.src.trainers.data_adapters.data_adapter import DataAdapter


class GeneratorDataAdapter(DataAdapter):
    """Adapter for Python generators."""

    def __init__(self, generator, num_batches=None):
        first_batches, generator = peek_and_restore(generator)
        self.generator = generator
        self._first_batches = first_batches
        self._output_signature = None
        self._num_batches = num_batches
        if not isinstance(first_batches[0], tuple):
            raise ValueError(
                "When passing a Python generator to a Synalinks program, "
                "the generator must return a tuple, either "
                "(input,) or (inputs, targets). "
                f"Received: {first_batches[0]}"
            )

    def get_numpy_iterator(self):
        """Get a Python iterable that yields Numpy object arrays.

        Returns:
            A Python iterator yielding batches from the wrapped generator,
            with each element converted to Numpy object arrays.
        """
        return data_adapter_utils.get_numpy_iterator(self.generator())

    @property
    def num_batches(self):
        """Return the number of batches in the dataset.

        Returns:
            int, the number of batches if it was supplied at construction,
            or ``None`` when unknown (a generator may have no end state).
        """
        return self._num_batches

    @property
    def batch_size(self):
        """Return the batch size of the dataset.

        Returns:
            ``None``, since the batch size of a generator is not known
            without consuming it.
        """
        return None


def peek_and_restore(generator):
    batches = list(itertools.islice(generator, data_adapter_utils.NUM_BATCHES_FOR_SPEC))
    return batches, lambda: itertools.chain(batches, generator)
