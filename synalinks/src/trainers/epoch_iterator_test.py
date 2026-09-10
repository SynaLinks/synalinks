# Modified from: keras/src/trainers/epoch_iterator_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import warnings

import numpy as np

from synalinks.src import testing
from synalinks.src.testing.test_utils import AnswerWithRationale
from synalinks.src.testing.test_utils import Query
from synalinks.src.testing.test_utils import load_test_data
from synalinks.src.trainers.epoch_iterator import EpochIterator


def _batches(count):
    """A generator of `count` single-example batches, its length unknown."""
    for i in range(count):
        yield (np.array([Query(query=str(i))], dtype="object"),)


class _SizedBatches:
    """An iterable that declares `declared` batches but yields `count`."""

    def __init__(self, count, declared):
        self.count = count
        self.declared = declared

    def __iter__(self):
        return _batches(self.count)

    def __len__(self):
        return self.declared


def _drain(epoch_iterator):
    """Iterate one epoch to exhaustion, returning the warnings it emitted."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with epoch_iterator.catch_stop_iteration():
            for _, iterator in epoch_iterator:
                iterator[0]
    return [str(w.message) for w in caught if "ran out of data" in str(w.message)]


class EpochIteratorTest(testing.TestCase):
    def test_basic_flow(self):
        (x_train, y_train), (x_test, y_test) = load_test_data()

        epoch_iterator = EpochIterator(
            x=x_train,
            y=y_train,
            batch_size=1,
        )

        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator:
                data = iterator[0]
                x_batch, y_batch = data
                self.assertIsInstance(x_batch, np.ndarray)
                self.assertIsInstance(y_batch, np.ndarray)
                self.assertIsInstance(x_batch[0], Query)
                self.assertIsInstance(y_batch[0], AnswerWithRationale)

    def test_unknown_length_source_ends_quietly(self):
        # A streaming source without a limit ends when it ends: that is the
        # end of the epoch, not an interruption.
        epoch_iterator = EpochIterator(x=_batches(3))
        self.assertIsNone(epoch_iterator.num_batches)
        self.assertEqual(_drain(epoch_iterator), [])
        # The length is known from then on.
        self.assertEqual(epoch_iterator.num_batches, 3)

    def test_declared_length_running_short_warns(self):
        epoch_iterator = EpochIterator(x=_SizedBatches(count=2, declared=5))
        self.assertEqual(epoch_iterator.num_batches, 5)
        self.assertEqual(len(_drain(epoch_iterator)), 1)

    def test_steps_per_epoch_beyond_data_warns(self):
        epoch_iterator = EpochIterator(x=_batches(2), steps_per_epoch=5)
        self.assertEqual(len(_drain(epoch_iterator)), 1)
