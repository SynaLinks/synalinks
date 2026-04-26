# Modified from: keras/src/trainers/epoch_iterator_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import numpy as np

from synalinks.src import testing
from synalinks.src.testing.test_utils import AnswerWithRationale
from synalinks.src.testing.test_utils import Query
from synalinks.src.testing.test_utils import load_test_data
from synalinks.src.trainers.epoch_iterator import EpochIterator


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
