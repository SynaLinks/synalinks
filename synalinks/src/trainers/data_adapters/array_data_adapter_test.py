# Modified from: keras/src/trainers/data_adapters/array_data_adapter_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from absl.testing import parameterized

from synalinks.src import testing
from synalinks.src.testing.test_utils import AnswerWithRationale
from synalinks.src.testing.test_utils import Query
from synalinks.src.testing.test_utils import load_test_data
from synalinks.src.testing.test_utils import named_product
from synalinks.src.trainers.data_adapters.array_data_adapter import ArrayDataAdapter


class TestArrayDataAdapter(testing.TestCase):
    @parameterized.named_parameters(
        named_product(
            shuffle=[False, "batch", True],
        )
    )
    def test_basic_flow(self, shuffle):
        (x, y), (x_test, y_test) = load_test_data()

        adapter = ArrayDataAdapter(
            x,
            y=y,
            batch_size=32,
            steps=None,
            shuffle=shuffle,
        )
        self.assertEqual(adapter.num_batches, 1)
        self.assertEqual(adapter.batch_size, 32)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 15)

        it = adapter.get_numpy_iterator()
        for i, batch in enumerate(it):
            self.assertEqual(len(batch), 2)
            x, y = batch
            self.assertIsInstance(batch, tuple)
            self.assertIsInstance(x[0], Query)
            self.assertIsInstance(x[1], Query)
            self.assertIsInstance(y[0], AnswerWithRationale)
            self.assertIsInstance(y[1], AnswerWithRationale)
