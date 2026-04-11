# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.trainers.data_adapters.generator_data_adapter import (
    GeneratorDataAdapter,
)
from synalinks.src.trainers.data_adapters.generator_data_adapter import peek_and_restore


class PeekAndRestoreTest(testing.TestCase):
    def test_peek_and_restore(self):
        def gen():
            for i in range(5):
                yield i

        batches, restored_gen = peek_and_restore(gen())
        # Should peek NUM_BATCHES_FOR_SPEC (2) items
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0], 0)
        self.assertEqual(batches[1], 1)

        # Restored generator should yield all items
        all_items = list(restored_gen())
        self.assertEqual(all_items, [0, 1, 2, 3, 4])

    def test_peek_and_restore_short_gen(self):
        def gen():
            yield 42

        batches, restored_gen = peek_and_restore(gen())
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0], 42)

        all_items = list(restored_gen())
        self.assertEqual(all_items, [42])


class GeneratorDataAdapterTest(testing.TestCase):
    def test_valid_tuple_generator(self):
        def gen():
            for i in range(5):
                yield ({"input": i},)

        adapter = GeneratorDataAdapter(gen())
        self.assertIsNone(adapter.num_batches)
        self.assertIsNone(adapter.batch_size)

    def test_valid_xy_tuple_generator(self):
        def gen():
            for i in range(5):
                yield ({"input": i}, {"target": i * 2})

        adapter = GeneratorDataAdapter(gen())
        self.assertIsNone(adapter.num_batches)
        self.assertIsNone(adapter.batch_size)

    def test_non_tuple_raises(self):
        def gen():
            for i in range(5):
                yield {"input": i}

        with self.assertRaises(ValueError) as ctx:
            GeneratorDataAdapter(gen())
        self.assertIn("must return a tuple", str(ctx.exception))
