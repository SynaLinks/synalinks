# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import numpy as np

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


class _Batches:
    """A re-iterable Dataset-like source: a fresh pass on every `iter()`."""

    def __init__(self, n):
        self.n = n
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        for i in range(self.n):
            yield (np.array([i], dtype="object"), np.array([i], dtype="object"))


class ReiterableSourceTest(testing.TestCase):
    def test_reiterable_yields_every_batch_on_every_epoch(self):
        source = _Batches(5)
        adapter = GeneratorDataAdapter(source)
        for _ in range(3):  # three "epochs"
            self.assertEqual(len(list(adapter.get_numpy_iterator())), 5)

    def test_get_data_adapter_keeps_iterable_reiterable(self):
        from synalinks.src.trainers.data_adapters import get_data_adapter

        source = _Batches(4)
        adapter = get_data_adapter(source)
        self.assertEqual(len(list(adapter.get_numpy_iterator())), 4)
        self.assertEqual(len(list(adapter.get_numpy_iterator())), 4)

    def test_epoch_iterator_runs_all_batches_after_first_epoch(self):
        from synalinks.src.trainers.epoch_iterator import EpochIterator

        it = EpochIterator(_Batches(6))
        for epoch in range(3):
            steps = 0
            with it.catch_stop_iteration():
                for _step, _data in it:
                    steps += 1
            self.assertEqual(steps, 6, f"epoch {epoch} ran {steps} steps")

    def test_one_shot_generator_is_still_accepted(self):
        def gen():
            for i in range(3):
                yield (np.array([i], dtype="object"),)

        adapter = GeneratorDataAdapter(gen())
        self.assertEqual(len(list(adapter.get_numpy_iterator())), 3)
