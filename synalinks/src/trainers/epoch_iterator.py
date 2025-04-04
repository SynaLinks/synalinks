# Modified from: keras/src/trainers/epoch_iterator.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import contextlib
import warnings

from synalinks.src.trainers import data_adapters


class EpochIterator:
    def __init__(
        self,
        x,
        y=None,
        batch_size=None,
        steps_per_epoch=None,
        shuffle=False,
        steps_per_execution=1,
    ):
        self.steps_per_epoch = steps_per_epoch
        self.steps_per_execution = steps_per_execution
        self._current_iterator = None
        self._epoch_iterator = None
        self._steps_seen = 0
        self.data_adapter = data_adapters.get_data_adapter(
            x=x,
            y=y,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
        )
        self._num_batches = self.data_adapter.num_batches

    def _get_iterator(self):
        return self.data_adapter.get_numpy_iterator()

    def _interrupted_warning(self):
        warnings.warn(
            "Your input ran out of data; interrupting training. "
            "Make sure that your dataset or generator can generate "
            "at least `steps_per_epoch * epochs` batches. ",
            stacklevel=2,
        )

    def reset(self):
        self._current_iterator = None
        self._num_batches = self.data_adapter.num_batches
        self._steps_seen = 0
        self._epoch_iterator = None
        self.data_adapter.on_epoch_end()

    def _enumerate_iterator(self):
        self.data_adapter.on_epoch_begin()
        steps_per_epoch = self.steps_per_epoch or self._num_batches or -1

        if steps_per_epoch > 0:
            if self._current_iterator is None or self.steps_per_epoch is None:
                self._current_iterator = iter(self._get_iterator())
                self._steps_seen = 0
            for step in range(0, steps_per_epoch, self.steps_per_execution):
                if self._num_batches and self._steps_seen >= self._num_batches:
                    if self.steps_per_epoch:
                        self._interrupted_warning()
                    break
                self._steps_seen += self.steps_per_execution
                yield step, self._current_iterator
            if self._num_batches and self._steps_seen >= self._num_batches:
                self._current_iterator = iter(self._get_iterator())
                self._steps_seen = 0
        else:
            iterator = iter(self._get_iterator())
            step = -self.steps_per_execution
            while True:
                step += self.steps_per_execution
                self._steps_seen = step + self.steps_per_execution
                yield step, iterator
        self.data_adapter.on_epoch_end()

    def __iter__(self):
        self._epoch_iterator = self._enumerate_iterator()
        return self

    def __next__(self):
        buffer = []
        step, iterator = next(self._epoch_iterator)
        with self.catch_stop_iteration():
            for _ in range(self.steps_per_execution):
                data = next(iterator)
                buffer.append(data)
            return step, buffer
        if buffer:
            return step, buffer
        raise StopIteration

    def enumerate_epoch(self):
        for step, data in self:
            yield step, data

    @contextlib.contextmanager
    def catch_stop_iteration(self):
        """Catches errors when an iterator runs out of data."""
        try:
            yield
        except StopIteration:
            if self._num_batches is None:
                self._num_batches = self._steps_seen
            self._interrupted_warning()
            self._current_iterator = None
            self.data_adapter.on_epoch_end()

    @property
    def num_batches(self):
        if self.steps_per_epoch:
            return self.steps_per_epoch
        # Either copied from the data_adapter, or
        # inferred at the end of an iteration.
        return self._num_batches
