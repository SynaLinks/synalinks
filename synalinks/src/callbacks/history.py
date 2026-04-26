# Modified from: keras/src/callbacks/history.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.callbacks.callback import Callback


@synalinks_export("synalinks.callbacks.History")
class History(Callback):
    """Callback that records events into a `History` object.

    This callback is automatically applied to
    every Synalinks program. The `History` object
    gets returned by the `fit()` method of programs.
    """

    def __init__(self):
        super().__init__()
        self.history = {}

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # Set the history attribute on the program after the epoch ends. This will
        # make sure that the state which is set is the latest one.
        self.program.history = self
