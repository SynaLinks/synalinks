import warnings

import numpy as np

from synalinks.src.api_export import synalinks_export
from synalinks.src.callbacks.callback import Callback
from synalinks.src.trainers import compile_utils
from synalinks.src.utils import io_utils


@synalinks_export("synalinks.callbacks.EarlyStopping")
class EarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.

    Assuming the goal of a training is to maximize the reward.
    With this, the metric to be monitored would be `reward`,
    and mode would be `max`. A `program.fit()` training loop
    will check at end of every epoch whether the reward is no
    longer augmenting, considering the `min_delta` and `patience`
    if applicable. Once it's found no longer increasing,
    `program.stop_training` is marked True and the training terminates.

    The quantity to be monitored needs to be available in `logs` dict
    To make it so, pass the reward or metrics at `program.compile`.

    Example:

    >>> callback = synalinks.callbacks.EarlyStopping(monitor='reward', patience=3)
    >>> # This callback will stop the training when there is no improvement in
    >>> # the loss for three consecutive epochs.
    >>> program = synalinks.programs.Sequential([synalinks.modules.Generator(data_model=Answer)])
    >>> program.compile(synalinks.optimizers.RandomFewShot(), reward=synalinks.rewards.ExactMatch())
    >>> history = program.fit(..., epochs=10, batch_size=1, callbacks=[callback], verbose=0)
    >>> len(history.history['reward'])  # Only 4 epochs are run.
    4

    Args:
        monitor: Quantity to be monitored. Defaults to `val_reward`.
        min_delta: Minimum change in the monitored quantity to qualify as an
            improvement, i.e. an absolute change of less than min_delta, will
            count as no improvement. Defaults to `0`.
        stop_at: The value at which we should stop training.
        patience: Number of epochs with no improvement after which training will
            be stopped. Defaults to `0`.
        verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1 displays
            messages when the callback takes an action. Defaults to `0`.
        mode: One of `{"auto", "min", "max"}`. In `min` mode, training will stop
            when the quantity monitored has stopped decreasing; in `"max"` mode
            it will stop when the quantity monitored has stopped increasing; in
            `"auto"` mode, the direction is automatically inferred from the name
            of the monitored quantity. Defaults to `"auto"`.
        baseline: Baseline value for the monitored quantity. If not `None`,
            training will stop if the program doesn't show improvement over the
            baseline. Defaults to `None`.
        restore_best_variables: Whether to restore program variables from the epoch
            with the best value of the monitored quantity. If `False`, the program
            variables obtained at the last step of training are used. An epoch
            will be restored regardless of the performance relative to the
            `baseline`. If no epoch improves on `baseline`, training will run
            for `patience` epochs and restore variables from the best epoch in
            that set. Defaults to `False`.
        start_from_epoch: Number of epochs to wait before starting to monitor
            improvement. This allows for a warm-up period in which no
            improvement is expected and thus training will not be stopped.
            Defaults to `0`.

    """

    def __init__(
        self,
        monitor="val_reward",
        min_delta=0,
        stop_at=1.0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_variables=False,
        start_from_epoch=0,
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.stop_at = stop_at
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_variables = restore_best_variables
        self.best_variables = None
        self.start_from_epoch = start_from_epoch

        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"EarlyStopping mode {mode} is unknown, fallback to auto mode.",
                stacklevel=2,
            )
            mode = "auto"
        self.mode = mode
        self.monitor_op = None

    def _set_monitor_op(self):
        if self.mode == "min":
            self.monitor_op = np.less
        elif self.mode == "max":
            self.monitor_op = np.greater
        else:
            metric_name = self.monitor.removeprefix("val_")
            if metric_name == "reward":
                self.monitor_op = np.greater
            if hasattr(self.program, "metrics"):
                all_metrics = []
                for m in self.program.metrics:
                    if isinstance(
                        m,
                        (
                            compile_utils.CompileMetrics,
                            compile_utils.MetricsList,
                        ),
                    ):
                        all_metrics.extend(m.metrics)
                for m in all_metrics:
                    if m.name == metric_name:
                        if hasattr(m, "_direction"):
                            if m._direction == "up":
                                self.monitor_op = np.greater
                            else:
                                self.monitor_op = np.less
        if self.monitor_op is None:
            raise ValueError(
                f"EarlyStopping callback received monitor={self.monitor} "
                "but Synalinks isn't able to automatically determine whether "
                "that metric should be maximized or minimized. "
                "Pass `mode='max'` in order to do early stopping based "
                "on the highest metric value, or pass `mode='min'` "
                "in order to use the lowest value."
            )
        if self.monitor_op == np.less:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = None
        self.best_variables = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.monitor_op is None:
            # Delay setup until the model's metrics are all built
            self._set_monitor_op()

        current = self.get_monitor_value(logs)
        if current >= self.stop_at:
            self.program.stop_training = True
            return
        if current is None or epoch < self.start_from_epoch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_variables and self.best_variables is None:
            # If best variables were never set,
            # then the current variables are the best.
            self.best_variables = self.program.get_variables()
            self.best_epoch = epoch

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            if self.restore_best_variables:
                self.best_variables = self.program.get_variables()
            # Only restart wait if we beat both the baseline and our previous
            # best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0
            return

        if self.wait >= self.patience and epoch > 0:
            # Patience has been exceeded: stop training
            self.stopped_epoch = epoch
            self.program.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            io_utils.print_msg(f"Epoch {self.stopped_epoch + 1}: early stopping")
        if self.restore_best_variables and self.best_variables is not None:
            if self.verbose > 0:
                io_utils.print_msg(
                    "Restoring model variables from "
                    "the end of the best epoch: "
                    f"{self.best_epoch + 1}."
                )
            self.program.set_variables(self.best_variables)

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                (
                    f"Early stopping conditioned on metric `{self.monitor}` "
                    "which is not available. "
                    f"Available metrics are: {','.join(list(logs.keys()))}"
                ),
                stacklevel=2,
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        if reference_value is None:
            return True
        return self.monitor_op(monitor_value - self.min_delta, reference_value)
