# Modified from: keras/src/callbacks/callback.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import utils
from synalinks.src.api_export import synalinks_export


@synalinks_export("synalinks.callbacks.Callback")
class Callback:
    """Base class used to build new callbacks.

    Callbacks can be passed to synalinks methods such as `fit()`, `evaluate()`, and
    `predict()` in order to hook into the various stages of the program training,
    evaluation, and inference lifecycle.

    To create a custom callback, subclass `synalinks.callbacks.Callback` and
    override the method associated with the stage of interest.

    If you want to use `Callback` objects in a custom training loop:

    1. You should pack all your callbacks into a single `callbacks.CallbackList`
       so they can all be called together.
    2. You will need to manually call all the `on_*` methods at the appropriate
       locations in your loop.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch (see method-specific docstrings).

    Attributes:
        params (dict): Training parameters
            (eg. verbosity, batch size, number of epochs...).
        program (Program): Instance of `Program`.
            Reference of the program being trained.
    """

    def __init__(self):
        self.params = None
        self._program = None

    def set_params(self, params):
        self.params = params

    def set_program(self, program):
        self._program = program

    @property
    def program(self):
        return self._program

    @utils.default
    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""

    @utils.default
    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""

    @utils.default
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch (int): Index of epoch.
            logs (dict): Currently no data is passed to this argument for this
                method but that may change in the future.
        """

    @utils.default
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch (int): Index of epoch.
            logs (dict): Metric results for this training epoch, and for the
                validation epoch if validation is performed. Validation result
                keys are prefixed with `val_`. For training epoch, the values of
                the `Program`'s metrics are returned. Example:
                `{'reward': 0.2, 'accuracy': 0.7}`.
        """

    @utils.default
    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Program` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch (int): Index of batch within the current epoch.
            logs (dict): Currently no data is passed to this argument for this
                method but that may change in the future.
        """
        # For backwards compatibility.
        self.on_batch_begin(batch, logs=logs)

    @utils.default
    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Program` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch (int): Index of batch within the current epoch.
            logs (dict): Aggregated metric results up until this batch.
        """
        # For backwards compatibility.
        self.on_batch_end(batch, logs=logs)

    @utils.default
    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.

        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Program` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch (int): Index of batch within the current epoch.
            logs (dict): Currently no data is passed to this argument for this
                method but that may change in the future.
        """

    @utils.default
    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Program` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch (int): Index of batch within the current epoch.
            logs (dict): Aggregated metric results up until this batch.
        """

    @utils.default
    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Program` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch (int): Index of batch within the current epoch.
            logs (dict): Currently no data is passed to this argument for this
                method but that may change in the future.
        """

    @utils.default
    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Program` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch (int): Index of batch within the current epoch.
            logs (dict): Aggregated metric results up until this batch.
        """

    @utils.default
    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs (dict): Currently no data is passed to this argument for this
                method but that may change in the future.
        """

    @utils.default
    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs (dict): Currently the output of the last call to
                `on_epoch_end()` is passed to this argument for this method but
                that may change in the future.
        """

    @utils.default
    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs (dict): Currently no data is passed to this argument for this
                method but that may change in the future.
        """

    @utils.default
    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs (dict): Currently the output of the last call to
                `on_test_batch_end()` is passed to this argument for this method
                but that may change in the future.
        """

    @utils.default
    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs (dict): Currently no data is passed to this argument for this
                method but that may change in the future.
        """

    @utils.default
    def on_predict_end(self, logs=None):
        """Called at the end of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs (dict): Currently no data is passed to this argument for this
                method but that may change in the future.
        """
