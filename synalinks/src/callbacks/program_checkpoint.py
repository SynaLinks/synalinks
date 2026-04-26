# Modified from: keras/src/callbacks/model_checkpoint.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import re
import warnings

import numpy as np

from synalinks.src.api_export import synalinks_export
from synalinks.src.callbacks.callback import Callback
from synalinks.src.utils import file_utils
from synalinks.src.utils import io_utils


@synalinks_export("synalinks.callbacks.ProgramCheckpoint")
class ProgramCheckpoint(Callback):
    """Callback to save the Synalinks program or program variables at some frequency.

    `ProgramCheckpoint` callback is used in conjunction with training using
    `program.fit()` to save a program or variables (in a checkpoint file) at some
    interval, so the program or variables can be loaded later to continue the
    training from the state saved.

    A few options this callback provides include:

    - Whether to only keep the program that has achieved the "best performance" so
      far, or whether to save the program at the end of every epoch regardless of
      performance.
    - Definition of "best"; which quantity to monitor and whether it should be
      maximized or minimized.
    - The frequency it should save at. Currently, the callback supports saving
      at the end of every epoch, or after a fixed number of training batches.
    - Whether only variables are saved, or the whole program is saved.

    Example:

    ```python
    program.compile(
        reward=...,
        optimizer=...,
        metrics=[
            ...
        ],
    )

    EPOCHS = 10
    checkpoint_filepath = '/tmp/synalinks/checkpoint.program.json'

    program_checkpoint_callback = synalinks.callbacks.ProgramCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_reward',
        mode='max',
        save_best_only=True,
    )

    # Program is saved at the end of every epoch, if it's the best seen so far.
    program.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        callbacks=[program_checkpoint_callback]
    )

    # The program (that are considered the best) can be loaded as -
    synalinks.programs.load_program(checkpoint_filepath)

    # Alternatively, one could checkpoint just the program variables as -
    checkpoint_filepath = '/tmp/synalinks/checkpoint.variables.json'
    program_checkpoint_callback = synalinks.callbacks.ProgramCheckpoint(
        filepath=checkpoint_filepath,
        save_variables_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
    )

    # Program variables are saved at the end of every epoch, if it's the best seen
    # so far.
    program.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        callbacks=[program_checkpoint_callback]
    )

    # The program variables (that are considered the best) can be loaded as -
    program.load_variables(checkpoint_filepath)
    ```

    Args:
        filepath (str | os.PathLike): string or `PathLike`, path to save the program file.
            `filepath` can contain named formatting options,
            which will be filled the value of `epoch` and keys in `logs`
            (passed in `on_epoch_end`).
            The `filepath` name needs to end with `".variables.json"` when
            `save_variables_only=True` or should end with `".json"`
            when checkpoint saving the whole program (default).
            For example, if `filepath` is `"{epoch:02d}-{val_loss:.2f}.json"` or
            "{epoch:02d}-{val_loss:.2f}.variables.json"`, then the program
            checkpoints will be saved with the epoch number and the validation
            loss in the filename. The directory of the filepath
            should not be reused by any other callbacks to avoid conflicts.
        monitor (str): The metric name to monitor. Typically the metrics are set by
            the `Program.compile` method. Note:
            * Prefix the name with `"val_"` to monitor validation metrics.
            * Use `"reward"` or `"val_reward"` to monitor the program's total reward.
            * If you specify metrics as strings, like `"accuracy"`, pass the
                same string (with or without the `"val_"` prefix).
            * If you pass `metrics.Metric` objects, `monitor` should be set to
                `metric.name`
            * If you're not sure about the metric names you can check the
                contents of the `history.history` dictionary returned by
                `history = program.fit()`
            * Multi-output programs set additional prefixes on the metric names.
        verbose (str | int): Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
            displays messages when the callback takes an action.
        save_best_only (bool): if `save_best_only=True`, it only saves when the program
            is considered the "best" and the latest best program according to the
            quantity monitored will not be overwritten. If `filepath` doesn't
            contain formatting options like `{epoch}` then `filepath` will be
            overwritten by each new better program.
        mode (str): one of {`"auto"`, `"min"`, `"max"`}. If `save_best_only=True`, the
            decision to overwrite the current save file is made based on either
            the maximization or the minimization of the monitored quantity.
            In `"auto"` mode, the mode is set to `"max"`.
        save_variables_only (bool): if `True`, then only the program's variables will be
            saved (`program.save_variables(filepath)`), else the full program is
            saved (`program.save(filepath)`).
        save_freq (str | int): `"epoch"` or integer. When using `"epoch"`, the callback
            saves the program after each epoch. When using integer, the callback
            saves the program at end of this many batches. If the `Program` is
            compiled with `steps_per_execution=N`, then the saving criteria will
            be checked every Nth batch. Note that if the saving isn't aligned to
            epochs, the monitored metric may potentially be less reliable (it
            could reflect as little as 1 batch, since the metrics get reset
            every epoch). Defaults to `"epoch"`.
        initial_value_threshold (float): Floating point initial "best" value of the
            metric to be monitored. Only applies if `save_best_value=True`. Only
            overwrites the program variables already saved if the performance of
            current program is better than this value.
    """

    def __init__(
        self,
        filepath,
        monitor="val_reward",
        verbose=0,
        save_best_only=False,
        save_variables_only=False,
        mode="auto",
        save_freq="epoch",
        initial_value_threshold=None,
    ):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = file_utils.path_to_string(filepath)
        self.save_best_only = save_best_only
        self.save_variables_only = save_variables_only
        self.save_freq = save_freq
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self.best = initial_value_threshold

        if mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"ProgramCheckpoint mode '{mode}' is unknown, fallback to auto mode.",
                stacklevel=2,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            if self.best is None:
                self.best = np.inf
        elif mode == "max":
            self.monitor_op = np.greater
            if self.best is None:
                self.best = -np.inf
        else:
            self.monitor_op = np.greater
            if self.best is None:
                self.best = -np.inf

        if self.save_freq != "epoch" and not isinstance(self.save_freq, int):
            raise ValueError(
                f"Unrecognized save_freq: {self.save_freq}. "
                "Expected save_freq are 'epoch' or integer values"
            )

        if save_variables_only:
            if not self.filepath.endswith(".variables.json"):
                raise ValueError(
                    "When using `save_variables_only=True` in `ProgramCheckpoint`"
                    ", the filepath provided must end in `.variables.json` "
                    "(Synalinks variables format). Received: "
                    f"filepath={self.filepath}"
                )
        else:
            if not self.filepath.endswith(".json"):
                raise ValueError(
                    "The filepath provided must end in `.json` "
                    "(Synalinks program format). Received: "
                    f"filepath={self.filepath}"
                )

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._save_program(epoch=self._current_epoch, batch=batch, logs=logs)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        if self.save_freq == "epoch":
            self._save_program(epoch=epoch, batch=None, logs=logs)

    def _should_save_on_batch(self, batch):
        """Handles batch-level saving logic, supports steps_per_execution."""
        if self.save_freq == "epoch":
            return False
        if batch <= self._last_batch_seen:  # New epoch.
            add_batches = batch + 1  # batches are zero-indexed.
        else:
            add_batches = batch - self._last_batch_seen
        self._batches_seen_since_last_saving += add_batches
        self._last_batch_seen = batch

        if self._batches_seen_since_last_saving >= self.save_freq:
            self._batches_seen_since_last_saving = 0
            return True
        return False

    def _save_program(self, epoch, batch, logs):
        """Saves the program.

        Args:
            epoch (int): the epoch this iteration is in.
            batch (int): the batch this iteration is in. `None` if the `save_freq`
                is set to `"epoch"`.
            logs (dict): the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        filepath = self._get_file_path(epoch, batch, logs)
        # Create host directory if it doesn't exist.
        dirname = os.path.dirname(filepath)
        if dirname and not file_utils.exists(dirname):
            file_utils.makedirs(dirname)

        try:
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn(
                        f"Can save best model only with {self.monitor} "
                        "available, skipping.",
                        stacklevel=2,
                    )
                elif isinstance(current, np.ndarray) and len(current.shape) > 0:
                    warnings.warn(
                        "Can save best model only when `monitor` is "
                        f"a scalar value. Received: {current}. "
                        "Falling back to `save_best_only=False`."
                    )
                    self.program.save(filepath, overwrite=True)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f"\nEpoch {epoch + 1}: {self.monitor} "
                                "improved "
                                f"from {self.best:.5f} to {current:.5f}, "
                                f"saving program to {filepath}"
                            )
                        self.best = current
                        if self.save_variables_only:
                            self.program.save_variables(filepath, overwrite=True)
                        else:
                            self.program.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f"\nEpoch {epoch + 1}: "
                                f"{self.monitor} did not improve "
                                f"from {self.best:.5f}"
                            )
            else:
                if self.verbose > 0:
                    io_utils.print_msg(f"\nEpoch {epoch + 1}: saving model to {filepath}")
                if self.save_variables_only:
                    self.program.save_variables(filepath, overwrite=True)
                else:
                    self.program.save(filepath, overwrite=True)
        except IsADirectoryError:  # h5py 3.x
            raise IOError(
                "Please specify a non-directory filepath for "
                "ProgramCheckpoint. Filepath used is an existing "
                f"directory: {filepath}"
            )
        except IOError as e:  # h5py 2.x
            # `e.errno` appears to be `None` so checking the content of
            # `e.args[0]`.
            if "is a directory" in str(e.args[0]).lower():
                raise IOError(
                    "Please specify a non-directory filepath for "
                    "ModelCheckpoint. Filepath used is an existing "
                    f"directory: {filepath}"
                )
            # Re-throw the error for any other causes.
            raise e

    def _get_file_path(self, epoch, batch, logs):
        """Returns the file path for checkpoint."""

        try:
            # `filepath` may contain placeholders such as
            # `{epoch:02d}`,`{batch:02d}` and `{mape:.2f}`. A mismatch between
            # logged metrics and the path's placeholders can cause formatting to
            # fail.
            if batch is None or "batch" in logs:
                file_path = self.filepath.format(epoch=epoch + 1, **logs)
            else:
                file_path = self.filepath.format(epoch=epoch + 1, batch=batch + 1, **logs)
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback filepath: "{self.filepath}". Reason: {e}'
            )
        return file_path

    def _checkpoint_exists(self, filepath):
        """Returns whether the checkpoint `filepath` refers to exists."""
        return file_utils.exists(filepath)

    def _get_most_recently_modified_file_matching_pattern(self, pattern):
        """Returns the most recently modified filepath matching pattern.

        In the rare case where there are more than one pattern-matching file
        having the same modified time that is most recent among all, return the
        filepath that is largest (by `>` operator, lexicographically using the
        numeric equivalents). This provides a tie-breaker when multiple files
        are most recent. Note that a larger `filepath` can sometimes indicate a
        later time of modification (for instance, when epoch/batch is used as
        formatting option), but not necessarily (when accuracy or loss is used).
        The tie-breaker is put in the logic as best effort to return the most
        recent, and to avoid nondeterministic result.

        Modified time of a file is obtained with `os.path.getmtime()`.

        This utility function is best demonstrated via an example:

        ```python
        file_pattern = 'batch{batch:02d}epoch{epoch:02d}.json'
        test_dir = self.get_temp_dir()
        path_pattern = os.path.join(test_dir, file_pattern)
        file_paths = [
            os.path.join(test_dir, file_name) for file_name in
            ['batch03epoch02.json',
             'batch02epoch02.json', 'batch01epoch01.json']
        ]
        for file_path in file_paths:
            # Write something to each of the files
            ...
        self.assertEqual(
            _get_most_recently_modified_file_matching_pattern(path_pattern),
            file_paths[-1])
        ```

        Args:
            pattern (str): The file pattern that may optionally contain python
                placeholder such as `{epoch:02d}`.

        Returns:
            (str): The most recently modified file's full filepath matching `pattern`.
                If `pattern` does not contain any placeholder, this returns the
                filepath that exactly matches `pattern`. Returns `None` if no match
                is found.
        """
        dir_name = os.path.dirname(pattern)
        base_name = os.path.basename(pattern)
        base_name_regex = "^" + re.sub(r"{.*}", r".*", base_name) + "$"

        latest_mod_time = 0
        file_path_with_latest_mod_time = None
        n_file_with_latest_mod_time = 0
        file_path_with_largest_file_name = None

        if file_utils.exists(dir_name):
            for file_name in os.listdir(dir_name):
                # Only consider if `file_name` matches the pattern.
                if re.match(base_name_regex, file_name):
                    file_path = os.path.join(dir_name, file_name)
                    mod_time = os.path.getmtime(file_path)
                    if (
                        file_path_with_largest_file_name is None
                        or file_path > file_path_with_largest_file_name
                    ):
                        file_path_with_largest_file_name = file_path
                    if mod_time > latest_mod_time:
                        latest_mod_time = mod_time
                        file_path_with_latest_mod_time = file_path
                        # In the case a file with later modified time is found,
                        # reset the counter for the number of files with latest
                        # modified time.
                        n_file_with_latest_mod_time = 1
                    elif mod_time == latest_mod_time:
                        # In the case a file has modified time tied with the
                        # most recent, increment the counter for the number of
                        # files with latest modified time by 1.
                        n_file_with_latest_mod_time += 1

        if n_file_with_latest_mod_time == 1:
            # Return the sole file that has most recent modified time.
            return file_path_with_latest_mod_time
        else:
            # If there are more than one file having latest modified time,
            # return the file path with the largest file name.
            return file_path_with_largest_file_name
