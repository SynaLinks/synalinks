# Modified from: synalinks/src/callbacks/backup_and_restore.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json

from synalinks.src.api_export import synalinks_export
from synalinks.src.callbacks.callback import Callback
from synalinks.src.utils import file_utils
from synalinks.src.utils.async_utils import run_maybe_nested


@synalinks_export("synalinks.callbacks.BackupAndRestore")
class BackupAndRestore(Callback):
    """Callback to back up and restore the training state.

    `BackupAndRestore` callback is intended to recover training from an
    interruption that has happened in the middle of a `Program.fit` execution, by
    backing up the training states in a temporary checkpoint file, at the end of
    each epoch. Each backup overwrites the previously written checkpoint file,
    so at any given time there is at most one such checkpoint file for
    backup/restoring purpose.

    If training restarts before completion, the training state (which includes
    the `Program` weights and epoch number) is restored to the most recently saved
    state at the beginning of a new `Program.fit` run. At the completion of a
    `Program.fit` run, the temporary checkpoint file is deleted.

    Note that the user is responsible to bring jobs back after the interruption.
    This callback is important for the backup and restore mechanism for fault
    tolerance purpose, and the program to be restored from a previous checkpoint
    is expected to be the same as the one used to back up. If user changes
    arguments passed to compile or fit, the checkpoint saved for fault tolerance
    can become invalid.

    Example:

    >>> class InterruptingCallback(synalinks.callbacks.Callback):
    ...   def on_epoch_begin(self, epoch, logs=None):
    ...     if epoch == 4:
    ...       raise RuntimeError('Interrupting!')
    >>> callback = synalinks.callbacks.BackupAndRestore(backup_dir="/tmp/backup")
    >>> program = synalinks.programs.Sequential([synalinks.Generator(data_model=Answer)])
    >>> program.compile(synalinks.optimizers.RandomFewShot(), reward=synalinks.reward.ExactMatch())
    >>> program.build(Query)
    >>> try:
    ...   program.fit(..., callbacks=[callback, InterruptingCallback()],
    ...             verbose=0)
    ... except:
    ...   pass
    >>> history = program.fit(..., epochs=10, batch_size=1,
    ...                       callbacks=[callback], verbose=0)
    >>> # Only 6 more epochs are run, since first training got interrupted at
    >>> # zero-indexed epoch 4, second training will continue from 4 to 9.
    >>> len(history.history['reward'])
    >>> 6

    Args:
        backup_dir: String, path of directory where to store the data
            needed to restore the program. The directory
            cannot be reused elsewhere to store other files, e.g. by the
            `BackupAndRestore` callback of another training run,
            or by another callback (e.g. `ProgramCheckpoint`)
            of the same training run.
        save_freq: `"epoch"`, integer, or `False`. When set to `"epoch"`
          the callback saves the checkpoint at the end of each epoch.
          When set to an integer, the callback saves the checkpoint every
          `save_freq` batches. Set `save_freq=False` only if using
          preemption checkpointing (i.e. with `save_before_preemption=True`).
        double_checkpoint: Boolean. If enabled, `BackupAndRestore` callback
          will save 2 last training states (current and previous). After
          interruption if current state can't be loaded due to IO error
          (e.g. file corrupted) it will try to restore previous one. Such
          behaviour will consume twice more space on disk, but increase fault
          tolerance. Defaults to `False`.
        delete_checkpoint: Boolean. This `BackupAndRestore`
          callback works by saving a checkpoint to back up the training state.
          If `delete_checkpoint=True`, the checkpoint will be deleted after
          training is finished. Use `False` if you'd like to keep the checkpoint
          for future usage. Defaults to `True`.
    """

    def __init__(
        self,
        backup_dir,
        save_freq="epoch",
        double_checkpoint=False,
        delete_checkpoint=True,
    ):
        super().__init__()
        self.save_freq = save_freq
        self.double_checkpoint = double_checkpoint
        self.delete_checkpoint = delete_checkpoint
        self._batches_seen_since_last_saving = 0
        self._last_batch_seen = 0
        self._current_epoch = 0

        if not backup_dir:
            raise ValueError("Empty `backup_dir` argument passed")
        self.backup_dir = backup_dir
        self._variables_path = file_utils.join(backup_dir, "latest.variables.json")
        self._training_metadata_path = file_utils.join(
            backup_dir, "training_metadata.json"
        )
        self._prev_variables_path = self._variables_path + ".bkp"
        self._prev_training_metadata_path = self._training_metadata_path + ".bkp"
        if save_freq != "epoch" and not isinstance(save_freq, int):
            raise ValueError(
                "Invalid value for argument `save_freq`. "
                f"Received: save_freq={save_freq}. "
                "Expected either 'epoch' or an integer value."
            )

    def on_train_begin(self, logs=None):
        try:
            self._load_program()
        except OSError as e:
            # Weights may be corrupted. Trying to load previous one.
            if not file_utils.exists(self._prev_variables_path):
                raise e
            file_utils.copy(self._prev_variables_path, self._variables_path)
            if file_utils.exists(self._prev_training_metadata_path):
                file_utils.copy(
                    self._prev_training_metadata_path,
                    self._training_metadata_path,
                )
            elif file_utils.exists(self._training_metadata_path):
                file_utils.remove(self._training_metadata_path)
            self._load_program()

    def _load_program(self):
        """Get training state from temporary file and restore it."""
        if not self.program.built:
            raise ValueError(
                "To use the BackupAndRestore callback, "
                "you program must be built before you call `fit()`. "
                f"Program {self.program} is unbuilt. You can build it "
                "beforehand by calling it on a batch of data."
            )
        if file_utils.exists(self._variables_path):
            if self.program.optimizer is not None and not self.program.optimizer.built:
                # Make sure optimizer variables exist before loading.
                run_maybe_nested(
                    self.program.optimizer.build(self.program.trainable_variables)
                )
            self.program.load_variables(self._variables_path)

        if file_utils.exists(self._training_metadata_path):
            with file_utils.File(self._training_metadata_path, "r") as f:
                training_metadata = json.loads(f.read())
            epoch = training_metadata["epoch"]
            self.program._initial_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self._current_epoch = epoch + 1
        self._last_batch_seen = 0
        if self.save_freq == "epoch":
            self._save_program()

    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            self._save_program()

    def _save_program(self):
        """Saves the program.

        Args:
            epoch: the epoch this iteration is in.
            batch: the batch this iteration is in. `None` if the `save_freq`
                is set to `"epoch"`.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        # Create host directory if it doesn't exist.
        if not file_utils.exists(self.backup_dir):
            file_utils.makedirs(self.backup_dir)
        if self.double_checkpoint and file_utils.exists(self._variables_path):
            file_utils.copy(self._variables_path, self._prev_variables_path)
        if self.double_checkpoint and file_utils.exists(self._training_metadata_path):
            file_utils.copy(
                self._training_metadata_path, self._prev_training_metadata_path
            )
        self.program.save_variables(filepath=self._variables_path, overwrite=True)
        with file_utils.File(self._training_metadata_path, "w") as f:
            training_metadata = {
                "epoch": self._current_epoch,
                "batch": self._last_batch_seen,
            }
            f.write(json.dumps(training_metadata))

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

    def on_train_end(self, logs=None):
        if self.delete_checkpoint and file_utils.exists(self.backup_dir):
            file_utils.rmtree(self.backup_dir)
