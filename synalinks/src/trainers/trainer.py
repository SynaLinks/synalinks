# Modified from: keras/src/trainers/trainer.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import inspect
import warnings

import numpy as np

from synalinks.src import backend
from synalinks.src import callbacks as callbacks_module
from synalinks.src import metrics as metrics_module
from synalinks.src import optimizers as optimizers_module
from synalinks.src.backend.common import numpy
from synalinks.src.saving import serialization_lib
from synalinks.src.trainers.compile_utils import CompileMetrics
from synalinks.src.trainers.compile_utils import CompileReward
from synalinks.src.trainers.data_adapters import array_slicing
from synalinks.src.trainers.data_adapters import data_adapter_utils
from synalinks.src.trainers.epoch_iterator import EpochIterator
from synalinks.src.utils import python_utils
from synalinks.src.utils import tracking
from synalinks.src.utils.async_utils import run_maybe_nested


class Trainer:
    def __init__(self):
        self._lock = False
        self._run_eagerly = False
        self.compiled = False
        self.reward = None
        self.steps_per_execution = 1
        # Can be set by callbacks in on_train_begin
        self._initial_epoch = None
        self._compute_reward_has_training_arg = (
            "training" in inspect.signature(self.compute_reward).parameters
        )
        # Placeholders used in `compile`
        self._optimizer = None
        self._meta_optimizer = None
        self._compile_reward = None
        self._compile_metrics = None
        self._reward_tracker = None

    @tracking.no_automatic_dependency_tracking
    def compile(
        self,
        optimizer=None,
        meta_optimizer=None,
        reward=None,
        reward_weights=None,
        metrics=None,
        run_eagerly=False,
        steps_per_execution=1,
    ):
        """Configures the program for training.

        Example:

        ```python
        program.compile(
            optimizer=synalinks.optimizers.RandomFewShot(),
            reward=synalinks.rewards.ExactMatch(),
            metrics=[
                synalinks.metrics.MeanMetricWrapper(synalinks.rewards.exact_match),
            ],
        )
        ```

        Args:
            optimizer (Optimizer): Optimizer instance. See `synalinks.optimizers`.
            meta_optimizer (Optimizer): Experimental feature, use at your own risk.
                Optional. Optimizer instance. See `synalinks.optimizers`.
            reward (Reward): Reward function. A `synalinks.rewards.Reward`
                instance. See `synalinks.rewards`. A reward function is
                any callable with the signature `reward = fn(y_true, y_pred)`,
                where `y_true` are the ground truth values, and `y_pred`
                are the program's predictions.
                `y_true` should be a list of batch size length `[d0, .. dN]`.
                `y_pred` should be a list of batch size length `[d0, .. dN]`.
                The reward function should return a float.
            reward_weights (list): Optional list specifying scalar coefficients
                (Python floats) to weight the reward contributions of
                different program outputs. The reward value that will be maximized
                by the program will then be the *weighted sum* of all individual
                rewards, weighted by the `reward_weights` coefficients. It is
                expected to have a 1:1 mapping to the program's outputs.
            metrics (list): List of metrics to be evaluated by the program during
                training and testing. Each of it is a `synalinks.metrics.Metric`
                instance. See `synalinks.metrics`. A function is any callable with the
                signature `result = fn(y_true, y_pred)`.
            run_eagerly (bool): If `True`, this program's forward pass
                will never be compiled. It is recommended to leave this
                as `False` when training (for best performance),
                and to set it to `True` when debugging.
            steps_per_execution (int): The number of batches to run
                during each a single compiled function call. Running multiple
                batches inside a single compiled function call can
                greatly improve performance on TPUs or small programs with a large
                Python overhead. At most, one full epoch will be run each
                execution. If a number larger than the size of the epoch is
                passed, the execution will be truncated to the size of the
                epoch. Note that if `steps_per_execution` is set to `N`,
                `Callback.on_batch_begin` and `Callback.on_batch_end` methods
                will only be called every `N` batches (i.e. before/after
                each compiled function execution).
        """
        self._clear_previous_trainer_metrics()
        self._optimizer = optimizer
        self._meta_optimizer = meta_optimizer
        self._optimizer.set_program(self)
        self._optimizer.set_meta_optimizer(meta_optimizer)

        if hasattr(self, "output_names"):
            output_names = self.output_names
        else:
            output_names = None
        if reward is not None:
            self._compile_reward = CompileReward(
                reward, reward_weights, output_names=output_names
            )
            self.reward = reward
        if metrics is not None:
            self._compile_metrics = CompileMetrics(metrics, output_names=output_names)
        self.run_eagerly = run_eagerly
        self.stop_training = False
        self.compiled = True
        self._reward_tracker = metrics_module.Mean(name="reward")
        self.steps_per_execution = steps_per_execution

        self._compile_config = serialization_lib.SerializableDict(
            optimizer=optimizer,
            reward=reward,
            reward_weights=reward_weights,
            metrics=metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
        )

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def meta_optimizer(self):
        return self._meta_optimizer

    @property
    def metrics(self):
        # Order: reward tracker, individual reward trackers, compiled metrics,
        # custom metrcis, submodule metrics.
        metrics = []
        if self.compiled:
            if self._reward_tracker is not None:
                metrics.append(self._reward_tracker)
            if self._compile_metrics is not None:
                metrics.append(self._compile_metrics)
            if self._compile_reward is not None:
                metrics.extend(self._compile_reward.metrics)
        metrics.extend(self._metrics)
        for module in self._flatten_modules(include_self=False):
            if isinstance(module, Trainer):
                # All Trainer-related metrics in submodules should be ignored
                # because a new Trainer has been instantiated.
                continue
            metrics.extend(module.metrics)
        return metrics

    @property
    def metrics_names(self):
        return [m.name for m in self.metrics]

    def reset_metrics(self):
        for m in self.metrics:
            m.reset_state()

    def _get_own_metrics(self):
        metrics = []
        if self._reward_tracker is not None:
            metrics.append(self._reward_tracker)
        if self._compile_metrics is not None:
            metrics.append(self._compile_metrics)
        if self._compile_reward is not None:
            metrics.extend(self._compile_reward.metrics)
        metrics.extend(self._metrics)
        return metrics

    def _clear_previous_trainer_metrics(self):
        for module in self._flatten_modules(include_self=False):
            if not isinstance(module, Trainer):
                continue
            # A submodule might be a Trainer. In that case, we need to clear
            # the Trainer-related metrics, as they are not usable when a
            # new Trainer is instantiated.
            for m in self._get_own_metrics():
                module._tracker.untrack(m)
            module._reward_tracker = None
            module._compile_metrics = None
            if module._compile_reward is not None:
                module._compile_reward._metrics.clear()
            module._metrics.clear()

    @property
    def run_eagerly(self):
        return self._run_eagerly

    @run_eagerly.setter
    def run_eagerly(self, value):
        self._run_eagerly = value

    async def compute_reward(
        self,
        x=None,
        y=None,
        y_pred=None,
        training=True,
    ):
        """Compute the total reward, validate it, and return it.

        Subclasses can optionally override this method to provide custom reward
        computation logic.

        Args:
            x (list): Input data.
            y (list): Target data.
            y_pred (list): Predictions returned by the program (output of `program(x)`).
            training (bool): Whether we are training or evaluating the program.

        Returns:
            (float | None): The total reward as a scalar, or `None` if no reward results
                (which is the case when called by `Program.test_step`).
        """
        # The default implementation does not use `x` or `training`.
        del x
        del training
        rewards = []
        if self._compile_reward is not None:
            for y_t, y_p in zip(y, y_pred):
                reward = await self._compile_reward(y_t, y_p)
                if reward is not None:
                    rewards.append(reward)
        for reward in self.rewards:
            rewards.append(numpy.sum(reward))
        if len(rewards) == 1:
            total_reward = rewards[0]
        elif len(rewards) == 0:
            total_reward = numpy.zeros(())
        else:
            total_reward = numpy.mean(rewards)
        return float(total_reward)

    def stateless_compute_reward(
        self,
        trainable_variables,
        non_trainable_variables,
        metrics_variables,
        x=None,
        y=None,
        y_pred=None,
        training=True,
    ):
        var_mapping = list(zip(self.trainable_variables, trainable_variables))
        var_mapping.extend(zip(self.non_trainable_variables, non_trainable_variables))
        var_mapping.extend(zip(self.metrics_variables, metrics_variables))
        with backend.StatelessScope(state_mapping=var_mapping) as scope:
            # Note that this is needed for the regularization reward, which need
            # the latest value of train/non-trainable variables.
            reward = self._compute_reward(
                x,
                y,
                y_pred,
                training=training,
            )

        # Update non trainable vars (may have been updated in compute_reward)
        non_trainable_variables = []
        for v in self.non_trainable_variables:
            new_v = scope.get_current_value(v)
            non_trainable_variables.append(new_v)

        # Update metrics vars (may have been updated in compute_reward)
        metrics_variables = []
        for v in self.metrics_variables:
            new_v = scope.get_current_value(v)
            metrics_variables.append(new_v)
        return reward, (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        )

    async def compute_metrics(self, x, y, y_pred):
        """Update metric states and collect all metrics to be returned.

        Subclasses can optionally override this method to provide custom metric
        updating and collection logic. Custom metrics are not passed in
        `compile()`, they can be created in `__init__` or `build`. They are
        automatically tracked and returned by `self.metrics`.
        ```

        Args:
            x: Input data.
            y: Target data.
            y_pred: Predictions returned by the program output of `program.call(x)`.

        Returns:
            A `dict` containing values that will be passed to
                `synalinks.callbacks.CallbackList.on_train_batch_end()`. Typically,
                the values of the metrics listed in `self.metrics` are returned.
                Example: `{'reward': 0.2, 'accuracy': 0.7}`.
        """
        del x  # The default implementation does not use `x`.
        if self._compile_metrics is not None:
            for y_t, y_p in zip(y, y_pred):
                await self._compile_metrics.update_state(y_t, y_p)
        return self.get_metrics_result()

    def get_metrics_result(self):
        """Returns the program's metrics values as a dict.

        If any of the metric result is a dict (containing multiple metrics),
        each of them gets added to the top level returned dict of this method.

        Returns:
            (dict): A `dict` containing values of the metrics listed in `self.metrics`.
                Example: `{'reward': 0.2, 'accuracy': 0.7}`.
        """
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return python_utils.pythonify_logs(return_metrics)

    async def fit(
        self,
        x=None,
        y=None,
        batch_size=1,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.1,
        validation_data=None,
        shuffle=True,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=32,
        validation_freq=1,
    ):
        """Trains the program for a fixed number of epochs (dataset iterations).

        Args:
            x (np.ndarray | generator): Input data. It can be:
                - A NumPy array (or array-like), or a list of `DataModel` arrays
                    (in case the model has multiple inputs).
                - A list of dict mapping input names to the corresponding `DataModel`s,
                    if the program has named inputs.
                - A Python generator function yielding `(inputs, targets)`.
            y (np.ndarray): Target data. Like the input data `x`, it can be either NumPy
                array(s) of `DataModel`(s). If `x` is a Python generator function,
                `y` should not be specified since targets will be obtained from
                `x`.
            batch_size (int): Integer or `None`.
                Number of samples per batch of computation.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your input data `x` is a
                Python generator function since they generate batches.
            epochs (int): Integer. Number of epochs to train the program.
                An epoch is an iteration over the entire `x` and `y`
                data provided (unless the `steps_per_epoch` flag is set to
                something other than None).
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The program is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose (int): `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                "auto" becomes 1 for most cases.
                Note that the progress bar is not
                particularly useful when logged to a file,
                so `verbose=2` is recommended when not running interactively
                (e.g., in a production environment). Defaults to `"auto"`.
            callbacks (list): List of `synalinks.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `synalinks.callbacks`. Note
                `synalinks.callbacks.ProgbarLogger` and
                `synalinks.callbacks.History` callbacks are created
                automatically and need not be passed to `program.fit()`.
                `synalinks.callbacks.ProgbarLogger` is created
                or not based on the `verbose` argument in `program.fit()`.
            validation_split (float): Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The program will set apart this fraction of the training data,
                will not train on it, and will evaluate the reward and any program
                metrics on this data at the end of each epoch. The validation
                data is selected from the last samples in the `x` and `y` data
                provided, before shuffling.
                This argument is only supported when `x` and `y` are made of
                data_models.
                If both `validation_data` and `validation_split` are provided,
                `validation_data` will override `validation_split`.
            validation_data (tuple | iterator): Data on which to evaluate
                the reward and any program metrics at the end of each epoch.
                The program will not be trained on this data.
                `validation_data` will override `validation_split`.
                It can be:
                - A tuple `(x_val, y_val)` of `DataModel`s lists.
            shuffle (bool): Whether to shuffle the training data before each
                epoch. This argument is ignored when `x` is a Python generator function.
            initial_epoch (int): Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch (int): Integer or `None`.
                Total number of steps (batches of samples) before declaring one
                epoch finished and starting the next epoch. When training with
                input data_models arrays, the default `None` means that the
                value used is the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
                If `x` is a Python generator function, the
                epoch will run until the input dataset is exhausted. When
                passing an infinitely repeating dataset, you must specify the
                `steps_per_epoch` argument, otherwise the training will run
                indefinitely.
            validation_steps (int): Integer or `None`.
                Only relevant if `validation_data` is provided.
                Total number of steps (batches of samples) to draw before
                stopping when performing validation at the end of every epoch.
                If `validation_steps` is `None`, validation will run until the
                `validation_data` dataset is exhausted. In the case of an
                infinitely repeating dataset, it will run indefinitely. If
                `validation_steps` is specified and only part of the dataset
                is consumed, the evaluation will start from the beginning of the
                dataset at each epoch. This ensures that the same validation
                samples are used every time.
            validation_batch_size (int): Integer or `None`.
                Number of samples per validation batch.
                If unspecified, will default to `batch_size`.
                Do not specify the `validation_batch_size` if your data is a
                `synalinks.utils.PyDataset`, `tf.data.Dataset`,
                `torch.utils.data.DataLoader` or Python generator function
                since they generate batches.
            validation_freq (int): Only relevant if validation data is provided.
                Specifies how many training epochs to run
                before a new validation run is performed,
                e.g. `validation_freq=2` runs validation every 2 epochs.

        Returns:
            (History): A `History` object. Its `History.history` attribute is
                a record of training reward values and metrics values
                at successive epochs, as well as validation reward values
                and validation metrics values (if applicable).
        """
        self._assert_compile_called("fit")
        self._eval_epoch_iterator = None
        val_y, val_y = None, None

        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for numpy arrays.
            (x, y), validation_data = array_slicing.train_validation_split(
                (x, y), validation_split=validation_split
            )

        if validation_data is not None:
            (val_x, val_y) = data_adapter_utils.unpack_x_y(validation_data)
        # Create an iterator that yields batches of input/target data.
        epoch_iterator = EpochIterator(
            x=x,
            y=y,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=False,
            steps_per_execution=self.steps_per_execution,
        )

        if not all(module.built for module in self._flatten_modules()):
            # Build the model on one batch of data.
            for _, data in epoch_iterator:
                data_batch = data[0]
                self._auto_build(
                    iterator=epoch_iterator,
                    data_batch=data_batch,
                )
                break
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=steps_per_epoch,
                program=self,
            )

        self.stop_training = False
        callbacks.on_train_begin()
        training_logs = None
        logs = {}
        initial_epoch = self._initial_epoch or initial_epoch

        if self.trainable_variables and isinstance(
            self.optimizer, optimizers_module.Optimizer
        ):
            await self.optimizer.on_train_begin(
                self.trainable_variables,
            )

        if self.optimizer.trainable_variables and isinstance(
            self.meta_optimizer, optimizers_module.Optimizer
        ):
            await self.meta_optimizer.on_train_begin(
                self.optimizer.trainable_variables,
            )

        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()

            if self.trainable_variables and isinstance(
                self.optimizer, optimizers_module.Optimizer
            ):
                await self.optimizer.on_epoch_begin(
                    epoch,
                    self.trainable_variables,
                )

            if self.optimizer.trainable_variables and isinstance(
                self.meta_optimizer, optimizers_module.Optimizer
            ):
                await self.meta_optimizer.on_epoch_begin(
                    epoch,
                    self.optimizer.trainable_variables,
                )

            callbacks.on_epoch_begin(epoch)
            with epoch_iterator.catch_stop_iteration():
                for step, iterator in epoch_iterator:
                    data = iterator[0]
                    x_batch, y_batch = data_adapter_utils.unpack_x_y(data)

                    if self.trainable_variables and isinstance(
                        self.optimizer, optimizers_module.Optimizer
                    ):
                        await self.optimizer.on_batch_begin(
                            step,
                            epoch,
                            self.trainable_variables,
                        )

                    if self.optimizer.trainable_variables and isinstance(
                        self.meta_optimizer, optimizers_module.Optimizer
                    ):
                        await self.meta_optimizer.on_batch_begin(
                            step,
                            epoch,
                            self.optimizer.trainable_variables,
                        )

                    callbacks.on_train_batch_begin(step)

                    logs = await self.train_on_batch(
                        step=step,
                        x=x_batch,
                        y=y_batch,
                        val_x=val_x,
                        val_y=val_y,
                        return_dict=True,
                    )

                    val_logs = await self.evaluate(
                        x=val_x,
                        y=val_y,
                        batch_size=validation_batch_size or batch_size,
                        steps=validation_steps,
                        callbacks=callbacks,
                        _use_cached_eval_dataset=False,
                    )

                    if self.trainable_variables and isinstance(
                        self.optimizer, optimizers_module.Optimizer
                    ):
                        await self.optimizer.on_batch_end(
                            step,
                            epoch,
                            self.trainable_variables,
                        )

                    if self.optimizer.trainable_variables and isinstance(
                        self.meta_optimizer, optimizers_module.Optimizer
                    ):
                        await self.meta_optimizer.on_batch_end(
                            step,
                            epoch,
                            self.optimizer.trainable_variables,
                        )

                    callbacks.on_train_batch_end(step, logs)
                    if self.stop_training:
                        break

            # Override with model metrics instead of last step logs if needed.
            epoch_logs = dict(self._get_metrics_result_or_logs(logs))

            # Run validation.
            if validation_data is not None and self._should_eval(epoch, validation_freq):
                # Create EpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = EpochIterator(
                        x=val_x,
                        y=val_y,
                        batch_size=validation_batch_size or batch_size,
                        steps_per_execution=self.steps_per_execution,
                        steps_per_epoch=validation_steps,
                        shuffle=False,
                    )

                val_logs = await self.evaluate(
                    x=val_x,
                    y=val_y,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    _use_cached_eval_dataset=True,
                )
                val_logs = {"val_" + name: val for name, val in val_logs.items()}
                epoch_logs.update(val_logs)

            if self.trainable_variables and isinstance(
                self.optimizer, optimizers_module.Optimizer
            ):
                await self.optimizer.on_epoch_end(
                    epoch,
                    self.trainable_variables,
                )

            if self.optimizer.trainable_variables and isinstance(
                self.meta_optimizer, optimizers_module.Optimizer
            ):
                await self.meta_optimizer.on_epoch_end(
                    epoch,
                    self.optimizer.trainable_variables,
                )

            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break

        # If _eval_epoch_iterator exists, delete it after all epochs are done.
        if getattr(self, "_eval_epoch_iterator", None) is not None:
            del self._eval_epoch_iterator

        if self.trainable_variables and isinstance(
            self.optimizer, optimizers_module.Optimizer
        ):
            await self.optimizer.on_train_end(self.trainable_variables)

        callbacks.on_train_end(logs=training_logs)
        return self.history

    async def evaluate(
        self,
        x=None,
        y=None,
        batch_size=32,
        verbose="auto",
        steps=None,
        callbacks=None,
        return_dict=True,
        **kwargs,
    ):
        """Returns the reward value & metrics values for the program in test mode.

        Computation is done in batches (see the `batch_size` arg.)

        Args:
            x (np.ndarray | generator): Input data. It can be:
                - A NumPy array (or array-like), or a list of `DataModel` arrays
                    (in case the model has multiple inputs).
                - A list of dict mapping input names to the corresponding `DataModel`s,
                    if the program has named inputs.
                - A Python generator function yielding `(inputs, targets)`.
            y (np.ndarray): Target data. Like the input data `x`, it can be either NumPy
                array(s) of `DataModel`(s). If `x` is a Python generator function,
                `y` should not be specified since targets will be obtained from
                `x`.
            batch_size (int): Integer or `None`.
                Number of samples per batch of computation.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your input data `x` is a
                Python generator function since they generate batches.
            verbose (int | str): `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = single line.
                `"auto"` becomes 1 for most cases.
                Note that the progress bar is not
                particularly useful when logged to a file, so `verbose=2` is
                recommended when not running interactively
                (e.g. in a production environment). Defaults to `"auto"`.
            steps (int): Integer or `None`.
                Total number of steps (batches of samples) to draw before
                declaring the evaluation round finished. If `steps` is `None`,
                it will run until `x` is exhausted. In the case of an infinitely
                repeating dataset, it will run indefinitely.
            callbacks (list): List of `synalinks.callbacks.Callback` instances.
                List of callbacks to apply during evaluation.
            return_dict (bool): If `True`, reward and metric results are returned as a
                dict, with each key being the name of the metric.
                If `False`, they are returned as a list.

        Returns:
            (float | list | dict): Scalar test reward
                (if the program has a single output and no metrics)
                or list of scalars (if the program has multiple outputs
                and/or metrics). The attribute `program.metrics_names` will give you
                the display labels for the scalar outputs.
        """
        self._assert_compile_called("evaluate")
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")
        # Create an iterator that yields batches of input/target data.
        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            epoch_iterator = EpochIterator(
                x=x,
                y=y,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                steps_per_execution=self.steps_per_execution,
            )

        if not all(module.built for module in self._flatten_modules()):
            # Build the model on one batch of data.
            for _, data in epoch_iterator:
                data_batch = data[0]
                self._auto_build(
                    iterator=epoch_iterator,
                    data_batch=data_batch,
                )
                break
        epoch_iterator.reset()

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=False,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                program=self,
            )

        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = {}
        self.reset_metrics()
        for step, iterator in epoch_iterator:
            callbacks.on_test_batch_begin(step)
            data = iterator[0]
            x_batch, y_batch = data_adapter_utils.unpack_x_y(data)
            logs = await self.test_on_batch(
                x=x_batch,
                y=y_batch,
                return_dict=True,
            )
            callbacks.on_test_batch_end(step, logs)
            if self.stop_evaluating:
                break
        logs = self.get_metrics_result()
        callbacks.on_test_end(logs)

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    async def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        """Generates output predictions for the input samples.

        Computation is done in batches. This method is designed for batch
        processing of large numbers of inputs. It is not intended for use inside
        of loops that iterate over your data and process small numbers of inputs
        at a time.

        For small numbers of inputs that fit in one batch,
        directly use `__call__()` for faster execution, e.g.,
        `program(x)`, or `program(x, training=False)` if you have modules
        that behave differently during inference.

        Args:
            x (np.ndarray | generator): Input data. It can be:
                - A NumPy array (or array-like), or a list of `DataModel` arrays
                    (in case the model has multiple inputs).
                - A list of dict mapping input names to the corresponding `DataModel`s,
                    if the program has named inputs.
                - A Python generator function yielding `(inputs, targets)`.
            batch_size (int): Integer or `None`.
                Number of samples per batch of computation.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your input data `x` is a
                `synalinks.utils.PyDataset`, `tf.data.Dataset`,
                `torch.utils.data.DataLoader` or Python generator function
                since they generate batches.
            verbose (int): `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = single line.
                `"auto"` becomes 1 for most cases. Note that the progress bar
                is not particularly useful when logged to a file,
                so `verbose=2` is recommended when not running interactively
                (e.g. in a production environment). Defaults to `"auto"`.
            steps (int): Total number of steps (batches of samples) to draw before
                declaring the prediction round finished. If `steps` is `None`,
                it will run until `x` is exhausted. In the case of an infinitely
                repeating dataset, it will run indefinitely.
            callbacks (list): List of `synalinks.callbacks.Callback` instances.
                List of callbacks to apply during prediction.

        Returns:
            (list): `JsonDataModel` array(s) of predictions.
                If the pipeline failed, a None is added to the predictions.
        """
        # Create an iterator that yields batches of input data.
        epoch_iterator = EpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            steps_per_execution=self.steps_per_execution,
        )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.stop_predicting = False
        callbacks.on_test_begin()
        outputs = []
        for step, iterator in epoch_iterator:
            callbacks.on_predict_batch_begin(step)
            data = iterator[0]
            x_batch, _ = data_adapter_utils.unpack_x_y(data)
            batch_outputs = await self.predict_on_batch(x_batch)
            outputs.extend(batch_outputs)
            callbacks.on_predict_batch_end(step, {"outputs": batch_outputs})
            if self.stop_predicting:
                break
        callbacks.on_predict_end()
        return np.array(outputs, dtype="object")

    async def train_on_batch(
        self,
        step,
        x,
        y=None,
        val_x=None,
        val_y=None,
        return_dict=False,
    ):
        """Runs a single optimization step on a single batch of data.

        Args:
            step (int): The training step.
            x (np.ndarray): Input data. Must be array-like.
            y (np.ndarray): Target data. Must be array-like.
            val_x (np.ndarray): Input validation data. Must be array-like.
            val_y (np.ndarray): Target validation data. Must be array-like.
            return_dict (bool): If `True`, reward and metric results are returned as a
                dict, with each key being the name of the metric. If `False`,
                they are returned as a list.

        Returns:
            (float | list | dict): A scalar reward value
                (when no metrics and `return_dict=False`), a list of reward
                and metric values (if there are metrics and `return_dict=False`),
                or a dict of metric and reward values (if `return_dict=True`).
        """
        if self.trainable_variables and isinstance(
            self.optimizer, optimizers_module.Optimizer
        ):
            metrics = await self.optimizer.optimize(
                step,
                self.trainable_variables,
                x=x,
                y=y,
                val_x=val_x,
                val_y=val_y,
            )
        else:
            warnings.warn("The program does not have any trainable variables.")
            y_pred = await self.predict_on_batch(val_x)
            reward = await self.compute_reward(
                x=val_x,
                y=val_y,
                y_pred=y_pred,
            )
            await self._reward_tracker.update_state(reward)
            metrics = await self.compute_metrics(val_x, val_y, y_pred)

        if return_dict:
            return metrics
        return self._flatten_metrics_in_order(metrics)

    async def test_on_batch(
        self,
        x,
        y=None,
        return_dict=False,
    ):
        """Test the program on a single batch of samples.

        Args:
            x (np.ndarray): Input data. Must be array-like.
            y (np.ndarray): Target data. Must be array-like.
            return_dict (bool): If `True`, reward and metric results are returned as a
                dict, with each key being the name of the metric. If `False`,
                they are returned as a list.

        Returns:
            (float | list | dict): A scalar reward value
                (when no metrics and `return_dict=False`), a list of reward
                and metric values (if there are metrics and `return_dict=False`),
                or a dict of metric and reward values (if `return_dict=True`).
        """
        y_pred = await self.predict_on_batch(x)

        reward = await self.compute_reward(
            x=x,
            y=y,
            y_pred=y_pred,
            training=False,
        )
        await self._reward_tracker.update_state(reward)

        metrics = await self.compute_metrics(x, y, y_pred)

        if return_dict:
            return metrics
        return self._flatten_metrics_in_order(metrics)

    async def predict_on_batch(self, x, training=False):
        """Returns predictions for a single batch of samples.

        Args:
            x (np.ndarray): Input data. Must be array-like.
            training (bool): Boolean. True if training.

        Returns:
            (list): list(s) of JsonDataModel predictions.
        """
        tasks = []
        for inputs in x:
            tasks.append(self(inputs, training=training))
        y_pred = await asyncio.gather(*tasks)
        return y_pred

    def get_compile_config(self):
        """Returns a serialized config with information for compiling the program.

        This method returns a config dictionary containing all the information
        (optimizer, reward, metrics, etc.) with which the program was compiled.

        Returns:
            (dict): A dict containing information for compiling the program.
        """
        if self.compiled and hasattr(self, "_compile_config"):
            return self._compile_config.serialize()

    def compile_from_config(self, config):
        """Compiles the program with the information given in config.

        This method uses the information in the config (optimizer, reward,
        metrics, etc.) to compile the program.

        Args:
            config (dict): Dict containing information for compiling the program.
        """
        has_overridden_compile = self.__class__.compile != Trainer.compile
        if has_overridden_compile:
            warnings.warn(
                "`compile()` was not called as part of program loading "
                "because the program's `compile()` method is custom. "
                "All subclassed Models that have `compile()` "
                "overridden should also override "
                "`get_compile_config()` and `compile_from_config(config)`. "
                "Alternatively, you can "
                "call `compile()` manually after loading.",
                stacklevel=2,
            )
            return
        config = serialization_lib.deserialize_synalinks_object(config)
        self.compile(**config)
        if hasattr(self, "optimizer") and self.built:
            # Create optimizer variables/programs.
            if not self.optimizer.built:
                run_maybe_nested(self.optimizer.build(self.trainable_variables))
            if hasattr(self, "meta_optimizer") and self.built:
                # Create optimizer variables/programs.
                if self.meta_optimizer and not self.meta_optimizer.built:
                    run_maybe_nested(
                        self.meta_optimizer.build(self.optimizer.trainable_variables)
                    )

    def _should_reward(self, epoch, validation_freq):
        epoch = epoch + 1  # one-index the user-facing epoch.
        if isinstance(validation_freq, int):
            return epoch % validation_freq == 0
        elif isinstance(validation_freq, list):
            return epoch in validation_freq
        else:
            raise ValueError(
                "Expected `validation_freq` to be a list or int. "
                f"Received: validation_freq={validation_freq} of the "
                f"type {type(validation_freq)}."
            )

    def _get_metrics_result_or_logs(self, logs):
        """Returns program metrics as a dict if the keys match with input logs.

        When the training / evaluation is performed with an asynchronous steps,
        the last scheduled `train / test_step` may not give the latest metrics
        because it is not guaranteed to be executed the last. This method gets
        metrics from the program directly instead of relying on the return from
        last step function.

        When the user has custom train / test step functions, the metrics
        returned may be different from `Program.metrics`. In those instances,
        this function will be no-op and return the logs passed in.

        Args:
            logs (dict): A `dict` of metrics returned by train / test step function.

        Returns:
            (dict): A `dict` containing values of the metrics listed in `self.metrics`
                when logs and program metrics keys match. Otherwise it returns input
                `logs`.
        """
        metric_logs = self.get_metrics_result()
        # Verify that train / test step logs passed and metric logs have
        # matching keys. It could be different when using custom step functions,
        # in which case we return the logs from the last step.
        if isinstance(logs, dict) and set(logs.keys()) == set(metric_logs.keys()):
            return metric_logs
        return logs

    def _flatten_metrics_in_order(self, logs):
        """Turns `logs` dict into a list as per key order of `metrics_names`."""
        metric_names = []
        for metric in self.metrics:
            if isinstance(metric, CompileMetrics):
                metric_names += [sub_metric.name for sub_metric in metric.metrics]
            else:
                metric_names.append(metric.name)
        results = []
        for name in metric_names:
            if name in logs:
                results.append(logs[name])
        for key in sorted(logs.keys()):
            if key not in metric_names:
                results.append(logs[key])
        if len(results) == 1:
            return results[0]
        return results

    def _assert_compile_called(self, method_name=None):
        if not self.compiled:
            msg = "You must call `compile()` before "
            if metrics_module:
                msg += "using the program."
            else:
                msg += f"calling `{method_name}()`."
            raise ValueError(msg)

    def _auto_build(self, iterator=None, data_batch=None):
        program_unbuilt = not all(module.built for module in self._flatten_modules())
        compile_metrics_unbuilt = (
            self._compile_metrics is not None and not self._compile_metrics.built
        )
        compile_reward_unbuilt = (
            self._compile_reward is not None and not self._compile_reward.built
        )
        optimizer_unbuilt = self.optimizer is not None and not self.optimizer.built
        meta_optimizer_unbuilt = (
            self.meta_optimizer is not None and not self.meta_optimizer.built
        )
        if program_unbuilt or compile_metrics_unbuilt or compile_reward_unbuilt:
            if data_batch is None:
                for _, data_or_iterator in iterator:
                    if isinstance(data_or_iterator, (list, tuple)):
                        data_batch = data_or_iterator[0]
                    else:
                        data_batch = next(data_or_iterator)
                    break
            (x, y) = data_batch
            try:
                y_pred = run_maybe_nested(self.predict_on_batch(x))
            except Exception as e:
                raise RuntimeError(
                    "Unable to automatically build the program. "
                    "Please build it yourself before calling "
                    "fit/evaluate/predict. "
                    "A program is 'built' when its variables have "
                    "been created and its `self.built` attribute "
                    "is True. Usually, calling the program on a batch "
                    "of data is the right way to build it.\n"
                    "Exception encountered:\n"
                    f"'{e}'"
                )
            if compile_metrics_unbuilt:
                # Build all metric state with `backend.compute_output_spec`.
                run_maybe_nested(
                    backend.compute_output_spec(
                        self.compute_metrics,
                        x,
                        y,
                        y_pred,
                    )
                )
            if compile_reward_unbuilt:
                # Build `CompileReward` state with `backend.compute_output_spec`.
                run_maybe_nested(
                    backend.compute_output_spec(
                        self.compute_reward,
                        x,
                        y,
                        y_pred,
                        training=False,
                    )
                )
        if optimizer_unbuilt:
            # Build optimizer
            run_maybe_nested(self.optimizer.build(self.trainable_variables))
        if meta_optimizer_unbuilt:
            run_maybe_nested(
                self.meta_optimizer.build(self.optimizer.trainable_variables)
            )
        self._post_build()

    def _assert_compile_called(self, method_name=None):
        if not self.compiled:
            msg = "You must call `compile()` before "
            if metrics_module:
                msg += "using the model."
            else:
                msg += f"calling `{method_name}()`."
            raise ValueError(msg)

    def _should_eval(self, epoch, validation_freq):
        epoch = epoch + 1  # one-index the user-facing epoch.
        if isinstance(validation_freq, int):
            return epoch % validation_freq == 0
        elif isinstance(validation_freq, list):
            return epoch in validation_freq
        else:
            raise ValueError(
                "Expected `validation_freq` to be a list or int. "
                f"Received: validation_freq={validation_freq} of the "
                f"type {type(validation_freq)}."
            )
