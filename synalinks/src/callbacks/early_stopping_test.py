# Modified from: keras/src/callbacks/early_stopping_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import warnings

import numpy as np

from synalinks.src import testing
from synalinks.src.callbacks.early_stopping import EarlyStopping


class _FakeProgram:
    def __init__(self, variables=None):
        self.stop_training = False
        self._variables = variables if variables is not None else {"v": 0}
        self._set_calls = []

    def get_variables(self):
        return dict(self._variables)

    def set_variables(self, variables):
        self._set_calls.append(variables)
        self._variables = dict(variables)


def _run(cb, sequence, program=None, start_logs=None):
    """Drive the callback through a sequence of monitor values."""
    if program is None:
        program = _FakeProgram()
    cb.set_program(program)
    cb.on_train_begin(logs=start_logs)
    monitor_key = cb.monitor
    for epoch, value in enumerate(sequence):
        cb.on_epoch_end(epoch, logs={monitor_key: value})
        if program.stop_training:
            break
    cb.on_train_end()
    return program


class EarlyStoppingModeTest(testing.TestCase):
    def test_explicit_min_mode(self):
        cb = EarlyStopping(monitor="val_loss", mode="min", patience=2)
        program = _run(cb, [0.5, 0.4, 0.4, 0.4, 0.4])
        self.assertTrue(program.stop_training)
        self.assertIs(cb.monitor_op, np.less)

    def test_explicit_max_mode(self):
        cb = EarlyStopping(monitor="val_score", mode="max", patience=1)
        program = _run(cb, [0.1, 0.2, 0.2, 0.2])
        self.assertTrue(program.stop_training)
        self.assertIs(cb.monitor_op, np.greater)

    def test_unknown_mode_falls_back_to_auto(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cb = EarlyStopping(monitor="val_reward", mode="banana")
            self.assertTrue(any("unknown" in str(x.message) for x in w))
        self.assertEqual(cb.mode, "auto")

    def test_auto_reward_uses_greater(self):
        cb = EarlyStopping(monitor="val_reward", patience=1)
        program = _run(cb, [0.1, 0.2, 0.2, 0.2])
        self.assertTrue(program.stop_training)
        self.assertIs(cb.monitor_op, np.greater)

    def test_auto_unknown_metric_raises(self):
        cb = EarlyStopping(monitor="val_mystery")
        cb.set_program(_FakeProgram())
        with self.assertRaises(ValueError):
            cb._set_monitor_op()


class EarlyStoppingBehaviorTest(testing.TestCase):
    def test_min_delta_applies_in_min_mode(self):
        cb = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.1, patience=1)
        # Tiny improvement of 0.05 should NOT count when min_delta = 0.1.
        program = _run(cb, [1.0, 0.95, 0.92])
        self.assertTrue(program.stop_training)

    def test_patience_resets_after_improvement(self):
        cb = EarlyStopping(monitor="val_reward", mode="max", patience=2)
        program = _run(cb, [0.1, 0.1, 0.2, 0.2])
        self.assertFalse(program.stop_training)
        self.assertEqual(cb.best, 0.2)

    def test_start_from_epoch_warmup(self):
        cb = EarlyStopping(
            monitor="val_reward", mode="max", patience=1, start_from_epoch=3
        )
        # No improvement at all, but stopping is suppressed before epoch 3.
        program = _run(cb, [0.1, 0.1, 0.1, 0.1])
        self.assertFalse(program.stop_training)

    def test_stop_at_threshold_triggers_immediately(self):
        cb = EarlyStopping(monitor="val_reward", mode="max", stop_at=0.9)
        program = _run(cb, [0.5, 0.95])
        self.assertTrue(program.stop_training)
        # Stopped via stop_at branch, so stopped_epoch remains 0.
        self.assertEqual(cb.stopped_epoch, 0)

    def test_baseline_required_to_reset_wait(self):
        # The first value sets `best`. Subsequent values fail to improve on
        # `best`, and since the baseline is never beaten the wait counter
        # keeps growing until patience is exceeded.
        cb = EarlyStopping(
            monitor="val_reward", mode="max", baseline=10.0, patience=1
        )
        program = _run(cb, [0.5, 0.4, 0.3])
        self.assertTrue(program.stop_training)

    def test_missing_monitor_warns(self):
        cb = EarlyStopping(monitor="val_reward", mode="max")
        cb.set_program(_FakeProgram())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = cb.get_monitor_value({"other": 0.5})
            self.assertIsNone(value)
            self.assertTrue(any("not available" in str(x.message) for x in w))


class EarlyStoppingRestoreVariablesTest(testing.TestCase):
    def test_restores_best_variables(self):
        cb = EarlyStopping(
            monitor="val_reward",
            mode="max",
            patience=1,
            restore_best_variables=True,
            verbose=1,
        )
        program = _FakeProgram(variables={"step": "start"})

        cb.set_program(program)
        cb.on_train_begin()

        program._variables = {"step": "epoch0"}
        cb.on_epoch_end(0, logs={"val_reward": 0.1})
        program._variables = {"step": "epoch1_best"}
        cb.on_epoch_end(1, logs={"val_reward": 0.5})
        program._variables = {"step": "epoch2"}
        cb.on_epoch_end(2, logs={"val_reward": 0.3})
        program._variables = {"step": "epoch3"}
        cb.on_epoch_end(3, logs={"val_reward": 0.2})

        self.assertTrue(program.stop_training)
        cb.on_train_end()
        self.assertEqual(program._variables, {"step": "epoch1_best"})
        self.assertEqual(cb.best_epoch, 1)
