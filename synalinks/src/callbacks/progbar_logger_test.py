# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.callbacks import progbar_logger as progbar_module
from synalinks.src.callbacks.progbar_logger import ProgbarLogger


class _FakeProgram:
    def __init__(self):
        self.stop_training = False
        self.stop_evaluating = False
        self.stop_predicting = False


class ProgbarLoggerEarlyStopTest(testing.TestCase):
    def _logger(self, steps=6):
        cb = ProgbarLogger()
        cb.set_params({"verbose": 1, "epochs": 1, "steps": steps})
        cb.set_program(_FakeProgram())
        return cb

    def _final_update(self, progbar_cls):
        bar = progbar_cls.return_value
        finals = [c for c in bar.update.call_args_list if c.kwargs.get("finalize")]
        self.assertEqual(len(finals), 1)
        return finals[0].args[0]

    def test_prediction_stopped_early_finalizes_at_seen_steps(self):
        cb = self._logger()
        with patch.object(progbar_module, "Progbar") as progbar_cls:
            cb.on_predict_begin()
            cb.on_predict_batch_end(0)
            cb.on_predict_batch_end(1)
            cb.program.stop_predicting = True
            cb.on_predict_end()
        self.assertEqual(self._final_update(progbar_cls), 2)

    def test_completed_prediction_finalizes_at_target(self):
        cb = self._logger()
        with patch.object(progbar_module, "Progbar") as progbar_cls:
            cb.on_predict_begin()
            for batch in range(6):
                cb.on_predict_batch_end(batch)
            cb.on_predict_end()
        self.assertEqual(self._final_update(progbar_cls), 6)

    def test_epoch_stopped_mid_way_finalizes_at_seen_steps(self):
        cb = self._logger()
        with patch.object(progbar_module, "Progbar") as progbar_cls:
            cb.on_train_begin()
            cb.on_epoch_begin(0)
            cb.on_train_batch_end(0)
            cb.program.stop_training = True
            cb.on_epoch_end(0, logs={"reward": 0.5})
        self.assertEqual(self._final_update(progbar_cls), 1)

    def test_early_stopping_after_a_full_epoch_keeps_target(self):
        cb = self._logger(steps=2)
        with patch.object(progbar_module, "Progbar") as progbar_cls:
            cb.on_train_begin()
            cb.on_epoch_begin(0)
            cb.on_train_batch_end(0)
            cb.on_train_batch_end(1)
            # EarlyStopping runs before the progbar and flags the stop after
            # the epoch completed: the bar still reads 2/2.
            cb.program.stop_training = True
            cb.on_epoch_end(0, logs={"reward": 0.5})
        self.assertEqual(self._final_update(progbar_cls), 2)

    def test_evaluation_stopped_early_finalizes_at_seen_steps(self):
        cb = self._logger()
        with patch.object(progbar_module, "Progbar") as progbar_cls:
            cb.on_test_begin()
            cb.on_test_batch_end(0, logs={"reward": 1.0})
            cb.program.stop_evaluating = True
            cb.on_test_end(logs={"reward": 1.0})
        self.assertEqual(self._final_update(progbar_cls), 1)
