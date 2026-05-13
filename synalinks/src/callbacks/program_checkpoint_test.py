# Modified from: keras/src/callbacks/model_checkpoint_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import time
import warnings
from pathlib import Path

import numpy as np

from synalinks.src import testing
from synalinks.src.callbacks.program_checkpoint import ProgramCheckpoint


class _FakeProgram:
    def __init__(self):
        self.saved = []
        self.saved_variables = []

    def save(self, filepath, overwrite=True):
        self.saved.append(filepath)
        with open(filepath, "w") as f:
            f.write("program")

    def save_variables(self, filepath, overwrite=True):
        self.saved_variables.append(filepath)
        with open(filepath, "w") as f:
            f.write("variables")


class _RaisingProgram:
    def __init__(self, exc):
        self.exc = exc

    def save(self, filepath, overwrite=True):
        raise self.exc

    def save_variables(self, filepath, overwrite=True):
        raise self.exc


class ProgramCheckpointInitTest(testing.TestCase):
    def test_unknown_mode_falls_back_to_auto(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cb = ProgramCheckpoint(filepath="/tmp/x.json", mode="banana")
            self.assertTrue(any("unknown" in str(x.message) for x in w))
        # Auto mode uses np.greater with -inf baseline.
        self.assertIs(cb.monitor_op, np.greater)
        self.assertEqual(cb.best, -np.inf)

    def test_min_mode_initializes_best(self):
        cb = ProgramCheckpoint(filepath="/tmp/x.json", mode="min")
        self.assertIs(cb.monitor_op, np.less)
        self.assertEqual(cb.best, np.inf)

    def test_max_mode_initializes_best(self):
        cb = ProgramCheckpoint(filepath="/tmp/x.json", mode="max")
        self.assertIs(cb.monitor_op, np.greater)
        self.assertEqual(cb.best, -np.inf)

    def test_initial_value_threshold_overrides_best(self):
        cb = ProgramCheckpoint(
            filepath="/tmp/x.json", mode="max", initial_value_threshold=0.7
        )
        self.assertEqual(cb.best, 0.7)

    def test_invalid_save_freq_raises(self):
        with self.assertRaisesRegex(ValueError, "Unrecognized save_freq"):
            ProgramCheckpoint(filepath="/tmp/x.json", save_freq="batch")

    def test_save_variables_only_requires_correct_suffix(self):
        with self.assertRaisesRegex(ValueError, "variables.json"):
            ProgramCheckpoint(filepath="/tmp/x.json", save_variables_only=True)

    def test_full_save_requires_json_suffix(self):
        with self.assertRaisesRegex(ValueError, ".json"):
            ProgramCheckpoint(filepath="/tmp/x.bin")

    def test_pathlike_filepath_is_converted_to_string(self):
        cb = ProgramCheckpoint(filepath=Path("/tmp/foo.json"))
        self.assertIsInstance(cb.filepath, str)
        self.assertTrue(cb.filepath.endswith("foo.json"))


class ProgramCheckpointPathFormattingTest(testing.TestCase):
    def test_format_with_epoch_and_metrics(self):
        cb = ProgramCheckpoint(filepath="/tmp/ckpt-{epoch:02d}-{reward:.2f}.json")
        path = cb._get_file_path(epoch=4, batch=None, logs={"reward": 0.456})
        self.assertEqual(path, "/tmp/ckpt-05-0.46.json")

    def test_format_with_batch(self):
        cb = ProgramCheckpoint(filepath="/tmp/b{batch:02d}e{epoch:02d}.json")
        path = cb._get_file_path(epoch=0, batch=3, logs={})
        self.assertEqual(path, "/tmp/b04e01.json")

    def test_missing_key_raises(self):
        cb = ProgramCheckpoint(filepath="/tmp/ckpt-{nope}.json")
        with self.assertRaisesRegex(KeyError, "Failed to format"):
            cb._get_file_path(epoch=0, batch=None, logs={})


class ProgramCheckpointBatchScheduleTest(testing.TestCase):
    def test_should_save_on_batch_schedules_every_n(self):
        cb = ProgramCheckpoint(filepath="/tmp/x.json", save_freq=3)
        # Batches are zero-indexed; saving fires after every 3rd batch.
        self.assertFalse(cb._should_save_on_batch(0))  # +1 → 1
        self.assertFalse(cb._should_save_on_batch(1))  # +1 → 2
        self.assertTrue(cb._should_save_on_batch(2))  # +1 → 3 → save & reset
        self.assertFalse(cb._should_save_on_batch(3))  # +1 → 1
        self.assertFalse(cb._should_save_on_batch(4))  # +1 → 2
        self.assertTrue(cb._should_save_on_batch(5))  # +1 → 3 → save

    def test_should_save_on_batch_returns_false_for_epoch_freq(self):
        cb = ProgramCheckpoint(filepath="/tmp/x.json", save_freq="epoch")
        self.assertFalse(cb._should_save_on_batch(7))

    def test_epoch_rollover_handled(self):
        cb = ProgramCheckpoint(filepath="/tmp/x.json", save_freq=2)
        cb._should_save_on_batch(0)  # 1
        self.assertTrue(cb._should_save_on_batch(1))  # 2 → save
        # New epoch — batch goes back down to 0, treated as "batch + 1" added.
        self.assertFalse(cb._should_save_on_batch(0))  # 1
        self.assertTrue(cb._should_save_on_batch(1))  # 2 → save


class ProgramCheckpointSaveTest(testing.TestCase):
    def test_save_every_epoch(self):
        tmp = self.get_temp_dir()
        filepath = os.path.join(tmp, "ckpt-{epoch:02d}.json")
        cb = ProgramCheckpoint(filepath=filepath, save_best_only=False, verbose=1)
        program = _FakeProgram()
        cb.set_program(program)
        cb.on_epoch_begin(0)
        cb.on_epoch_end(0, logs={"val_reward": 0.5})

        out = os.path.join(tmp, "ckpt-01.json")
        self.assertIn(out, program.saved)
        self.assertTrue(os.path.exists(out))

    def test_save_best_only_improvement(self):
        tmp = self.get_temp_dir()
        filepath = os.path.join(tmp, "best.json")
        cb = ProgramCheckpoint(
            filepath=filepath, save_best_only=True, mode="max", verbose=1
        )
        program = _FakeProgram()
        cb.set_program(program)

        cb.on_epoch_end(0, logs={"val_reward": 0.5})
        cb.on_epoch_end(1, logs={"val_reward": 0.3})  # no improvement
        cb.on_epoch_end(2, logs={"val_reward": 0.7})  # improvement

        # Two improvements means two saves.
        self.assertEqual(len(program.saved), 2)
        self.assertEqual(cb.best, 0.7)

    def test_save_best_only_warns_when_monitor_missing(self):
        tmp = self.get_temp_dir()
        filepath = os.path.join(tmp, "best.json")
        cb = ProgramCheckpoint(
            filepath=filepath, save_best_only=True, monitor="val_reward"
        )
        program = _FakeProgram()
        cb.set_program(program)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cb.on_epoch_end(0, logs={"other": 0.5})
            self.assertTrue(any("Can save best" in str(x.message) for x in w))
        self.assertEqual(program.saved, [])

    def test_save_variables_only(self):
        tmp = self.get_temp_dir()
        filepath = os.path.join(tmp, "ckpt.variables.json")
        cb = ProgramCheckpoint(filepath=filepath, save_variables_only=True)
        program = _FakeProgram()
        cb.set_program(program)
        cb.on_epoch_end(0, logs={"val_reward": 0.5})
        self.assertEqual(program.saved_variables, [filepath])
        self.assertEqual(program.saved, [])

    def test_save_creates_missing_directories(self):
        tmp = self.get_temp_dir()
        nested = os.path.join(tmp, "a", "b", "c")
        filepath = os.path.join(nested, "ckpt.json")
        cb = ProgramCheckpoint(filepath=filepath)
        program = _FakeProgram()
        cb.set_program(program)
        cb.on_epoch_end(0, logs={"val_reward": 0.5})
        self.assertTrue(os.path.isdir(nested))
        self.assertTrue(os.path.exists(filepath))

    def test_on_train_batch_end_respects_batch_schedule(self):
        tmp = self.get_temp_dir()
        filepath = os.path.join(tmp, "ckpt.json")
        cb = ProgramCheckpoint(filepath=filepath, save_freq=2)
        program = _FakeProgram()
        cb.set_program(program)
        cb.on_epoch_begin(0)
        cb.on_train_batch_end(0, logs={})  # 1 — no save
        self.assertEqual(program.saved, [])
        cb.on_train_batch_end(1, logs={})  # 2 — save
        self.assertEqual(program.saved, [filepath])

    def test_io_error_directory_raised_as_helpful_message(self):
        tmp = self.get_temp_dir()
        filepath = os.path.join(tmp, "ckpt.json")
        cb = ProgramCheckpoint(filepath=filepath)
        cb.set_program(_RaisingProgram(IsADirectoryError("dir")))
        with self.assertRaisesRegex(IOError, "non-directory filepath"):
            cb.on_epoch_end(0, logs={})

    def test_save_best_only_with_array_monitor_falls_back(self):
        tmp = self.get_temp_dir()
        filepath = os.path.join(tmp, "ckpt.json")
        cb = ProgramCheckpoint(
            filepath=filepath, save_best_only=True, mode="max"
        )
        program = _FakeProgram()
        cb.set_program(program)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cb.on_epoch_end(0, logs={"val_reward": np.array([0.1, 0.2])})
            self.assertTrue(any("scalar value" in str(x.message) for x in w))
        # Falls back to unconditional save.
        self.assertEqual(program.saved, [filepath])


class ProgramCheckpointPatternMatchingTest(testing.TestCase):
    def test_no_matches_returns_none(self):
        tmp = self.get_temp_dir()
        cb = ProgramCheckpoint(filepath=os.path.join(tmp, "ckpt.json"))
        pattern = os.path.join(tmp, "missing-{epoch:02d}.json")
        self.assertIsNone(
            cb._get_most_recently_modified_file_matching_pattern(pattern)
        )

    def test_returns_most_recent_file(self):
        tmp = self.get_temp_dir()
        cb = ProgramCheckpoint(filepath=os.path.join(tmp, "ckpt.json"))
        pattern = os.path.join(tmp, "ckpt-{epoch:02d}.json")

        older = os.path.join(tmp, "ckpt-01.json")
        newer = os.path.join(tmp, "ckpt-02.json")
        with open(older, "w") as f:
            f.write("a")
        time.sleep(0.05)
        with open(newer, "w") as f:
            f.write("b")

        match = cb._get_most_recently_modified_file_matching_pattern(pattern)
        self.assertEqual(match, newer)

    def test_tiebreak_returns_largest_file_name(self):
        tmp = self.get_temp_dir()
        cb = ProgramCheckpoint(filepath=os.path.join(tmp, "ckpt.json"))
        pattern = os.path.join(tmp, "ckpt-{epoch:02d}.json")

        a = os.path.join(tmp, "ckpt-01.json")
        b = os.path.join(tmp, "ckpt-02.json")
        for p in (a, b):
            with open(p, "w") as f:
                f.write("x")
        # Force identical mtimes to trigger the tie-break path.
        ts = time.time()
        os.utime(a, (ts, ts))
        os.utime(b, (ts, ts))

        match = cb._get_most_recently_modified_file_matching_pattern(pattern)
        # When tied, the file path with the lexicographically largest name wins.
        self.assertEqual(match, b)

    def test_checkpoint_exists(self):
        tmp = self.get_temp_dir()
        cb = ProgramCheckpoint(filepath=os.path.join(tmp, "ckpt.json"))
        present = os.path.join(tmp, "ckpt.json")
        with open(present, "w") as f:
            f.write("x")
        self.assertTrue(cb._checkpoint_exists(present))
        self.assertFalse(cb._checkpoint_exists(os.path.join(tmp, "absent.json")))
