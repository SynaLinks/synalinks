# Modified from: synalinks/src/callbacks/backup_and_restore_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)


from unittest.mock import patch

from synalinks.src import callbacks
from synalinks.src import modules
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.callbacks.backup_and_restore import BackupAndRestore
from synalinks.src.language_models import LanguageModel
from synalinks.src.optimizers.random_few_shot import RandomFewShot
from synalinks.src.programs import Sequential
from synalinks.src.rewards import ExactMatch
from synalinks.src.testing.test_utils import load_test_data
from synalinks.src.testing.test_utils import mock_completion_data
from synalinks.src.testing.test_utils import mock_incorrect_completion_data
from synalinks.src.utils import file_utils


class Query(DataModel):
    query: str


class AnswerWithRationale(DataModel):
    rationale: str
    answer: str


class InterruptingCallback(callbacks.Callback):
    """A callback to intentionally interrupt training."""

    def __init__(self, steps_int, epoch_int):
        self.batch_count = 0
        self.epoch_count = 0
        self.steps_int = steps_int
        self.epoch_int = epoch_int

    def on_epoch_end(self, epoch, log=None):
        self.epoch_count += 1
        if self.epoch_int is not None and self.epoch_count == self.epoch_int:
            raise RuntimeError("EpochInterruption")

    def on_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.steps_int is not None and self.batch_count == self.steps_int:
            raise RuntimeError("StepsInterruption")


class Counter(DataModel):
    counter: int = 0


class CanaryModule(modules.Module):
    def __init__(self):
        super().__init__()
        self.counter = self.add_variable(data_model=Counter)

    async def call(self, x):
        counter = self.counter.get("counter")
        self.counter.update({"counter": counter + 1})
        return x.clone(name="clone_" + self.name)

    async def compute_output_spec(self, x):
        return x.clone(name="clone_" +self.name)


class BackupAndRestoreCallbackTest(testing.TestCase):
    def make_model(self):
        language_model = LanguageModel(
            model="ollama/mistral",
        )

        program = Sequential(
            modules=[
                modules.Input(data_model=Query),
                CanaryModule(),
                modules.Generator(
                    data_model=AnswerWithRationale,
                    language_model=language_model,
                ),
            ],
            name="test_program",
            description="A test program",
        )
        program.compile(
            reward=ExactMatch(in_mask=["answer"]),
            optimizer=RandomFewShot(),
        )
        return program

    # Check invalid save_freq, both string and non integer
    def test_save_freq_unknown_error(self):
        with self.assertRaisesRegex(ValueError, expected_regex="Invalid value"):
            BackupAndRestore(backup_dir="backup_dir", save_freq="batch")

        with self.assertRaisesRegex(ValueError, expected_regex="Invalid value"):
            BackupAndRestore(backup_dir="backup_dir", save_freq=0.15)

    # Checking if after interruption, correct program params and
    # weights are loaded in step-wise backup
    @patch("litellm.acompletion")
    def test_best_case_step(self, mock_completion):
        temp_dir = self.get_temp_dir()
        backup_dir = file_utils.join(temp_dir, "subdir")
        self.assertFalse(file_utils.exists(backup_dir))

        program = self.make_model()
        cbk = BackupAndRestore(backup_dir, save_freq=1)

        (x_train, y_train), _ = load_test_data()

        mock_responses = []
        mock_responses.extend(mock_incorrect_completion_data())
        mock_responses.extend(mock_completion_data())

        mock_completion.side_effect = mock_responses

        try:
            program.fit(
                x_train,
                y_train,
                batch_size=8,
                callbacks=[
                    cbk,
                    InterruptingCallback(steps_int=2, epoch_int=None),
                ],
                epochs=2,
                verbose=0,
            )
        except RuntimeError:
            self.assertTrue(file_utils.exists(backup_dir))
            self.assertEqual(cbk._current_epoch, 0)
            self.assertEqual(cbk._last_batch_seen, 1)
            self.assertEqual(int(program.modules[1].counter.get("counter")), 2)

            hist = program.fit(x_train, y_train, batch_size=4, callbacks=[cbk], epochs=5)

            self.assertEqual(cbk._current_epoch, 5)
            self.assertEqual(hist.epoch[-1], 4)
            self.assertEqual(int(program.modules[1].counter.get("counter")), 17)

    # Checking if after interruption, correct program params and
    # variables are loaded in epoch-wise backup
    @patch("litellm.acompletion")
    def test_best_case_epoch(self, mock_completion):
        temp_dir = self.get_temp_dir()
        backup_dir = file_utils.join(temp_dir, "subdir")
        self.assertFalse(file_utils.exists(backup_dir))

        program = self.make_model()
        self.assertEqual(int(program.modules[1].counter.get("counter")), 0)
        cbk = BackupAndRestore(backup_dir=backup_dir, save_freq="epoch")

        (x_train, y_train), _ = load_test_data()

        mock_responses = []
        mock_responses.extend(mock_incorrect_completion_data())
        mock_responses.extend(mock_completion_data())

        mock_completion.side_effect = mock_responses

        try:
            program.fit(
                x_train,
                y_train,
                batch_size=8,
                callbacks=[
                    cbk,
                    InterruptingCallback(steps_int=None, epoch_int=2),
                ],
                epochs=6,
                verbose=0,
            )
        except RuntimeError:
            self.assertEqual(cbk._current_epoch, 2)
            self.assertTrue(file_utils.exists(backup_dir))
            self.assertEqual(int(program.modules[1].counter.get("counter")), 6)

            hist = program.fit(x_train, y_train, batch_size=4, callbacks=[cbk], epochs=5)
            self.assertEqual(cbk._current_epoch, 5)
            self.assertEqual(hist.epoch[-1], 4)
            self.assertEqual(int(program.modules[1].counter.get("counter")), 5 * 3)

    # Checking if after interruption and weights corruption, previous program
    # params and weights are loaded
    @patch("litellm.acompletion")
    def test_backup_corrupted(self, mock_completion):
        temp_dir = self.get_temp_dir()
        backup_dir = file_utils.join(temp_dir, "subdir")
        self.assertFalse(file_utils.exists(backup_dir))

        program = self.make_model()
        self.assertEqual(int(program.modules[1].counter.get("counter")), 0)
        cbk = BackupAndRestore(
            backup_dir=backup_dir, save_freq="epoch", double_checkpoint=True
        )

        (x_train, y_train), _ = load_test_data()

        mock_responses = []
        mock_responses.extend(mock_incorrect_completion_data())
        mock_responses.extend(mock_completion_data())

        mock_completion.side_effect = mock_responses

        try:
            program.fit(
                x_train,
                y_train,
                batch_size=8,
                callbacks=[
                    cbk,
                    InterruptingCallback(steps_int=None, epoch_int=2),
                ],
                epochs=6,
                verbose=0,
            )
        except RuntimeError:
            self.assertEqual(cbk._current_epoch, 2)
            self.assertTrue(file_utils.exists(backup_dir))
            self.assertTrue(file_utils.exists(cbk._weights_path))
            self.assertTrue(file_utils.exists(cbk._training_metadata_path))
            self.assertTrue(file_utils.exists(cbk._prev_weights_path))
            self.assertTrue(file_utils.exists(cbk._prev_training_metadata_path))
            self.assertEqual(int(program.modules[1].counter.get("counter")), 6)

            # Corruption weights
            with file_utils.File(cbk._weights_path, "w") as f:
                f.write("0")

            hist = program.fit(x_train, y_train, batch_size=4, callbacks=[cbk], epochs=5)
            self.assertEqual(cbk._current_epoch, 5)
            self.assertEqual(hist.epoch[-1], 4)
            self.assertEqual(int(program.modules[1].counter.get("counter")), 5 * 3)

    # Checking if after interruption, when program is deleted
    @patch("litellm.acompletion")
    def test_model_deleted_case_epoch(self, mock_completion):
        temp_dir = self.get_temp_dir()
        backup_dir = file_utils.join(temp_dir, "subdir")
        self.assertFalse(file_utils.exists(backup_dir))

        program = self.make_model()
        cbk = BackupAndRestore(backup_dir, save_freq="epoch")

        (x_train, y_train), _ = load_test_data()

        mock_messages = []

        mock_responses = []
        mock_responses.extend(mock_incorrect_completion_data())
        mock_responses.extend(mock_completion_data())

        mock_completion.side_effect = mock_responses

        program.fit(
            x_train,
            y_train,
            batch_size=8,
            callbacks=[cbk],
            epochs=2,
            verbose=0,
        )
        self.assertFalse(file_utils.exists(backup_dir))

    def test_backup_dir_empty_error(self):
        with self.assertRaisesRegex(
            ValueError, expected_regex="Empty `backup_dir` argument passed"
        ):
            BackupAndRestore(backup_dir="", save_freq="epoch")

    def test_backup_dir_none_error(self):
        with self.assertRaisesRegex(
            ValueError, expected_regex="Empty `backup_dir` argument passed"
        ):
            BackupAndRestore(backup_dir=None, save_freq="epoch")
