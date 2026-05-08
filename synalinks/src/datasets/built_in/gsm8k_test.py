# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.datasets import huggingface_dataset as hf_module
from synalinks.src.datasets.built_in.gsm8k import MathQuestion
from synalinks.src.datasets.built_in.gsm8k import NumericalAnswerWithThinking
from synalinks.src.datasets.built_in.gsm8k import get_input_data_model
from synalinks.src.datasets.built_in.gsm8k import get_output_data_model
from synalinks.src.datasets.built_in.gsm8k import load_data


class _FakeListDataset(list):
    """Stand-in for HF ``datasets.Dataset``."""


_TRAIN_ROWS = [
    {
        "question": "Janet has 3 apples and gets 4 more. How many?",
        "answer": "She had 3, got 4 more.\n#### 7",
    },
    {
        "question": "A book costs $1,250. After 10% off?",
        "answer": "1250 - 125 = 1125.\n#### 1,125",
    },
]

_TEST_ROWS = [
    {
        "question": "What is 6 * 7?",
        "answer": "Multiply.\n#### 42",
    },
]


def _fake_load_dataset(path, name=None, split=None, **kwargs):
    if split == "train":
        return _FakeListDataset(_TRAIN_ROWS)
    if split == "test":
        return _FakeListDataset(_TEST_ROWS)
    raise ValueError(f"Unexpected split: {split!r}")


class GSM8KTest(testing.TestCase):
    def test_data_models(self):
        self.assertIs(get_input_data_model(), MathQuestion)
        self.assertIs(get_output_data_model(), NumericalAnswerWithThinking)

    def test_load_data_renders_via_huggingface_dataset(self):
        with patch.object(hf_module, "load_dataset", side_effect=_fake_load_dataset):
            (x_train, y_train), (x_test, y_test) = load_data()

        self.assertEqual(len(x_train), 2)
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(x_test), 1)
        self.assertEqual(len(y_test), 1)

    def test_thinking_split_strips_marker(self):
        with patch.object(hf_module, "load_dataset", side_effect=_fake_load_dataset):
            (_, y_train), _ = load_data()
        # ``thinking`` is everything before "####", stripped.
        self.assertEqual(y_train[0].thinking, "She had 3, got 4 more.")
        self.assertNotIn("####", y_train[0].thinking)

    def test_answer_handles_comma_thousands_separator(self):
        with patch.object(hf_module, "load_dataset", side_effect=_fake_load_dataset):
            (_, y_train), _ = load_data()
        # "1,125" → 1125.0 (commas dropped before float coercion).
        self.assertEqual(y_train[1].answer, 1125.0)

    def test_question_field_round_trips(self):
        with patch.object(hf_module, "load_dataset", side_effect=_fake_load_dataset):
            (x_train, _), _ = load_data()
        self.assertEqual(x_train[0].question, _TRAIN_ROWS[0]["question"])
        self.assertEqual(x_train[1].question, _TRAIN_ROWS[1]["question"])
