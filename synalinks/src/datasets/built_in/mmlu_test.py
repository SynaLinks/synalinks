# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

import pytest

from synalinks.src import testing
from synalinks.src.datasets import huggingface_dataset as hf_module
from synalinks.src.datasets.built_in.mmlu import MMLUAnswer
from synalinks.src.datasets.built_in.mmlu import MMLUQuestion
from synalinks.src.datasets.built_in.mmlu import get_input_data_model
from synalinks.src.datasets.built_in.mmlu import get_output_data_model
from synalinks.src.datasets.built_in.mmlu import load_data


class _FakeListDataset(list):
    """Stand-in for HF ``datasets.Dataset``."""


_VALIDATION_ROWS = [
    {
        "question": "What is 2+2?",
        "choices": ["3", "4", "5", "6"],
        "answer": 1,
        "subject": "elementary_mathematics",
    },
    {
        "question": "Capital of France?",
        "choices": ["Berlin", "Madrid", "Paris", "Rome"],
        "answer": 2,
        "subject": "world_geography",
    },
]

_TEST_ROWS = [
    {
        "question": "Who wrote Hamlet?",
        "choices": ["Dickens", "Shakespeare", "Austen", "Tolkien"],
        "answer": 1,
        "subject": "literature",
    },
]


def _fake_load_dataset(path, name=None, split=None, **kwargs):
    if split == "validation":
        return _FakeListDataset(_VALIDATION_ROWS)
    if split == "test":
        return _FakeListDataset(_TEST_ROWS)
    raise ValueError(f"Unexpected split: {split!r}")


class MMLUTest(testing.TestCase):
    def test_data_models(self):
        self.assertIs(get_input_data_model(), MMLUQuestion)
        self.assertIs(get_output_data_model(), MMLUAnswer)

    def test_load_data_shapes(self):
        with patch.object(hf_module, "load_dataset", side_effect=_fake_load_dataset):
            (x_train, y_train), (x_test, y_test) = load_data()
        self.assertEqual(len(x_train), 2)
        self.assertEqual(len(y_train), 2)
        self.assertEqual(len(x_test), 1)
        self.assertEqual(len(y_test), 1)

    def test_input_fields_round_trip(self):
        with patch.object(hf_module, "load_dataset", side_effect=_fake_load_dataset):
            (x_train, _), _ = load_data()
        self.assertEqual(x_train[0].question, "What is 2+2?")
        self.assertEqual(x_train[0].choices, ["3", "4", "5", "6"])
        self.assertEqual(x_train[0].subject, "elementary_mathematics")

    def test_answer_index_mapped_to_letter(self):
        # 1 → "B", 2 → "C", 1 → "B" — verifies the [A,B,C,D][answer]
        # template indexing.
        with patch.object(hf_module, "load_dataset", side_effect=_fake_load_dataset):
            (_, y_train), (_, y_test) = load_data()
        self.assertEqual([a.answer for a in y_train], ["B", "C"])
        self.assertEqual(y_test[0].answer, "B")

    def test_answer_literal_rejects_invalid_letter(self):
        # The DataModel uses Literal["A","B","C","D"] — pydantic must
        # refuse anything else.
        with pytest.raises(Exception):
            MMLUAnswer(answer="E")

    def test_choices_must_be_list_of_strings(self):
        # Type guard on MMLUQuestion.choices.
        with pytest.raises(Exception):
            MMLUQuestion(question="q", choices="not a list", subject="x")
