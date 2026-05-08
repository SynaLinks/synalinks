# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.datasets import huggingface_dataset as hf_module
from synalinks.src.datasets.built_in.hotpotqa import Answer
from synalinks.src.datasets.built_in.hotpotqa import Document
from synalinks.src.datasets.built_in.hotpotqa import Question
from synalinks.src.datasets.built_in.hotpotqa import get_input_data_model
from synalinks.src.datasets.built_in.hotpotqa import get_knowledge_data_model
from synalinks.src.datasets.built_in.hotpotqa import get_output_data_model
from synalinks.src.datasets.built_in.hotpotqa import load_data
from synalinks.src.datasets.built_in.hotpotqa import load_knowledge


class _FakeListDataset(list):
    """Stand-in for HF ``datasets.Dataset``."""


_TRAIN_ROWS = [
    {
        "question": "Where was Einstein born?",
        "answer": "Ulm",
        "level": "easy",
        "context": {
            "title": ["Albert Einstein", "Ulm"],
            "sentences": [
                ["Einstein was a physicist.", "Born in Ulm."],
                ["Ulm is in Germany.", "Population about 130k."],
            ],
        },
    },
    {
        "question": "What did Curie discover?",
        "answer": "Radium",
        "level": "medium",
        "context": {
            "title": ["Marie Curie"],
            "sentences": [["Discovered radium and polonium."]],
        },
    },
]

_VAL_ROWS = [
    {"question": "Hard question 1?", "answer": "A1", "level": "hard"},
    {"question": "Easy question?", "answer": "A2", "level": "easy"},
    {"question": "Hard question 2?", "answer": "A3", "level": "hard"},
]


def _fake_load_dataset(path, name=None, split=None, **kwargs):
    if split == "train":
        return _FakeListDataset(_TRAIN_ROWS)
    if split == "validation":
        return _FakeListDataset(_VAL_ROWS)
    raise ValueError(f"Unexpected split: {split!r}")


class HotpotQATest(testing.TestCase):
    def test_data_models(self):
        self.assertIs(get_input_data_model(), Question)
        self.assertIs(get_output_data_model(), Answer)
        self.assertIs(get_knowledge_data_model(), Document)

    def test_load_data_train_and_filtered_validation(self):
        with patch.object(hf_module, "load_dataset", side_effect=_fake_load_dataset):
            (x_train, y_train), (x_test, y_test) = load_data()

        # All train rows pass through.
        self.assertEqual(len(x_train), 2)
        self.assertEqual(len(y_train), 2)
        self.assertEqual(x_train[0].question, _TRAIN_ROWS[0]["question"])
        self.assertEqual(y_train[0].answer, _TRAIN_ROWS[0]["answer"])

        # Only ``level == "hard"`` rows survive in the test split.
        self.assertEqual(len(x_test), 2)
        self.assertEqual(
            [q.question for q in x_test], ["Hard question 1?", "Hard question 2?"]
        )
        self.assertEqual([a.answer for a in y_test], ["A1", "A3"])

    def test_load_knowledge_skips_rows_without_context(self):
        # Hits the ``continue`` branch in _HotpotKnowledge._iter_rows.
        rows_with_gap = [
            {"context": None},
            _TRAIN_ROWS[0],
        ]

        def _fake(path, name=None, split=None, **kwargs):
            if split == "train":
                return _FakeListDataset(rows_with_gap)
            raise ValueError(f"Unexpected split: {split!r}")

        with patch.object(hf_module, "load_dataset", side_effect=_fake):
            documents = load_knowledge()
        # First row was skipped, second produced 2 documents.
        self.assertEqual(len(documents), 2)

    def test_load_knowledge_explodes_contexts_into_documents(self):
        with patch.object(hf_module, "load_dataset", side_effect=_fake_load_dataset):
            documents = load_knowledge()

        # Row 1 → 2 docs, row 2 → 1 doc, total = 3.
        self.assertEqual(len(documents), 3)
        self.assertEqual(documents[0].title, "Albert Einstein")
        # Sentences are joined with newline (matches the original behavior).
        self.assertEqual(documents[0].text, "Einstein was a physicist.\nBorn in Ulm.")
        self.assertEqual(documents[1].title, "Ulm")
        self.assertEqual(documents[2].title, "Marie Curie")
        self.assertEqual(documents[2].text, "Discovered radium and polonium.")
