# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Smoke tests for the new HF benchmark loaders.

Each test mocks ``load_dataset`` with a tiny set of realistic rows and
verifies that the dataset's templates render valid JSON, the DataModels
parse it, and the train/test split shape is right.
"""

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.datasets import huggingface_dataset as hf_module


class _FakeListDataset(list):
    """Stand-in for HF ``datasets.Dataset``."""


def _patch(rows_by_split):
    def _fake(path, name=None, split=None, **kwargs):
        if split in rows_by_split:
            return _FakeListDataset(rows_by_split[split])
        raise ValueError(f"Unexpected split: {split!r}")

    return patch.object(hf_module, "load_dataset", side_effect=_fake)


# --- Each test loads its dataset, checks shape + a couple of fields. ---


class HellaSwagTest(testing.TestCase):
    def test_load_data(self):
        from synalinks.src.datasets.built_in.hellaswag import load_data

        rows = [
            {
                "ctx": "He poured the milk and",
                "endings": ["sang", "drank it", "ran away", "exploded"],
                "label": "1",
            }
        ]
        with _patch({"train": rows, "validation": rows}):
            (x_train, y_train), (x_test, y_test) = load_data()
        self.assertEqual(len(x_train), 1)
        self.assertEqual(len(x_test), 1)
        self.assertIn("A) sang", x_train[0].question)
        self.assertEqual(y_train[0].answer, "B")


class ARCChallengeTest(testing.TestCase):
    def test_load_data_variable_choice_count(self):
        from synalinks.src.datasets.built_in.arc_challenge import load_data

        rows = [
            {
                "question": "What melts ice?",
                "choices": {"text": ["heat", "cold"], "label": ["A", "B"]},
                "answerKey": "A",
            },
            {
                "question": "What is H2O?",
                "choices": {
                    "text": ["water", "salt", "iron", "oil", "gas"],
                    "label": ["1", "2", "3", "4", "5"],
                },
                "answerKey": "1",
            },
        ]
        with _patch({"train": rows, "test": rows}):
            (x_train, y_train), _ = load_data()
        self.assertIn("A) heat", x_train[0].question)
        self.assertIn("5) gas", x_train[1].question)
        self.assertEqual(y_train[0].answer, "A")
        self.assertEqual(y_train[1].answer, "1")


class WinoGrandeTest(testing.TestCase):
    def test_load_data_literal_int(self):
        from synalinks.src.datasets.built_in.winogrande import load_data

        rows = [
            {
                "sentence": "The cup is on the _ because it is heavy.",
                "option1": "table",
                "option2": "shelf",
                "answer": "1",
            }
        ]
        with _patch({"train": rows, "validation": rows}):
            (x_train, y_train), _ = load_data()
        self.assertIn("1) table", x_train[0].question)
        self.assertEqual(y_train[0].answer, 1)


class BoolQTest(testing.TestCase):
    def test_load_data_bool_answer(self):
        from synalinks.src.datasets.built_in.boolq import load_data

        rows = [
            {"passage": "The sky is blue.", "question": "is the sky blue", "answer": True}
        ]
        with _patch({"train": rows, "validation": rows}):
            (x_train, y_train), _ = load_data()
        self.assertIn("Question: is the sky blue?", x_train[0].question)
        self.assertEqual(y_train[0].answer, True)


class BBHTest(testing.TestCase):
    def test_load_data_target_string_to_bool(self):
        from synalinks.src.datasets.built_in.bbh import load_data

        rows = [
            {"input": "True and False or True", "target": "True"},
            {"input": "not (True and True)", "target": "False"},
            {"input": "False or False", "target": "False"},
            {"input": "True and True", "target": "True"},
            {"input": "not False", "target": "True"},
        ]
        with _patch({"test": rows}):
            (x_train, y_train), (x_test, y_test) = load_data()
        # 5 rows × 0.8 = 4 train, 1 test.
        self.assertEqual(len(x_train), 4)
        self.assertEqual(len(x_test), 1)
        self.assertEqual(y_train[0].answer, True)
        self.assertEqual(y_train[1].answer, False)


class DROPTest(testing.TestCase):
    def test_load_data_first_span(self):
        from synalinks.src.datasets.built_in.drop import load_data

        rows = [
            {
                "passage": "John ran 5 miles. Mary ran 3 miles.",
                "question": "How far did John run?",
                "answers_spans": {"spans": ["5 miles", "five miles"]},
            }
        ]
        with _patch({"train": rows, "validation": rows}):
            (x_train, y_train), _ = load_data()
        self.assertEqual(y_train[0].answer, "5 miles")


class TruthfulQATest(testing.TestCase):
    def test_load_data_letter_from_one_hot_label(self):
        from synalinks.src.datasets.built_in.truthfulqa import load_data

        rows = [
            {
                "question": "What's 2+2?",
                "mc1_targets": {
                    "choices": ["3", "4", "5"],
                    "labels": [0, 1, 0],
                },
            },
            {
                "question": "Capital of France?",
                "mc1_targets": {
                    "choices": ["Berlin", "Paris", "Rome"],
                    "labels": [0, 1, 0],
                },
            },
            {
                "question": "Sky color?",
                "mc1_targets": {
                    "choices": ["red", "green", "blue"],
                    "labels": [0, 0, 1],
                },
            },
            {
                "question": "Largest ocean?",
                "mc1_targets": {
                    "choices": ["Pacific", "Atlantic"],
                    "labels": [1, 0],
                },
            },
            {
                "question": "Closest star?",
                "mc1_targets": {
                    "choices": ["Sun", "Sirius"],
                    "labels": [1, 0],
                },
            },
        ]
        with _patch({"validation": rows}):
            (x_train, y_train), (x_test, y_test) = load_data()
        # 5 × 0.8 = 4 train.
        self.assertEqual(len(x_train), 4)
        # Index of 1 in labels → letter mapping.
        self.assertEqual(y_train[0].answer, "B")
        self.assertEqual(y_train[2].answer, "C")
        self.assertEqual(y_train[3].answer, "A")


class HumanEvalTest(testing.TestCase):
    def test_load_data(self):
        from synalinks.src.datasets.built_in.humaneval import load_data

        rows = [
            {
                "prompt": f"def f{i}(x): pass\n",
                "canonical_solution": f"    return x + {i}\n",
            }
            for i in range(5)
        ]
        with _patch({"test": rows}):
            (x_train, y_train), (x_test, y_test) = load_data()
        self.assertEqual(len(x_train), 4)
        self.assertEqual(x_train[0].prompt, "def f0(x): pass\n")
        self.assertEqual(y_train[0].completion, "    return x + 0\n")


class IFEvalTest(testing.TestCase):
    def test_load_data_response_mirrors_prompt(self):
        from synalinks.src.datasets.built_in.ifeval import load_data

        rows = [{"prompt": f"Do task {i}."} for i in range(5)]
        with _patch({"train": rows}):
            (x_train, y_train), _ = load_data()
        # Gold response is the prompt itself, by design.
        self.assertEqual(y_train[0].response, "Do task 0.")


class SQuADTest(testing.TestCase):
    def test_load_data_first_answer(self):
        from synalinks.src.datasets.built_in.squad import load_data

        rows = [
            {
                "context": "The Eiffel Tower is in Paris.",
                "question": "Where is the Eiffel Tower?",
                "answers": {"text": ["Paris", "in Paris"]},
            }
        ]
        with _patch({"train": rows, "validation": rows}):
            (x_train, y_train), _ = load_data()
        self.assertEqual(y_train[0].answer, "Paris")


class LogiQATest(testing.TestCase):
    def test_load_data_letter_answer(self):
        from synalinks.src.datasets.built_in.logiqa import load_data

        rows = [
            {
                "context": "All cats are mammals.",
                "question": "Are cats mammals?",
                "options": [
                    "A. Yes",
                    "B. No",
                    "C. Sometimes",
                    "D. Unknown",
                ],
                "answer": "A",
            }
        ]
        with _patch({"train": rows, "test": rows}):
            (x_train, y_train), _ = load_data()
        self.assertIn("A. Yes", x_train[0].question)
        self.assertEqual(y_train[0].answer, "A")


class BBQTest(testing.TestCase):
    def test_load_data_from_age_split(self):
        from synalinks.src.datasets.built_in.bbq import load_data

        rows = [
            {
                "context": "An older man and a younger man.",
                "question": "Who was confused?",
                "choices": ["The older man", "The younger man", "Cannot be determined"],
                "answer": 2,
            }
        ] * 5
        with _patch({"age": rows}):
            (x_train, y_train), (x_test, y_test) = load_data()
        self.assertEqual(len(x_train), 4)
        self.assertEqual(y_train[0].answer, "C")


class LAMBADATest(testing.TestCase):
    def test_load_data_blanks_last_word(self):
        from synalinks.src.datasets.built_in.lambada import load_data

        rows = [{"text": f"This is a sentence number {i} ending here."} for i in range(5)]
        with _patch({"test": rows}):
            (x_train, y_train), _ = load_data()
        # Last word is "here." for all rows; rest forms the prompt with " ___".
        self.assertTrue(x_train[0].question.endswith("___"))
        self.assertEqual(y_train[0].answer, "here.")


class IterableDatasetTest(testing.TestCase):
    """``iterable_dataset()`` returns a streaming HuggingFaceDataset that
    respects ``repeat`` / ``batch_size`` / ``limit`` knobs.
    """

    async def test_gsm8k_iterable_repeat_for_grpo(self):
        from synalinks.src.datasets.built_in.gsm8k import iterable_dataset

        rows = [{"question": f"q{i}", "answer": f"r{i}\n#### {i}"} for i in range(3)]
        with _patch({"train": rows}):
            ds = iterable_dataset(repeat=2, batch_size=2, limit=2)
            batches = list(ds)
        # 2 raw rows × repeat=2 = 4 examples; batch_size=2 → 2 batches.
        sizes = [len(b[0]) for b in batches]
        self.assertEqual(sizes, [2, 2])
        # Repeats are consecutive — needed for GRPO grouping semantics.
        questions = [r.question for batch in batches for r in batch[0]]
        self.assertEqual(questions, ["q0", "q0", "q1", "q1"])

    async def test_validation_split_override(self):
        # ``validation_split`` is honored end-to-end via split_train_test.
        from synalinks.src.datasets.built_in.bbh import load_data

        rows = [
            {"input": f"e{i}", "target": "True" if i % 2 else "False"} for i in range(10)
        ]
        with _patch({"test": rows}):
            (x_train, _), (x_test, _) = load_data(validation_split=0.5)
        # 10 × 0.5 = 5 train / 5 test (vs. 8/2 with the default 0.2).
        self.assertEqual(len(x_train), 5)
        self.assertEqual(len(x_test), 5)


# All built-in HF benchmarks share the same three-function surface:
# ``get_input_data_model``, ``get_output_data_model``, ``iterable_dataset``.
# A single parametrized class exercises the surface for each module so we
# don't carry 16 copies of the same trivial test.
_BENCHMARKS = [
    "arc_challenge",
    "bbh",
    "bbq",
    "boolq",
    "drop",
    "gsm8k",
    "hellaswag",
    "hotpotqa",
    "humaneval",
    "ifeval",
    "lambada",
    "logiqa",
    "mmlu",
    "squad",
    "truthfulqa",
    "winogrande",
]


class GettersAndIterableSurfaceTest(testing.TestCase):
    """Cover ``get_input_data_model`` / ``get_output_data_model`` /
    ``iterable_dataset`` for every built-in benchmark.
    """

    def test_getters_return_datamodel_classes(self):
        import importlib

        from synalinks.src.backend import DataModel

        for name in _BENCHMARKS:
            mod = importlib.import_module(f"synalinks.src.datasets.built_in.{name}")
            inp = mod.get_input_data_model()
            out = mod.get_output_data_model()
            self.assertTrue(
                issubclass(inp, DataModel), f"{name}: input is not DataModel"
            )
            self.assertTrue(
                issubclass(out, DataModel), f"{name}: output is not DataModel"
            )

    def test_iterable_dataset_constructs(self):
        import importlib

        from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset

        # Empty list satisfies the streaming-path constructor without
        # reaching the templates (we're proving the function body runs,
        # not iterating rows).
        with patch.object(
            hf_module, "load_dataset", return_value=_FakeListDataset()
        ):
            for name in _BENCHMARKS:
                mod = importlib.import_module(
                    f"synalinks.src.datasets.built_in.{name}"
                )
                ds = mod.iterable_dataset(repeat=1, batch_size=1, limit=2)
                self.assertIsInstance(
                    ds, HuggingFaceDataset, f"{name}: not a HuggingFaceDataset"
                )
