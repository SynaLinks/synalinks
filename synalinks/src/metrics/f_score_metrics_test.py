# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json

from synalinks.src import backend
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.metrics.f_score_metrics import BinaryF1Score
from synalinks.src.metrics.f_score_metrics import BinaryFBetaScore
from synalinks.src.metrics.f_score_metrics import CategoricalF1Score
from synalinks.src.metrics.f_score_metrics import CategoricalFBetaScore
from synalinks.src.metrics.f_score_metrics import F1Score
from synalinks.src.metrics.f_score_metrics import FBetaScore
from synalinks.src.metrics.f_score_metrics import ListF1Score
from synalinks.src.metrics.f_score_metrics import ListFBetaScore


class FBetaScoreTest(testing.TestCase):
    async def test_same_field(self):
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Toulouse is the French city of aeronautics and space.")
        y_true = Answer(answer="Toulouse is the French city of aeronautics and space.")

        metric = FBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_field(self):
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Toulouse is the French city of aeronautics and space.")
        y_true = Answer(answer="Paris is the capital of France.")

        metric = FBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())


class F1ScoreTest(testing.TestCase):
    async def test_same_field(self):
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Toulouse is the French city of aeronautics and space.")
        y_true = Answer(answer="Toulouse is the French city of aeronautics and space.")

        metric = F1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_field(self):
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Toulouse is the French city of aeronautics and space.")
        y_true = Answer(answer="Paris is the capital of France.")

        metric = F1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_identical_with_repeated_tokens(self):
        # Regression: SQuAD-style F1 uses Counter intersection, so identical
        # strings with repeated tokens (LMs do this all the time) score 1.0.
        # With set-based TP and list-based FP/FN this would erroneously be < 1.
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="John and Mary went to school. John was happy.")
        y_true = Answer(answer="John and Mary went to school. John was happy.")

        metric = F1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_squad_multiset_partial_overlap(self):
        # `normalize_and_tokenize` strips "is" (treated as a stopword), so:
        # pred → ["paris", "paris", "france"]  (len 3)
        # true → ["paris", "france"]            (len 2)
        # multiset intersection = {"paris": 1, "france": 1} → 2
        # precision = 2/3, recall = 2/2 = 1.0, F1 = 4/5 = 0.8
        # Under the old (buggy) set-based code this would have been
        # set ∩ = 2 with FP = 3-2 = 1, FN = 2-2 = 0 → identical here, but
        # the multiset formulation is what makes the previous
        # `test_identical_with_repeated_tokens` case pass.
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Paris Paris is France")
        y_true = Answer(answer="Paris is France")

        metric = F1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.8, delta=1e-3)


class BinaryFBetaScoreTest(testing.TestCase):
    async def test_same_boolean_fields(self):
        class MultiLabels(DataModel):
            label: bool
            label_1: bool
            label_2: bool

        y_pred = MultiLabels(label=False, label_1=True, label_2=True)
        y_true = MultiLabels(label=False, label_1=True, label_2=True)

        metric = BinaryFBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_boolean_fields(self):
        class MultiLabels(DataModel):
            label: bool
            label_1: bool
            label_2: bool

        y_pred = MultiLabels(label=True, label_1=True, label_2=False)
        y_true = MultiLabels(label=False, label_1=False, label_2=True)

        metric = BinaryFBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())


class BinaryF1ScoreTest(testing.TestCase):
    async def test_same_boolean_fields(self):
        class MultiLabels(DataModel):
            label: bool
            label_1: bool
            label_2: bool

        y_pred = MultiLabels(label=False, label_1=True, label_2=True)
        y_true = MultiLabels(label=False, label_1=True, label_2=True)

        metric = BinaryF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_boolean_fields(self):
        class MultiLabels(DataModel):
            label: bool
            label_1: bool
            label_2: bool

        y_pred = MultiLabels(label=True, label_1=True, label_2=False)
        y_true = MultiLabels(label=False, label_1=False, label_2=True)

        metric = BinaryF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_reset_state(self):
        class MultiLabels(DataModel):
            label: bool
            label_1: bool
            label_2: bool

        y_pred = MultiLabels(label=False, label_1=True, label_2=True)
        y_true = MultiLabels(label=False, label_1=True, label_2=True)

        metric = BinaryF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())
        metric.reset_state()
        score = metric.result()
        self.assertEqual(score, 0.0)

    async def test_variable_serialization(self):
        class MultiLabels(DataModel):
            label: bool
            label_1: bool
            label_2: bool

        y_pred = MultiLabels(label=False, label_1=True, label_2=True)
        y_true = MultiLabels(label=False, label_1=True, label_2=True)
        metric = BinaryF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())
        state = metric.variables[0]
        # Try to dump it so we can test if the state is serializable
        _ = json.dumps(state.get_json())


class CategoricalFBetaScoreTest(testing.TestCase):
    async def test_same_single_label(self):
        from typing import Literal

        class SingleLabel(DataModel):
            label: Literal["label_1", "label_2", "label_3"]

        y_pred = SingleLabel(label="label_1")
        y_true = SingleLabel(label="label_1")

        metric = CategoricalFBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        print(score)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_single_label(self):
        from typing import Literal

        class SingleLabel(DataModel):
            label: Literal["label_1", "label_2", "label_3"]

        y_pred = SingleLabel(label="label_1")
        y_true = SingleLabel(label="label_2")

        metric = CategoricalFBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_same_multi_label(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1", "label_2"])
        y_true = MultiLabel(labels=["label_1", "label_2"])

        metric = CategoricalFBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_partial_overlap_multi_label(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1", "label_2"])
        y_true = MultiLabel(labels=["label_2", "label_3"])

        metric = CategoricalFBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        # Should have partial overlap (label_2 matches)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    async def test_no_overlap_multi_label(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1"])
        y_true = MultiLabel(labels=["label_2", "label_3"])

        metric = CategoricalFBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_retrieval_sources(self):
        from typing import List

        class AnswerWithSources(DataModel):
            sources: List[str]
            answer: str

        y_pred = AnswerWithSources(
            sources=["source1", "source2", "source3"], answer="This is an answer"
        )
        y_true = AnswerWithSources(
            sources=["source1", "source2"], answer="This is an answer"
        )

        # Test with in_mask to only evaluate sources
        metric = CategoricalFBetaScore(average="weighted", in_mask=["sources"])
        score = await metric(y_true, y_pred)
        # Should have partial match (2 out of 3 predicted, 2 out of 2 true)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


class CategoricalF1ScoreTest(testing.TestCase):
    async def test_same_single_label(self):
        from typing import Literal

        class SingleLabel(DataModel):
            label: Literal["label_1", "label_2", "label_3"]

        y_pred = SingleLabel(label="label_1")
        y_true = SingleLabel(label="label_1")

        metric = CategoricalF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_single_label(self):
        from typing import Literal

        class SingleLabel(DataModel):
            label: Literal["label_1", "label_2", "label_3"]

        y_pred = SingleLabel(label="label_1")
        y_true = SingleLabel(label="label_2")

        metric = CategoricalF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_same_multi_label(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1", "label_2", "label_3"])
        y_true = MultiLabel(labels=["label_1", "label_2", "label_3"])

        metric = CategoricalF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_multi_label(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1"])
        y_true = MultiLabel(labels=["label_2", "label_3"])

        metric = CategoricalF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_reset_state(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1", "label_2"])
        y_true = MultiLabel(labels=["label_1", "label_2"])

        metric = CategoricalF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())
        metric.reset_state()
        score = metric.result()
        self.assertEqual(score, 0.0)

    async def test_variable_serialization(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1", "label_2"])
        y_true = MultiLabel(labels=["label_1", "label_2"])

        metric = CategoricalF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())
        state = metric.variables[0]
        # Try to dump it so we can test if the state is serializable
        _ = json.dumps(state.get_json())

    async def test_with_out_mask(self):
        from typing import List

        class AnswerWithSources(DataModel):
            sources: List[str]
            answer: str

        y_pred = AnswerWithSources(
            sources=["source1", "source2"], answer="Different answer"
        )
        y_true = AnswerWithSources(
            sources=["source1", "source2"], answer="This is an answer"
        )

        # Test with out_mask to exclude answer field
        metric = CategoricalF1Score(average="weighted", out_mask=["answer"])
        score = await metric(y_true, y_pred)
        # Sources match perfectly, answer is excluded
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_averaging_modes(self):
        from typing import Literal

        class MultiField(DataModel):
            field1: Literal["a", "b", "c"]
            field2: Literal["x", "y", "z"]

        y_pred = MultiField(field1="a", field2="x")
        y_true = MultiField(field1="a", field2="y")

        # No-labels path is a single global set-F1, so average=None returns
        # one scalar. Per-label breakdown requires labels=... (see
        # CategoricalF1ScoreWithLabelsTest.test_labels_dict_result).
        metric_none = CategoricalF1Score(average=None)
        score_none = await metric_none(y_true, y_pred)
        self.assertIsInstance(score_none, float)

        metric_micro = CategoricalF1Score(average="micro")
        score_micro = await metric_micro(y_true, y_pred)
        self.assertIsInstance(score_micro, float)

        metric_macro = CategoricalF1Score(average="macro")
        score_macro = await metric_macro(y_true, y_pred)
        self.assertIsInstance(score_macro, float)


class CategoricalF1ScoreWithLabelsTest(testing.TestCase):
    async def test_labels_dict_result(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["a", "b", "c"]]

        y_pred = MultiLabel(labels=["a", "b"])
        y_true = MultiLabel(labels=["a", "c"])

        metric = CategoricalF1Score(labels=["a", "b", "c"], average=None)
        score = await metric(y_true, y_pred)
        # average=None + labels → dict {label: score}
        self.assertIsInstance(score, dict)
        self.assertEqual(set(score.keys()), {"a", "b", "c"})
        # Label "a": tp=1, fp=0, fn=0 → F1=1.0
        self.assertAlmostEqual(score["a"], 1.0, delta=1e-3)
        # Label "b": tp=0, fp=1, fn=0 → F1=0.0
        self.assertAlmostEqual(score["b"], 0.0, delta=1e-3)
        # Label "c": tp=0, fp=0, fn=1 → F1=0.0
        self.assertAlmostEqual(score["c"], 0.0, delta=1e-3)

    async def test_labels_macro_average(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["a", "b", "c"]]

        y_pred = MultiLabel(labels=["a", "b"])
        y_true = MultiLabel(labels=["a", "b"])

        metric = CategoricalF1Score(labels=["a", "b", "c"], average="macro")
        score = await metric(y_true, y_pred)
        # Labels "a","b" perfect (F1=1.0 each); "c" absent from both →
        # tp=fp=fn=0, F1≈0 → macro ≈ 2/3
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    async def test_labels_serialization(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["a", "b"]]

        metric = CategoricalF1Score(labels=["a", "b"], average="macro")
        config = metric.get_config()
        self.assertEqual(config["labels"], ["a", "b"])
        clone = CategoricalF1Score.from_config(config)
        self.assertEqual(clone.labels, ["a", "b"])

        y_pred = MultiLabel(labels=["a", "b"])
        y_true = MultiLabel(labels=["a", "b"])
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())
        # State must be JSON-serializable.
        _ = json.dumps(metric.variables[0].get_json())


class ListAliasTest(testing.TestCase):
    def test_aliases_resolve_to_categorical(self):
        # Backward-compatibility: the legacy names import as the new classes.
        self.assertIs(ListFBetaScore, CategoricalFBetaScore)
        self.assertIs(ListF1Score, CategoricalF1Score)


class RaggedStructureFScoreTest(testing.TestCase):
    """Variable-length structures: the extraction-bench regression (F1 side).

    Same shape bug as the accuracy family: per-update leaf counts vary with
    the number of items the model extracted, so the positional TP/FP/FN state
    must zero-pad across updates instead of crashing."""

    async def test_accumulates_across_different_leaf_counts(self):
        from typing import List

        class Extraction(DataModel):
            relations: List[str]

        metric = F1Score(average="micro")
        await metric(
            Extraction(relations=["alpha beta", "gamma delta", "epsilon zeta"]),
            Extraction(relations=["alpha beta", "gamma delta", "epsilon zeta"]),
        )
        # Pre-fix this raised "operands could not be broadcast together"
        await metric(
            Extraction(relations=["alpha beta"]),
            Extraction(relations=["alpha beta"]),
        )
        self.assertAlmostEqual(metric.result(), 1.0, delta=3 * backend.epsilon())

    async def test_unmatched_leaves_are_penalized_not_dropped(self):
        from typing import List

        class Extraction(DataModel):
            relations: List[str]

        metric = F1Score(average="micro")
        # Gold has 2 relations, model extracted 1 perfectly: the missed one
        # is pure false negatives -> micro F1 = 2/3, not a truncated 1.0.
        await metric(
            Extraction(relations=["alpha beta", "gamma delta"]),
            Extraction(relations=["alpha beta"]),
        )
        self.assertAlmostEqual(metric.result(), 2.0 / 3.0, delta=3 * backend.epsilon())
