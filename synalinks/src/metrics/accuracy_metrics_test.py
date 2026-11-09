# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json

from synalinks.src import backend
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.metrics.accuracy_metrics import Accuracy
from synalinks.src.metrics.accuracy_metrics import BinaryAccuracy
from synalinks.src.metrics.accuracy_metrics import CategoricalAccuracy


class AccuracyTest(testing.TestCase):
    async def test_same_field(self):
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Toulouse is the French city of aeronautics and space.")
        y_true = Answer(answer="Toulouse is the French city of aeronautics and space.")

        metric = Accuracy(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_field(self):
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Toulouse is the French city of aeronautics and space.")
        y_true = Answer(answer="Paris is the capital of France.")

        metric = Accuracy(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_partial_overlap(self):
        class Answer(DataModel):
            answer: str

        # 2 tokens shared ("paris", "is"), 3 unique to y_true, 3 unique to y_pred
        y_pred = Answer(answer="Paris is in northern France region")
        y_true = Answer(answer="Paris is the capital of France")

        metric = Accuracy(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    async def test_reset_state(self):
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Same string here")
        y_true = Answer(answer="Same string here")

        metric = Accuracy(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())
        metric.reset_state()
        score = metric.result()
        self.assertEqual(score, 0.0)

    async def test_variable_serialization(self):
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Same string here")
        y_true = Answer(answer="Same string here")
        metric = Accuracy(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())
        state = metric.variables[0]
        _ = json.dumps(state.get_json())

    async def test_single_letter_answer_with_keep_articles(self):
        class Answer(DataModel):
            answer: str

        # "A" is an article: by default it is stripped and nothing is left to
        # compare. `keep_articles=True` scores multiple-choice letters correctly.
        for letter in ("A", "B"):
            metric = Accuracy(average="weighted", keep_articles=True)
            score = await metric(Answer(answer=letter), Answer(answer=letter))
            self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

        metric = Accuracy(average="weighted", keep_articles=True)
        score = await metric(Answer(answer="A"), Answer(answer="B"))
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_articles_removed_by_default(self):
        class Answer(DataModel):
            answer: str

        y_true = Answer(answer="The capital")
        y_pred = Answer(answer="capital")

        metric = Accuracy(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

        # With the article kept, "the" is an extra token: 1 shared over 2.
        metric = Accuracy(average="weighted", keep_articles=True)
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5, delta=3 * backend.epsilon())

    async def test_tokens_are_summed_over_rows(self):
        class Answer(DataModel):
            answer: str

        # Documents the aggregation described in the class docstring: tokens
        # are summed over rows, so the wrong row adds 2 to the denominator and
        # the score is 1 / (1 + 2), not the mean of 1.0 and 0.0.
        metric = Accuracy(average="weighted", keep_articles=True)
        await metric.update_state(Answer(answer="B"), Answer(answer="B"))
        await metric.update_state(Answer(answer="C"), Answer(answer="D"))
        self.assertAlmostEqual(metric.result(), 1.0 / 3.0, delta=3 * backend.epsilon())

    def test_keep_articles_config_round_trip(self):
        metric = Accuracy(keep_articles=True)
        clone = Accuracy.from_config(metric.get_config())
        self.assertTrue(clone.keep_articles)
        self.assertFalse(Accuracy.from_config(Accuracy().get_config()).keep_articles)

    def test_subclasses_config_round_trip(self):
        # The subclasses do not tokenize: the `keep_articles` option must not
        # leak into their config, or `from_config` would reject it.
        for metric_class in (BinaryAccuracy, CategoricalAccuracy):
            config = metric_class().get_config()
            self.assertNotIn("keep_articles", config)
            metric_class.from_config(config)


class BinaryAccuracyTest(testing.TestCase):
    async def test_same_boolean_fields(self):
        class MultiLabels(DataModel):
            label: bool
            label_1: bool
            label_2: bool

        y_pred = MultiLabels(label=False, label_1=True, label_2=True)
        y_true = MultiLabels(label=False, label_1=True, label_2=True)

        metric = BinaryAccuracy(average="macro")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_boolean_fields(self):
        class MultiLabels(DataModel):
            label: bool
            label_1: bool
            label_2: bool

        y_pred = MultiLabels(label=True, label_1=True, label_2=False)
        y_true = MultiLabels(label=False, label_1=False, label_2=True)

        metric = BinaryAccuracy(average="macro")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_partial_boolean_fields(self):
        class MultiLabels(DataModel):
            label: bool
            label_1: bool
            label_2: bool

        y_pred = MultiLabels(label=True, label_1=True, label_2=False)
        y_true = MultiLabels(label=True, label_1=False, label_2=False)

        metric = BinaryAccuracy(average="macro")
        score = await metric(y_true, y_pred)
        # 2 out of 3 fields agree
        self.assertAlmostEqual(score, 2.0 / 3.0, delta=1e-3)

    async def test_reset_state(self):
        class MultiLabels(DataModel):
            label: bool
            label_1: bool
            label_2: bool

        y_pred = MultiLabels(label=False, label_1=True, label_2=True)
        y_true = MultiLabels(label=False, label_1=True, label_2=True)

        metric = BinaryAccuracy(average="macro")
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
        metric = BinaryAccuracy(average="macro")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())
        state = metric.variables[0]
        _ = json.dumps(state.get_json())


class CategoricalAccuracyTest(testing.TestCase):
    async def test_same_single_label(self):
        from typing import Literal

        class SingleLabel(DataModel):
            label: Literal["label_1", "label_2", "label_3"]

        y_pred = SingleLabel(label="label_1")
        y_true = SingleLabel(label="label_1")

        metric = CategoricalAccuracy(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_single_label(self):
        from typing import Literal

        class SingleLabel(DataModel):
            label: Literal["label_1", "label_2", "label_3"]

        y_pred = SingleLabel(label="label_1")
        y_true = SingleLabel(label="label_2")

        metric = CategoricalAccuracy(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_same_multi_label(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1", "label_2", "label_3"])
        y_true = MultiLabel(labels=["label_1", "label_2", "label_3"])

        metric = CategoricalAccuracy(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_no_overlap_multi_label(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1"])
        y_true = MultiLabel(labels=["label_2", "label_3"])

        metric = CategoricalAccuracy(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_reset_state(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1", "label_2"])
        y_true = MultiLabel(labels=["label_1", "label_2"])

        metric = CategoricalAccuracy(average="weighted")
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

        metric = CategoricalAccuracy(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())
        state = metric.variables[0]
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

        metric = CategoricalAccuracy(average="weighted", out_mask=["answer"])
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_averaging_modes(self):
        from typing import Literal

        class MultiField(DataModel):
            field1: Literal["a", "b", "c"]
            field2: Literal["x", "y", "z"]

        y_pred = MultiField(field1="a", field2="x")
        y_true = MultiField(field1="a", field2="y")

        # No-labels path is a single global set-Jaccard, so average=None
        # returns one scalar. Per-label breakdown requires labels=...
        # (see CategoricalAccuracyWithLabelsTest.test_labels_dict_result).
        metric_none = CategoricalAccuracy(average=None)
        score_none = await metric_none(y_true, y_pred)
        self.assertIsInstance(score_none, float)

        metric_micro = CategoricalAccuracy(average="micro")
        score_micro = await metric_micro(y_true, y_pred)
        self.assertIsInstance(score_micro, float)

        metric_macro = CategoricalAccuracy(average="macro")
        score_macro = await metric_macro(y_true, y_pred)
        self.assertIsInstance(score_macro, float)


class CategoricalAccuracyWithLabelsTest(testing.TestCase):
    async def test_labels_dict_result(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["a", "b", "c"]]

        y_pred = MultiLabel(labels=["a", "b"])
        y_true = MultiLabel(labels=["a", "c"])

        metric = CategoricalAccuracy(labels=["a", "b", "c"], average=None)
        score = await metric(y_true, y_pred)
        self.assertIsInstance(score, dict)
        self.assertEqual(set(score.keys()), {"a", "b", "c"})
        # "a": present in both → correct. "b": only in pred → wrong.
        # "c": only in true → wrong.
        self.assertAlmostEqual(score["a"], 1.0, delta=1e-3)
        self.assertAlmostEqual(score["b"], 0.0, delta=1e-3)
        self.assertAlmostEqual(score["c"], 0.0, delta=1e-3)

    async def test_labels_macro_average(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["a", "b", "c"]]

        y_pred = MultiLabel(labels=["a", "b"])
        y_true = MultiLabel(labels=["a", "b"])

        metric = CategoricalAccuracy(labels=["a", "b", "c"], average="macro")
        score = await metric(y_true, y_pred)
        # All three labels match presence/absence → macro = 1.0
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_labels_serialization(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["a", "b"]]

        metric = CategoricalAccuracy(labels=["a", "b"], average="macro")
        config = metric.get_config()
        self.assertEqual(config["labels"], ["a", "b"])

        y_pred = MultiLabel(labels=["a", "b"])
        y_true = MultiLabel(labels=["a", "b"])
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())
        _ = json.dumps(metric.variables[0].get_json())


class RaggedStructureAccuracyTest(testing.TestCase):
    """Variable-length structures: the extraction-bench regression.

    Golds whose arrays hold a different number of items per sample produce a
    different leaf count per update; the positional state must zero-pad, not
    crash (np broadcast error), and unmatched leaves within one sample must
    be scored, not silently dropped by zip truncation."""

    async def test_accumulates_across_different_leaf_counts(self):
        from typing import List

        class Extraction(DataModel):
            relations: List[str]

        metric = Accuracy(average="micro")
        # 3 leaves, all perfect
        await metric(
            Extraction(relations=["alpha beta", "gamma delta", "epsilon zeta"]),
            Extraction(relations=["alpha beta", "gamma delta", "epsilon zeta"]),
        )
        # 1 leaf, perfect; pre-fix this raised
        # "operands could not be broadcast together"
        await metric(
            Extraction(relations=["alpha beta"]),
            Extraction(relations=["alpha beta"]),
        )
        self.assertAlmostEqual(metric.result(), 1.0, delta=3 * backend.epsilon())

    async def test_unmatched_leaves_are_penalized_not_dropped(self):
        from typing import List

        class Extraction(DataModel):
            relations: List[str]

        metric = Accuracy(average="micro")
        # Gold has 2 relations, the model only extracted 1 (perfectly):
        # the missing one must count against the score (0.5), not be
        # zip-truncated away (1.0).
        await metric(
            Extraction(relations=["alpha beta", "gamma delta"]),
            Extraction(relations=["alpha beta"]),
        )
        self.assertAlmostEqual(metric.result(), 0.5, delta=3 * backend.epsilon())
