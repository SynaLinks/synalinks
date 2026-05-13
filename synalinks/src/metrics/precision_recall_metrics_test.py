# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json

from synalinks.src import backend
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.metrics.precision_recall_metrics import BinaryPrecision
from synalinks.src.metrics.precision_recall_metrics import BinaryRecall
from synalinks.src.metrics.precision_recall_metrics import CategoricalPrecision
from synalinks.src.metrics.precision_recall_metrics import CategoricalRecall
from synalinks.src.metrics.precision_recall_metrics import Precision
from synalinks.src.metrics.precision_recall_metrics import Recall


class PrecisionTest(testing.TestCase):
    async def test_identical(self):
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Toulouse is the French city of aeronautics.")
        y_true = Answer(answer="Toulouse is the French city of aeronautics.")

        metric = Precision(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_disjoint(self):
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Aeronautics planes Toulouse")
        y_true = Answer(answer="Paris cuisine bistros")

        metric = Precision(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_squad_multiset(self):
        # pred → ["paris", "paris", "france"] (3 tokens, "is" stripped)
        # true → ["paris", "france"]          (2 tokens)
        # multiset ∩ = {"paris": 1, "france": 1} → num_common = 2
        # precision = 2/3
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Paris Paris is France")
        y_true = Answer(answer="Paris is France")

        metric = Precision(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 2.0 / 3.0, delta=1e-3)

    async def test_get_config_drops_beta(self):
        metric = Precision(average="macro")
        cfg = metric.get_config()
        self.assertNotIn("beta", cfg)
        clone = Precision.from_config(cfg)
        self.assertEqual(clone.average, "macro")


class RecallTest(testing.TestCase):
    async def test_identical(self):
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Toulouse is the French city of aeronautics.")
        y_true = Answer(answer="Toulouse is the French city of aeronautics.")

        metric = Recall(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_squad_multiset(self):
        # pred → ["paris", "paris", "france"] (3 tokens)
        # true → ["paris", "france"]          (2 tokens)
        # multiset ∩ = 2 ; recall = 2/2 = 1.0
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Paris Paris is France")
        y_true = Answer(answer="Paris is France")

        metric = Recall(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=1e-3)

    async def test_partial(self):
        # pred → ["aeronautics"] ; true → ["aeronautics", "planes"]
        # recall = 1/2
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="aeronautics")
        y_true = Answer(answer="aeronautics planes")

        metric = Recall(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5, delta=1e-3)


class BinaryPrecisionTest(testing.TestCase):
    async def test_all_correct_micro(self):
        # Keras/sklearn "micro" sums TP/FP/FN across fields first, then
        # divides. With pred==true everywhere, ΣTP=2, ΣFP=0 → precision=1.0,
        # even though field `c` has no positives.
        class MultiLabel(DataModel):
            a: bool
            b: bool
            c: bool

        y_pred = MultiLabel(a=True, b=True, c=False)
        y_true = MultiLabel(a=True, b=True, c=False)

        metric = BinaryPrecision(average="micro")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_micro_vs_macro_differ(self):
        # Two correct fields and one all-negative field separate the two
        # aggregations:
        #   per-field precision ≈ [1, 1, 0]
        #   macro = mean([1, 1, 0]) = 2/3
        #   micro = ΣTP / Σ(TP+FP) = 2 / (2+0) = 1.0
        class MultiLabel(DataModel):
            a: bool
            b: bool
            c: bool

        y_pred = MultiLabel(a=True, b=True, c=False)
        y_true = MultiLabel(a=True, b=True, c=False)

        micro = BinaryPrecision(average="micro")
        macro = BinaryPrecision(average="macro")
        s_micro = await micro(y_true, y_pred)
        s_macro = await macro(y_true, y_pred)
        self.assertAlmostEqual(s_micro, 1.0, delta=1e-3)
        self.assertAlmostEqual(s_macro, 2.0 / 3.0, delta=1e-3)

    async def test_one_false_positive(self):
        # Per-field TP, FP, FN over (true, pred):
        #   a: (1,1) → TP
        #   b: (0,1) → FP
        #   c: (1,1) → TP
        # micro precision = TP / (TP+FP) per field; with current implementation
        # the per-field precision is [1, 0, 1] (epsilon-stable) and micro-avg
        # mean is 2/3.
        class MultiLabel(DataModel):
            a: bool
            b: bool
            c: bool

        y_pred = MultiLabel(a=True, b=True, c=True)
        y_true = MultiLabel(a=True, b=False, c=True)

        metric = BinaryPrecision(average="micro")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 2.0 / 3.0, delta=1e-3)


class BinaryRecallTest(testing.TestCase):
    async def test_one_false_negative(self):
        # a: (1,0)→FN  b: (1,1)→TP  c: (0,0)→neither
        # Keras/sklearn micro: ΣTP=1, ΣFN=1 → recall = 1/(1+1) = 1/2.
        class MultiLabel(DataModel):
            a: bool
            b: bool
            c: bool

        y_pred = MultiLabel(a=False, b=True, c=False)
        y_true = MultiLabel(a=True, b=True, c=False)

        metric = BinaryRecall(average="micro")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.5, delta=1e-3)

    async def test_serialization(self):
        class MultiLabel(DataModel):
            a: bool
            b: bool

        metric = BinaryRecall(average="weighted", threshold=0.5)
        cfg = metric.get_config()
        self.assertNotIn("beta", cfg)
        self.assertEqual(cfg["threshold"], 0.5)

        y_pred = MultiLabel(a=True, b=True)
        y_true = MultiLabel(a=True, b=True)
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())
        _ = json.dumps(metric.variables[0].get_json())


class CategoricalPrecisionTest(testing.TestCase):
    async def test_labels_dict_result(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["a", "b", "c"]]

        y_pred = MultiLabel(labels=["a", "b"])
        y_true = MultiLabel(labels=["a", "c"])

        metric = CategoricalPrecision(labels=["a", "b", "c"], average=None)
        score = await metric(y_true, y_pred)
        # tp/fp per label:
        #   a: tp=1, fp=0 → precision=1.0
        #   b: tp=0, fp=1 → precision=0.0
        #   c: tp=0, fp=0 → precision=0.0 (epsilon-stable)
        self.assertIsInstance(score, dict)
        self.assertAlmostEqual(score["a"], 1.0, delta=1e-3)
        self.assertAlmostEqual(score["b"], 0.0, delta=1e-3)
        self.assertAlmostEqual(score["c"], 0.0, delta=1e-3)

    async def test_macro(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["a", "b", "c"]]

        y_pred = MultiLabel(labels=["a", "b"])
        y_true = MultiLabel(labels=["a", "b"])

        metric = CategoricalPrecision(labels=["a", "b", "c"], average="macro")
        score = await metric(y_true, y_pred)
        # a,b perfect (1.0 each); c absent everywhere → 0 / eps ≈ 0 → macro 2/3
        self.assertGreater(score, 0.5)
        self.assertLess(score, 1.0)


class CategoricalRecallTest(testing.TestCase):
    async def test_labels_dict_result(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["a", "b", "c"]]

        y_pred = MultiLabel(labels=["a", "b"])
        y_true = MultiLabel(labels=["a", "c"])

        metric = CategoricalRecall(labels=["a", "b", "c"], average=None)
        score = await metric(y_true, y_pred)
        # tp/fn per label:
        #   a: tp=1, fn=0 → recall=1.0
        #   b: tp=0, fn=0 → recall=0.0 (epsilon-stable; b is not in y_true)
        #   c: tp=0, fn=1 → recall=0.0
        self.assertIsInstance(score, dict)
        self.assertAlmostEqual(score["a"], 1.0, delta=1e-3)
        self.assertAlmostEqual(score["c"], 0.0, delta=1e-3)

    async def test_serialization(self):
        from typing import List
        from typing import Literal

        class MultiLabel(DataModel):
            labels: List[Literal["a", "b"]]

        metric = CategoricalRecall(labels=["a", "b"], average="macro")
        cfg = metric.get_config()
        self.assertNotIn("beta", cfg)
        self.assertEqual(cfg["labels"], ["a", "b"])
        clone = CategoricalRecall.from_config(cfg)
        self.assertEqual(clone.labels, ["a", "b"])

        y_pred = MultiLabel(labels=["a", "b"])
        y_true = MultiLabel(labels=["a", "b"])
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())
