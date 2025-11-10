# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json

from synalinks.src import backend
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.metrics.f_score_metrics import BinaryF1Score
from synalinks.src.metrics.f_score_metrics import BinaryFBetaScore
from synalinks.src.metrics.f_score_metrics import F1Score
from synalinks.src.metrics.f_score_metrics import FBetaScore
from synalinks.src.metrics.f_score_metrics import ListFBetaScore
from synalinks.src.metrics.f_score_metrics import ListF1Score


class FBetaScoreTest(testing.TestCase):
    async def test_same_field(self):
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Toulouse is the French city of aeronautics and space.")
        y_true = Answer(answer="Toulouse is the French city of aeronautics and space.")

        metric = FBetaScore()
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_field(self):
        class Answer(DataModel):
            answer: str

        y_pred = Answer(answer="Toulouse is the French city of aeronautics and space.")
        y_true = Answer(answer="Paris is the capital of France.")

        metric = FBetaScore()
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


class ListFBetaScoreTest(testing.TestCase):
    async def test_same_single_label(self):
        from typing import Literal
        
        class SingleLabel(DataModel):
            label: Literal["label_1", "label_2", "label_3"]

        y_pred = SingleLabel(label="label_1")
        y_true = SingleLabel(label="label_1")

        metric = ListFBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        print(score)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_single_label(self):
        from typing import Literal
        
        class SingleLabel(DataModel):
            label: Literal["label_1", "label_2", "label_3"]

        y_pred = SingleLabel(label="label_1")
        y_true = SingleLabel(label="label_2")

        metric = ListFBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_same_multi_label(self):
        from typing import List, Literal
        
        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1", "label_2"])
        y_true = MultiLabel(labels=["label_1", "label_2"])

        metric = ListFBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_partial_overlap_multi_label(self):
        from typing import List, Literal
        
        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1", "label_2"])
        y_true = MultiLabel(labels=["label_2", "label_3"])

        metric = ListFBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        # Should have partial overlap (label_2 matches)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    async def test_no_overlap_multi_label(self):
        from typing import List, Literal
        
        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1"])
        y_true = MultiLabel(labels=["label_2", "label_3"])

        metric = ListFBetaScore(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_retrieval_sources(self):
        from typing import List
        
        class AnswerWithSources(DataModel):
            sources: List[str]
            answer: str

        y_pred = AnswerWithSources(
            sources=["source1", "source2", "source3"],
            answer="This is an answer"
        )
        y_true = AnswerWithSources(
            sources=["source1", "source2"],
            answer="This is an answer"
        )

        # Test with in_mask to only evaluate sources
        metric = ListFBetaScore(average="weighted", in_mask=["sources"])
        score = await metric(y_true, y_pred)
        # Should have partial match (2 out of 3 predicted, 2 out of 2 true)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)


class ListF1ScoreTest(testing.TestCase):
    async def test_same_single_label(self):
        from typing import Literal
        
        class SingleLabel(DataModel):
            label: Literal["label_1", "label_2", "label_3"]

        y_pred = SingleLabel(label="label_1")
        y_true = SingleLabel(label="label_1")

        metric = ListF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_single_label(self):
        from typing import Literal
        
        class SingleLabel(DataModel):
            label: Literal["label_1", "label_2", "label_3"]

        y_pred = SingleLabel(label="label_1")
        y_true = SingleLabel(label="label_2")

        metric = ListF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_same_multi_label(self):
        from typing import List, Literal
        
        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1", "label_2", "label_3"])
        y_true = MultiLabel(labels=["label_1", "label_2", "label_3"])

        metric = ListF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_different_multi_label(self):
        from typing import List, Literal
        
        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1"])
        y_true = MultiLabel(labels=["label_2", "label_3"])

        metric = ListF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, delta=3 * backend.epsilon())

    async def test_reset_state(self):
        from typing import List, Literal
        
        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1", "label_2"])
        y_true = MultiLabel(labels=["label_1", "label_2"])

        metric = ListF1Score(average="weighted")
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())
        metric.reset_state()
        score = metric.result()
        self.assertEqual(score, 0.0)

    async def test_variable_serialization(self):
        from typing import List, Literal
        
        class MultiLabel(DataModel):
            labels: List[Literal["label_1", "label_2", "label_3"]]

        y_pred = MultiLabel(labels=["label_1", "label_2"])
        y_true = MultiLabel(labels=["label_1", "label_2"])
        
        metric = ListF1Score(average="weighted")
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
            sources=["source1", "source2"],
            answer="Different answer"
        )
        y_true = AnswerWithSources(
            sources=["source1", "source2"],
            answer="This is an answer"
        )

        # Test with out_mask to exclude answer field
        metric = ListF1Score(average="weighted", out_mask=["answer"])
        score = await metric(y_true, y_pred)
        # Sources match perfectly, answer is excluded
        self.assertAlmostEqual(score, 1.0, delta=3 * backend.epsilon())

    async def test_averaging_modes(self):
        from typing import List, Literal
        
        class MultiField(DataModel):
            field1: Literal["a", "b", "c"]
            field2: Literal["x", "y", "z"]

        y_pred = MultiField(field1="a", field2="x")
        y_true = MultiField(field1="a", field2="y")

        # Test with no averaging
        metric_none = ListF1Score(average=None)
        score_none = await metric_none(y_true, y_pred)
        self.assertIsInstance(score_none, list)

        # Test with micro averaging
        metric_micro = ListF1Score(average="micro")
        score_micro = await metric_micro(y_true, y_pred)
        self.assertIsInstance(score_micro, float)

        # Test with macro averaging
        metric_macro = ListF1Score(average="macro")
        score_macro = await metric_macro(y_true, y_pred)
        self.assertIsInstance(score_macro, float)