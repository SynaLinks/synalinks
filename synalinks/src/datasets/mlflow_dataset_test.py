# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.datasets import mlflow_dataset as mlflow_module
from synalinks.src.datasets.mlflow_dataset import MLflowDataset

INPUT_TEMPLATE = '{"question": {{ inputs.question | tojson }}}'
OUTPUT_TEMPLATE = '{"answer": {{ expectations.answer | tojson }}}'


class Question(DataModel):
    question: str


class Answer(DataModel):
    answer: str


class _FakeEvaluationDataset:
    """Stand-in for ``mlflow.genai.datasets.EvaluationDataset``."""

    def __init__(self, records, name="qa", dataset_id="d-1"):
        self.name = name
        self.dataset_id = dataset_id
        self.records = records
        self.merged = []

    def to_dict(self):
        return {"name": self.name, "dataset_id": self.dataset_id, "records": self.records}

    def merge_records(self, records):
        self.merged.extend(records)
        return self


def _record(record_id, inputs, expectations=None, tags=None):
    return {
        "dataset_record_id": record_id,
        "inputs": inputs,
        "expectations": expectations,
        "outputs": {},
        "tags": tags or {},
    }


def _make_ds(records, **kwargs):
    """Construct a loader with ``get_dataset`` patched to serve ``records``."""
    fake = _FakeEvaluationDataset(records)
    with patch.object(mlflow_module, "get_dataset", return_value=fake) as mock_get:
        ds = MLflowDataset(name="qa", input_template=INPUT_TEMPLATE, **kwargs)
    return ds, mock_get


class MLflowDatasetTest(testing.TestCase):
    def test_get_dataset_called_with_name_or_id(self):
        _, mock_get = _make_ds([], input_data_model=Question)
        mock_get.assert_called_once_with(name="qa", dataset_id=None)

        fake = _FakeEvaluationDataset([], name="qa", dataset_id="d-42")
        with patch.object(mlflow_module, "get_dataset", return_value=fake) as mock_get:
            ds = MLflowDataset(
                dataset_id="d-42",
                input_data_model=Question,
                input_template=INPUT_TEMPLATE,
            )
        mock_get.assert_called_once_with(name=None, dataset_id="d-42")
        # The loader reports the resolved name and id.
        self.assertEqual(ds.name, "qa")
        self.assertEqual(ds.dataset_id, "d-42")

    def test_name_or_id_required(self):
        with pytest.raises(ValueError, match="`name` or `dataset_id`"):
            MLflowDataset(input_data_model=Question, input_template=INPUT_TEMPLATE)

    def test_input_template_required(self):
        with pytest.raises(ValueError, match="`input_template` is required"):
            MLflowDataset(name="qa", input_data_model=Question)

    def test_tracking_uri_is_set(self):
        fake = _FakeEvaluationDataset([])
        with (
            patch.object(mlflow_module, "get_dataset", return_value=fake),
            patch.object(mlflow_module.mlflow, "set_tracking_uri") as mock_set,
        ):
            MLflowDataset(
                name="qa",
                tracking_uri="sqlite:///probe.db",
                input_data_model=Question,
                input_template=INPUT_TEMPLATE,
            )
        mock_set.assert_called_once_with("sqlite:///probe.db")

    def test_inputs_and_targets(self):
        records = [
            _record("dr-1", {"question": "1+1?"}, {"answer": "2"}),
            _record("dr-2", {"question": "2+2?"}, {"answer": "4"}),
        ]
        ds, _ = _make_ds(
            records,
            input_data_model=Question,
            output_data_model=Answer,
            output_template=OUTPUT_TEMPLATE,
            batch_size=2,
        )
        batches = list(ds)
        self.assertEqual(len(batches), 1)
        x, y = batches[0]
        self.assertEqual([item.question for item in x], ["1+1?", "2+2?"])
        self.assertEqual([item.answer for item in y], ["2", "4"])

    def test_inputs_only_ignores_expectations(self):
        records = [
            _record("dr-1", {"question": "1+1?"}, {"answer": "2"}),
            _record("dr-2", {"question": "2+2?"}),
        ]
        ds, _ = _make_ds(records, input_data_model=Question, batch_size=2)
        (x,) = next(iter(ds))
        self.assertEqual([item.question for item in x], ["1+1?", "2+2?"])

    def test_templates_see_the_whole_record(self):
        records = [
            _record("dr-1", {"question": 'why "X"?'}, {"a": "because"}, {"lvl": "easy"}),
        ]
        ds, _ = _make_ds(
            records,
            input_data_model=Question,
            output_data_model=Answer,
            output_template=(
                '{"answer": {{ (expectations.a ~ " (" ~ tags.lvl ~ ")") | tojson }}}'
            ),
        )
        x, y = next(iter(ds))
        # ``tojson`` escapes the embedded quotes safely.
        self.assertEqual(x[0].question, 'why "X"?')
        self.assertEqual(y[0].answer, "because (easy)")

    def test_labeled_dataset_requires_expectations_on_every_record(self):
        records = [
            _record("dr-1", {"question": "1+1?"}, {"answer": "2"}),
            _record("dr-2", {"question": "2+2?"}),
        ]
        ds, _ = _make_ds(
            records,
            input_data_model=Question,
            output_data_model=Answer,
            output_template=OUTPUT_TEMPLATE,
            batch_size=2,
        )
        with pytest.raises(ValueError, match="'dr-2'.*no expectations"):
            list(ds)

    def test_len_honors_batch_size_limit_and_repeat(self):
        records = [_record(f"dr-{i}", {"question": str(i)}) for i in range(7)]
        ds, _ = _make_ds(records, input_data_model=Question, batch_size=3)
        # ceil(7 / 3) = 3.
        self.assertEqual(len(ds), 3)
        ds, _ = _make_ds(records, input_data_model=Question, batch_size=2, limit=3)
        # ceil(min(7, 3) / 2) = 2.
        self.assertEqual(len(ds), 2)
        ds, _ = _make_ds(records, input_data_model=Question, batch_size=2, repeat=2)
        # 7 rows × 2 repeats = 14, batch_size 2 → 7 batches.
        self.assertEqual(len(ds), 7)
        self.assertEqual(len(list(ds)), 7)

    def test_create_pushes_one_record_per_example(self):
        fake = _FakeEvaluationDataset([], name="qa", dataset_id="d-7")
        x = [Question(question="1+1?"), Question(question="2+2?")]
        y = [Answer(answer="2"), Answer(answer="4")]
        with patch.object(
            mlflow_module, "create_dataset", return_value=fake
        ) as mock_create:
            dataset = MLflowDataset.create(
                "qa", x, y, experiment_id="1", tags={"kind": "qa"}
            )
        mock_create.assert_called_once_with(
            name="qa", experiment_id="1", tags={"kind": "qa"}
        )
        self.assertIs(dataset, fake)
        # Inputs and expectations land as plain JSON.
        self.assertEqual(
            fake.merged,
            [
                {"inputs": {"question": "1+1?"}, "expectations": {"answer": "2"}},
                {"inputs": {"question": "2+2?"}, "expectations": {"answer": "4"}},
            ],
        )

    def test_create_inputs_only(self):
        fake = _FakeEvaluationDataset([], name="qa", dataset_id="d-7")
        with patch.object(mlflow_module, "create_dataset", return_value=fake):
            MLflowDataset.create("qa", [Question(question="1+1?")])
        self.assertEqual(fake.merged, [{"inputs": {"question": "1+1?"}}])

    def test_create_validates_arrays(self):
        with pytest.raises(ValueError, match="at least one"):
            MLflowDataset.create("qa", [])
        with pytest.raises(ValueError, match="same length"):
            MLflowDataset.create("qa", [Question(question="q")], [])

    def test_requires_mlflow(self):
        with patch.object(mlflow_module, "MLFLOW_AVAILABLE", False):
            with pytest.raises(ImportError, match="pip install mlflow"):
                MLflowDataset(
                    name="qa", input_data_model=Question, input_template=INPUT_TEMPLATE
                )

    def test_round_trip_through_a_sqlite_store(self):
        # End to end against a real MLflow store: evaluation datasets need a
        # SQL-backed tracking URI, so use a throwaway sqlite file.
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)
        tracking_uri = "sqlite:///" + os.path.join(tmpdir, "mlflow.db")
        mlflow_module.mlflow.set_tracking_uri(tracking_uri)
        experiment_id = mlflow_module.mlflow.set_experiment("qa").experiment_id

        x = [Question(question="1+1?"), Question(question="2+2?")]
        y = [Answer(answer="2"), Answer(answer="4")]
        dataset = MLflowDataset.create(
            "golden", x, y, experiment_id=experiment_id, tracking_uri=tracking_uri
        )
        self.assertEqual(dataset.name, "golden")

        ds = MLflowDataset(
            name="golden",
            tracking_uri=tracking_uri,
            input_data_model=Question,
            input_template=INPUT_TEMPLATE,
            output_data_model=Answer,
            output_template=OUTPUT_TEMPLATE,
            batch_size=None,
        )
        self.assertEqual(len(ds), 1)
        x_back, y_back = ds.materialize()
        self.assertEqual(sorted(item.question for item in x_back), ["1+1?", "2+2?"])
        self.assertEqual(sorted(item.answer for item in y_back), ["2", "4"])
