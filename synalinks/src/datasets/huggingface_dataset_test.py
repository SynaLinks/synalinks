# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

import pytest

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.datasets import huggingface_dataset as hf_module
from synalinks.src.datasets.huggingface_dataset import HuggingFaceDataset


class Question(DataModel):
    question: str


class Answer(DataModel):
    answer: str


class _FakeListDataset(list):
    """Stand-in for ``datasets.Dataset``: a list with ``__len__``."""


class _FakeStreamingDataset:
    """Stand-in for ``datasets.IterableDataset``: iterable, no length."""

    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)


class _FakeDatasetDict(dict):
    """Stand-in for ``datasets.DatasetDict``."""


def _make_ds(load_return, **kwargs):
    """Construct an HF dataset with ``load_dataset`` patched to ``load_return``."""
    with patch.object(hf_module, "load_dataset", return_value=load_return) as m:
        ds = HuggingFaceDataset(path="dummy", **kwargs)
    return ds, m


class HuggingFaceDatasetTest(testing.TestCase):
    def test_load_dataset_called_with_forwarded_kwargs(self):
        rows = [{"q": "a"}]
        _, mock_load = _make_ds(
            _FakeListDataset(rows),
            name="cfg",
            split="train",
            revision="abc",
            streaming=False,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            data_files="x.json",
            trust_remote_code=True,
        )
        mock_load.assert_called_once_with(
            "dummy",
            name="cfg",
            split="train",
            revision="abc",
            streaming=False,
            data_files="x.json",
            trust_remote_code=True,
        )

    def test_iter_single_split(self):
        rows = [{"q": "1+1?"}, {"q": "2+2?"}]
        ds, _ = _make_ds(
            _FakeListDataset(rows),
            split="train",
            streaming=False,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=2,
        )
        batches = list(ds)
        self.assertEqual(len(batches), 1)
        (x,) = batches[0]
        self.assertEqual([item.question for item in x], ["1+1?", "2+2?"])

    def test_iter_empty_dataset_dict_yields_nothing(self):
        # Empty DatasetDict: the for-loop in _iter_rows runs zero times and
        # the iterator exits immediately. Covers the 123->exit arc.
        ds, _ = _make_ds(
            _FakeDatasetDict(),
            split=None,
            streaming=False,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=1,
        )
        self.assertEqual(list(ds), [])

    def test_iter_dataset_dict_concatenates_splits_in_order(self):
        # When ``split`` is None and the source is a DatasetDict, iterate
        # all splits in their declared order.
        train = _FakeListDataset([{"q": "t1"}, {"q": "t2"}])
        test = _FakeListDataset([{"q": "v1"}])
        ds_dict = _FakeDatasetDict()
        ds_dict["train"] = train
        ds_dict["test"] = test

        ds, _ = _make_ds(
            ds_dict,
            split=None,
            streaming=False,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=3,
        )
        (x,) = next(iter(ds))
        self.assertEqual([item.question for item in x], ["t1", "t2", "v1"])

    def test_streaming_iter_terminates_when_source_exhausted(self):
        rows = [{"q": "a"}, {"q": "b"}, {"q": "c"}]
        ds, _ = _make_ds(
            _FakeStreamingDataset(rows),
            split="train",
            streaming=True,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=2,
        )
        sizes = [len(b[0]) for b in ds]
        # Two batches: [2, 1] — trailing partial batch gets flushed.
        self.assertEqual(sizes, [2, 1])

    def test_len_streaming_without_limit_raises(self):
        ds, _ = _make_ds(
            _FakeStreamingDataset([{"q": "a"}]),
            streaming=True,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=1,
        )
        with pytest.raises(NotImplementedError, match="unknown length"):
            len(ds)

    def test_len_streaming_with_limit_uses_limit(self):
        ds, _ = _make_ds(
            _FakeStreamingDataset([{"q": "a"}] * 100),
            streaming=True,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=4,
            limit=10,
        )
        # ceil(10 / 4) = 3.
        self.assertEqual(len(ds), 3)

    def test_len_non_streaming_dataset_dict(self):
        train = _FakeListDataset([{"q": str(i)} for i in range(5)])
        test = _FakeListDataset([{"q": str(i)} for i in range(2)])
        ds_dict = _FakeDatasetDict()
        ds_dict["train"] = train
        ds_dict["test"] = test
        ds, _ = _make_ds(
            ds_dict,
            split=None,
            streaming=False,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=3,
        )
        # 5 + 2 = 7 rows, batch_size 3 → ceil(7/3) = 3 batches.
        self.assertEqual(len(ds), 3)

    def test_len_non_streaming_single_split(self):
        rows = _FakeListDataset([{"q": str(i)} for i in range(4)])
        ds, _ = _make_ds(
            rows,
            split="train",
            streaming=False,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=2,
            repeat=2,
        )
        # 4 rows × 2 repeats = 8, batch_size 2 → 4 batches.
        self.assertEqual(len(ds), 4)

    def test_inputs_and_targets_with_tojson_filter(self):
        rows = [{"q": 'why "X"?', "a": "1+1=2"}, {"q": "ok", "a": "fine"}]
        ds, _ = _make_ds(
            _FakeListDataset(rows),
            split="train",
            streaming=False,
            input_data_model=Question,
            # ``tojson`` escapes the embedded quotes safely.
            input_template='{"question": {{ q | tojson }}}',
            output_data_model=Answer,
            output_template='{"answer": {{ a | tojson }}}',
            batch_size=2,
        )
        x, y = next(iter(ds))
        self.assertEqual(x[0].question, 'why "X"?')
        self.assertEqual(y[0].answer, "1+1=2")
