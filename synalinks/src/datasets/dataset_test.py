# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json

import numpy as np
import pytest

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.datasets.dataset import Dataset


class Question(DataModel):
    question: str


class Answer(DataModel):
    answer: str = Field(description="Final short answer.")


class _RowDataset(Dataset):
    """Minimal subclass for testing — rows come from the constructor."""

    def __init__(self, rows, **kwargs):
        super().__init__(**kwargs)
        self._rows = rows

    def _iter_rows(self):
        for row in self._rows:
            yield row

    def __len__(self):
        return self._total_batches(len(self._rows))


class DatasetCoreTest(testing.TestCase):
    def test_inputs_only_yields_single_element_tuple(self):
        rows = [{"q": "what is 1+1?"}, {"q": "capital of France?"}]
        ds = _RowDataset(
            rows,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=2,
        )
        batches = list(ds)
        self.assertEqual(len(batches), 1)
        (x,) = batches[0]
        self.assertEqual(len(x), 2)
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.dtype, np.dtype("object"))
        self.assertEqual(x[0].question, "what is 1+1?")

    def test_inputs_and_targets_yield_pair(self):
        rows = [{"q": "1+1?", "a": "2"}, {"q": "2+2?", "a": "4"}]
        ds = _RowDataset(
            rows,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            output_data_model=Answer,
            output_template=json.dumps({"answer": "{{ a }}"}),
            batch_size=2,
        )
        batches = list(ds)
        x, y = batches[0]
        self.assertEqual([item.question for item in x], ["1+1?", "2+2?"])
        self.assertEqual([item.answer for item in y], ["2", "4"])

    def test_repeat_expands_each_row_in_place(self):
        rows = [{"q": "a"}, {"q": "b"}]
        ds = _RowDataset(
            rows,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=4,
            repeat=2,
        )
        (x,) = next(iter(ds))
        # Repeats are consecutive — needed for GRPO grouping semantics.
        self.assertEqual([item.question for item in x], ["a", "a", "b", "b"])

    def test_limit_caps_raw_rows_before_repeat(self):
        rows = [{"q": str(i)} for i in range(10)]
        ds = _RowDataset(
            rows,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=2,
            limit=3,
            repeat=2,
        )
        # 3 raw rows × 2 repeats = 6 total; with batch_size=2 → 3 batches.
        batches = list(ds)
        self.assertEqual(len(batches), 3)
        flat = [item.question for batch in batches for item in batch[0]]
        self.assertEqual(flat, ["0", "0", "1", "1", "2", "2"])

    def test_trailing_partial_batch_is_flushed(self):
        rows = [{"q": str(i)} for i in range(5)]
        ds = _RowDataset(
            rows,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=2,
        )
        sizes = [len(b[0]) for b in ds]
        self.assertEqual(sizes, [2, 2, 1])

    def test_batch_size_none_yields_single_batch(self):
        rows = [{"q": str(i)} for i in range(4)]
        ds = _RowDataset(
            rows,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=None,
        )
        batches = list(ds)
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0][0]), 4)

    def test_call_returns_fresh_generator(self):
        rows = [{"q": "x"}]
        ds = _RowDataset(
            rows,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=1,
        )
        first = list(ds())
        second = list(ds())
        # Same content on repeat iteration — generator is fresh each call.
        self.assertEqual(len(first), len(second))
        self.assertEqual(first[0][0][0].question, second[0][0][0].question)

    def test_len_via_total_batches(self):
        rows = [{"q": str(i)} for i in range(7)]
        ds = _RowDataset(
            rows,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=3,
            repeat=2,
        )
        # 7 × 2 = 14 examples, batch_size=3 → ceil(14/3) = 5 batches.
        self.assertEqual(len(ds), 5)


class DatasetSchemaTest(testing.TestCase):
    """Raw JSON Schema path (``input_schema``/``output_schema``)."""

    def test_input_schema_dict_yields_json_data_model(self):
        rows = [{"q": "hello"}]
        schema = {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        }
        ds = _RowDataset(
            rows,
            input_schema=schema,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=1,
        )
        (x,) = next(iter(ds))
        # JsonDataModel exposes .get_json() / .get_schema() — see
        # synalinks/src/backend/common/json_data_model.py.
        self.assertEqual(x[0].get_json(), {"question": "hello"})
        self.assertEqual(x[0].get_schema()["properties"]["question"]["type"], "string")

    def test_output_schema_yields_json_data_model(self):
        rows = [{"q": "hello", "a": "hi"}]
        in_schema = {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        }
        out_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        ds = _RowDataset(
            rows,
            input_schema=in_schema,
            input_template=json.dumps({"question": "{{ q }}"}),
            output_schema=out_schema,
            output_template=json.dumps({"answer": "{{ a }}"}),
            batch_size=1,
        )
        x, y = next(iter(ds))
        # Hits the schema path of _make_target.
        self.assertEqual(y[0].get_json(), {"answer": "hi"})

    def test_input_schema_str_is_parsed(self):
        rows = [{"q": "hello"}]
        schema = json.dumps(
            {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            }
        )
        ds = _RowDataset(
            rows,
            input_schema=schema,
            input_template=json.dumps({"question": "{{ q }}"}),
            batch_size=1,
        )
        (x,) = next(iter(ds))
        self.assertEqual(x[0].get_json(), {"question": "hello"})


class DatasetValidationTest(testing.TestCase):
    def test_missing_input_template_raises(self):
        with pytest.raises(ValueError, match="`input_template` is required"):
            _RowDataset([], input_data_model=Question)

    def test_input_data_model_and_schema_mutually_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            _RowDataset(
                [],
                input_data_model=Question,
                input_schema={"type": "object"},
                input_template="{}",
            )

    def test_output_data_model_and_schema_mutually_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            _RowDataset(
                [],
                input_data_model=Question,
                input_template="{}",
                output_data_model=Answer,
                output_schema={"type": "object"},
                output_template="{}",
            )

    def test_output_without_template_raises(self):
        with pytest.raises(ValueError, match="`output_data_model`"):
            _RowDataset(
                [],
                input_data_model=Question,
                input_template="{}",
                output_data_model=Answer,
            )

    def test_repeat_must_be_positive_int(self):
        with pytest.raises(ValueError, match="positive int"):
            _RowDataset(
                [],
                input_data_model=Question,
                input_template="{}",
                repeat=0,
            )

    def test_jinja_strict_undefined_raises(self):
        rows = [{"q": "hi"}]  # template references {{ missing }}
        ds = _RowDataset(
            rows,
            input_data_model=Question,
            input_template=json.dumps({"question": "{{ missing }}"}),
            batch_size=1,
        )
        with pytest.raises(Exception):
            next(iter(ds))

    def test_invalid_schema_type_raises(self):
        with pytest.raises(TypeError, match="dict or JSON string"):
            _RowDataset(
                [],
                input_schema=123,
                input_template="{}",
            )


class DatasetDefaultsTest(testing.TestCase):
    """Defaults to ChatMessages / ChatMessage when no data_model/schema given."""

    def test_default_output_is_chat_message(self):
        # ``output_template`` set, no output_data_model / output_schema →
        # ``ChatMessage`` default kicks in.
        rows = [
            {
                "role_in": "user",
                "content_in": "hi",
                "role_out": "assistant",
                "content_out": "hello",
            }
        ]
        in_template = json.dumps(
            {"messages": [{"role": "{{ role_in }}", "content": "{{ content_in }}"}]}
        )
        out_template = json.dumps(
            {"role": "{{ role_out }}", "content": "{{ content_out }}"}
        )
        ds = _RowDataset(
            rows,
            input_template=in_template,
            output_template=out_template,
            batch_size=1,
        )
        x, y = next(iter(ds))
        self.assertEqual(y[0].role, "assistant")
        self.assertEqual(y[0].content, "hello")

    def test_default_input_is_chat_messages(self):
        # No input_data_model / input_schema → defaults to ChatMessages.
        rows = [{"role": "user", "content": "hi"}]
        template = json.dumps(
            {
                "messages": [
                    {"role": "{{ role }}", "content": "{{ content }}"},
                ]
            }
        )
        ds = _RowDataset(
            rows,
            input_template=template,
            batch_size=1,
        )
        (x,) = next(iter(ds))
        self.assertEqual(x[0].messages[0].role, "user")
        self.assertEqual(x[0].messages[0].content, "hi")
