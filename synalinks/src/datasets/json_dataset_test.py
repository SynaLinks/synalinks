# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
import os
import tempfile

import pytest

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.datasets.json_dataset import JSONDataset
from synalinks.src.datasets.json_dataset import JSONLDataset


class Question(DataModel):
    question: str


class Answer(DataModel):
    answer: str


def _write_json(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


class JSONDatasetTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def test_reads_single_file(self):
        path = os.path.join(self.tmp, "qa.json")
        _write_json(
            path,
            [
                {"question": "1+1?", "answer": "2"},
                {"question": "2+2?", "answer": "4"},
            ],
        )
        ds = JSONDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ question | tojson }}}',
            output_data_model=Answer,
            output_template='{"answer": {{ answer | tojson }}}',
            batch_size=2,
        )
        x, y = next(iter(ds))
        self.assertEqual([item.question for item in x], ["1+1?", "2+2?"])
        self.assertEqual([item.answer for item in y], ["2", "4"])

    def test_reads_multiple_files_in_order(self):
        a = os.path.join(self.tmp, "a.json")
        b = os.path.join(self.tmp, "b.json")
        _write_json(a, [{"q": "a1"}, {"q": "a2"}])
        _write_json(b, [{"q": "b1"}])
        ds = JSONDataset(
            path=[a, b],
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=10,
        )
        (x,) = next(iter(ds))
        self.assertEqual([item.question for item in x], ["a1", "a2", "b1"])

    def test_columns_filter_drops_unrequested_keys(self):
        # Directly inspect the row dicts the loader yields. Previously
        # this test only asserted the resulting DataModel field — which
        # would have passed even if no projection happened, since the
        # template ignored the unrequested keys.
        path = os.path.join(self.tmp, "wide.json")
        _write_json(
            path,
            [
                {"id": "1", "question": "qa", "answer": "ans", "extra": "junk"},
                {"id": "2", "question": "qb", "answer": "ans2", "extra": "junk2"},
            ],
        )
        ds = JSONDataset(
            path=path,
            columns=["question"],
            input_data_model=Question,
            input_template='{"question": {{ question | tojson }}}',
            batch_size=2,
        )
        rows = list(ds._iter_rows())
        self.assertEqual(rows, [{"question": "qa"}, {"question": "qb"}])

    def test_limit_caps_total_rows(self):
        path = os.path.join(self.tmp, "many.json")
        _write_json(path, [{"q": f"q{i}"} for i in range(20)])
        ds = JSONDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=3,
            limit=7,
        )
        sizes = [len(b[0]) for b in ds]
        self.assertEqual(sizes, [3, 3, 1])

    def test_repeat_expands_each_row(self):
        path = os.path.join(self.tmp, "rep.json")
        _write_json(path, [{"q": "alpha"}, {"q": "beta"}])
        ds = JSONDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=4,
            repeat=2,
        )
        (x,) = next(iter(ds))
        self.assertEqual(
            [item.question for item in x],
            ["alpha", "alpha", "beta", "beta"],
        )

    def test_inputs_only_no_output_template(self):
        path = os.path.join(self.tmp, "io.json")
        _write_json(path, [{"q": "only-input"}])
        ds = JSONDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=1,
        )
        batch = next(iter(ds))
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0][0].question, "only-input")

    def test_len_requires_limit(self):
        path = os.path.join(self.tmp, "len.json")
        _write_json(path, [{"q": "a"}])
        ds = JSONDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=1,
        )
        with pytest.raises(NotImplementedError, match="unknown length"):
            len(ds)

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            JSONDataset(
                path="/nonexistent/path/x.json",
                input_data_model=Question,
                input_template='{"question": {{ q | tojson }}}',
            )

    def test_raises_on_empty_path_list(self):
        with pytest.raises(ValueError, match="at least one JSON file"):
            JSONDataset(
                path=[],
                input_data_model=Question,
                input_template='{"question": {{ q | tojson }}}',
            )

    def test_materialize_returns_numpy_arrays(self):
        path = os.path.join(self.tmp, "mat.json")
        _write_json(
            path,
            [
                {"q": "a", "a": "1"},
                {"q": "b", "a": "2"},
            ],
        )
        ds = JSONDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            output_data_model=Answer,
            output_template='{"answer": {{ a | tojson }}}',
            batch_size=1,
        )
        x, y = ds.materialize()
        self.assertEqual(len(x), 2)
        self.assertEqual(x[0].question, "a")
        self.assertEqual(y[1].answer, "2")

    # ---- Partial / malformed file behaviour ----

    def test_empty_array_yields_no_rows(self):
        # A file that's just `[]` (e.g. the producer wrote the array
        # delimiters but no rows). Loader returns no rows, no
        # exception.
        path = os.path.join(self.tmp, "empty.json")
        _write_json(path, [])
        ds = JSONDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=4,
        )
        self.assertEqual([b for b in ds], [])

    def test_top_level_object_instead_of_array_raises(self):
        # A common user mistake: writing a single object instead of
        # an array. Surface as a clear ValueError pointing at JSONL.
        path = os.path.join(self.tmp, "obj.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"q": "single object"}')
        ds = JSONDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=1,
        )
        with pytest.raises(ValueError, match="JSONLDataset"):
            [b for b in ds]

    def test_non_object_array_element_raises(self):
        path = os.path.join(self.tmp, "mixed.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write('[{"q": "good"}, "not-an-object"]')
        ds = JSONDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=2,
        )
        with pytest.raises(ValueError, match="must be an object"):
            [b for b in ds]

    def test_truncated_file_raises_json_decode_error(self):
        # Producer crashed mid-write — the file ends mid-token. json.load
        # surfaces the standard decode error so the user can find the
        # offending byte offset.
        path = os.path.join(self.tmp, "trunc.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write('[{"q": "good"}, {"q":')  # truncated
        ds = JSONDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=2,
        )
        with pytest.raises(json.JSONDecodeError):
            [b for b in ds]


class JSONLDatasetTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def test_reads_single_file(self):
        path = os.path.join(self.tmp, "qa.jsonl")
        _write_jsonl(
            path,
            [
                {"question": "1+1?", "answer": "2"},
                {"question": "2+2?", "answer": "4"},
            ],
        )
        ds = JSONLDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ question | tojson }}}',
            output_data_model=Answer,
            output_template='{"answer": {{ answer | tojson }}}',
            batch_size=2,
        )
        x, y = next(iter(ds))
        self.assertEqual([item.question for item in x], ["1+1?", "2+2?"])
        self.assertEqual([item.answer for item in y], ["2", "4"])

    def test_reads_multiple_files_in_order(self):
        a = os.path.join(self.tmp, "a.jsonl")
        b = os.path.join(self.tmp, "b.jsonl")
        _write_jsonl(a, [{"q": "a1"}, {"q": "a2"}])
        _write_jsonl(b, [{"q": "b1"}])
        ds = JSONLDataset(
            path=[a, b],
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=10,
        )
        (x,) = next(iter(ds))
        self.assertEqual([item.question for item in x], ["a1", "a2", "b1"])

    def test_blank_lines_skipped(self):
        # Trailing newlines and split-then-rejoined chunks leave blank
        # lines; JSONL parsers conventionally tolerate them. Verify
        # nothing's silently treated as a JSON-null or empty-dict.
        path = os.path.join(self.tmp, "blanks.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"q": "alpha"}\n')
            f.write("\n")
            f.write('{"q": "beta"}\n')
            f.write("   \n")  # whitespace-only
            f.write('{"q": "gamma"}\n')
        ds = JSONLDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=10,
        )
        (x,) = next(iter(ds))
        self.assertEqual(
            [item.question for item in x], ["alpha", "beta", "gamma"]
        )

    def test_columns_filter_drops_unrequested_keys(self):
        # Directly inspect the projection — see the JSONDataset
        # equivalent for why the original "round-trip via template"
        # form was too weak to catch a missing filter.
        path = os.path.join(self.tmp, "wide.jsonl")
        _write_jsonl(
            path,
            [
                {"id": "1", "q": "qa", "extra": "junk"},
                {"id": "2", "q": "qb", "extra": "junk2"},
            ],
        )
        ds = JSONLDataset(
            path=path,
            columns=["q"],
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=2,
        )
        rows = list(ds._iter_rows())
        self.assertEqual(rows, [{"q": "qa"}, {"q": "qb"}])

    def test_limit_caps_total_rows(self):
        path = os.path.join(self.tmp, "many.jsonl")
        _write_jsonl(path, [{"q": f"q{i}"} for i in range(20)])
        ds = JSONLDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=3,
            limit=7,
        )
        sizes = [len(b[0]) for b in ds]
        self.assertEqual(sizes, [3, 3, 1])

    def test_malformed_line_raises_json_decode_error(self):
        # One bad line in the middle. The standard JSONDecodeError
        # propagates so callers can find the offending row.
        path = os.path.join(self.tmp, "bad.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"q": "good"}\n')
            f.write("not-json\n")
            f.write('{"q": "after-bad"}\n')
        ds = JSONLDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=10,
        )
        with pytest.raises(json.JSONDecodeError):
            [b for b in ds]

    def test_non_object_line_raises(self):
        path = os.path.join(self.tmp, "scalar.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write('{"q": "good"}\n')
            f.write('"a string at the top level"\n')
        ds = JSONLDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=2,
        )
        with pytest.raises(ValueError, match="must decode to a JSON object"):
            [b for b in ds]

    def test_empty_file_yields_no_rows(self):
        path = os.path.join(self.tmp, "empty.jsonl")
        with open(path, "w", encoding="utf-8"):
            pass
        ds = JSONLDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=4,
        )
        self.assertEqual([b for b in ds], [])

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            JSONLDataset(
                path="/nonexistent/path/x.jsonl",
                input_data_model=Question,
                input_template='{"question": {{ q | tojson }}}',
            )

    def test_len_requires_limit(self):
        path = os.path.join(self.tmp, "len.jsonl")
        _write_jsonl(path, [{"q": "a"}])
        ds = JSONLDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=1,
        )
        with pytest.raises(NotImplementedError, match="unknown length"):
            len(ds)
