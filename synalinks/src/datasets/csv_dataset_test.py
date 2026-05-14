# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import csv as _csv
import json
import os
import tempfile

import pytest

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.datasets.csv_dataset import CSVDataset


class Question(DataModel):
    question: str


class Answer(DataModel):
    answer: str


def _write_csv(path, rows, header=True, delimiter=","):
    """Write a list of dicts as a CSV without depending on pandas/pyarrow.

    Tests use this to lay down small fixtures on disk; the production
    code reads them back via pyarrow.csv.
    """
    cols = list(rows[0].keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        if header:
            f.write(delimiter.join(cols) + "\n")
        for row in rows:
            f.write(delimiter.join(str(row[c]) for c in cols) + "\n")


class CSVDatasetTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def test_reads_single_file_with_header(self):
        path = os.path.join(self.tmp, "qa.csv")
        _write_csv(
            path,
            [
                {"question": "1+1?", "answer": "2"},
                {"question": "2+2?", "answer": "4"},
            ],
        )
        ds = CSVDataset(
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
        # Multi-path: each file is consumed in the order given. Order
        # matters because rows are not shuffled — callers slicing or
        # splitting downstream rely on it being deterministic.
        a = os.path.join(self.tmp, "a.csv")
        b = os.path.join(self.tmp, "b.csv")
        _write_csv(a, [{"q": "a1"}, {"q": "a2"}])
        _write_csv(b, [{"q": "b1"}])
        ds = CSVDataset(
            path=[a, b],
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=10,
        )
        (x,) = next(iter(ds))
        self.assertEqual([item.question for item in x], ["a1", "a2", "b1"])

    def test_custom_delimiter(self):
        path = os.path.join(self.tmp, "tsv.csv")
        _write_csv(
            path,
            [{"q": "tab-sep", "a": "yes"}],
            delimiter="\t",
        )
        ds = CSVDataset(
            path=path,
            delimiter="\t",
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            output_data_model=Answer,
            output_template='{"answer": {{ a | tojson }}}',
            batch_size=1,
        )
        x, y = next(iter(ds))
        self.assertEqual(x[0].question, "tab-sep")
        self.assertEqual(y[0].answer, "yes")

    def test_headerless_with_explicit_column_names(self):
        # No header row in the file; column_names tells pyarrow what
        # to call the columns. Used when the source CSV doesn't have
        # a header, or to alias non-identifier column names into
        # identifier-shaped ones that Jinja can reference.
        path = os.path.join(self.tmp, "noheader.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("1,foo\n2,bar\n")
        ds = CSVDataset(
            path=path,
            column_names=["id", "q"],
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=2,
        )
        (x,) = next(iter(ds))
        self.assertEqual([item.question for item in x], ["foo", "bar"])

    def test_columns_pushdown_restricts_returned_fields(self):
        # ``columns`` filters at the pyarrow layer — the omitted columns
        # never reach the row dict, so a template referencing one would
        # raise. We exercise this by including only the column the
        # template uses.
        path = os.path.join(self.tmp, "wide.csv")
        _write_csv(
            path,
            [
                {"id": "1", "question": "qa", "answer": "ans", "extra": "junk"},
                {"id": "2", "question": "qb", "answer": "ans2", "extra": "junk2"},
            ],
        )
        ds = CSVDataset(
            path=path,
            columns=["question"],
            input_data_model=Question,
            input_template='{"question": {{ question | tojson }}}',
            batch_size=2,
        )
        (x,) = next(iter(ds))
        self.assertEqual([item.question for item in x], ["qa", "qb"])

    def test_limit_caps_total_rows(self):
        path = os.path.join(self.tmp, "many.csv")
        _write_csv(path, [{"q": f"q{i}"} for i in range(20)])
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=3,
            limit=7,
        )
        # 7 rows, batch_size 3 → ceil(7/3) = 3 batches: [3, 3, 1].
        batches = list(ds)
        sizes = [len(b[0]) for b in batches]
        self.assertEqual(sizes, [3, 3, 1])
        self.assertEqual(sum(sizes), 7)

    def test_repeat_expands_each_row(self):
        path = os.path.join(self.tmp, "rep.csv")
        _write_csv(path, [{"q": "alpha"}, {"q": "beta"}])
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=4,
            repeat=2,
        )
        (x,) = next(iter(ds))
        # Each raw row appears `repeat` times consecutively.
        self.assertEqual(
            [item.question for item in x],
            ["alpha", "alpha", "beta", "beta"],
        )

    def test_trailing_partial_batch_is_flushed(self):
        path = os.path.join(self.tmp, "odd.csv")
        _write_csv(path, [{"q": f"q{i}"} for i in range(5)])
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=2,
        )
        sizes = [len(b[0]) for b in ds]
        # 5 rows, batch_size 2 → [2, 2, 1] — partial batch isn't dropped.
        self.assertEqual(sizes, [2, 2, 1])

    def test_inputs_only_no_output_template(self):
        path = os.path.join(self.tmp, "io.csv")
        _write_csv(path, [{"q": "only-input"}])
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=1,
        )
        batch = next(iter(ds))
        # Inputs-only datasets yield (x,) — a one-tuple.
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0][0].question, "only-input")

    def test_len_requires_limit(self):
        path = os.path.join(self.tmp, "len.csv")
        _write_csv(path, [{"q": "a"}])
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=1,
        )
        with pytest.raises(NotImplementedError, match="unknown length"):
            len(ds)

    def test_len_with_limit(self):
        path = os.path.join(self.tmp, "len.csv")
        _write_csv(path, [{"q": "a"}] * 50)
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=4,
            limit=10,
        )
        # ceil(10 / 4) = 3.
        self.assertEqual(len(ds), 3)

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            CSVDataset(
                path="/nonexistent/path/x.csv",
                input_data_model=Question,
                input_template='{"question": {{ q | tojson }}}',
            )

    def test_raises_on_empty_path_list(self):
        with pytest.raises(ValueError, match="at least one CSV file"):
            CSVDataset(
                path=[],
                input_data_model=Question,
                input_template='{"question": {{ q | tojson }}}',
            )

    def test_iter_is_repeatable(self):
        # Each call to ``ds()`` should produce a fresh generator from
        # the top of the source file. Without this, callers that
        # evaluate then refit (or run a second epoch via materialize)
        # would silently see an empty second pass.
        path = os.path.join(self.tmp, "rep.csv")
        _write_csv(path, [{"q": "x"}, {"q": "y"}])
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=2,
        )
        first = [b[0].tolist() for b in ds()]
        second = [b[0].tolist() for b in ds()]
        self.assertEqual(
            [[x.question for x in b] for b in first],
            [[x.question for x in b] for b in second],
        )

    def test_materialize_returns_numpy_arrays(self):
        path = os.path.join(self.tmp, "mat.csv")
        _write_csv(
            path,
            [
                {"q": "a", "a": "1"},
                {"q": "b", "a": "2"},
            ],
        )
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            output_data_model=Answer,
            output_template='{"answer": {{ a | tojson }}}',
            batch_size=1,
        )
        x, y = ds.materialize()
        self.assertEqual(len(x), 2)
        self.assertEqual(len(y), 2)
        self.assertEqual(x[0].question, "a")
        self.assertEqual(y[1].answer, "2")

    # ---- Partial / malformed file behaviour ----
    #
    # Real CSV inputs aren't always clean: a producer may have crashed
    # mid-flush, a hand-edited file may have rows with missing trailing
    # columns, or the file may have been written with embedded
    # newlines inside quoted fields. The loader should not blow up on
    # these shapes — instead the failure mode should either be
    # silent-empty (empty / header-only file) or surface as a clear
    # downstream validation error from Pydantic / Jinja, not a parser
    # crash inside csv.DictReader.

    def test_empty_file_yields_no_rows(self):
        # A zero-byte file is the classic "the producer crashed before
        # writing anything" failure. DictReader treats it as "no
        # header, no rows" — iterating returns nothing. Use iter(ds)
        # explicitly because the streaming dataset has no __len__
        # without a limit and ``list(ds)`` would call __len__.
        path = os.path.join(self.tmp, "empty.csv")
        with open(path, "w", encoding="utf-8") as f:
            pass  # touch
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=4,
        )
        self.assertEqual([b for b in ds], [])

    def test_header_only_file_yields_no_rows(self):
        # Some pipelines write the header and then fail before any
        # data rows land. Loader should report zero examples rather
        # than raise.
        path = os.path.join(self.tmp, "headeronly.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("q,a\n")
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=4,
        )
        self.assertEqual([b for b in ds], [])

    def test_truncated_last_row_yields_partial_dict(self):
        # The most common "half-completed" shape: writer crashed
        # partway through the last row. DictReader fills the missing
        # trailing columns with None, so the row reaches the template
        # with `None` for the absent field. Verify the loader doesn't
        # drop the row silently — it surfaces a Pydantic error so the
        # user knows the file is corrupt.
        path = os.path.join(self.tmp, "trunc.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("q,a\n")
            f.write("q1,ans1\n")
            f.write("q2\n")  # truncated — no `a` field
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            output_data_model=Answer,
            output_template='{"answer": {{ a | tojson }}}',
            batch_size=2,
        )
        # The good row consumes fine, but the truncated row's `None`
        # answer fails the string-typed Answer validation. The error
        # is a clear ValidationError, not a parser crash.
        with pytest.raises(Exception) as exc_info:
            [b for b in ds]
        self.assertIn("Answer", str(exc_info.value))

    def test_truncated_row_works_when_template_tolerates_missing_field(self):
        # Same truncated file, but the input-only template only
        # references the field that *is* present. The loader yields
        # the good row and the partial row without complaint —
        # consistent with "the loader doesn't crash, the template
        # decides what's required".
        path = os.path.join(self.tmp, "trunc2.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("q,a\n")
            f.write("q1,ans1\n")
            f.write("q2\n")
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=10,
        )
        (x,) = next(iter(ds))
        self.assertEqual([item.question for item in x], ["q1", "q2"])

    def test_row_with_empty_string_values_passes_through(self):
        # Quoted-empty values are a legitimate "missing-but-present"
        # encoding. DictReader returns "" — not None — so a
        # string-typed DataModel accepts them.
        path = os.path.join(self.tmp, "blanks.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("q,a\n")
            f.write('q1,""\n')
            f.write(",ans2\n")
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            output_data_model=Answer,
            output_template='{"answer": {{ a | tojson }}}',
            batch_size=2,
        )
        x, y = next(iter(ds))
        self.assertEqual([item.question for item in x], ["q1", ""])
        self.assertEqual([item.answer for item in y], ["", "ans2"])

    def test_embedded_commas_and_newlines_in_quoted_fields(self):
        # CSV's signature gotcha: a quoted field can contain the
        # delimiter or a newline. The csv module handles this; verify
        # the loader doesn't accidentally split or truncate.
        path = os.path.join(self.tmp, "tricky.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = _csv.writer(f, quoting=_csv.QUOTE_ALL)
            writer.writerow(["q", "a"])
            writer.writerow(["has, commas", "yes"])
            writer.writerow(["has\nnewline", "yes2"])
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            output_data_model=Answer,
            output_template='{"answer": {{ a | tojson }}}',
            batch_size=2,
        )
        x, y = next(iter(ds))
        self.assertEqual(x[0].question, "has, commas")
        self.assertEqual(x[1].question, "has\nnewline")
        self.assertEqual([item.answer for item in y], ["yes", "yes2"])

    def test_extra_columns_are_silently_ignored_by_template(self):
        # A row with more columns than the header is unusual but
        # legal. DictReader collects the extras under the `None` key.
        # Templates that only reference declared columns ignore the
        # extras — the loader doesn't fail just because there's
        # surplus data on a row.
        path = os.path.join(self.tmp, "extras.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("q\n")
            f.write("hello,extra1,extra2\n")
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=1,
        )
        (x,) = next(iter(ds))
        self.assertEqual(x[0].question, "hello")

    def test_blank_lines_in_middle_are_skipped_by_dictreader(self):
        # ``csv.DictReader`` skips empty lines (rows that produce ``[]``
        # from the underlying reader) — that's standard stdlib
        # behavior, not anything our loader adds. A blank line in the
        # middle of a file therefore doesn't poison the stream with a
        # bogus all-empty row. Pin that behavior down so a future
        # switch to a different reader doesn't silently change it.
        path = os.path.join(self.tmp, "blanks2.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("q\n")
            f.write("alpha\n")
            f.write("\n")  # blank line in the middle
            f.write("beta\n")
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=10,
        )
        (x,) = next(iter(ds))
        self.assertEqual([item.question for item in x], ["alpha", "beta"])

    def test_encoding_mismatch_raises_clearly(self):
        # File written as latin-1 but opened as utf-8. The user should
        # see a UnicodeDecodeError, not a silent corrupted row.
        path = os.path.join(self.tmp, "latin.csv")
        with open(path, "wb") as f:
            # `é` in latin-1 is 0xE9, which is not a valid utf-8 byte
            # sequence on its own.
            f.write(b"q\ncaf\xe9\n")
        ds = CSVDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=1,
        )
        with pytest.raises(UnicodeDecodeError):
            [b for b in ds]

        # Re-opened with the correct encoding the same bytes decode fine.
        ds2 = CSVDataset(
            path=path,
            encoding="latin-1",
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=1,
        )
        (x,) = next(iter(ds2))
        self.assertEqual(x[0].question, "café")

    def test_input_schema_path_does_not_require_data_model(self):
        # The base class allows specifying an explicit JSON schema
        # instead of a DataModel class. Verify the CSV loader respects
        # that path too (schema-driven structured outputs).
        path = os.path.join(self.tmp, "sch.csv")
        _write_csv(path, [{"q": "a"}])
        schema = json.dumps(
            {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            }
        )
        ds = CSVDataset(
            path=path,
            input_schema=schema,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=1,
        )
        (x,) = next(iter(ds))
        self.assertEqual(x[0].get_json()["question"], "a")
