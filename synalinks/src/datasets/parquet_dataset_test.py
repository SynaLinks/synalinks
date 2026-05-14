# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pa_parquet
import pytest

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.datasets.parquet_dataset import ParquetDataset


class Question(DataModel):
    question: str


class Answer(DataModel):
    answer: str


def _write_parquet(path, columns):
    """Write a list-of-columns dict to a Parquet file.

    Using pyarrow directly (rather than going through pandas) keeps
    the test fixture independent of the pandas dep.
    """
    table = pa.table(columns)
    pa_parquet.write_table(table, path)


class ParquetDatasetTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def test_reads_single_file(self):
        path = os.path.join(self.tmp, "qa.parquet")
        _write_parquet(
            path,
            {
                "question": ["1+1?", "2+2?"],
                "answer": ["2", "4"],
            },
        )
        ds = ParquetDataset(
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
        a = os.path.join(self.tmp, "a.parquet")
        b = os.path.join(self.tmp, "b.parquet")
        _write_parquet(a, {"q": ["a1", "a2"]})
        _write_parquet(b, {"q": ["b1"]})
        ds = ParquetDataset(
            path=[a, b],
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=10,
        )
        (x,) = next(iter(ds))
        self.assertEqual([item.question for item in x], ["a1", "a2", "b1"])

    def test_columns_pushdown_avoids_unused_columns(self):
        # Parquet's column-store means that asking for a subset of
        # columns means the rest is never read from disk. Functionally
        # this is the same observable contract as the CSV loader:
        # only the requested keys reach the template.
        path = os.path.join(self.tmp, "wide.parquet")
        _write_parquet(
            path,
            {
                "id": ["1", "2"],
                "question": ["qa", "qb"],
                "answer": ["ans", "ans2"],
                "extra": ["junk", "junk2"],
            },
        )
        ds = ParquetDataset(
            path=path,
            columns=["question"],
            input_data_model=Question,
            input_template='{"question": {{ question | tojson }}}',
            batch_size=2,
        )
        (x,) = next(iter(ds))
        self.assertEqual([item.question for item in x], ["qa", "qb"])

    def test_limit_caps_total_rows(self):
        path = os.path.join(self.tmp, "many.parquet")
        _write_parquet(path, {"q": [f"q{i}" for i in range(20)]})
        ds = ParquetDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=3,
            limit=7,
        )
        batches = list(ds)
        sizes = [len(b[0]) for b in batches]
        self.assertEqual(sizes, [3, 3, 1])
        self.assertEqual(sum(sizes), 7)

    def test_repeat_expands_each_row(self):
        path = os.path.join(self.tmp, "rep.parquet")
        _write_parquet(path, {"q": ["alpha", "beta"]})
        ds = ParquetDataset(
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

    def test_trailing_partial_batch_is_flushed(self):
        path = os.path.join(self.tmp, "odd.parquet")
        _write_parquet(path, {"q": [f"q{i}" for i in range(5)]})
        ds = ParquetDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=2,
        )
        sizes = [len(b[0]) for b in ds]
        self.assertEqual(sizes, [2, 2, 1])

    def test_inputs_only_no_output_template(self):
        path = os.path.join(self.tmp, "io.parquet")
        _write_parquet(path, {"q": ["only-input"]})
        ds = ParquetDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=1,
        )
        batch = next(iter(ds))
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0][0].question, "only-input")

    def test_len_from_metadata_without_limit(self):
        # Unlike CSV, Parquet records the row count in the footer
        # metadata, so __len__ works *without* a limit. This is the
        # main user-visible benefit of Parquet over CSV here.
        path = os.path.join(self.tmp, "len.parquet")
        _write_parquet(path, {"q": ["a"] * 50})
        ds = ParquetDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=4,
        )
        # 50 rows / 4 = ceil → 13 batches.
        self.assertEqual(len(ds), 13)

    def test_len_with_limit_overrides_metadata(self):
        path = os.path.join(self.tmp, "len.parquet")
        _write_parquet(path, {"q": ["a"] * 50})
        ds = ParquetDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=4,
            limit=10,
        )
        # Limit caps the source — 10 rows / 4 = ceil → 3.
        self.assertEqual(len(ds), 3)

    def test_len_sums_multiple_files(self):
        a = os.path.join(self.tmp, "a.parquet")
        b = os.path.join(self.tmp, "b.parquet")
        _write_parquet(a, {"q": ["x"] * 7})
        _write_parquet(b, {"q": ["y"] * 5})
        ds = ParquetDataset(
            path=[a, b],
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=4,
        )
        # 12 rows / 4 = 3 batches.
        self.assertEqual(len(ds), 3)

    def test_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            ParquetDataset(
                path="/nonexistent/path/x.parquet",
                input_data_model=Question,
                input_template='{"question": {{ q | tojson }}}',
            )

    def test_raises_on_empty_path_list(self):
        with pytest.raises(ValueError, match="at least one Parquet file"):
            ParquetDataset(
                path=[],
                input_data_model=Question,
                input_template='{"question": {{ q | tojson }}}',
            )

    def test_iter_is_repeatable(self):
        path = os.path.join(self.tmp, "rep.parquet")
        _write_parquet(path, {"q": ["x", "y"]})
        ds = ParquetDataset(
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
        path = os.path.join(self.tmp, "mat.parquet")
        _write_parquet(
            path,
            {"q": ["a", "b"], "a": ["1", "2"]},
        )
        ds = ParquetDataset(
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

    def test_batch_size_rows_controls_reader_buffer(self):
        # `batch_size_rows` controls the pyarrow reader's chunk size,
        # not the yielded batch size. The dataset still emits batches
        # of `batch_size` rows — but internally it consumes the
        # underlying record-batch reader in chunks of
        # `batch_size_rows`. Verify both knobs work independently.
        path = os.path.join(self.tmp, "buf.parquet")
        _write_parquet(path, {"q": [f"q{i}" for i in range(10)]})
        ds = ParquetDataset(
            path=path,
            batch_size_rows=3,  # internal pyarrow chunk size
            batch_size=4,  # dataset-level yielded batch size
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
        )
        sizes = [len(b[0]) for b in ds]
        # 10 rows, batch_size 4 → [4, 4, 2] regardless of internal chunking.
        self.assertEqual(sizes, [4, 4, 2])

    # ---- Partial / malformed file behaviour ----
    #
    # Parquet files come from real producers that crash, get
    # truncated, or write inconsistent schemas across runs. The
    # failure modes are very different from CSV — Parquet's footer
    # holds the schema and row count, so a truncated file is
    # detectable at open time, and per-row malformation isn't
    # really possible. We still want predictable behaviour on the
    # cases that DO occur in practice: zero-row files, schema
    # drift across a multi-file batch, and corrupt files.

    def test_empty_file_with_schema_yields_no_rows(self):
        # A 0-row parquet file (valid header + footer, no row
        # groups). This is what `pq.write_table` of an empty table
        # produces — a perfectly legal Parquet artifact. The
        # loader must iterate cleanly and report __len__ as zero.
        path = os.path.join(self.tmp, "empty.parquet")
        empty = pa.table({"q": pa.array([], type=pa.string())})
        pa_parquet.write_table(empty, path)

        ds = ParquetDataset(
            path=path,
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=4,
        )
        self.assertEqual([b for b in ds], [])
        self.assertEqual(len(ds), 0)

    def test_multi_file_with_extra_column_in_some_files(self):
        # Real-world case: file A has columns {q, a}, file B was
        # written by a later pipeline version with {q, a, score}.
        # When we don't restrict ``columns``, each file is iterated
        # with its own schema and the row dicts carry whatever
        # columns that file has — the template only references
        # what it needs, so the surplus is ignored.
        a = os.path.join(self.tmp, "old.parquet")
        b = os.path.join(self.tmp, "new.parquet")
        _write_parquet(a, {"q": ["a1"], "a": ["x"]})
        _write_parquet(b, {"q": ["b1"], "a": ["y"], "score": [0.9]})
        ds = ParquetDataset(
            path=[a, b],
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            output_data_model=Answer,
            output_template='{"answer": {{ a | tojson }}}',
            batch_size=10,
        )
        x, y = next(iter(ds))
        self.assertEqual([item.question for item in x], ["a1", "b1"])
        self.assertEqual([item.answer for item in y], ["x", "y"])

    def test_multi_file_missing_column_raises_clearly(self):
        # If one file in a multi-file batch lacks a column the
        # template requires, the failure must surface — silently
        # filling None would corrupt downstream training data. We
        # leave the raise to pyarrow when ``columns=`` includes a
        # missing one, and to Jinja's StrictUndefined otherwise.
        a = os.path.join(self.tmp, "has.parquet")
        b = os.path.join(self.tmp, "missing.parquet")
        _write_parquet(a, {"q": ["a1"], "a": ["x"]})
        _write_parquet(b, {"q": ["b1"]})  # missing `a`

        # Path 1: explicit columns including `a` — pyarrow raises
        # when reading the second file.
        ds_pushed = ParquetDataset(
            path=[a, b],
            columns=["q", "a"],
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            output_data_model=Answer,
            output_template='{"answer": {{ a | tojson }}}',
            batch_size=10,
        )
        with pytest.raises(Exception):
            [batch for batch in ds_pushed]

        # Path 2: no columns pushdown — Jinja's StrictUndefined
        # raises on the second file's row because `a` isn't there.
        ds_loose = ParquetDataset(
            path=[a, b],
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            output_data_model=Answer,
            output_template='{"answer": {{ a | tojson }}}',
            batch_size=10,
        )
        with pytest.raises(Exception):
            [batch for batch in ds_loose]

    def test_corrupt_file_raises_at_construction(self):
        # A file with garbage bytes can't be opened as Parquet —
        # ParquetFile validates the magic-number footer at
        # construction. The error must surface immediately so the
        # caller fails fast rather than after iteration starts.
        path = os.path.join(self.tmp, "garbage.parquet")
        with open(path, "wb") as f:
            f.write(b"this is not a parquet file at all")
        with pytest.raises(Exception):
            ParquetDataset(
                path=path,
                input_data_model=Question,
                input_template='{"question": {{ q | tojson }}}',
            )

    def test_truncated_file_raises_at_construction(self):
        # Real "half-completed" failure: writer crashed mid-flush
        # so the footer is missing. ParquetFile reads the footer
        # at construction; a truncated file can't be opened.
        path = os.path.join(self.tmp, "trunc.parquet")
        good = os.path.join(self.tmp, "good.parquet")
        _write_parquet(good, {"q": ["a", "b", "c"]})
        with open(good, "rb") as f:
            content = f.read()
        # Drop the trailing footer / magic bytes — anything past the
        # first 32 bytes is fine for tearing the footer off.
        with open(path, "wb") as f:
            f.write(content[: max(32, len(content) // 4)])
        with pytest.raises(Exception):
            ParquetDataset(
                path=path,
                input_data_model=Question,
                input_template='{"question": {{ q | tojson }}}',
            )

    def test_multi_file_empty_in_the_middle_is_concatenated_cleanly(self):
        # Multi-file batch where one of the files is the empty-but-
        # valid case: iteration must continue past it without
        # losing the other files' rows.
        a = os.path.join(self.tmp, "a.parquet")
        empty = os.path.join(self.tmp, "empty.parquet")
        b = os.path.join(self.tmp, "b.parquet")
        _write_parquet(a, {"q": ["a1"]})
        pa_parquet.write_table(
            pa.table({"q": pa.array([], type=pa.string())}), empty
        )
        _write_parquet(b, {"q": ["b1"]})

        ds = ParquetDataset(
            path=[a, empty, b],
            input_data_model=Question,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=10,
        )
        (x,) = next(iter(ds))
        self.assertEqual([item.question for item in x], ["a1", "b1"])
        # len from metadata: 1 + 0 + 1 = 2, batch_size 10 → 1 batch.
        self.assertEqual(len(ds), 1)

    def test_input_schema_path_does_not_require_data_model(self):
        path = os.path.join(self.tmp, "sch.parquet")
        _write_parquet(path, {"q": ["a"]})
        schema = json.dumps(
            {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            }
        )
        ds = ParquetDataset(
            path=path,
            input_schema=schema,
            input_template='{"question": {{ q | tojson }}}',
            batch_size=1,
        )
        (x,) = next(iter(ds))
        self.assertEqual(x[0].get_json()["question"], "a")
