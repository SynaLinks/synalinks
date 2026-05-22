# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
import os
import shutil
import tempfile

import pytest

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.datasets.text_dataset import TextDataset
from synalinks.src.datasets.text_dataset import TextDocument


def _write(path, text, encoding="utf-8"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding=encoding) as f:
        f.write(text)


class TextDatasetTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def test_reads_single_file_as_text_document(self):
        _write(os.path.join(self.tmp, "a.txt"), "hello world")
        ds = TextDataset(root=self.tmp, batch_size=4)
        (x,) = next(iter(ds))
        # Default shape is TextDocument with filepath + text.
        self.assertIsInstance(x[0], TextDocument)
        self.assertEqual(x[0].filepath, "a.txt")
        self.assertEqual(x[0].text, "hello world")

    def test_filepath_is_relative_to_root(self):
        # The PK is the path relative to the corpus root, not absolute,
        # so reruns from a different cwd upsert deterministically.
        _write(os.path.join(self.tmp, "sub", "deep.txt"), "nested")
        ds = TextDataset(root=self.tmp, batch_size=4)
        (x,) = next(iter(ds))
        self.assertEqual(x[0].filepath, os.path.join("sub", "deep.txt"))

    def test_filenames_sorted_within_a_directory(self):
        for name in ["c.txt", "a.txt", "b.txt"]:
            _write(os.path.join(self.tmp, name), name)
        ds = TextDataset(root=self.tmp, batch_size=10)
        (x,) = next(iter(ds))
        self.assertEqual([item.filepath for item in x], ["a.txt", "b.txt", "c.txt"])

    def test_only_matching_suffix_is_read(self):
        _write(os.path.join(self.tmp, "keep.txt"), "yes")
        _write(os.path.join(self.tmp, "skip.md"), "no")
        _write(os.path.join(self.tmp, "skip.csv"), "no")
        ds = TextDataset(root=self.tmp, batch_size=10)
        (x,) = next(iter(ds))
        self.assertEqual([item.filepath for item in x], ["keep.txt"])

    def test_glob_pattern_is_case_insensitive(self):
        _write(os.path.join(self.tmp, "upper.TXT"), "u")
        _write(os.path.join(self.tmp, "lower.txt"), "l")
        ds = TextDataset(root=self.tmp, batch_size=10)
        (x,) = next(iter(ds))
        self.assertEqual(
            sorted(item.filepath for item in x), ["lower.txt", "upper.TXT"]
        )

    def test_custom_glob_pattern(self):
        _write(os.path.join(self.tmp, "note.text"), "body")
        _write(os.path.join(self.tmp, "note.txt"), "skip")
        ds = TextDataset(root=self.tmp, glob_pattern=".text", batch_size=10)
        (x,) = next(iter(ds))
        self.assertEqual([item.filepath for item in x], ["note.text"])

    def test_non_recursive_only_reads_direct_children(self):
        _write(os.path.join(self.tmp, "top.txt"), "top")
        _write(os.path.join(self.tmp, "sub", "nested.txt"), "nested")
        ds = TextDataset(root=self.tmp, recursive=False, batch_size=10)
        (x,) = next(iter(ds))
        self.assertEqual([item.filepath for item in x], ["top.txt"])

    def test_recursive_descends_into_subdirectories(self):
        _write(os.path.join(self.tmp, "top.txt"), "top")
        _write(os.path.join(self.tmp, "sub", "nested.txt"), "nested")
        ds = TextDataset(root=self.tmp, recursive=True, batch_size=10)
        (x,) = next(iter(ds))
        # os.walk dir order is filesystem-dependent across subdirs, so
        # assert membership rather than order here.
        self.assertEqual(
            {item.filepath for item in x},
            {"top.txt", os.path.join("sub", "nested.txt")},
        )

    def test_batching_and_trailing_partial(self):
        for i in range(5):
            _write(os.path.join(self.tmp, f"{i}.txt"), str(i))
        ds = TextDataset(root=self.tmp, batch_size=2)
        sizes = [len(b[0]) for b in ds]
        self.assertEqual(sizes, [2, 2, 1])

    def test_limit_caps_total_files(self):
        for i in range(10):
            _write(os.path.join(self.tmp, f"{i:02d}.txt"), str(i))
        ds = TextDataset(root=self.tmp, batch_size=3, limit=7)
        sizes = [len(b[0]) for b in ds]
        self.assertEqual(sizes, [3, 3, 1])
        self.assertEqual(sum(sizes), 7)

    def test_repeat_expands_each_file(self):
        _write(os.path.join(self.tmp, "a.txt"), "alpha")
        _write(os.path.join(self.tmp, "b.txt"), "beta")
        ds = TextDataset(root=self.tmp, batch_size=4, repeat=2)
        (x,) = next(iter(ds))
        self.assertEqual(
            [item.text for item in x], ["alpha", "alpha", "beta", "beta"]
        )

    def test_inputs_only_yields_one_tuple(self):
        _write(os.path.join(self.tmp, "a.txt"), "x")
        ds = TextDataset(root=self.tmp, batch_size=1)
        batch = next(iter(ds))
        self.assertEqual(len(batch), 1)

    def test_iter_is_repeatable(self):
        _write(os.path.join(self.tmp, "a.txt"), "x")
        _write(os.path.join(self.tmp, "b.txt"), "y")
        ds = TextDataset(root=self.tmp, batch_size=2)
        first = [[i.filepath for i in b[0]] for b in ds()]
        second = [[i.filepath for i in b[0]] for b in ds()]
        self.assertEqual(first, second)

    def test_materialize_returns_numpy_array(self):
        _write(os.path.join(self.tmp, "a.txt"), "one")
        _write(os.path.join(self.tmp, "b.txt"), "two")
        ds = TextDataset(root=self.tmp, batch_size=1)
        (x,) = ds.materialize()
        self.assertEqual(len(x), 2)
        self.assertEqual([item.text for item in x], ["one", "two"])

    def test_len_requires_limit(self):
        _write(os.path.join(self.tmp, "a.txt"), "x")
        ds = TextDataset(root=self.tmp, batch_size=1)
        with pytest.raises(NotImplementedError, match="unknown length"):
            len(ds)

    def test_len_with_limit(self):
        for i in range(50):
            _write(os.path.join(self.tmp, f"{i:02d}.txt"), str(i))
        ds = TextDataset(root=self.tmp, batch_size=4, limit=10)
        self.assertEqual(len(ds), 3)  # ceil(10 / 4)

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError, match="Corpus root not found"):
            TextDataset(root="/nonexistent/corpus/dir")

    def test_encoding_is_honored(self):
        _write(os.path.join(self.tmp, "fr.txt"), "café", encoding="latin-1")
        # Wrong encoding surfaces a decode error rather than corrupt text.
        ds_utf8 = TextDataset(root=self.tmp, batch_size=1)
        with pytest.raises(UnicodeDecodeError):
            [b for b in ds_utf8]
        ds_latin = TextDataset(root=self.tmp, encoding="latin-1", batch_size=1)
        (x,) = next(iter(ds_latin))
        self.assertEqual(x[0].text, "café")

    def test_empty_directory_yields_no_rows(self):
        ds = TextDataset(root=self.tmp, batch_size=4)
        self.assertEqual([b for b in ds], [])

    def test_custom_input_schema_path(self):
        # The base class accepts a raw JSON schema instead of a DataModel.
        _write(os.path.join(self.tmp, "a.txt"), "body")
        schema = json.dumps(
            {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["filepath", "text"],
            }
        )
        ds = TextDataset(root=self.tmp, input_schema=schema, batch_size=1)
        (x,) = next(iter(ds))
        self.assertEqual(x[0].get_json()["text"], "body")

    def test_custom_input_data_model_and_template(self):
        # Override the row shape: only keep the text under a renamed field.
        class Doc(DataModel):
            body: str

        _write(os.path.join(self.tmp, "a.txt"), "hi")
        ds = TextDataset(
            root=self.tmp,
            input_data_model=Doc,
            input_template='{"body": {{ text | tojson }}}',
            batch_size=1,
        )
        (x,) = next(iter(ds))
        self.assertEqual(x[0].body, "hi")
