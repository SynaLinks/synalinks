# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
import shutil
import tempfile

import pytest

from synalinks.src import testing
from synalinks.src.datasets.markdown_dataset import MarkdownDataset
from synalinks.src.datasets.markdown_dataset import MarkdownDocument
from synalinks.src.datasets.markdown_dataset import MarkdownSection
from synalinks.src.datasets.markdown_dataset import parse_markdown_sections


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class ParseMarkdownSectionsTest(testing.TestCase):
    def test_single_heading_with_body(self):
        sections = parse_markdown_sections("# Title\n\nbody text\n")
        self.assertEqual(len(sections), 1)
        s = sections[0]
        self.assertEqual(s["section_name"], "Title")
        self.assertEqual(s["level"], 1)
        self.assertEqual(s["path"], "Title")
        self.assertEqual(s["text"], "body text")

    def test_nested_headings_build_breadcrumb_path(self):
        text = "# A\n\nintro\n\n## B\n\nb body\n\n### C\n\nc body\n"
        sections = parse_markdown_sections(text)
        paths = [(s["section_name"], s["level"], s["path"]) for s in sections]
        self.assertEqual(
            paths,
            [
                ("A", 1, "A"),
                ("B", 2, "A / B"),
                ("C", 3, "A / B / C"),
            ],
        )

    def test_sibling_heading_pops_to_parent(self):
        # A second h2 closes the first h2 but stays under the same h1.
        text = "# A\n\n## B1\n\nx\n\n## B2\n\ny\n"
        sections = parse_markdown_sections(text)
        self.assertEqual(
            [s["path"] for s in sections],
            ["A", "A / B1", "A / B2"],
        )

    def test_preamble_before_first_heading(self):
        text = "preamble content\n\n# Heading\n\nbody\n"
        sections = parse_markdown_sections(text)
        self.assertEqual(sections[0]["section_name"], "")
        self.assertEqual(sections[0]["level"], 0)
        self.assertEqual(sections[0]["path"], "")
        self.assertEqual(sections[0]["text"], "preamble content")
        self.assertEqual(sections[1]["section_name"], "Heading")

    def test_no_headings_yields_whole_document_as_preamble(self):
        sections = parse_markdown_sections("just\nsome\nprose\n")
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["level"], 0)
        self.assertEqual(sections[0]["section_name"], "")
        self.assertEqual(sections[0]["text"], "just\nsome\nprose")

    def test_empty_document_yields_no_sections(self):
        self.assertEqual(parse_markdown_sections(""), [])
        self.assertEqual(parse_markdown_sections("   \n\n  \n"), [])

    def test_front_matter_is_stripped(self):
        text = "---\ntitle: My Doc\ntags: [a, b]\n---\n\n# Real Heading\n\nbody\n"
        sections = parse_markdown_sections(text)
        # The YAML block must not leak into the preamble or any section.
        self.assertEqual([s["section_name"] for s in sections], ["Real Heading"])
        self.assertNotIn("title: My Doc", sections[0]["text"])

    def test_hash_inside_code_fence_is_not_a_heading(self):
        text = "# Real\n\n```python\n# this is a comment, not a heading\nx = 1\n```\n"
        sections = parse_markdown_sections(text)
        # Only the real ATX heading is detected.
        self.assertEqual([s["section_name"] for s in sections], ["Real"])
        self.assertIn("# this is a comment", sections[0]["text"])

    def test_setext_heading_is_detected(self):
        # `===` underline is an h1 in CommonMark.
        text = "Setext Title\n===\n\nbody\n"
        sections = parse_markdown_sections(text)
        self.assertEqual(sections[0]["section_name"], "Setext Title")
        self.assertEqual(sections[0]["level"], 1)
        self.assertEqual(sections[0]["text"], "body")

    def test_deeper_then_shallower_heading_resets_stack(self):
        # h1 -> h3 -> h2: the h2 should pop back past the h3.
        text = "# A\n\n### Deep\n\nx\n\n## Shallow\n\ny\n"
        sections = parse_markdown_sections(text)
        self.assertEqual(
            [s["path"] for s in sections],
            ["A", "A / Deep", "A / Shallow"],
        )


class MarkdownDatasetTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def test_reads_file_as_markdown_document(self):
        _write(os.path.join(self.tmp, "doc.md"), "# Title\n\nbody\n\n## Sub\n\nmore\n")
        ds = MarkdownDataset(root=self.tmp, batch_size=4)
        (x,) = next(iter(ds))
        doc = x[0]
        self.assertIsInstance(doc, MarkdownDocument)
        self.assertEqual(doc.filepath, "doc.md")
        self.assertEqual(doc.title, "Title")  # first h1
        self.assertEqual([s.section_name for s in doc.sections], ["Title", "Sub"])
        self.assertIsInstance(doc.sections[0], MarkdownSection)

    def test_section_id_format(self):
        _write(os.path.join(self.tmp, "doc.md"), "# A\n\n## B\n\nx\n")
        ds = MarkdownDataset(root=self.tmp, batch_size=4)
        (x,) = next(iter(ds))
        ids = [s.section_id for s in x[0].sections]
        # "<relpath>#<breadcrumb-path>"
        self.assertEqual(ids, ["doc.md#A", "doc.md#A / B"])

    def test_title_falls_back_to_basename_without_h1(self):
        _write(os.path.join(self.tmp, "no-h1.md"), "## Only h2\n\nbody\n")
        ds = MarkdownDataset(root=self.tmp, batch_size=4)
        (x,) = next(iter(ds))
        self.assertEqual(x[0].title, "no-h1")

    def test_filenames_sorted_within_directory(self):
        _write(os.path.join(self.tmp, "b.md"), "# B\n")
        _write(os.path.join(self.tmp, "a.md"), "# A\n")
        ds = MarkdownDataset(root=self.tmp, batch_size=10)
        (x,) = next(iter(ds))
        self.assertEqual([doc.filepath for doc in x], ["a.md", "b.md"])

    def test_only_md_suffix_read_case_insensitive(self):
        _write(os.path.join(self.tmp, "keep.md"), "# K\n")
        _write(os.path.join(self.tmp, "upper.MD"), "# U\n")
        _write(os.path.join(self.tmp, "skip.txt"), "nope")
        ds = MarkdownDataset(root=self.tmp, batch_size=10)
        (x,) = next(iter(ds))
        self.assertEqual(
            sorted(doc.filepath for doc in x), ["keep.md", "upper.MD"]
        )

    def test_non_recursive(self):
        _write(os.path.join(self.tmp, "top.md"), "# T\n")
        _write(os.path.join(self.tmp, "sub", "nested.md"), "# N\n")
        ds = MarkdownDataset(root=self.tmp, recursive=False, batch_size=10)
        (x,) = next(iter(ds))
        self.assertEqual([doc.filepath for doc in x], ["top.md"])

    def test_batching_and_trailing_partial(self):
        for i in range(5):
            _write(os.path.join(self.tmp, f"{i}.md"), f"# H{i}\n")
        ds = MarkdownDataset(root=self.tmp, batch_size=2)
        sizes = [len(b[0]) for b in ds]
        self.assertEqual(sizes, [2, 2, 1])

    def test_limit_and_len(self):
        for i in range(10):
            _write(os.path.join(self.tmp, f"{i:02d}.md"), f"# H{i}\n")
        ds = MarkdownDataset(root=self.tmp, batch_size=4, limit=10)
        self.assertEqual(len(ds), 3)  # ceil(10 / 4)
        self.assertEqual(sum(len(b[0]) for b in ds), 10)

    def test_len_requires_limit(self):
        _write(os.path.join(self.tmp, "a.md"), "# A\n")
        ds = MarkdownDataset(root=self.tmp, batch_size=1)
        with pytest.raises(NotImplementedError, match="unknown length"):
            len(ds)

    def test_missing_root_raises(self):
        with pytest.raises(FileNotFoundError, match="Corpus root not found"):
            MarkdownDataset(root="/nonexistent/docs")

    def test_materialize(self):
        _write(os.path.join(self.tmp, "a.md"), "# A\n\nbody\n")
        ds = MarkdownDataset(root=self.tmp, batch_size=1)
        (x,) = ds.materialize()
        self.assertEqual(len(x), 1)
        self.assertEqual(x[0].title, "A")

    def test_empty_directory_yields_no_rows(self):
        ds = MarkdownDataset(root=self.tmp, batch_size=4)
        self.assertEqual([b for b in ds], [])
