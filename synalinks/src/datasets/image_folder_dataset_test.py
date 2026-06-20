# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import base64
import os
import shutil
import tempfile

from synalinks.src import testing
from synalinks.src.backend.pydantic.media import resolve_content_media
from synalinks.src.datasets.image_folder_dataset import ImageFolderDataset


def _write_bytes(path, data=b"\x89PNG\r\n\x1a\n"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


class ImageFolderDatasetTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def _content(self, x):
        """The chat content list of the first message of input `x`."""
        return x.get_json()["messages"][0]["content"]

    def test_yields_lightweight_file_ref_not_bytes(self):
        _write_bytes(os.path.join(self.tmp, "a.png"))
        ds = ImageFolderDataset(root=self.tmp, prompt="Describe.", batch_size=4)
        (x,) = next(iter(ds))
        content = self._content(x[0])
        self.assertEqual(content[0], {"type": "text", "text": "Describe."})
        # The image is a file:// reference, not an inlined data: URI.
        url = content[1]["image_url"]["url"]
        self.assertTrue(url.startswith("file://"))
        self.assertTrue(url.endswith("a.png"))
        self.assertNotIn("base64", url)

    def test_image_path_and_name_template_variables(self):
        # All four row variables must be available to a custom template.
        _write_bytes(os.path.join(self.tmp, "sub", "photo.png"))
        ds = ImageFolderDataset(
            root=self.tmp,
            input_template='{"messages":[{"role":"user","content":{{ name | tojson }}}]}',
            output_template='{"role":"assistant","content":{{ image_path | tojson }}}',
            batch_size=4,
        )
        (x, y) = next(iter(ds))
        self.assertEqual(x[0].get_json()["messages"][0]["content"], "photo")
        self.assertEqual(y[0].get_json()["content"], os.path.join("sub", "photo.png"))

    def test_only_image_extensions_matched(self):
        _write_bytes(os.path.join(self.tmp, "keep.PNG"))  # case-insensitive
        _write_bytes(os.path.join(self.tmp, "skip.txt"))
        _write_bytes(os.path.join(self.tmp, "skip.json"))
        ds = ImageFolderDataset(root=self.tmp, batch_size=10)
        (x,) = next(iter(ds))
        self.assertEqual(len(x), 1)
        self.assertTrue(self._content(x[0])[1]["image_url"]["url"].endswith("keep.PNG"))

    def test_label_is_parent_folder_then_empty_at_root(self):
        # torchvision ImageFolder convention: class = subdirectory name.
        _write_bytes(os.path.join(self.tmp, "cat", "x.jpg"))
        _write_bytes(os.path.join(self.tmp, "top.jpg"))
        ds = ImageFolderDataset(
            root=self.tmp,
            output_template='{"role":"assistant","content":{{ label | tojson }}}',
            batch_size=10,
        )
        x, y = next(iter(ds))
        labels = sorted(t.get_json()["content"] for t in y)
        self.assertEqual(labels, ["", "cat"])

    def test_non_recursive_skips_subdirs(self):
        _write_bytes(os.path.join(self.tmp, "top.png"))
        _write_bytes(os.path.join(self.tmp, "sub", "deep.png"))
        ds = ImageFolderDataset(root=self.tmp, recursive=False, batch_size=10)
        (x,) = next(iter(ds))
        self.assertEqual(len(x), 1)

    def test_missing_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            ImageFolderDataset(root=os.path.join(self.tmp, "nope"))

    def test_len_requires_limit(self):
        _write_bytes(os.path.join(self.tmp, "a.png"))
        with self.assertRaises(NotImplementedError):
            len(ImageFolderDataset(root=self.tmp, batch_size=4))
        self.assertEqual(len(ImageFolderDataset(root=self.tmp, limit=5, batch_size=2)), 3)

    async def test_per_batch_resolution_reads_the_real_file(self):
        payload = b"\x89PNG\r\n\x1a\nREALBYTES"
        _write_bytes(os.path.join(self.tmp, "a.png"), payload)
        ds = ImageFolderDataset(root=self.tmp, batch_size=4)
        (x,) = next(iter(ds))
        # Resolve this batch: the file is read from disk and inlined now.
        messages = x[0].get_json()["messages"]
        await resolve_content_media(messages)
        url = messages[0]["content"][1]["image_url"]["url"]
        expected = base64.b64encode(payload).decode("ascii")
        self.assertEqual(url, f"data:image/png;base64,{expected}")
