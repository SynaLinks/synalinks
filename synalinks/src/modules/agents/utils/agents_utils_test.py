# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Tests for the AGENTS.md helpers in ``agents_utils`` (agents.md standard):
root + nested (monorepo nearest-wins) discovery and the prompt rendering."""

import json
import os
import tempfile

from synalinks.src import testing
from synalinks.src.modules.agents.utils.agents_utils import discover_agents_md
from synalinks.src.modules.agents.utils.agents_utils import find_agents_md
from synalinks.src.modules.agents.utils.agents_utils import tool_message_content


class AgentsMdTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.root, ignore_errors=True)
        super().tearDown()

    def _write(self, relpath, text):
        full = os.path.join(self.root, relpath)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as fh:
            fh.write(text)

    def test_find_agents_md(self):
        self.assertIsNone(find_agents_md(self.root))
        self._write("AGENTS.md", "# root")
        self.assertIsNotNone(find_agents_md(self.root))

    def test_discover_root_and_nested_ordered(self):
        self._write("AGENTS.md", "# root conventions")
        self._write("pkg/AGENTS.md", "# pkg conventions")
        self._write("pkg/sub/AGENTS.md", "# sub conventions")
        items = discover_agents_md(self.root)
        # root first, then by depth/path
        self.assertEqual([a.directory for a in items], ["", "pkg", "pkg/sub"])
        self.assertEqual(items[0].content, "# root conventions")

    def test_skips_vendored_and_empty(self):
        self._write("AGENTS.md", "# root")
        self._write("node_modules/lib/AGENTS.md", "# vendored, ignore")
        self._write("empty/AGENTS.md", "   \n")  # empty after strip -> skipped
        dirs = [a.directory for a in discover_agents_md(self.root)]
        self.assertEqual(dirs, [""])

    def test_missing_workdir(self):
        self.assertEqual(discover_agents_md(None), [])
        self.assertEqual(discover_agents_md(os.path.join(self.root, "nope")), [])


class ToolMessageContentTest(testing.TestCase):
    def test_images_are_attached_after_the_json(self):
        part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
        content = tool_message_content({"ok": True, "images": [part, part]})
        self.assertEqual(
            json.loads(content[0]), {"ok": True, "images": ["<image 1>", "<image 2>"]}
        )
        self.assertEqual(content[1:], [part, part])
        self.assertEqual(
            json.loads(tool_message_content({"image": part})[0]), {"image": "<image 1>"}
        )

    def test_audio_is_attached_after_the_json(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ"}}
        clip = {"type": "input_audio", "input_audio": {"data": "QUJD", "format": "wav"}}
        content = tool_message_content({"image": image, "audio": clip})
        self.assertEqual(
            json.loads(content[0]), {"image": "<image 1>", "audio": "<audio 1>"}
        )
        self.assertEqual(content[1:], [image, clip])

    def test_results_without_images_are_unchanged(self):
        self.assertEqual(tool_message_content({"ok": True}), {"ok": True})
        self.assertEqual(tool_message_content("error: boom"), "error: boom")
