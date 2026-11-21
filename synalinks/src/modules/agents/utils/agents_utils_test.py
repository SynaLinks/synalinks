# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Tests for the AGENTS.md helpers in ``agents_utils`` (agents.md standard):
the root file lookup."""

import json
import os
import tempfile

from synalinks.src import testing
from synalinks.src.modules.agents.utils.agents_utils import call_tool
from synalinks.src.modules.agents.utils.agents_utils import find_agents_md
from synalinks.src.modules.agents.utils.agents_utils import tool_arguments
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

    def test_results_without_images_are_json_strings(self):
        self.assertEqual(tool_message_content({"ok": True}), '{"ok": true}')
        self.assertEqual(tool_message_content([1, "é"]), '[1, "é"]')
        self.assertEqual(tool_message_content("error: boom"), "error: boom")


class ToolArgumentsTest(testing.TestCase):
    def test_the_json_string_is_parsed(self):
        function = {"name": "calculate", "arguments": '{"expression": "1 + 1"}'}
        self.assertEqual(tool_arguments(function), {"expression": "1 + 1"})

    def test_no_arguments_are_none(self):
        self.assertEqual(tool_arguments({"name": "now", "arguments": ""}), {})

    def test_arguments_that_are_no_json_object_say_so(self):
        with self.assertRaisesRegex(ValueError, "'calculate' are not valid JSON"):
            tool_arguments({"name": "calculate", "arguments": '{"expression": '})
        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            tool_arguments({"name": "calculate", "arguments": "[1, 2]"})

    async def test_a_call_with_bad_arguments_fails_like_the_tool(self):
        async def calculate(expression):
            return {"result": eval(expression)}

        result = await call_tool(
            calculate, {"name": "calculate", "arguments": '{"expression": "2 * 3"}'}
        )
        self.assertEqual(result, {"result": 6})
        with self.assertRaises(ValueError):
            await call_tool(calculate, {"name": "calculate", "arguments": "not json"})
