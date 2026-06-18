# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Tests for the AGENTS.md helpers in ``agents_utils`` (agents.md standard):
root + nested (monorepo nearest-wins) discovery and the prompt rendering."""

import os
import tempfile

from synalinks.src import testing
from synalinks.src.modules.agents.utils.agents_utils import AgentsMd
from synalinks.src.modules.agents.utils.agents_utils import agents_md_prompt
from synalinks.src.modules.agents.utils.agents_utils import discover_agents_md
from synalinks.src.modules.agents.utils.agents_utils import find_agents_md


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

    def test_prompt_returns_root_content_verbatim(self):
        items = [
            AgentsMd(path="/w/AGENTS.md", content="Always be terse.", directory=""),
            AgentsMd(path="/w/pkg/AGENTS.md", content="pkg rule", directory="pkg"),
        ]
        out = agents_md_prompt(items)
        # Root body returned verbatim, with no added framing.
        self.assertEqual(out, "Always be terse.")
        # Nested files are not surfaced.
        self.assertNotIn("pkg rule", out)
        self.assertNotIn("pkg/AGENTS.md", out)

    def test_prompt_no_root_returns_empty(self):
        items = [AgentsMd(path="/w/pkg/AGENTS.md", content="pkg rule", directory="pkg")]
        self.assertEqual(agents_md_prompt(items), "")

    def test_prompt_empty(self):
        self.assertEqual(agents_md_prompt([]), "")
