# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.rewards.rubrics import get_rubric
from synalinks.src.rewards.rubrics import list_rubrics


class RubricsTest(testing.TestCase):
    def test_list_rubrics(self):
        names = list_rubrics()
        self.assertIn("answer_relevancy", names)
        self.assertIn("faithfulness", names)
        self.assertIn("tool_correctness", names)
        self.assertEqual(names, sorted(names))

    def test_get_rubric_returns_copy(self):
        rubric = get_rubric("faithfulness")
        rubric[0]["name"] = "changed"
        self.assertNotEqual(get_rubric("faithfulness")[0]["name"], "changed")

    def test_get_rubric_unknown(self):
        with self.assertRaisesRegex(ValueError, "Unknown rubric preset"):
            get_rubric("unknown")
