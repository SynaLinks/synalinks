from synalinks.src import testing
from synalinks.src.utils.progbar import Progbar


class ProgbarPlainModeTest(testing.TestCase):
    """When stdout is not a terminal (log file, pipe), the bar prints one plain
    line per update: no escape sequences, no backspaces, no blank lines."""

    def _run(self, target):
        import io
        from contextlib import redirect_stdout

        out = io.StringIO()
        with redirect_stdout(out):  # a StringIO is not a tty
            bar = Progbar(target=target, verbose=1, unit_name="step", interval=0)
            self.assertFalse(bar._dynamic_display)
            bar.update(1, [("reward", 0.5)])
            bar.update(2, [("reward", 0.7)])
            if target is not None:
                bar.update(2, [("reward", 0.7)], finalize=True)
        return out.getvalue()

    def test_known_target_prints_clean_lines(self):
        text = self._run(target=2)
        self.assertNotIn("\x1b", text)
        self.assertNotIn("\r", text)
        self.assertNotIn("\b", text)
        lines = text.splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(line.strip() for line in lines))  # no blank lines
        self.assertTrue(lines[0].startswith("1/2 - "))
        self.assertIn("reward: 0.5000", lines[0])
        self.assertTrue(lines[2].startswith("2/2 - "))
        self.assertIn("s/step", lines[2])

    def test_unknown_target_uses_question_mark(self):
        text = self._run(target=None)
        self.assertNotIn("Unknown", text)
        self.assertTrue(text.splitlines()[0].startswith("1/? - "))
