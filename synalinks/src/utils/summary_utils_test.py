# Modified from: keras/src/utils/summary_utils_test.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.modules.core.identity import Identity
from synalinks.src.modules.core.input_module import Input
from synalinks.src.programs.program import Program
from synalinks.src.programs.sequential import Sequential
from synalinks.src.utils import summary_utils


class Query(DataModel):
    query: str


class SummaryHelpersTest(testing.TestCase):
    def test_count_params_returns_len(self):
        self.assertEqual(summary_utils.count_params([1, 2, 3]), 3)
        self.assertEqual(summary_utils.count_params([]), 0)

    def test_highlight_number_none_uses_distinct_color(self):
        self.assertIn("color(45)", summary_utils.highlight_number(None))
        self.assertIn("None", summary_utils.highlight_number(None))

    def test_highlight_number_non_none(self):
        out = summary_utils.highlight_number(42)
        self.assertIn("color(34)", out)
        self.assertIn("42", out)

    def test_highlight_symbol(self):
        self.assertIn("color(33)", summary_utils.highlight_symbol("Identity"))

    def test_bold_text_plain(self):
        out = summary_utils.bold_text("hi")
        self.assertEqual(out, "[bold]hi[/]")

    def test_bold_text_with_color(self):
        out = summary_utils.bold_text("hi", color=9)
        self.assertEqual(out, "[bold][color(9)]hi[/][/]")


class FormatModuleSchemaTest(testing.TestCase):
    async def test_unbuilt_module_returns_question_mark(self):
        # A bare Identity has no inbound nodes and no built schemas dict.
        self.assertEqual(summary_utils.format_module_schema(Identity()), "?")

    async def test_built_module_returns_schema(self):
        inputs = Input(data_model=Query)
        outputs = await Identity()(inputs)
        program = Program(inputs=inputs, outputs=outputs)
        # Inspect Identity inside the program — it has an inbound node.
        ident = program.modules[-1]
        schema = summary_utils.format_module_schema(ident)
        self.assertIn("query", schema)
        # Schemas are JSON-formatted strings, $defs is stripped.
        self.assertNotIn("$defs", schema)


class GetModuleIndexBoundsTest(testing.TestCase):
    class _M:
        def __init__(self, name):
            self.name = name

    def _modules(self):
        return [self._M("input"), self._M("hidden_1"), self._M("hidden_2"), self._M("out")]

    def test_none_range_returns_full_span(self):
        mods = self._modules()
        self.assertEqual(
            summary_utils.get_module_index_bound_by_module_name(mods, None),
            [0, len(mods)],
        )

    def test_string_range_inclusive(self):
        mods = self._modules()
        idx = summary_utils.get_module_index_bound_by_module_name(
            mods, ("hidden_1", "hidden_2")
        )
        self.assertEqual(idx, [1, 3])

    def test_regex_range(self):
        mods = self._modules()
        idx = summary_utils.get_module_index_bound_by_module_name(
            mods, ("hidden_.*", "out")
        )
        # First match for "hidden_.*" is index 1; matches the last one (2) too,
        # but the function returns [min(lower), max(upper)+1].
        self.assertEqual(idx, [1, 4])

    def test_wrong_length_raises(self):
        mods = self._modules()
        with self.assertRaisesRegex(ValueError, "length 2"):
            summary_utils.get_module_index_bound_by_module_name(mods, ("only_one",))

    def test_non_string_raises(self):
        mods = self._modules()
        with self.assertRaisesRegex(ValueError, "string type only"):
            summary_utils.get_module_index_bound_by_module_name(mods, (0, 1))

    def test_no_match_raises(self):
        mods = self._modules()
        with self.assertRaisesRegex(ValueError, "do not match"):
            summary_utils.get_module_index_bound_by_module_name(
                mods, ("not_present", "out")
            )

    def test_lower_after_upper_swaps(self):
        mods = self._modules()
        # When the lower bound appears after the upper bound, the function
        # returns the reversed span so the caller gets a non-empty slice.
        idx = summary_utils.get_module_index_bound_by_module_name(
            mods, ("out", "input")
        )
        self.assertEqual(idx, [0, 4])


class PrintSummarySequentialTest(testing.TestCase):
    async def test_sequential_program_summary_runs(self):
        program = Sequential(
            modules=[
                Input(data_model=Query),
                Identity(),
            ],
            name="seq_program",
            description="A test sequential program",
        )
        # The sequential-like branch should render without errors.
        program.summary(print_fn=lambda _: None)

    async def test_functional_program_summary_runs(self):
        inputs = Input(data_model=Query)
        outputs = await Identity()(inputs)
        program = Program(
            inputs=inputs,
            outputs=outputs,
            name="functional_program",
            description="A test program",
        )
        # Exercises expand_nested and show_trainable branches.
        program.summary(
            print_fn=lambda _: None,
            expand_nested=True,
            show_trainable=True,
        )

    async def test_summary_with_module_range(self):
        program = Sequential(
            modules=[
                Input(data_model=Query),
                Identity(),
            ],
            name="ranged",
            description="ranged summary",
        )
        # Limiting the printed range exercises the module_range branch.
        program.summary(
            print_fn=lambda _: None,
            module_range=("identity.*", "identity.*"),
        )

    async def test_summary_without_print_fn(self):
        # Without an explicit print_fn the function uses rich's default
        # console; just confirm it doesn't crash.
        program = Sequential(
            modules=[
                Input(data_model=Query),
                Identity(),
            ],
            name="default_console",
            description="d",
        )
        program.summary()
