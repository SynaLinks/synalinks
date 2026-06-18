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


class BeautifySchemaTest(testing.TestCase):
    """`beautify_schema` renders a JSON schema as a compact, Pydantic-like
    class definition rather than a verbose indented JSON dump."""

    def test_simple_object_renders_fields_as_annotations(self):
        schema = {
            "title": "Query",
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        self.assertEqual(summary_utils.beautify_schema(schema), "Query:\n  query: str")

    def test_primitive_type_mapping(self):
        schema = {
            "title": "M",
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "number"},
                "c": {"type": "boolean"},
            },
            "required": ["a", "b", "c"],
        }
        out = summary_utils.beautify_schema(schema)
        self.assertIn("a: int", out)
        self.assertIn("b: float", out)
        self.assertIn("c: bool", out)

    def test_optional_field_without_default_marked_with_question_mark(self):
        # Not in `required` and no explicit `default` (e.g. a default_factory):
        # flagged with `?`, no invented value.
        schema = {
            "title": "M",
            "type": "object",
            "properties": {"req": {"type": "string"}, "opt": {"type": "string"}},
            "required": ["req"],
        }
        out = summary_utils.beautify_schema(schema)
        self.assertIn("req: str", out)
        self.assertIn("opt?: str", out)

    def test_explicit_default_is_rendered(self):
        schema = {
            "title": "M",
            "type": "object",
            "properties": {
                "count": {"type": "integer", "default": 5},
                "label": {"type": "string", "default": "hi"},
                "maybe": {
                    "anyOf": [{"type": "number"}, {"type": "null"}],
                    "default": None,
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
            "required": [],
        }
        out = summary_utils.beautify_schema(schema)
        self.assertIn("count: int = 5", out)
        self.assertIn("label: str = 'hi'", out)
        # A default carries the optionality, so no redundant `?` marker.
        self.assertIn("maybe: float | None = None", out)
        self.assertNotIn("maybe?", out)
        self.assertIn("tags: list[str] = []", out)

    def test_long_default_is_truncated(self):
        schema = {
            "title": "M",
            "type": "object",
            "properties": {
                "blob": {"type": "string", "default": "x" * 100},
            },
            "required": [],
        }
        out = summary_utils.beautify_schema(schema)
        self.assertIn("blob: str = ", out)
        self.assertIn("…", out)
        # The rendered line must not carry the full 100-char default.
        self.assertNotIn("x" * 100, out)

    def test_array_renders_as_list_of_item_type(self):
        schema = {
            "title": "M",
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        }
        self.assertIn("items: list[str]", summary_utils.beautify_schema(schema))

    def test_anyof_with_null_renders_as_union(self):
        schema = {
            "title": "M",
            "type": "object",
            "properties": {
                "x": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["x"],
        }
        self.assertIn("x: str | None", summary_utils.beautify_schema(schema))

    def test_enum_renders_as_literal(self):
        schema = {
            "title": "M",
            "type": "object",
            "properties": {"color": {"enum": ["red", "green"]}},
            "required": ["color"],
        }
        out = summary_utils.beautify_schema(schema)
        self.assertIn("color: Literal['red', 'green']", out)

    def test_ref_renders_nested_model_as_its_own_block(self):
        schema = {
            "title": "Answer",
            "type": "object",
            "properties": {"doc": {"$ref": "#/$defs/Document"}},
            "required": ["doc"],
            "$defs": {
                "Document": {
                    "title": "Document",
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                }
            },
        }
        out = summary_utils.beautify_schema(schema)
        self.assertIn("Answer:\n  doc: Document", out)
        self.assertIn("Document:\n  text: str", out)
        # A dependency is defined *before* the model that uses it — the order
        # you'd write it in code, root last.
        self.assertLess(out.index("Document:"), out.index("Answer:"))
        # `$defs` plumbing never leaks into the rendered output.
        self.assertNotIn("$defs", out)
        self.assertNotIn("$ref", out)

    def test_dict_with_typed_values(self):
        schema = {
            "title": "M",
            "type": "object",
            "properties": {
                "meta": {"type": "object", "additionalProperties": {"type": "integer"}}
            },
            "required": ["meta"],
        }
        self.assertIn("meta: dict[str, int]", summary_utils.beautify_schema(schema))

    def test_non_object_schema_renders_single_line(self):
        schema = {"title": "Color", "enum": ["a", "b"]}
        self.assertEqual(
            summary_utils.beautify_schema(schema), "Color: Literal['a', 'b']"
        )


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
        # Rendered as a compact Pydantic-like block, no raw JSON-schema keys.
        self.assertIn("query: str", schema)
        self.assertNotIn("$defs", schema)
        self.assertNotIn("properties", schema)


class GetModuleIndexBoundsTest(testing.TestCase):
    class _M:
        def __init__(self, name):
            self.name = name

    def _modules(self):
        return [
            self._M("input"),
            self._M("hidden_1"),
            self._M("hidden_2"),
            self._M("out"),
        ]

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
