# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import pyarrow as pa

from synalinks.src import testing
from synalinks.src.knowledge_bases.adapters_utils import format_schema_columns
from synalinks.src.knowledge_bases.adapters_utils import format_search_results
from synalinks.src.knowledge_bases.adapters_utils import unknown_title_error


class _Model:
    def __init__(self, title):
        self.title = title

    def get_schema(self):
        return {"title": self.title}


class SchemaColumnsTest(testing.TestCase):
    def test_columns_with_type_const_fallback_and_skip(self):
        properties = {
            "label": {"const": "Person"},
            "name": {"type": "string", "description": "Full name"},
            "kind": {"const": "a"},
            "tag": {},
        }
        self.assertEqual(
            format_schema_columns(properties, skip=("label",)),
            ["  - name (string): Full name", "  - kind (a): ", "  - tag (unknown): "],
        )


class UnknownTitleErrorTest(testing.TestCase):
    def test_known_title_is_none(self):
        self.assertIsNone(unknown_title_error([_Model("Doc")], "Doc", "Table"))

    def test_unknown_title_lists_the_available_ones(self):
        self.assertEqual(
            unknown_title_error([_Model("Doc"), _Model("Note")], "Docs", "Table"),
            {"error": "Table 'Docs' not found. Available: ['Doc', 'Note']"},
        )


class FormatSearchResultsTest(testing.TestCase):
    def test_csv_writes_nested_columns_as_json(self):
        table = pa.table(
            {
                "id": ["d1"],
                "embedding": pa.array([[1.0, 0.0]], pa.list_(pa.float32(), 2)),
                "node": [{"name": "a", "tags": ["x"]}],
            }
        )
        out = format_search_results(table, "csv")
        self.assertIn('"[1.0,0.0]"', out)
        self.assertIn('"{""name"":""a"",""tags"":[""x""]}"', out)
