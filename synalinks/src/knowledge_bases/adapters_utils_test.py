# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import pyarrow as pa

from synalinks.src import testing
from synalinks.src.knowledge_bases.adapters_utils import format_search_results


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
