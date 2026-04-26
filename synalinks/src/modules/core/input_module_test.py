# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import standardize_schema
from synalinks.src.modules import Input
from synalinks.src.modules import InputModule


class InputLayerTest(testing.TestCase):
    def test_input_basic(self):
        class Query(DataModel):
            query: str

        values = InputModule(input_data_model=SymbolicDataModel(data_model=Query))
        self.assertEqual(values.name, "input_module")
        self.assertEqual(values.get_schema(), standardize_schema(Query.get_schema()))

    def test_input(self):
        class Query(DataModel):
            query: str

        inputs = Input(data_model=Query)
        self.assertIsInstance(inputs, SymbolicDataModel)
        self.assertEqual(inputs.get_schema(), standardize_schema(Query.get_schema()))
