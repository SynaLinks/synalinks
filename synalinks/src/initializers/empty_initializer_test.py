# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from typing import List

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.initializers.empty_initializer import Empty


class EmptyInitializerTest(testing.TestCase):
    def test_empty_initializer(self):
        class Instructions(DataModel):
            instructions: List[str] = []

        initializer = Empty(data_model=Instructions)
        empty_data_model = initializer()
        self.assertEqual(initializer.get_schema(), Instructions.get_schema())
        self.assertEqual(empty_data_model, Instructions().get_json())

    def test_empty_initializer_from_config(self):
        class Instructions(DataModel):
            instructions: List[str] = []

        initializer = Empty(data_model=Instructions)
        config = initializer.get_config()
        initializer = Empty.from_config(config)
        empty_data_model = initializer()
        self.assertEqual(initializer.get_schema(), Instructions.get_schema())
        self.assertEqual(empty_data_model, Instructions().get_json())
