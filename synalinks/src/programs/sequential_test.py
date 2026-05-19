# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import modules
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import standardize_schema
from synalinks.src.modules import Input
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Sequential
from synalinks.src.saving.serialization_lib import deserialize_synalinks_object
from synalinks.src.saving.serialization_lib import serialize_synalinks_object


class SequentialTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_basic_flow_with_input(self, mock_completion):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(
            model="ollama_chat/deepseek-r1",
        )

        expected_string = (
            """{"rationale": "Toulouse hosts numerous research institutions """
            """and universities that specialize in aerospace engineering and """
            """robotics, such as the Institut Supérieur de l'Aéronautique et """
            """de l'Espace (ISAE-SUPAERO) and the French National Centre for """
            """Scientific Research (CNRS)","""
            """ "answer": "Toulouse"}"""
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        program = Sequential(
            name="chain_of_thought",
            description="Useful to answer in a step by step manner",
        )
        program.add(Input(data_model=Query))
        program.add(
            modules.Generator(
                data_model=AnswerWithRationale,
                language_model=language_model,
            )
        )

        self.assertEqual(len(program.modules), 2)
        self.assertTrue(program.built)
        self.assertEqual(len(program.variables), 1)

        # Test eager call
        result = await program(
            Query(query="What is the french city of aerospace and robotics?")
        )
        self.assertEqual(result.get_json(), json.loads(expected_string))

        # Test symbolic call
        x = SymbolicDataModel(data_model=Query)
        y = await program(x)
        self.assertIsInstance(y, SymbolicDataModel)
        self.assertEqual(
            y.get_schema(), standardize_schema(AnswerWithRationale.get_schema())
        )

        # Test `modules` constructor arg
        program = Sequential(
            modules=[
                Input(data_model=Query),
                modules.Generator(
                    data_model=AnswerWithRationale,
                    language_model=language_model,
                ),
            ],
            name="chain_of_thought",
            description="Useful to answer in a step by step manner",
        )
        self.assertEqual(len(program.modules), 2)
        self.assertTrue(program.built)
        self.assertEqual(len(program.variables), 1)

        result = await program(
            Query(query="What is the french city of aerospace and robotics?")
        )
        self.assertEqual(result.get_json(), json.loads(expected_string))

        # Test pop
        program.pop()
        self.assertEqual(len(program.modules), 1)
        self.assertFalse(program.built)
        self.assertEqual(len(program.variables), 0)

    def test_representation(self):
        program = Sequential(
            name="chain_of_thought",
            description="Useful to answer in a step by step manner",
        )
        self.assertEqual(
            str(program),
            "<Sequential name=chain_of_thought, "
            "description='Useful to answer in a step by step manner', "
            "built=False>",
        )

    def test_add_rejects_non_module(self):
        program = Sequential(description="t")
        with self.assertRaisesRegex(ValueError, "Only instances of `synalinks.Module`"):
            program.add("not-a-module")

    def test_add_rejects_duplicate_name(self):
        class Query(DataModel):
            query: str

        program = Sequential(description="t")
        program.add(Input(data_model=Query, name="shared"))
        with self.assertRaisesRegex(ValueError, "should have unique names"):
            program.add(Input(data_model=Query, name="shared"))

    def test_add_rejects_second_input_module(self):
        class Query(DataModel):
            query: str

        class Query2(DataModel):
            other: str

        program = Sequential(description="t")
        program.add(Input(data_model=Query))
        with self.assertRaisesRegex(ValueError, "already been configured"):
            program.add(Input(data_model=Query2))

    def test_modules_attribute_is_read_only(self):
        program = Sequential(description="t")
        with self.assertRaisesRegex(AttributeError, "reserved"):
            program.modules = []

    def test_property_errors_when_not_built(self):
        program = Sequential(name="empty", description="t")
        with self.assertRaisesRegex(AttributeError, "no defined input schema"):
            _ = program.input_schema
        with self.assertRaisesRegex(AttributeError, "no defined output schema"):
            _ = program.output_schema
        with self.assertRaisesRegex(AttributeError, "no defined inputs"):
            _ = program.inputs
        with self.assertRaisesRegex(AttributeError, "no defined outputs"):
            _ = program.outputs

    async def test_build_without_modules_raises(self):
        class Query(DataModel):
            query: str

        program = Sequential(description="t")
        with self.assertRaisesRegex(ValueError, "no modules"):
            await program.build(Input(data_model=Query))

    async def test_build_silently_returns_when_inputs_have_no_schema(self):
        # `inputs.get_schema()` raises → build() should bail without error and
        # leave the program in its pre-build state (no `_functional`).
        program = Sequential(description="t")

        class _NoSchema:
            def get_schema(self):
                raise AttributeError("nope")

        await program.build(_NoSchema())
        self.assertIsNone(program._functional)

    @patch("litellm.acompletion")
    async def test_get_config_from_config_round_trip(self, mock_completion):
        class Query(DataModel):
            query: str

        class Answer(DataModel):
            answer: str

        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"answer":"Toulouse"}'}}]
        }

        program = Sequential(
            name="roundtrip",
            description="Round-trip serialization check.",
        )
        program.add(Input(data_model=Query))
        program.add(
            modules.Generator(
                data_model=Answer,
                language_model=LanguageModel(model="ollama_chat/deepseek-r1"),
            )
        )
        # Trigger eager build so `build_input_schema` ends up in the config.
        await program(Query(query="hello"))

        cfg = serialize_synalinks_object(program)
        restored = deserialize_synalinks_object(cfg)
        self.assertIsInstance(restored, Sequential)
        self.assertEqual(restored.name, "roundtrip")
        self.assertEqual(restored.description, "Round-trip serialization check.")
        self.assertEqual(len(restored.modules), len(program.modules))
