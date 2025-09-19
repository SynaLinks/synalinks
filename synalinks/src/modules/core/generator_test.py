# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks import modules
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Generator
from synalinks.src.modules import Input
from synalinks.src.programs import Program


class GeneratorModuleTest(testing.TestCase):
    def test_format_message(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        msgs = Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        ).format_messages(
            inputs=Query(query="What is the french city of aerospace and robotics?")
        )
        print(msgs.prettify_json())
        self.assertTrue(len(msgs.messages) == 2)

    def test_format_message_with_instructions(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        msgs = Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
            instructions="You are an helpfull assistant",
        ).format_messages(
            Query(query="What is the french city of aerospace and robotics?")
        )
        print(msgs.prettify_json())
        self.assertTrue(len(msgs.messages) == 2)

    def test_format_message_with_examples(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        msgs = Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
            examples=[
                (
                    {"query": "What is the capital of France?"},
                    {
                        "rationale": "The capital of France is well known",
                        "answer": "Paris",
                    },
                )
            ],
        ).format_messages(
            Query(query="What is the french city of aerospace and robotics?")
        )
        print(msgs.prettify_json())
        self.assertTrue(len(msgs.messages) == 2)

    def test_format_message_with_examples_and_instructions(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        msgs = Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
            examples=[
                (
                    {
                        "query": "What is the capital of France?",
                    },
                    {
                        "rationale": "The capital of France is well known",
                        "answer": "Paris",
                    },
                )
            ],
            instructions="You are an helpfull assistant",
        ).format_messages(
            Query(query="What is the french city of aerospace and robotics?")
        )
        print(msgs.prettify_json())
        self.assertTrue(len(msgs.messages) == 2)

    @patch("litellm.acompletion")
    async def test_basic_functional_setup(self, mock_completion):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

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

        x0 = Input(data_model=Query)
        x1 = await Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        )(x0)
        program = Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought",
        )

        result = await program(
            Query(query="What is the french city of aerospace and robotics?")
        )

        self.assertEqual(result.get_json(), json.loads(expected_string))

    @patch("litellm.acompletion")
    async def test_basic_functional_setup_with_schema(self, mock_completion):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama/mistral")

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

        x0 = Input(data_model=Query)
        x1 = await Generator(
            schema=AnswerWithRationale.get_schema(),
            language_model=language_model,
        )(x0)
        program = Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought",
        )

        result = await program(
            Query(query="What is the french city of aerospace and robotics?")
        )

        self.assertEqual(result.get_json(), json.loads(expected_string))

    @patch("litellm.acompletion")
    async def test_basic_subclassing_setup(self, mock_completion):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama/mistral")

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

        class ChainOfThought(Program):
            def __init__(self, language_model):
                super().__init__()
                self.answer = Generator(
                    data_model=AnswerWithRationale, language_model=language_model
                )

            async def call(self, inputs):
                x = await self.answer(inputs)
                return x

        program = ChainOfThought(language_model=language_model)

        result = await program(
            Query(query="What is the french city of aerospace and robotics?")
        )

        self.assertEqual(result.get_json(), json.loads(expected_string))

    def test_serialization(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama/mistral")

        generator = Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        )

        serialized_dict = modules.serialize(generator)
        new_generator = modules.deserialize(serialized_dict)
        # check that the nested object are good
        self.assertEqual(
            str(new_generator.language_model),
            str(generator.language_model),
        )
