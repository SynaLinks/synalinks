# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Input
from synalinks.src.modules.core.branch import Branch
from synalinks.src.modules.core.generator import Generator
from synalinks.src.programs import Program


class BranchTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_basic_branch(self, mock_completion):
        class Query(DataModel):
            query: str

        class Answer(DataModel):
            answer: str

        class AnswerWithCritique(DataModel):
            thinking: str
            critique: str
            answer: str

        language_model = LanguageModel("ollama_chat/deepseek-r1")

        x0 = Input(data_model=Query)
        x1, x2 = await Branch(
            question="What is the difficulty level of the given query?",
            labels=["easy", "difficult"],
            branches=[
                Generator(
                    data_model=Answer,
                    language_model=language_model,
                ),
                Generator(
                    data_model=AnswerWithCritique,
                    language_model=language_model,
                ),
            ],
            language_model=language_model,
        )(x0)
        x3 = x1 | x2

        program = Program(
            inputs=x0,
            outputs=x3,
            name="adaptative_chain_of_thought",
            description="Useful to answer step by step only when needed",
        )

        decision_response = (
            """{"thinking": "The question ask for the capital of France, """
            """the answer is straitforward as it is well known that Paris is the"""
            """ capital", "choice": "easy"}"""
        )

        inference_response = """{"answer": "Paris"}"""

        mock_responses = [
            {"choices": [{"message": {"content": decision_response}}]},
            {"choices": [{"message": {"content": inference_response}}]},
        ]

        mock_completion.side_effect = mock_responses

        result = await program(Query(query="What is the French capital?"))
        self.assertEqual(result.get("answer"), "Paris")
