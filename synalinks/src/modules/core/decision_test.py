# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Input
from synalinks.src.modules.core.decision import Decision
from synalinks.src.programs import Program


class DecisionTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_basic_decision(self, mock_completion):
        class Query(DataModel):
            query: str

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        expected_string = (
            """{"thinking": "The question ask for the capital of France, """
            """the answer is straitforward as it is well known that Paris is the"""
            """ capital", "choice": "easy"}"""
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        x0 = Input(data_model=Query)
        x1 = await Decision(
            question="What is the difficulty level of the above provided query?",
            labels=["easy", "difficult", "unkown"],
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
        )

        result = await program(Query(query="What is the French capital?"))

        self.assertEqual(result.get_json(), json.loads(expected_string))
