# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json
from enum import Enum
from typing import List
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel, Field
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Input
from synalinks.src.modules.agents.prebuilt import Agent, dynamic_enum_on_nested_property, ToolDecision as OGToolDecision
from synalinks.src.programs import Program


class AgentTest(testing.TestCase):
    def setUp(self):
        schema = OGToolDecision.get_schema()
        
        toolkit = ["calculate", "websearch"]

        self.dynamic_schema = dynamic_enum_on_nested_property(
            schema,
            "ToolChoice/properties/name",
            toolkit,
            description="The name of tool to run from available toolkit."
        )

    @patch("litellm.acompletion")
    async def test_flow_with_parallel_searches(self, mock_completion):
        class I(DataModel):
            query: str = Field(
                description="The user query",
            )

        class O(DataModel):
            answer: str = Field(
                description="The final answer to the query.",
            )

        async def websearch(query: str):
            """Perform a web search for the given query.
            
            Args:
                query (str): The search query to perform.
            """
            return {
                "results": [
                    f"Result 1 for {query}",
                    f"Result 2 for {query}",
                    f"Result 3 for {query}",
                ],
                "log": "Web search completed successfully."
            }

        language_model = LanguageModel(model="ollama_chat/deepseek-r1")

        tool_decision_1 = (
            "{"
                '"reasoning": "I need to search for ...", '
                '"choices": ['
                    '{"name": "websearch", "purpose": "Get ..."}, '
                    '{"name": "websearch", "purpose": "Get ..."}, '
                ']'
            "}"
        )

        action_1_1 = """{"query": "..."}"""
        action_1_2 = """{"query": "..."}"""

        tool_decision_end = (
            "{"
                '"reasoning": "I have all the information, so I can provide the answer.", '
                '"choices": []'
            "}"
        )

        response = (
            "{"
            '"answer": "..."'
            "}"
        )

        mock_responses = [
            {"choices": [{"message": {"content": tool_decision_1}}]},
            {"choices": [{"message": {"content": action_1_1}}]},
            {"choices": [{"message": {"content": action_1_2}}]},
            {"choices": [{"message": {"content": tool_decision_end}}]},
            {"choices": [{"message": {"content": response}}]},
        ]
        mock_completion.side_effect = mock_responses

        x0 = Input(data_model=I)
        x1 = await Agent(
            data_model=O,
            language_model=language_model,
            toolkit=[websearch],
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
        )

        result = await program(
            I(
                query=(
                    "I need comprehensive information about Tesla over the past week."
                )
            )
        )

    def test_dynamic_enum_schema_on_nested_property(self):
        class Name(str, Enum):
            """The name of tool to run from available toolkit."""
            calculate = "calculate"
            websearch = "websearch"

        class ToolChoice(DataModel):
            name: Name
            purpose: str = Field(
                description="A clear, specific explanation of what the tool should accomplish."
            )

        class ToolDecision(DataModel):
            reasoning: str = Field(
                description="A step-by-step analysis of the current state, what has been done, and what should be done next."
            )
            choices: List[ToolChoice] = Field(
                description="The array of tool choices to run in parallel with their specific purpose."
            )

        self.assertJsonEqual(json.dumps(self.dynamic_schema, indent=2), ToolDecision.prettify_schema())
