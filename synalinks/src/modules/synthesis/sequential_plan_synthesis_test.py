# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.language_models.language_model import LanguageModel
from synalinks.src.modules.synthesis.sequential_plan_synthesis import (
    SequentialPlanSynthesis,
)
from synalinks.src.programs.program import Program
from synalinks.src.saving.object_registration import register_synalinks_serializable
from synalinks.src.utils.tool_utils import Tool


@register_synalinks_serializable()
async def calculate(expression: str):
    """Calculate the result of a mathematical expression.

    Args:
        expression (str): The mathematical expression to calculate, such as
            '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
            parentheses, and spaces.
    """
    if not all(char in "0123456789+-*/(). " for char in expression):
        return {
            "result": None,
            "log": "Error: invalid characters in expression",
        }
    try:
        # Evaluate the mathematical expression safely
        result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
        return {
            "result": result,
            "log": "Successfully executed",
        }
    except Exception as e:
        return {
            "result": None,
            "log": f"Error: {e}",
        }


@register_synalinks_serializable()
async def thinking(thinking: str):
    """Think about something.

    Args:
        thinking (str): Your step by step thinking.
    """
    return {
        "thinking": thinking,
    }


class Query(DataModel):
    query: str = Field(
        description="The user query",
    )


class FinalReport(DataModel):
    report: str = Field(
        description="The final report",
    )


class TaskSummary(DataModel):
    summary: str = Field(
        description="The summary of the executed task",
    )


class SequentialPlanSynthesisTest(testing.TestCase):
    async def test_default_synthesis(self):
        language_model = LanguageModel(model="ollama/mistral")

        tools = [
            Tool(calculate),
            Tool(thinking),
        ]

        inputs = Input(data_model=Query)
        outputs = await SequentialPlanSynthesis(
            data_model=FinalReport,
            language_model=language_model,
            runner=FunctionCallingAgent(
                data_model=TaskSummary,
                language_model=language_model,
                tools=tools,
                return_inputs_with_trajectory=False,
            ),
        )(inputs)

        _program = Program(
            inputs=inputs,
            outputs=outputs,
            name="planner_agent",
            description="An agent that learn a step by step plan to achieve a task",
        )

        # Verify program was built successfully
        self.assertIsNotNone(_program)

    @patch("litellm.acompletion")
    async def test_each_step_reaches_the_runner(self, mock_completion):
        language_model = LanguageModel(model="ollama/mistral")
        mock_completion.side_effect = [
            {"choices": [{"message": {"content": '{"summary": "facts"}'}}]},
            {"choices": [{"message": {"content": '{"summary": "outline"}'}}]},
            {"choices": [{"message": {"content": '{"thinking": "t", "report": "r"}'}}]},
        ]

        inputs = Input(data_model=Query)
        outputs = await SequentialPlanSynthesis(
            data_model=FinalReport,
            language_model=language_model,
            runner=Generator(data_model=TaskSummary, language_model=language_model),
            steps=["STEP-ONE", "STEP-TWO"],
        )(inputs)
        program = Program(inputs=inputs, outputs=outputs)

        result = await program(Query(query="q"))

        prompts = [str(c.kwargs["messages"]) for c in mock_completion.call_args_list]
        self.assertIn("STEP-ONE", prompts[0])
        self.assertIn("STEP-TWO", prompts[1])
        self.assertEqual(result.get_schema(), outputs.get_schema())
        self.assertEqual(result.get("report"), "r")
