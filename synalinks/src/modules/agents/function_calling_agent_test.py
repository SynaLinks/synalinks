# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import is_chat_messages
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.tool import Tool
from synalinks.src.programs import Program
from synalinks.src.saving.object_registration import register_synalinks_serializable


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


@register_synalinks_serializable()
async def get_weather(location: str):
    """Get weather information for a location.
    Args:
        location (str): The location to get weather for.
    """
    # Mock weather data
    weather_data = {
        "New York": {"temp": 22, "condition": "Sunny"},
        "London": {"temp": 15, "condition": "Cloudy"},
        "Tokyo": {"temp": 28, "condition": "Rainy"},
    }

    if location in weather_data:
        return {
            "location": location,
            "temperature": weather_data[location]["temp"],
            "condition": weather_data[location]["condition"],
            "success": True,
        }
    else:
        return {"location": location, "error": "Location not found", "success": False}


@register_synalinks_serializable()
async def failing_tool(should_fail: bool = True):
    """A tool that intentionally fails for testing error handling.
    Args:
        should_fail (bool): Whether the tool should fail.
    """
    if should_fail:
        raise ValueError("This tool was designed to fail")
    return {"status": "success"}


class FunctionCallingAgentTest(testing.TestCase):
    async def test_agent_instantiation(self):
        """Test basic agent instantiation."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [
            Tool(calculate),
            Tool(thinking),
        ]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
        )(inputs)
        program = Program(
            inputs=inputs,
            outputs=outputs,
            name="function_calling_agent_test",
        )
        self.assertIsNotNone(program)

    async def test_agent_temperature_parameter(self):
        """Test that temperature is correctly passed to generators."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [Tool(calculate)]

        agent = FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            temperature=0.7,
            name="temp_test",
        )

        # Verify temperature is stored
        self.assertEqual(agent.temperature, 0.7)
        # Verify temperature is passed to tool_calls_generator
        self.assertEqual(agent.tool_calls_generator.temperature, 0.7)

    async def test_agent_reasoning_effort_parameter(self):
        """Test that reasoning_effort is correctly passed to generators."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [Tool(calculate)]

        agent = FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            reasoning_effort="low",
            name="reasoning_test",
        )

        # Verify reasoning_effort is stored
        self.assertEqual(agent.reasoning_effort, "low")
        # Verify reasoning_effort is passed to tool_calls_generator
        self.assertEqual(agent.tool_calls_generator.reasoning_effort, "low")

    async def test_agent_default_reasoning_effort_is_none(self):
        """Test that default reasoning_effort is None (not 'low' like ChainOfThought)."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [Tool(calculate)]

        agent = FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            name="default_test",
        )

        # FunctionCallingAgent should default to None (no reasoning)
        # But its ChainOfThought generators will default to "low"
        self.assertIsNone(agent.reasoning_effort)
        # The ChainOfThought inside defaults None to "low"
        self.assertEqual(agent.tool_calls_generator.reasoning_effort, "low")

    @patch("litellm.acompletion")
    async def test_autonomous_mode_simple_calculation(self, mock_completion):
        """Test autonomous mode with a simple calculation."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [
            Tool(calculate),
            Tool(thinking),
        ]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            autonomous=True,
            max_iterations=3,
        )(inputs)
        agent = Program(
            inputs=inputs,
            outputs=outputs,
            name="autonomous_calculation_test",
        )

        tool_calls = {
            "thinking": (
                "Perform simple arithmetic operation by adding the numbers "
                "given in the input."
            ),
            "tool_calls": [
                {
                    "tool_name": "calculate",
                    "expression": "152648 + 485",
                }
            ],
        }

        tool_calls_1 = {
            "thinking": (
                "The user has asked for a simple arithmetic operation, "
                "specifically adding 152648 and 485. I have already performed "
                "the calculation using the 'calculate' tool and obtained the "
                "result as 153133."
            ),
            "tool_calls": [],
        }

        mock_responses = [
            {"choices": [{"message": {"content": json.dumps(tool_calls)}}]},
            {"choices": [{"message": {"content": json.dumps(tool_calls_1)}}]},
        ]

        mock_completion.side_effect = mock_responses

        input_messages = ChatMessages(
            messages=[
                ChatMessage(
                    role="user",
                    content="How much is 152648 + 485?",
                ),
            ]
        )
        result = await agent(input_messages)

        print("Result:")
        print(result.prettify_json())

        # Verify result structure
        self.assertIsNotNone(result)
        messages = result.get("messages", [])
        self.assertGreater(len(messages), 0)
        self.assertTrue(is_chat_messages(result))

    @patch("litellm.acompletion")
    async def test_autonomous_mode_complex_calculation(self, mock_completion):
        """Test autonomous mode with a more complex calculation."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [
            Tool(calculate),
            Tool(thinking),
        ]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            autonomous=True,
            max_iterations=5,
        )(inputs)
        agent = Program(
            inputs=inputs,
            outputs=outputs,
            name="complex_calculation_test",
        )

        tool_calls = {
            "thinking": (
                "First, I will perform the arithmetic operation as instructed. "
                "Let's calculate (150 + 250) * 2 / 4. The order of operations "
                "is follow BIDMAS/BODMAS which means Brackets, Orders or "
                "Powers, Division and Multiplication, Addition and "
                "Subtraction. So, first I will add 150 and 250, then multiply "
                "the result by 2, divide it by 4 and finally add 100."
            ),
            "tool_calls": [
                {
                    "tool_name": "calculate",
                    "expression": "(150 + 250) * 2 / 4 + 100",
                }
            ],
        }

        tool_calls_1 = {
            "thinking": (
                "The user provided a mathematical expression to calculate. I "
                "performed the operation (150 + 250) * 2 / 4 and then added "
                "100 to the result. Now, the result is 300."
            ),
            "tool_calls": [
                {
                    "tool_name": "calculate",
                    "expression": "(150 + 250) * 2 / 4 + 100",
                }
            ],
        }

        mock_responses = [
            {"choices": [{"message": {"content": json.dumps(tool_calls)}}]},
            {"choices": [{"message": {"content": json.dumps(tool_calls_1)}}]},
        ]

        mock_completion.side_effect = mock_responses

        input_messages = ChatMessages(
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        "Calculate (150 + 250) * 2 / 4 and then add 100 to the result"
                    ),
                ),
            ]
        )
        result = await agent(input_messages)
        print("Result:")
        print(result.prettify_json())

    @patch("litellm.acompletion")
    async def test_autonomous_mode_returns_chat_messages_without_schema(
        self, mock_completion
    ):
        """Test that autonomous mode returns ChatMessages when no schema is provided."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [
            Tool(calculate),
        ]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            autonomous=True,
            max_iterations=3,
        )(inputs)
        agent = Program(
            inputs=inputs,
            outputs=outputs,
            name="autonomous_no_schema_test",
        )

        tool_calls = {
            "thinking": "I need to calculate 10 + 20.",
            "tool_calls": [
                {
                    "tool_name": "calculate",
                    "expression": "10 + 20",
                }
            ],
        }

        tool_calls_1 = {
            "thinking": "The calculation is complete. The result is 30.",
            "tool_calls": [],
        }

        mock_responses = [
            {"choices": [{"message": {"content": json.dumps(tool_calls)}}]},
            {"choices": [{"message": {"content": json.dumps(tool_calls_1)}}]},
        ]

        mock_completion.side_effect = mock_responses

        input_messages = ChatMessages(
            messages=[
                ChatMessage(
                    role="user",
                    content="How much is 10 + 20?",
                ),
            ]
        )
        result = await agent(input_messages)

        # Verify result is ChatMessages
        self.assertIsNotNone(result)
        self.assertTrue(is_chat_messages(result))

        # Verify messages include tool calls and tool results
        messages = result.get("messages", [])
        self.assertGreater(len(messages), 1)

        # Check that we have user, assistant with tool_calls, tool, and final assistant
        roles = [msg.get("role") for msg in messages]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)
        self.assertIn("tool", roles)

    @patch("litellm.acompletion")
    async def test_non_autonomous_mode_returns_chat_messages(self, mock_completion):
        """Test that non-autonomous mode returns ChatMessages."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [
            Tool(calculate),
        ]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            autonomous=False,
            return_inputs_with_trajectory=False,
        )(inputs)
        agent = Program(
            inputs=inputs,
            outputs=outputs,
            name="non_autonomous_test",
        )

        tool_calls = {
            "thinking": "I need to calculate 5 * 5.",
            "tool_calls": [
                {
                    "tool_name": "calculate",
                    "expression": "5 * 5",
                }
            ],
        }

        mock_responses = [
            {"choices": [{"message": {"content": json.dumps(tool_calls)}}]},
        ]

        mock_completion.side_effect = mock_responses

        input_messages = ChatMessages(
            messages=[
                ChatMessage(
                    role="user",
                    content="How much is 5 * 5?",
                ),
            ]
        )
        result = await agent(input_messages)

        # Verify result is ChatMessages (not ChatMessage)
        self.assertIsNotNone(result)
        self.assertTrue(is_chat_messages(result))

        # Verify messages structure - only assistant message since no prior tool calls
        messages = result.get("messages", [])
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].get("role"), "assistant")

    @patch("litellm.acompletion")
    async def test_non_autonomous_mode_returns_tool_and_assistant_messages(
        self, mock_completion
    ):
        """Test non-autonomous mode returns tool messages and assistant message."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [
            Tool(calculate),
        ]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            autonomous=False,
            return_inputs_with_trajectory=False,
        )(inputs)
        agent = Program(
            inputs=inputs,
            outputs=outputs,
            name="non_autonomous_tool_messages_test",
        )

        # Second LLM response after tool execution
        tool_calls_response = {
            "thinking": "The calculation result is 25. Task complete.",
            "tool_calls": [],
        }

        mock_responses = [
            {"choices": [{"message": {"content": json.dumps(tool_calls_response)}}]},
        ]

        mock_completion.side_effect = mock_responses

        # Input includes a previous assistant message with tool_calls
        # This simulates continuing the conversation after the user approved tool calls
        input_messages = ChatMessages(
            messages=[
                ChatMessage(
                    role="user",
                    content="How much is 5 * 5?",
                ),
                ChatMessage(
                    role="assistant",
                    content="I need to calculate 5 * 5.",
                    tool_calls=[
                        {
                            "id": "test-tool-call-id",
                            "name": "calculate",
                            "arguments": {"expression": "5 * 5"},
                        }
                    ],
                ),
            ]
        )
        result = await agent(input_messages)

        # Verify result is ChatMessages
        self.assertIsNotNone(result)
        self.assertTrue(is_chat_messages(result))

        # Verify messages include both tool result and new assistant message
        messages = result.get("messages", [])
        self.assertEqual(len(messages), 2)

        # First message should be the tool result
        self.assertEqual(messages[0].get("role"), "tool")
        self.assertEqual(messages[0].get("tool_call_id"), "test-tool-call-id")

        # Second message should be the assistant response
        self.assertEqual(messages[1].get("role"), "assistant")

    @patch("litellm.acompletion")
    async def test_non_autonomous_mode_with_trajectory(self, mock_completion):
        """Test non-autonomous mode with return_inputs_with_trajectory=True."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [
            Tool(calculate),
        ]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            autonomous=False,
            return_inputs_with_trajectory=True,
        )(inputs)
        agent = Program(
            inputs=inputs,
            outputs=outputs,
            name="non_autonomous_trajectory_test",
        )

        tool_calls = {
            "thinking": "I need to calculate 3 + 3.",
            "tool_calls": [
                {
                    "tool_name": "calculate",
                    "expression": "3 + 3",
                }
            ],
        }

        mock_responses = [
            {"choices": [{"message": {"content": json.dumps(tool_calls)}}]},
        ]

        mock_completion.side_effect = mock_responses

        input_messages = ChatMessages(
            messages=[
                ChatMessage(
                    role="user",
                    content="How much is 3 + 3?",
                ),
            ]
        )
        result = await agent(input_messages)

        # Verify result is ChatMessages
        self.assertIsNotNone(result)
        self.assertTrue(is_chat_messages(result))

        # Verify full trajectory is returned (user message + assistant message)
        messages = result.get("messages", [])
        self.assertGreaterEqual(len(messages), 2)
        self.assertEqual(messages[0].get("role"), "user")
        self.assertEqual(messages[-1].get("role"), "assistant")

    @patch("litellm.acompletion")
    async def test_interactive_mode_single_step(self, mock_completion):
        """Test interactive mode with single step execution."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [
            Tool(calculate),
            Tool(thinking),
        ]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            autonomous=False,
            return_inputs_with_trajectory=True,
        )(inputs)
        agent = Program(
            inputs=inputs,
            outputs=outputs,
            name="interactive_single_step_test",
        )

        tool_calls = {
            "thinking": "I need to calculate 152648 + 485.",
            "tool_calls": [
                {
                    "tool_name": "calculate",
                    "expression": "152648 + 485",
                }
            ],
        }

        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps(tool_calls)}}]
        }

        input_messages = ChatMessages(
            messages=[
                ChatMessage(
                    role="user",
                    content="How much is 152648 + 485?",
                )
            ]
        )
        result = await agent(input_messages)

        # Verify result structure
        self.assertIsNotNone(result)
        self.assertTrue(is_chat_messages(result))
        messages = result.get("messages", [])
        self.assertGreaterEqual(len(messages), 2)
        self.assertEqual(messages[0].get("role"), "user")
        self.assertEqual(messages[-1].get("role"), "assistant")

    @patch("litellm.acompletion")
    async def test_interactive_mode_multi_step(self, mock_completion):
        """Test interactive mode with multiple steps simulation."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [
            Tool(calculate),
            Tool(thinking),
        ]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            autonomous=False,
            return_inputs_with_trajectory=True,
        )(inputs)
        agent = Program(
            inputs=inputs,
            outputs=outputs,
            name="interactive_multi_step_test",
        )

        # First step: calculate 100 + 200
        tool_calls_step1 = {
            "thinking": "First, I need to calculate 100 + 200.",
            "tool_calls": [
                {
                    "tool_name": "calculate",
                    "expression": "100 + 200",
                }
            ],
        }

        # Second step: multiply result by 3
        tool_calls_step2 = {
            "thinking": "Now I need to multiply 300 by 3.",
            "tool_calls": [
                {
                    "tool_name": "calculate",
                    "expression": "300 * 3",
                }
            ],
        }

        # Final step: no more tool calls
        tool_calls_step3 = {
            "thinking": "Calculation complete. 100 + 200 = 300, then 300 * 3 = 900.",
            "tool_calls": [],
        }

        mock_responses = [
            {"choices": [{"message": {"content": json.dumps(tool_calls_step1)}}]},
            {"choices": [{"message": {"content": json.dumps(tool_calls_step2)}}]},
            {"choices": [{"message": {"content": json.dumps(tool_calls_step3)}}]},
        ]
        mock_completion.side_effect = mock_responses

        input_messages = ChatMessages(
            messages=[
                ChatMessage(
                    role="user",
                    content="I need to calculate 100 + 200, then multiply by 3",
                )
            ]
        )

        # Simulate multiple interaction steps
        max_steps = 3
        for step in range(max_steps):
            result = await agent(input_messages)

            # Verify result is ChatMessages
            self.assertIsNotNone(result)
            self.assertTrue(is_chat_messages(result))

            # Get the latest assistant message
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if last_message.get("role") == "assistant":
                    tool_calls = last_message.get("tool_calls", [])
                    if not tool_calls:
                        break
                    # Continue with the result as new input
                    input_messages = result
                else:
                    break
            else:
                break

        # Verify we completed all steps
        self.assertEqual(step, 2)  # 0, 1, 2 = 3 steps
