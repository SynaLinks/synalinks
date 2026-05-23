# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import is_chat_messages
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Program
from synalinks.src.saving.object_registration import register_synalinks_serializable


def _lm_response(*, content=None, reasoning_content=None, tool_calls=None):
    """Build a litellm-shaped response for the native function-calling path.

    `tool_calls` is a list of `{"name": ..., "arguments": {...}}` (flat
    synalinks shape); this helper wraps each into the OpenAI nested
    `{id, type, function}` envelope and JSON-encodes the arguments.
    """
    message = {"content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]),
                },
            }
            for i, tc in enumerate(tool_calls)
        ]
    return {"choices": [{"message": message}]}


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
            use_chain_of_thought=True,
            name="default_test",
        )

        # FunctionCallingAgent should default to None (no reasoning)
        # But its ChainOfThought generators will default to "low"
        self.assertIsNone(agent.reasoning_effort)
        # The ChainOfThought inside defaults None to "low"
        self.assertEqual(agent.tool_calls_generator.reasoning_effort, "low")

    @patch("litellm.acompletion")
    async def test_autonomous_mode_simple_calculation(self, mock_completion):
        """Autonomous mode issues a native tool call, runs it, then finalizes."""
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

        mock_completion.side_effect = [
            # Iter 1: native tool call to `calculate`.
            _lm_response(
                content="Adding the two numbers.",
                tool_calls=[
                    {"name": "calculate", "arguments": {"expression": "152648 + 485"}}
                ],
            ),
            # Iter 2: no tool calls -> loop breaks.
            _lm_response(content="The calculation is done; the result is 153133."),
            # final_generator turn.
            _lm_response(content="152648 + 485 = 153133."),
        ]

        input_messages = ChatMessages(
            messages=[
                ChatMessage(
                    role="user",
                    content="How much is 152648 + 485?",
                ),
            ]
        )
        result = await agent(input_messages)

        self.assertIsNotNone(result)
        self.assertTrue(is_chat_messages(result))
        messages = result.get("messages", [])
        roles = [m.get("role") for m in messages]
        # The tool actually executed: an assistant tool-call followed by a
        # tool-result message carrying `calculate`'s output (153133).
        self.assertIn("assistant", roles)
        self.assertIn("tool", roles)
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        self.assertTrue(
            any("153133" in json.dumps(m.get("content")) for m in tool_msgs),
            f"calculate result not found in tool messages: {tool_msgs}",
        )

    @patch("litellm.acompletion")
    async def test_autonomous_mode_complex_calculation(self, mock_completion):
        """Autonomous mode runs a multi-step expression through `calculate`."""
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

        # (150 + 250) * 2 / 4 + 100 == 300
        mock_completion.side_effect = [
            _lm_response(
                content="Evaluating the full expression in one shot.",
                tool_calls=[
                    {
                        "name": "calculate",
                        "arguments": {"expression": "(150 + 250) * 2 / 4 + 100"},
                    }
                ],
            ),
            _lm_response(content="The expression evaluates to 300."),
            _lm_response(content="The result is 300."),
        ]

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

        self.assertIsNotNone(result)
        self.assertTrue(is_chat_messages(result))
        messages = result.get("messages", [])
        self.assertIn("tool", [m.get("role") for m in messages])
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        self.assertTrue(
            any("300" in json.dumps(m.get("content")) for m in tool_msgs),
            f"calculate result not found in tool messages: {tool_msgs}",
        )

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

        mock_completion.side_effect = [
            # Iter 1: native tool call.
            _lm_response(
                content="I need to calculate 10 + 20.",
                tool_calls=[
                    {"name": "calculate", "arguments": {"expression": "10 + 20"}}
                ],
            ),
            # Iter 2: no tool calls → loop breaks.
            _lm_response(content="The calculation is complete. The result is 30."),
            # final_generator turn.
            _lm_response(content="The result is 30."),
        ]

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
    async def test_autonomous_mode_no_data_model_with_custom_input(self, mock_completion):
        """Autonomous agent with data_model=None and a non-ChatMessages input.

        Exercises the path where the final_generator has schema=None and the
        resulting ChatMessage is appended to the trajectory.
        """

        class Query(DataModel):
            query: str = Field(description="The user query")

        language_model = LanguageModel(model="ollama/mistral")
        tools = [Tool(calculate)]
        inputs = Input(data_model=Query)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            autonomous=True,
            max_iterations=3,
        )(inputs)
        agent = Program(
            inputs=inputs,
            outputs=outputs,
            name="autonomous_no_schema_custom_input_test",
        )

        mock_completion.side_effect = [
            # Iter 1: native tool call.
            _lm_response(
                content="I need to add 1 + 1.",
                tool_calls=[{"name": "calculate", "arguments": {"expression": "1 + 1"}}],
            ),
            # Iter 2: no tool calls → loop breaks.
            _lm_response(content="Result is 2."),
            # final_generator: thinking populated from reasoning_content.
            _lm_response(content="The answer is 2.", reasoning_content="1 + 1 equals 2."),
        ]

        result = await agent(Query(query="How much is 1 + 1?"))

        self.assertIsNotNone(result)
        self.assertTrue(is_chat_messages(result))
        messages = result.get("messages", [])
        # Final assistant message must come from the no-schema final_generator
        # and carry the thinking field populated from reasoning_content.
        last = messages[-1]
        self.assertEqual(last.get("role"), "assistant")
        self.assertEqual(last.get("content"), "The answer is 2.")
        self.assertEqual(last.get("thinking"), "1 + 1 equals 2.")

    @patch("litellm.acompletion")
    async def test_autonomous_mode_no_data_model_with_chain_of_thought(
        self, mock_completion
    ):
        """Autonomous agent with data_model=None and use_chain_of_thought=True.

        Exercises the CoT-based tool_calls_generator and the no-schema
        final_generator together.
        """
        language_model = LanguageModel(model="ollama/mistral")
        tools = [Tool(calculate)]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            autonomous=True,
            max_iterations=3,
            use_chain_of_thought=True,
        )(inputs)
        agent = Program(
            inputs=inputs,
            outputs=outputs,
            name="autonomous_no_schema_cot_test",
        )

        tool_calls_done = {
            "thinking": "Nothing to do.",
            "tool_calls": [],
        }
        final_message = {
            "choices": [
                {
                    "message": {
                        "content": "Done.",
                        "reasoning_content": "No work needed.",
                    }
                }
            ]
        }

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(tool_calls_done)}}]},
            final_message,
        ]

        input_messages = ChatMessages(messages=[ChatMessage(role="user", content="hi")])
        result = await agent(input_messages)

        self.assertIsNotNone(result)
        self.assertTrue(is_chat_messages(result))
        last = result.get("messages")[-1]
        self.assertEqual(last.get("role"), "assistant")
        self.assertEqual(last.get("content"), "Done.")
        self.assertEqual(last.get("thinking"), "No work needed.")

    @patch("litellm.acompletion")
    async def test_autonomous_mode_streaming_final_answer(self, mock_completion):
        """With streaming=True and no data_model, the agent returns a
        StreamingIterator from the final generator instead of a wrapped
        trajectory."""
        language_model = LanguageModel(model="ollama/mistral")
        tools = [Tool(calculate)]
        inputs = Input(data_model=ChatMessages)
        outputs = await FunctionCallingAgent(
            language_model=language_model,
            tools=tools,
            autonomous=True,
            max_iterations=2,
            streaming=True,
        )(inputs)
        agent = Program(
            inputs=inputs,
            outputs=outputs,
            name="autonomous_streaming_test",
        )

        # Tool-calling step: no tools to call, agent moves to final answer.
        no_more_tools = {"thinking": "done.", "tool_calls": []}
        # Final generator streams its response.
        stream_chunks = iter(
            [
                {"choices": [{"delta": {"content": "Hello "}}]},
                {"choices": [{"delta": {"content": "world."}}]},
            ]
        )

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(no_more_tools)}}]},
            stream_chunks,
        ]

        input_messages = ChatMessages(
            messages=[ChatMessage(role="user", content="say hi")]
        )
        stream = await agent(input_messages)

        # The result must be the StreamingIterator itself, not a wrapped
        # ChatMessages — the caller drains it.
        collected = ""
        async for chunk in stream:
            collected += chunk.get("content", "")
        self.assertEqual(collected, "Hello world.")

    @patch("litellm.acompletion")
    async def test_streaming_disabled_when_schema_provided(self, _mock_completion):
        """`streaming=True` alongside a structured `data_model` is silently
        downgraded, since structured output needs the full response."""

        class FinalAnswer(DataModel):
            answer: str = Field(description="The final answer")

        language_model = LanguageModel(model="ollama/mistral")
        agent = FunctionCallingAgent(
            language_model=language_model,
            tools=[Tool(calculate)],
            data_model=FinalAnswer,
            streaming=True,
            name="streaming_with_schema",
        )
        self.assertFalse(agent.streaming)
        self.assertFalse(agent.final_generator.streaming)

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

        mock_completion.side_effect = [
            _lm_response(
                content="I need to calculate 5 * 5.",
                tool_calls=[{"name": "calculate", "arguments": {"expression": "5 * 5"}}],
            ),
        ]

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

        # Second LLM response after tool execution (no more tool calls)
        tool_calls_response = {
            "thinking": "The calculation result is 25. Task complete.",
            "tool_calls": [],
        }

        # Final generator response (ChatMessage format since no schema)
        final_response = {
            "role": "assistant",
            "content": "The result of 5 * 5 is 25.",
        }

        mock_responses = [
            {"choices": [{"message": {"content": json.dumps(tool_calls_response)}}]},
            {"choices": [{"message": {"content": json.dumps(final_response)}}]},
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

        # Verify result contains the new messages only (tool result + final assistant)
        # With return_inputs_with_trajectory=False, only new messages are returned
        self.assertIsNotNone(result)
        self.assertTrue(is_chat_messages(result))

        # The result should contain only the new messages (tool result + final assistant)
        messages = result.get("messages", [])
        self.assertEqual(len(messages), 2)

        # First message should be the tool result
        self.assertEqual(messages[0].get("role"), "tool")
        self.assertEqual(messages[0].get("tool_call_id"), "test-tool-call-id")

        # Second message should be the final assistant response
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

        mock_completion.side_effect = [
            _lm_response(
                content="I need to calculate 3 + 3.",
                tool_calls=[{"name": "calculate", "arguments": {"expression": "3 + 3"}}],
            ),
        ]

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

        mock_completion.side_effect = [
            # Step 1: calculate 100 + 200
            _lm_response(
                content="First, I need to calculate 100 + 200.",
                tool_calls=[
                    {"name": "calculate", "arguments": {"expression": "100 + 200"}}
                ],
            ),
            # Step 2: multiply result by 3
            _lm_response(
                content="Now I need to multiply 300 by 3.",
                tool_calls=[
                    {"name": "calculate", "arguments": {"expression": "300 * 3"}}
                ],
            ),
            # Step 3: no more tool calls → loop terminates → final_generator.
            _lm_response(content="Calculation complete."),
            _lm_response(
                content=(
                    "The calculation is complete. 100 + 200 = 300, then 300 * 3 = 900."
                )
            ),
        ]

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
