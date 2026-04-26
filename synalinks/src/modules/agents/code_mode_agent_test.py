# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules.agents.code_mode_agent import CodeModeAgent
from synalinks.src.modules.agents.code_mode_agent import CodeStep
from synalinks.src.modules.agents.code_mode_agent import IterationInfo
from synalinks.src.modules.agents.code_mode_agent import _summarize_inputs
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.tool import Tool
from synalinks.src.programs import Program
from synalinks.src.sandboxes.monty_sandbox import MontySandbox
from synalinks.src.saving.object_registration import register_synalinks_serializable


class Query(DataModel):
    query: str


class Answer(DataModel):
    answer: str


@register_synalinks_serializable()
async def triple(x: int) -> int:
    """Triple an integer.

    Args:
        x (int): the integer to triple.
    """
    return x * 3


class CodeModeAgentTest(testing.TestCase):
    async def test_agent_instantiation_without_tools(self):
        language_model = LanguageModel(model="ollama/mistral")
        agent = CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=3,
        )
        self.assertEqual(agent.tools, {})
        self.assertEqual(agent.max_iterations, 3)

    async def test_agent_instantiation_with_tools(self):
        language_model = LanguageModel(model="ollama/mistral")
        agent = CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            tools=[Tool(triple)],
        )
        self.assertIn("triple", agent.tools)

    async def test_schema_is_optional(self):
        """No schema / data_model => schemaless mode, matching
        FunctionCallingAgent. The agent should instantiate cleanly."""
        language_model = LanguageModel(model="ollama/mistral")
        agent = CodeModeAgent(language_model=language_model)
        self.assertIsNone(agent.schema)

    @patch("litellm.acompletion")
    async def test_schemaless_autonomous_returns_trajectory(self, mock_completion):
        """Schemaless autonomous run with submit appends the submitted
        payload as an assistant ChatMessage and returns ChatMessages (no
        structured answer)."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await CodeModeAgent(
            language_model=language_model,
            max_iterations=2,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs, name="schemaless_agent")

        turn1 = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'note': 'all done'})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
        ]

        result = await agent(Query(query="hi"))
        # Output is a trajectory — has `messages`, no `answer` field.
        self.assertIn("messages", result.get_json())
        self.assertNotIn("answer", result.get_json())
        messages = result.get("messages")
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        self.assertTrue(
            any(
                isinstance(m.get("content"), dict)
                and m["content"].get("note") == "all done"
                for m in assistant_msgs
            ),
            f"submitted payload was not appended; got: {assistant_msgs}",
        )

    async def test_code_step_schema_has_code_field(self):
        schema = CodeStep.get_schema()
        self.assertIn("python_code", schema["properties"])

    @patch("litellm.acompletion")
    async def test_autonomous_flow_computes_answer_via_code(self, mock_completion):
        """LM emits code, sandbox executes, next turn submits the answer."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=3,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs, name="code_agent")

        # Turn 1: compute something in the REPL, print the result
        turn1 = {"python_code": 'answer = inputs.get("query").upper()\nprint(answer)'}
        # Turn 2: submit the structured answer
        turn2 = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': answer})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
            {"choices": [{"message": {"content": json.dumps(turn2)}}]},
        ]

        result = await agent(Query(query="hello"))
        self.assertEqual(result.get("answer"), "HELLO")
        # Trajectory includes the assistant code message and the tool observation
        messages = result.get("messages")
        self.assertGreaterEqual(len(messages), 2)
        # Observation from the executed code should contain stdout
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        self.assertTrue(
            any("HELLO" in (m.get("content") or "") for m in tool_messages),
            f"expected stdout 'HELLO' in tool messages, got: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_async_tool_invoked_from_sandbox(self, mock_completion):
        """Bound tool is callable from the generated script via await."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            tools=[Tool(triple)],
            max_iterations=3,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs, name="code_agent_tools")

        turn1 = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    return (await triple(x=7))['result']\n"
                "tripled = asyncio.run(main())\n"
                "print(tripled)"
            )
        }
        turn2 = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': str(tripled)})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
            {"choices": [{"message": {"content": json.dumps(turn2)}}]},
        ]

        result = await agent(Query(query="triple seven"))
        self.assertEqual(result.get("answer"), "21")
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        self.assertTrue(
            any("21" in (m.get("content") or "") for m in tool_messages),
            f"expected '21' in tool observation, got: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_repl_state_persists_across_turns(self, mock_completion):
        """Variable defined in turn 1 is still bound in turn 2."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=5,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        turn1 = {"python_code": "x = 100"}
        turn2 = {"python_code": "print(x + 1)"}
        turn3 = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': '101'})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
            {"choices": [{"message": {"content": json.dumps(turn2)}}]},
            {"choices": [{"message": {"content": json.dumps(turn3)}}]},
        ]

        result = await agent(Query(query="persistence"))
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        self.assertTrue(
            any("101" in (m.get("content") or "") for m in tool_messages),
            f"state did not persist; got: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_runtime_error_turns_into_observation(self, mock_completion):
        """A sandbox error is fed back as an observation, not raised."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=3,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        turn1 = {"python_code": "1 / 0"}
        turn2 = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': 'recovered'})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
            {"choices": [{"message": {"content": json.dumps(turn2)}}]},
        ]

        result = await agent(Query(query="divide by zero"))
        self.assertEqual(result.get("answer"), "recovered")
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        self.assertTrue(
            any("ZeroDivisionError" in (m.get("content") or "") for m in tool_messages),
            f"expected ZeroDivisionError observation, got: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_max_iterations_forces_final_answer(self, mock_completion):
        """Even if the LM never emits empty code, max_iterations caps the loop."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=2,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        # The LM keeps emitting non-empty code; we only supply 2 turns + final.
        turn = {"python_code": 'print("still going")'}
        final = {"answer": "capped"}

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn)}}]},
            {"choices": [{"message": {"content": json.dumps(turn)}}]},
            {"choices": [{"message": {"content": json.dumps(final)}}]},
        ]

        result = await agent(Query(query="loop"))
        self.assertEqual(result.get("answer"), "capped")

    async def test_chat_messages_passthrough(self):
        """If inputs is already a ChatMessages, no wrapping concat happens."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=ChatMessages)
        outputs = await CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=1,
        )(inputs)
        # Just checking the graph builds and accepts ChatMessages inputs
        self.assertIsNotNone(outputs)
        _ = ChatMessage(role="user", content="hi")  # noqa: F841

    async def test_interactive_mode_requires_chat_messages(self):
        """Interactive mode rejects non-ChatMessages inputs."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        with self.assertRaises(ValueError):
            await CodeModeAgent(
                data_model=Answer,
                language_model=language_model,
                autonomous=False,
            )(inputs)

    @patch("litellm.acompletion")
    async def test_interactive_mode_runs_single_turn(self, mock_completion):
        """autonomous=False runs exactly one code turn per call."""
        language_model = LanguageModel(model="ollama/mistral")

        agent = CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            autonomous=False,
            max_iterations=10,
        )

        user_msg = ChatMessage(role="user", content="hello").get_json()
        trajectory = ChatMessages(messages=[ChatMessage(**user_msg)])

        # Only one LM call should happen — one code step, no final generator
        mock_completion.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({"python_code": "x = 42\nprint(x)"})
                        }
                    }
                ]
            },
        ]

        result = await agent(trajectory)

        # Exactly one LM call (the code generator), no final generator call
        self.assertEqual(mock_completion.call_count, 1)
        # Output is the extended trajectory: user + assistant + tool observation
        messages = result.get("messages")
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[2]["role"], "tool")
        self.assertIn("42", messages[2]["content"])

    @patch("litellm.acompletion")
    async def test_injected_sandbox_carries_state_across_calls(self, mock_completion):
        """A caller-owned `MontySandbox` persists REPL state between
        successive interactive calls of the same agent."""
        language_model = LanguageModel(model="ollama/mistral")

        agent = CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            autonomous=False,
        )
        sandbox = MontySandbox()

        turn1 = {"python_code": "x = 7"}
        turn2 = {"python_code": "print(x * 6)"}

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
            {"choices": [{"message": {"content": json.dumps(turn2)}}]},
        ]

        trajectory_1 = ChatMessages(
            messages=[ChatMessage(role="user", content="multiply seven")]
        )
        result_1 = await agent(trajectory_1, sandbox=sandbox)

        # Feed the extended trajectory back in along with the same sandbox.
        trajectory_2 = ChatMessages(
            messages=[ChatMessage(**m) for m in result_1.get("messages")]
        )
        result_2 = await agent(trajectory_2, sandbox=sandbox)

        tool_messages = [m for m in result_2.get("messages") if m.get("role") == "tool"]
        self.assertTrue(
            any("42" in (m.get("content") or "") for m in tool_messages),
            f"x did not persist across calls; got: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_default_sandbox_is_fresh_each_call(self, mock_completion):
        """When no sandbox is supplied, each call builds a fresh one —
        state from a prior call must not leak in."""
        language_model = LanguageModel(model="ollama/mistral")

        agent = CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            autonomous=False,
        )

        turn1 = {"python_code": "x = 7"}
        turn2 = {"python_code": "print(x)"}

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
            {"choices": [{"message": {"content": json.dumps(turn2)}}]},
        ]

        await agent(
            ChatMessages(messages=[ChatMessage(role="user", content="first call")])
        )
        result_2 = await agent(
            ChatMessages(messages=[ChatMessage(role="user", content="second call")])
        )

        tool_messages = [m for m in result_2.get("messages") if m.get("role") == "tool"]
        self.assertTrue(
            any("NameError" in (m.get("content") or "") for m in tool_messages),
            f"default behaviour leaked state across calls; got: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_injected_sandbox_round_trips_through_dump_and_load(
        self, mock_completion
    ):
        """Serializing a sandbox between calls preserves REPL state."""
        language_model = LanguageModel(model="ollama/mistral")

        agent = CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            autonomous=False,
        )

        turn1 = {"python_code": "state = {'counter': 10}"}
        turn2 = {"python_code": "print(state['counter'] * 3)"}

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
            {"choices": [{"message": {"content": json.dumps(turn2)}}]},
        ]

        sandbox = MontySandbox()
        result_1 = await agent(
            ChatMessages(messages=[ChatMessage(role="user", content="init")]),
            sandbox=sandbox,
        )

        # Persist and rebuild the sandbox between turns — simulates an
        # orchestrator that stores sandbox state alongside the trajectory.
        blob = sandbox.dump()
        rehydrated = MontySandbox.load(blob)

        trajectory_2 = ChatMessages(
            messages=[ChatMessage(**m) for m in result_1.get("messages")]
        )
        result_2 = await agent(trajectory_2, sandbox=rehydrated)

        tool_messages = [m for m in result_2.get("messages") if m.get("role") == "tool"]
        self.assertTrue(
            any("30" in (m.get("content") or "") for m in tool_messages),
            f"sandbox state did not survive dump/load; got: {tool_messages}",
        )

    async def test_autonomous_flag_in_config(self):
        """The autonomous flag is preserved through serialization."""
        language_model = LanguageModel(model="ollama/mistral")

        agent = CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            autonomous=False,
        )
        config = agent.get_config()
        self.assertFalse(config["autonomous"])
        restored = CodeModeAgent.from_config(config)
        self.assertFalse(restored.autonomous)

    async def test_inputs_summary_truncates_long_values(self):
        """_summarize_inputs keeps a head preview and flags truncation
        for values longer than the char/item budget."""
        long_string = "x" * 5_000
        summary = _summarize_inputs({"doc": long_string})
        field = summary.fields[0]
        self.assertEqual(field["name"], "doc")
        self.assertEqual(field["type"], "str")
        self.assertEqual(field["size"], 5_000)
        self.assertTrue(field["truncated"])
        # Preview is bounded — full value is NOT in the prompt-facing summary.
        self.assertLess(len(field["preview"]), 500)

    @patch("litellm.acompletion")
    async def test_long_input_not_materialised_in_prompt(self, mock_completion):
        """A long input field only appears as a preview in what the code
        generator sees; the full value stays in the sandbox."""
        language_model = LanguageModel(model="ollama/mistral")

        class LongInput(DataModel):
            document: str

        inputs = Input(data_model=LongInput)
        outputs = await CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=2,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        # Turn 1: read the real value from the sandbox, print its length.
        # This is the "full value in sandbox, not in prompt" round-trip.
        turn1 = {"python_code": "print(len(inputs['document']))"}
        turn2 = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': 'done'})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
            {"choices": [{"message": {"content": json.dumps(turn2)}}]},
        ]

        long_doc = "y" * 10_000
        result = await agent(LongInput(document=long_doc))
        # The sandbox saw the full 10_000-char value (inputs['document'])
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        self.assertTrue(
            any("10000" in (m.get("content") or "") for m in tool_messages),
            f"sandbox did not see full value; got: {tool_messages}",
        )

        # The prompt sent to the LM (mock_completion's call args) must NOT
        # contain the raw 10_000-char document — only the summary preview.
        seen_calls = mock_completion.call_args_list
        # First call is the code generator, which receives the summary
        first_prompt = json.dumps(seen_calls[0].kwargs.get("messages", []))
        self.assertNotIn(long_doc, first_prompt)
        # The summary's preview length (defaults to 200 chars) is well
        # under the full value, so only a short slice appears.
        self.assertLess(first_prompt.count("y"), 1_000)

    @patch("litellm.acompletion")
    async def test_iteration_counter_visible_to_generator(self, mock_completion):
        """Each code-generator call receives an IterationInfo with a
        formatted `<current>/<max>` field."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=3,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps({"python_code": "pass"})}}]},
            {"choices": [{"message": {"content": json.dumps({"python_code": "pass"})}}]},
            {"choices": [{"message": {"content": json.dumps({"python_code": ""})}}]},
            {"choices": [{"message": {"content": json.dumps({"answer": "ok"})}}]},
        ]

        await agent(Query(query="test"))

        # The first three LM calls are the code generator (one per turn).
        # Each should carry the matching iteration counter in its prompt.
        prompts = [
            json.dumps(c.kwargs.get("messages", []))
            for c in mock_completion.call_args_list[:3]
        ]
        self.assertIn("1/3", prompts[0])
        self.assertIn("2/3", prompts[1])
        self.assertIn("3/3", prompts[2])

    async def test_iteration_info_schema_has_iteration_field(self):
        schema = IterationInfo.get_schema()
        self.assertIn("iteration", schema["properties"])

    @patch("litellm.acompletion")
    async def test_submit_skips_final_generator_on_valid_payload(self, mock_completion):
        """A successful `submit` short-circuits the loop and bypasses the
        final-formatting LM call entirely — saves one round trip."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=3,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs, name="submit_agent")

        turn1 = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': 'FORTY_TWO'})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
        ]

        result = await agent(Query(query="what's the answer?"))
        self.assertEqual(result.get("answer"), "FORTY_TWO")
        # Only ONE LM call — the code generator. final_generator is skipped.
        self.assertEqual(mock_completion.call_count, 1)
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        self.assertTrue(
            any("submit accepted" in (m.get("content") or "") for m in tool_messages),
            f"expected submit acceptance observation; got: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_submit_invalid_payload_feeds_error_back(self, mock_completion):
        """An invalid `submit` payload surfaces as a validation-error
        observation; the LM retries on the next turn."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=3,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs, name="submit_retry")

        bad_turn = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'wrong_field': 'oops'})\n"
                "asyncio.run(main())"
            )
        }
        good_turn = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': 'recovered'})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(bad_turn)}}]},
            {"choices": [{"message": {"content": json.dumps(good_turn)}}]},
        ]

        result = await agent(Query(query="test"))
        self.assertEqual(result.get("answer"), "recovered")
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        contents = [m.get("content") or "" for m in tool_messages]
        self.assertTrue(
            any("submit validation failed" in c for c in contents),
            f"expected validation-failure observation; got: {contents}",
        )
        self.assertTrue(
            any("submit accepted" in c for c in contents),
            f"expected submit acceptance after retry; got: {contents}",
        )

    @patch("litellm.acompletion")
    async def test_submit_schemaless_appends_to_trajectory(self, mock_completion):
        """In schemaless mode, `submit(result=...)` ends the run and the
        payload is appended to the trajectory as an assistant
        ChatMessage. No final-formatting LM call happens."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await CodeModeAgent(
            language_model=language_model,
            max_iterations=3,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs, name="schemaless_submit")

        turn1 = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'note': 'hi there'})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
        ]

        result = await agent(Query(query="whatever"))
        # Only one LM call (the code generator). No final_generator call.
        self.assertEqual(mock_completion.call_count, 1)
        messages = result.get("messages")
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        # The last assistant message should carry the submitted payload as content.
        self.assertTrue(
            any(
                isinstance(m.get("content"), dict)
                and m["content"].get("note") == "hi there"
                for m in assistant_msgs
            ),
            f"expected submitted payload appended as assistant message; got: {assistant_msgs}",
        )

    async def test_sandbox_type_round_trips_through_config(self):
        """The sandbox_type class reference is preserved via the
        serializable-object registry."""
        language_model = LanguageModel(model="ollama/mistral")

        agent = CodeModeAgent(
            data_model=Answer,
            language_model=language_model,
            sandbox_type=MontySandbox,
        )
        config = agent.get_config()
        # Registered name may be package-qualified (e.g. "Custom>MontySandbox").
        self.assertIn("MontySandbox", config["sandbox_type"])
        restored = CodeModeAgent.from_config(config)
        self.assertIs(restored.sandbox_type, MontySandbox)
