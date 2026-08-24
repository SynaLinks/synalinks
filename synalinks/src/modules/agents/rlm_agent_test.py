# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.modules.agents.rlm_agent import RecursiveLanguageModelAgent
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Program
from synalinks.src.sandboxes.mirage_sandbox import MirageSandbox
from synalinks.src.saving.object_registration import register_synalinks_serializable


class Query(DataModel):
    query: str


class Answer(DataModel):
    answer: str


@register_synalinks_serializable()
async def square(x: int) -> int:
    """Square an integer.

    Args:
        x (int): the integer to square.
    """
    return x * x


@register_synalinks_serializable()
async def cube(x: int) -> int:
    """Cube an integer.

    Args:
        x (int): the integer to cube.
    """
    return x * x * x


def _exec_tool_call(code, call_id="call_1"):
    """A litellm response where the LM calls `run_python_code` with the
    given `code`, the native tool-call transport the RLM uses each turn.
    """
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "run_python_code",
                                "arguments": json.dumps({"code": code}),
                            },
                        }
                    ],
                }
            }
        ]
    }


class RecursiveLanguageModelAgentTest(testing.TestCase):
    async def test_defaults_sub_lm_to_main_lm(self):
        language_model = LanguageModel(model="ollama/mistral")
        agent = RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
        )
        self.assertIs(agent.sub_language_model, language_model)
        self.assertEqual(agent.max_llm_calls, 50)

    async def test_separate_sub_lm_is_used(self):
        primary = LanguageModel(model="ollama/mistral")
        cheap = LanguageModel(model="ollama/llama3")
        agent = RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=primary,
            sub_language_model=cheap,
        )
        self.assertIs(agent.sub_language_model, cheap)
        self.assertIsNot(agent.sub_language_model, agent.language_model)

    async def test_reserved_tool_names_rejected(self):
        language_model = LanguageModel(model="ollama/mistral")

        @register_synalinks_serializable()
        async def llm_query(prompt: str) -> dict:
            """Reserved name.

            Args:
                prompt (str): the prompt.
            """
            return {}

        with self.assertRaises(ValueError):
            RecursiveLanguageModelAgent(
                data_model=Answer,
                language_model=language_model,
                tools=[Tool(llm_query)],
            )

    async def test_reserved_tool_names_rejected_when_non_recursive(self):
        """`llm_query` / `llm_query_batched` stay reserved with `recursive=False`."""
        language_model = LanguageModel(model="ollama/mistral")

        @register_synalinks_serializable()
        async def llm_query_batched(prompts: list[str]) -> dict:
            """Reserved name.

            Args:
                prompts (list[str]): the prompts.
            """
            return {}

        with self.assertRaises(ValueError):
            RecursiveLanguageModelAgent(
                data_model=Answer,
                language_model=language_model,
                recursive=False,
                tools=[Tool(llm_query_batched)],
            )

    async def test_user_tools_pass_through(self):
        language_model = LanguageModel(model="ollama/mistral")
        agent = RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            tools=[Tool(square)],
        )
        self.assertIn("square", agent.tools)
        # llm_query / llm_query_batched are built per-call, not stored
        # on `self.tools`.
        self.assertNotIn("llm_query", agent.tools)
        self.assertNotIn("llm_query_batched", agent.tools)

    async def test_tools_still_means_sandbox_tools(self):
        """Back-compat: `tools=` keeps its original meaning.

        Adding `native_tools=` must not silently move existing agents' tools
        out of the sandbox — `tools=` stays the sandbox half, and an agent
        that never mentions `native_tools` has none."""
        language_model = LanguageModel(model="ollama/mistral")
        agent = RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            tools=[Tool(square)],
        )
        self.assertIn("square", agent.tools)
        self.assertNotIn("square", agent.native_tools)
        self.assertEqual(agent.native_tools, {})

    async def test_tools_and_sandbox_tools_are_concatenated(self):
        language_model = LanguageModel(model="ollama/mistral")
        agent = RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            tools=[Tool(square)],
            sandbox_tools=[Tool(cube)],
        )
        self.assertIn("square", agent.tools)
        self.assertIn("cube", agent.tools)
        self.assertEqual(agent.native_tools, {})

    async def test_sandbox_tools_is_a_spelling_of_tools(self):
        language_model = LanguageModel(model="ollama/mistral")
        agent = RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            sandbox_tools=[Tool(square)],
        )
        self.assertIn("square", agent.tools)
        self.assertEqual(agent.native_tools, {})

    async def test_native_tools_are_kept_out_of_the_sandbox_set(self):
        """A native tool is callable directly, so it is not a sandbox tool.

        It must stay out of `self.tools`: that set is what gets bound into the
        sandbox namespace and advertised in the instructions as
        snippet-reachable.
        """
        language_model = LanguageModel(model="ollama/mistral")
        agent = RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            native_tools=[Tool(square)],
        )
        self.assertIn("square", agent.native_tools)
        self.assertNotIn("square", agent.tools)

    async def test_native_and_sandbox_tools_coexist(self):
        language_model = LanguageModel(model="ollama/mistral")
        agent = RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            sandbox_tools=[Tool(square)],
            native_tools=[Tool(cube)],
        )
        self.assertIn("square", agent.tools)
        self.assertNotIn("cube", agent.tools)
        self.assertIn("cube", agent.native_tools)

    async def test_same_name_in_both_halves_rejected(self):
        language_model = LanguageModel(model="ollama/mistral")
        with self.assertRaises(ValueError):
            RecursiveLanguageModelAgent(
                data_model=Answer,
                language_model=language_model,
                sandbox_tools=[Tool(square)],
                native_tools=[Tool(square)],
            )

    async def test_reserved_names_rejected_for_native_tools_too(self):
        language_model = LanguageModel(model="ollama/mistral")

        @register_synalinks_serializable()
        async def llm_query(prompt: str) -> dict:
            """Reserved name.

            Args:
                prompt (str): the prompt.
            """
            return {}

        with self.assertRaises(ValueError):
            RecursiveLanguageModelAgent(
                data_model=Answer,
                language_model=language_model,
                native_tools=[Tool(llm_query)],
            )

    @patch("litellm.acompletion")
    async def test_native_tool_is_dispatched_not_rejected(self, mock_completion):
        """The LM calls a `native_tools=` tool directly and gets its result.

        The same call against a sandbox tool comes back as `Unknown tool`,
        which is the whole point of the split.
        """
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            native_tools=[Tool(cube)],
            max_iterations=3,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        mock_completion.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "cube",
                                        "arguments": json.dumps({"x": 3}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            _exec_tool_call("submit(result={'answer': 'done'})", "call_2"),
        ]

        result = await agent(Query(query="hi"))
        self.assertEqual(result.get("answer"), "done")
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        self.assertTrue(
            any("27" in str(m.get("content")) for m in tool_messages),
            f"expected the native tool's result; got: {tool_messages}",
        )
        self.assertFalse(
            any("Unknown tool" in str(m.get("content")) for m in tool_messages),
            f"native tool was rejected: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_sandbox_tool_called_natively_is_still_rejected(self, mock_completion):
        """The other half of the contract: a sandbox tool stays snippet-only."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            sandbox_tools=[Tool(cube)],
            max_iterations=3,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        mock_completion.side_effect = [
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "cube",
                                        "arguments": json.dumps({"x": 3}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            _exec_tool_call("submit(result={'answer': 'done'})", "call_2"),
        ]

        result = await agent(Query(query="hi"))
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        self.assertTrue(
            any("Unknown tool" in str(m.get("content")) for m in tool_messages),
            f"expected the sandbox tool to be rejected natively; got: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_llm_query_visible_in_prompt(self, mock_completion):
        """llm_query and llm_query_batched are explained in the step
        generator's instructions, so they appear in its prompt."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=1,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        # Single empty turn, enough to inspect the prompt.
        mock_completion.side_effect = [
            _exec_tool_call(""),
            {"choices": [{"message": {"content": json.dumps({"answer": "x"})}}]},
        ]

        await agent(Query(query="hi"))

        first_prompt = json.dumps(
            mock_completion.call_args_list[0].kwargs.get("messages", [])
        )
        self.assertIn("llm_query", first_prompt)
        self.assertIn("llm_query_batched", first_prompt)

    @patch("litellm.acompletion")
    async def test_sandbox_functions_advertised_in_prompt(self, mock_completion):
        """Sandbox tools are advertised as Python `def` stubs in the
        prompt, never as a catalog in the trajectory.

        A JSON-schema catalog (name/description/parameters entries) reads
        like the provider's native tools array and lures the LM into
        emitting native tool calls for sandbox-only functions; the
        instructions present them as regular Python functions instead, and
        still carry the target output schema for `submit`, the only place
        the LM discovers the expected result shape.
        """
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            sandbox_tools=[Tool(square)],
            max_iterations=1,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        # Single empty turn, enough to inspect the prompt.
        mock_completion.side_effect = [
            _exec_tool_call(""),
            {"choices": [{"message": {"content": json.dumps({"answer": "x"})}}]},
        ]

        await agent(Query(query="hi"))

        first_prompt = json.dumps(
            mock_completion.call_args_list[0].kwargs.get("messages", [])
        )
        self.assertIn("def square(x: int) -> dict:", first_prompt)
        # The instructions still advertise the target output shape for
        # `submit`, as compact per-field lines rather than a raw schema dump.
        self.assertIn("`result` is a dict with these fields", first_prompt)
        self.assertIn("- answer (str)", first_prompt)
        # No catalog DataModel fields left in the trajectory's input turn.
        self.assertNotIn("'parameters'", first_prompt)
        self.assertNotIn("'functions':", first_prompt)

    async def test_sandbox_guidance_immune_to_instruction_optimization(self):
        """The sandbox-functions guidance rides on the step generator's
        `sandbox_functions` prompt variable (rendered by the agent prompt
        template), not on `instructions`: instructions are the generator's
        trainable state, so an in-context optimizer rewriting them during
        `fit()` could otherwise drop the `submit` schema and the tool stubs.
        """
        language_model = LanguageModel(model="ollama/mistral")
        agent = RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            sandbox_tools=[Tool(square)],
        )
        self.assertIn("{{ sandbox_functions }}", agent.prompt_template)
        for needle in ("`result` is a dict with these fields", "def square"):
            self.assertIn(needle, agent.prompt_variables["sandbox_functions"])
            self.assertNotIn(needle, agent.instructions)
            # The trainable variable itself must be clean too.
            self.assertNotIn(needle, agent.tool_calls_generator.state.get("instructions"))

    async def test_file_tools_advertised_only_with_workspace(self):
        """With a workspace (a `workdir`, or a supplied sandbox) the
        sandbox's file tools, DeepAgent's built-in set, are advertised as
        `def` stubs in the prompt; a workspace-less agent carries none."""
        language_model = LanguageModel(model="ollama/mistral")
        with tempfile.TemporaryDirectory() as workdir:
            agent = RecursiveLanguageModelAgent(
                data_model=Answer,
                language_model=language_model,
                workdir=workdir,
            )
            guidance = agent.prompt_variables["sandbox_functions"]
            for name in (
                "read_file",
                "list_files",
                "search_files",
                "write_file",
                "edit_file",
                "run_bash",
            ):
                self.assertIn(f"def {name}(", guidance)
                self.assertNotIn(f"def {name}(", agent.instructions)
        agent = RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
        )
        self.assertFalse(agent._file_tools_enabled)
        self.assertNotIn(
            "def read_file(", agent.prompt_variables.get("sandbox_functions", "")
        )

    @patch("litellm.acompletion")
    async def test_file_tools_callable_from_snippet(self, mock_completion):
        """A snippet calls a file tool directly and it operates on the
        call's sandbox filesystem, seeded from the workdir."""
        language_model = LanguageModel(model="ollama/mistral")
        with tempfile.TemporaryDirectory() as workdir:
            (Path(workdir) / "data.txt").write_text("magic-42\n")
            inputs = Input(data_model=Query)
            outputs = await RecursiveLanguageModelAgent(
                data_model=Answer,
                language_model=language_model,
                workdir=workdir,
                max_iterations=3,
            )(inputs)
            agent = Program(inputs=inputs, outputs=outputs)

            mock_completion.side_effect = [
                _exec_tool_call(
                    "out = read_file('/data.txt')\n"
                    "submit(result={'answer': out['content'].strip()})"
                ),
            ]

            result = await agent(Query(query="what is in data.txt?"))
            self.assertEqual(result.get("answer"), "magic-42")

    @patch("litellm.acompletion")
    async def test_llm_query_round_trip(self, mock_completion):
        """A snippet that calls `llm_query` triggers a sub-LM call and
        the response text is observable to the next turn."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=3,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        turn1 = {
            "code": ("out = llm_query(prompt='summarize this')\nprint(out['result'])")
        }
        turn2 = {"code": "submit(result={'answer': 'done'})"}

        # Order: code-generator (tool call), sub-LM (free-form),
        # code-generator (tool call).
        mock_completion.side_effect = [
            _exec_tool_call(turn1["code"], "call_1"),
            {"choices": [{"message": {"content": "the gist is X"}}]},
            _exec_tool_call(turn2["code"], "call_2"),
        ]

        result = await agent(Query(query="long doc here"))
        self.assertEqual(result.get("answer"), "done")
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        self.assertTrue(
            any("the gist is X" in str(m.get("content")) for m in tool_messages),
            f"expected sub-LM response in tool observation; got: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_llm_query_round_trip_positional_call(self, mock_completion):
        """`llm_query(prompt)` called positionally — the shape the built-in
        instructions themselves show (`llm_query(prompt)`, no `prompt=`) —
        must work exactly like the keyword form. Regression test: the
        sandbox's tool-call RPC bridge (bootstrap stub -> Unix socket -> host
        dispatcher -> `_adapt_tool_for_sandbox`) used to be keyword-only at
        every hop, so this raised `TypeError: _stub() takes 0 positional
        arguments but 1 was given` even though `Tool.__call__` itself already
        supports `*args`."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=3,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        turn1 = {"code": ("out = llm_query('summarize this')\nprint(out['result'])")}
        turn2 = {"code": "submit(result={'answer': 'done'})"}

        mock_completion.side_effect = [
            _exec_tool_call(turn1["code"], "call_1"),
            {"choices": [{"message": {"content": "the gist is X"}}]},
            _exec_tool_call(turn2["code"], "call_2"),
        ]

        result = await agent(Query(query="long doc here"))
        self.assertEqual(result.get("answer"), "done")
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        self.assertTrue(
            any("the gist is X" in str(m.get("content")) for m in tool_messages),
            f"expected sub-LM response in tool observation; got: {tool_messages}",
        )

    @patch("litellm.acompletion")
    async def test_user_tool_accepts_positional_call(self, mock_completion):
        """A plain user tool (not a recursive helper) called positionally
        through the sandbox — same RPC bridge, same regression."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            tools=[Tool(square)],
            max_iterations=2,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        turn1 = {"code": "out = square(5)\nsubmit(result={'answer': str(out['result'])})"}
        mock_completion.side_effect = [_exec_tool_call(turn1["code"], "call_1")]

        result = await agent(Query(query="square 5"))
        self.assertEqual(result.get("answer"), "25")

    @patch("litellm.acompletion")
    async def test_llm_query_batched_runs_concurrently(self, mock_completion):
        """`llm_query_batched` returns one response per prompt."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=3,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        turn1 = {
            "code": (
                "out = llm_query_batched(prompts=['a', 'b', 'c'])\nprint(out['result'])"
            )
        }
        turn2 = {"code": "submit(result={'answer': 'merged'})"}

        mock_completion.side_effect = [
            _exec_tool_call(turn1["code"], "call_1"),
            {"choices": [{"message": {"content": "ans-A"}}]},
            {"choices": [{"message": {"content": "ans-B"}}]},
            {"choices": [{"message": {"content": "ans-C"}}]},
            _exec_tool_call(turn2["code"], "call_2"),
        ]

        result = await agent(Query(query="batch"))
        self.assertEqual(result.get("answer"), "merged")
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        # FC tool results carry structured dict content ({"stdout", ...}); the
        # printed sub-LM output lives inside `stdout`, so stringify to search.
        joined = "\n".join(str(m.get("content")) for m in tool_messages)
        self.assertIn("ans-A", joined)
        self.assertIn("ans-B", joined)
        self.assertIn("ans-C", joined)

    @patch("litellm.acompletion")
    async def test_llm_query_budget_enforced(self, mock_completion):
        """Beyond `max_llm_calls`, llm_query returns an error string and
        does NOT call the sub-LM."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=3,
            max_llm_calls=1,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        turn1 = {
            "code": (
                "a = llm_query(prompt='first')\n"
                "b = llm_query(prompt='second')\n"
                "print(a, b)"
            )
        }
        turn2 = {"code": ("submit(result={'answer': 'capped'})")}

        mock_completion.side_effect = [
            _exec_tool_call(turn1["code"], "call_1"),
            # Only ONE sub-LM call should fire; the second is rejected.
            {"choices": [{"message": {"content": "first response"}}]},
            _exec_tool_call(turn2["code"], "call_2"),
        ]

        result = await agent(Query(query="overrun"))
        self.assertEqual(result.get("answer"), "capped")
        # Three completions total: 2 code-generator + 1 sub-LM.
        self.assertEqual(mock_completion.call_count, 3)
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        # FC tool results carry structured dict content ({"stdout", ...}); the
        # printed sub-LM output lives inside `stdout`, so stringify to search.
        joined = "\n".join(str(m.get("content")) for m in tool_messages)
        self.assertIn("budget exhausted", joined)

    @patch("litellm.acompletion")
    async def test_quota_resets_per_call(self, mock_completion):
        """A second call of the same agent gets a fresh sub-LM budget."""
        language_model = LanguageModel(model="ollama/mistral")

        agent_module = RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=2,
            max_llm_calls=1,
        )

        inputs = Input(data_model=Query)
        outputs = await agent_module(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        # Each agent call: one llm_query, then submit.
        per_call_turn1 = {"code": "out = llm_query(prompt='x')\nprint(out)"}
        per_call_turn2 = {"code": "submit(result={'answer': 'k'})"}

        mock_completion.side_effect = [
            # Call 1
            _exec_tool_call(per_call_turn1["code"], "call_1"),
            {"choices": [{"message": {"content": "first"}}]},
            _exec_tool_call(per_call_turn2["code"], "call_2"),
            # Call 2: quota should be reset, so the sub-LM fires again.
            _exec_tool_call(per_call_turn1["code"], "call_3"),
            {"choices": [{"message": {"content": "second"}}]},
            _exec_tool_call(per_call_turn2["code"], "call_4"),
        ]

        await agent(Query(query="round 1"))
        result_2 = await agent(Query(query="round 2"))
        # Six completions total: 2 calls × (2 code-gen + 1 sub-LM).
        self.assertEqual(mock_completion.call_count, 6)
        tool_messages = [m for m in result_2.get("messages") if m.get("role") == "tool"]
        # FC tool results carry structured dict content ({"stdout", ...}); the
        # printed sub-LM output lives inside `stdout`, so stringify to search.
        joined = "\n".join(str(m.get("content")) for m in tool_messages)
        # The second run's sub-LM call returned "second"; budget did NOT
        # leak from run 1 (otherwise we'd see "budget exhausted").
        self.assertIn("second", joined)
        self.assertNotIn("budget exhausted", joined)

    @patch("litellm.acompletion")
    async def test_schemaless_run_returns_trajectory(self, mock_completion):
        """Schemaless run: data_model=None => agent returns a trajectory of
        ChatMessages and never produces a structured `answer` field.
        Sub-LM calls (`llm_query`) still work in this mode."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await RecursiveLanguageModelAgent(
            language_model=language_model,
            max_iterations=2,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs, name="schemaless_recursive")

        turn1 = {
            "code": (
                "out = llm_query(prompt='gist?')\n"
                "submit(result={'answer': out['result']})"
            )
        }

        mock_completion.side_effect = [
            _exec_tool_call(turn1["code"], "call_1"),
            {"choices": [{"message": {"content": "the gist"}}]},
        ]

        result = await agent(Query(query="hi"))
        result_json = result.get_json()
        self.assertIn("messages", result_json)
        self.assertNotIn("answer", result_json)
        # Schemaless submit takes {"answer": "..."}; the answer string lands as
        # the content of the final assistant message.
        assistant_msgs = [
            m for m in result.get("messages") if m.get("role") == "assistant"
        ]
        self.assertTrue(
            any(m.get("content") == "the gist" for m in assistant_msgs),
            f"submitted answer not appended as message content; got: {assistant_msgs}",
        )

    async def test_config_round_trip(self):
        primary = LanguageModel(model="ollama/mistral")
        cheap = LanguageModel(model="ollama/llama3")
        agent = RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=primary,
            sub_language_model=cheap,
            max_llm_calls=7,
        )
        config = agent.get_config()
        self.assertEqual(config["max_llm_calls"], 7)
        self.assertIn("sub_language_model", config)
        restored = RecursiveLanguageModelAgent.from_config(config)
        self.assertEqual(restored.max_llm_calls, 7)
        self.assertEqual(restored.sub_language_model.model, "ollama_chat/llama3")


class RLMSubagentTest(testing.TestCase):
    """REPL-aware subagents: fork (REPL+files), parallel, parent-reviewed merge."""

    def _lm(self):
        return LanguageModel(model="ollama/mistral")

    def _agent(self, **kw):
        return RecursiveLanguageModelAgent(language_model=self._lm(), name="r", **kw)

    # -- wiring (no LM) ------------------------------------------------------

    async def test_subagents_off_by_default(self):
        self.assertFalse(self._agent()._subagents_enabled)

    async def test_subagents_enabled_flag_and_guidance(self):
        agent = self._agent(max_subagent_depth=1)
        self.assertTrue(agent._subagents_enabled)
        self.assertIn("delegate to parallel subagents", agent.instructions)

    async def test_subagent_at_max_depth_disabled(self):
        sub = self._agent(max_subagent_depth=1, _subagent_depth=1)
        self.assertFalse(sub._subagents_enabled)

    async def test_negative_depth_rejected(self):
        with self.assertRaises(ValueError):
            self._agent(max_subagent_depth=-1)

    async def test_subagent_tool_names_reserved(self):
        with self.assertRaises(ValueError):
            self._agent(tools=[Tool(square, name="spawn_subagents")])

    async def test_get_config_round_trips_max_subagent_depth(self):
        agent = self._agent(max_subagent_depth=2)
        config = agent.get_config()
        self.assertEqual(config["max_subagent_depth"], 2)
        restored = RecursiveLanguageModelAgent.from_config(config)
        self.assertEqual(restored.max_subagent_depth, 2)
        self.assertTrue(restored._subagents_enabled)
        # Guidance is appended idempotently, not doubled on round-trip.
        self.assertEqual(restored.instructions.count("delegate to parallel subagents"), 1)

    # -- merge / discard handlers (no LM, manual fork) -----------------------

    async def test_merge_subagent_files_and_repl(self):
        agent = self._agent(max_subagent_depth=1)
        sandbox = MirageSandbox()
        await sandbox.run("x = 1")
        await sandbox.write_file("/base.txt", "base")
        registry = {}
        tools = agent._build_subagent_tools(sandbox, registry, [0], {"adopted": False})

        # Stand in for a finished subagent: a fork that changed REPL + files.
        fork = sandbox.fork(copy_repl=True)
        await fork.run("y = 99")
        await fork.write_file("/new.txt", "child")
        registry["subagent_0"] = fork

        out = (
            await tools["merge_subagent"](handle="subagent_0", adopt_repl=True)
        ).get_json()
        self.assertTrue(out["repl_adopted"])
        self.assertIn("/new.txt", out["written"])
        # Files merged...
        self.assertEqual((await sandbox.read_file("/new.txt"))["content"], "child")
        # ...and the subagent's REPL var, alongside the parent's own.
        self.assertIn("99", (await sandbox.run("print(y)")).stdout)
        self.assertIn("1", (await sandbox.run("print(x)")).stdout)

    async def test_merge_subagent_files_only_leaves_repl(self):
        agent = self._agent(max_subagent_depth=1)
        sandbox = MirageSandbox()
        await sandbox.run("x = 1")
        registry = {}
        tools = agent._build_subagent_tools(sandbox, registry, [0], {"adopted": False})
        fork = sandbox.fork(copy_repl=True)
        await fork.run("x = 999")
        await fork.write_file("/f.txt", "child")
        registry["subagent_0"] = fork

        out = (await tools["merge_subagent"](handle="subagent_0")).get_json()
        self.assertFalse(out["repl_adopted"])
        self.assertIn("/f.txt", out["written"])
        # REPL untouched (no adoption).
        self.assertIn("1", (await sandbox.run("print(x)")).stdout)

    async def test_only_one_repl_adoption_per_turn(self):
        agent = self._agent(max_subagent_depth=1)
        sandbox = MirageSandbox()
        registry = {}
        repl_state = {"adopted": False}
        tools = agent._build_subagent_tools(sandbox, registry, [0], repl_state)
        fa = sandbox.fork(copy_repl=True)
        await fa.run("a = 1")
        registry["subagent_0"] = fa
        fb = sandbox.fork(copy_repl=True)
        await fb.run("b = 2")
        registry["subagent_1"] = fb

        r1 = (
            await tools["merge_subagent"](handle="subagent_0", adopt_repl=True)
        ).get_json()
        self.assertTrue(r1["repl_adopted"])
        r2 = (
            await tools["merge_subagent"](handle="subagent_1", adopt_repl=True)
        ).get_json()
        self.assertFalse(r2["repl_adopted"])
        self.assertIn("repl_warning", r2)
        # First adoption's var is present; the second's is not (REPL-wise).
        self.assertIn("1", (await sandbox.run("print(a)")).stdout)
        self.assertFalse((await sandbox.run("print(b)")).ok)

    async def test_merge_and_discard_unknown_handle(self):
        agent = self._agent(max_subagent_depth=1)
        sandbox = MirageSandbox()
        tools = agent._build_subagent_tools(sandbox, {}, [0], {"adopted": False})
        self.assertIn("error", (await tools["merge_subagent"](handle="nope")).get_json())
        self.assertIn(
            "error", (await tools["discard_subagent"](handle="nope")).get_json()
        )

    async def test_discard_subagent_drops_fork(self):
        agent = self._agent(max_subagent_depth=1)
        sandbox = MirageSandbox()
        registry = {"subagent_0": sandbox.fork(copy_repl=True)}
        tools = agent._build_subagent_tools(sandbox, registry, [0], {"adopted": False})
        out = (await tools["discard_subagent"](handle="subagent_0")).get_json()
        self.assertEqual(out, {"discarded": "subagent_0"})
        self.assertNotIn("subagent_0", registry)

    # -- spawn end-to-end (mocked LM) ----------------------------------------

    @patch("litellm.acompletion")
    async def test_spawn_runs_subagent_on_isolated_fork(self, mock_completion):
        # The subagent's single snippet sets a REPL var and submits. (Its
        # interpreter is an isolated subprocess, so it folds REPL state back to
        # the parent on merge, not in-snippet host file writes.)
        snippet = "subvar = 7\nsubmit(result={'answer': 'computed subvar'})\n"
        mock_completion.side_effect = lambda *a, **k: _exec_tool_call(snippet)

        agent = self._agent(max_subagent_depth=1)
        sandbox = MirageSandbox()
        await sandbox.run("parentvar = 1")
        registry = {}
        tools = agent._build_subagent_tools(sandbox, registry, [0], {"adopted": False})

        out = (await tools["spawn_subagents"](tasks=["do the thing"])).get_json()
        subs = out["subagents"]
        self.assertEqual(len(subs), 1)
        sub = subs[0]
        self.assertEqual(sub["handle"], "subagent_0")
        self.assertEqual(sub["result"], "computed subvar")
        # Parent REPL is untouched until merge: `subvar` is not defined there.
        self.assertFalse((await sandbox.run("print(subvar)")).ok)
        # The fork carries the subagent's REPL var.
        fork = registry["subagent_0"]
        self.assertIn("7", (await fork.run("print(subvar)")).stdout)
