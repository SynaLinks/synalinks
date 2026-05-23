# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.modules.agents.rlm_agent import RecursiveLanguageModelAgent
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Program
from synalinks.src.sandboxes.monty_sandbox import MontySandbox
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


def _exec_tool_call(code, call_id="call_1"):
    """A litellm response where the LM calls `run_python_code` with the
    given `code` — the native tool-call transport the RLM uses each turn.
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

    @patch("litellm.acompletion")
    async def test_llm_query_visible_in_prompt_catalog(self, mock_completion):
        """llm_query and llm_query_batched appear in the per-turn catalog
        the code generator sees in its prompt."""
        language_model = LanguageModel(model="ollama/mistral")

        inputs = Input(data_model=Query)
        outputs = await RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=language_model,
            max_iterations=1,
        )(inputs)
        agent = Program(inputs=inputs, outputs=outputs)

        # Single empty turn — enough to inspect the prompt.
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
            "code": (
                "import asyncio\n"
                "async def main():\n"
                "    out = await llm_query(prompt='summarize this')\n"
                "    print(out['result'])\n"
                "asyncio.run(main())"
            )
        }
        turn2 = {
            "code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': 'done'})\n"
                "asyncio.run(main())"
            )
        }

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
                "import asyncio\n"
                "async def main():\n"
                "    out = await llm_query_batched(prompts=['a', 'b', 'c'])\n"
                "    print(out['result'])\n"
                "asyncio.run(main())"
            )
        }
        turn2 = {
            "code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': 'merged'})\n"
                "asyncio.run(main())"
            )
        }

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
                "import asyncio\n"
                "async def main():\n"
                "    a = await llm_query(prompt='first')\n"
                "    b = await llm_query(prompt='second')\n"
                "    print(a, b)\n"
                "asyncio.run(main())"
            )
        }
        turn2 = {
            "code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': 'capped'})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            _exec_tool_call(turn1["code"], "call_1"),
            # Only ONE sub-LM call should fire — the second is rejected.
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
        per_call_turn1 = {
            "code": (
                "import asyncio\n"
                "async def main():\n"
                "    out = await llm_query(prompt='x')\n"
                "    print(out)\n"
                "asyncio.run(main())"
            )
        }
        per_call_turn2 = {
            "code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': 'k'})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            # Call 1
            _exec_tool_call(per_call_turn1["code"], "call_1"),
            {"choices": [{"message": {"content": "first"}}]},
            _exec_tool_call(per_call_turn2["code"], "call_2"),
            # Call 2 — quota should be reset, so the sub-LM fires again.
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
                "import asyncio\n"
                "async def main():\n"
                "    out = await llm_query(prompt='gist?')\n"
                "    await submit(result={'answer': out['result']})\n"
                "asyncio.run(main())"
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
        # Guidance is appended idempotently — not doubled on round-trip.
        self.assertEqual(restored.instructions.count("delegate to parallel subagents"), 1)

    # -- merge / discard handlers (no LM, manual fork) -----------------------

    async def test_merge_subagent_files_and_repl(self):
        agent = self._agent(max_subagent_depth=1)
        sandbox = MontySandbox()
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
        sandbox = MontySandbox()
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
        sandbox = MontySandbox()
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
        sandbox = MontySandbox()
        tools = agent._build_subagent_tools(sandbox, {}, [0], {"adopted": False})
        self.assertIn("error", (await tools["merge_subagent"](handle="nope")).get_json())
        self.assertIn(
            "error", (await tools["discard_subagent"](handle="nope")).get_json()
        )

    async def test_discard_subagent_drops_fork(self):
        agent = self._agent(max_subagent_depth=1)
        sandbox = MontySandbox()
        registry = {"subagent_0": sandbox.fork(copy_repl=True)}
        tools = agent._build_subagent_tools(sandbox, registry, [0], {"adopted": False})
        out = (await tools["discard_subagent"](handle="subagent_0")).get_json()
        self.assertEqual(out, {"discarded": "subagent_0"})
        self.assertNotIn("subagent_0", registry)

    # -- spawn end-to-end (mocked LM) ----------------------------------------

    @patch("litellm.acompletion")
    async def test_spawn_runs_subagent_on_isolated_fork(self, mock_completion):
        # The subagent's single snippet sets a REPL var, writes a file, submits.
        snippet = (
            "import asyncio\n"
            "import pathlib\n"
            "subvar = 7\n"
            "pathlib.Path('/sub.txt').write_text('hi from sub')\n"
            "async def main():\n"
            "    await submit(result={'answer': 'computed subvar'})\n"
            "asyncio.run(main())\n"
        )
        mock_completion.side_effect = lambda *a, **k: _exec_tool_call(snippet)

        agent = self._agent(max_subagent_depth=1)
        sandbox = MontySandbox()
        await sandbox.run("parentvar = 1")
        registry = {}
        tools = agent._build_subagent_tools(sandbox, registry, [0], {"adopted": False})

        out = (await tools["spawn_subagents"](tasks=["do the thing"])).get_json()
        subs = out["subagents"]
        self.assertEqual(len(subs), 1)
        sub = subs[0]
        self.assertEqual(sub["handle"], "subagent_0")
        self.assertEqual(sub["result"], "computed subvar")
        self.assertIn("/sub.txt", [w["path"] for w in sub["diff"]["written"]])
        # Parent sandbox is untouched until merge.
        self.assertIn("error", await sandbox.read_file("/sub.txt"))
        self.assertFalse((await sandbox.run("print(subvar)")).ok)
        # The fork carries the subagent's REPL var and files.
        fork = registry["subagent_0"]
        self.assertIn("7", (await fork.run("print(subvar)")).stdout)
