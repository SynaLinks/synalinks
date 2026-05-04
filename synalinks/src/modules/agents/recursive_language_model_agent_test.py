# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.modules.agents.recursive_language_model_agent import (
    RecursiveLanguageModelAgent,
)
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Program
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
            {"choices": [{"message": {"content": json.dumps({"python_code": ""})}}]},
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
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    out = await llm_query(prompt='summarize this')\n"
                "    print(out['result'])\n"
                "asyncio.run(main())"
            )
        }
        turn2 = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': 'done'})\n"
                "asyncio.run(main())"
            )
        }

        # Order: code-generator (structured), sub-LM (free-form),
        # code-generator (structured).
        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
            {"choices": [{"message": {"content": "the gist is X"}}]},
            {"choices": [{"message": {"content": json.dumps(turn2)}}]},
        ]

        result = await agent(Query(query="long doc here"))
        self.assertEqual(result.get("answer"), "done")
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        self.assertTrue(
            any("the gist is X" in (m.get("content") or "") for m in tool_messages),
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
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    out = await llm_query_batched(prompts=['a', 'b', 'c'])\n"
                "    print(out['result'])\n"
                "asyncio.run(main())"
            )
        }
        turn2 = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': 'merged'})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
            {"choices": [{"message": {"content": "ans-A"}}]},
            {"choices": [{"message": {"content": "ans-B"}}]},
            {"choices": [{"message": {"content": "ans-C"}}]},
            {"choices": [{"message": {"content": json.dumps(turn2)}}]},
        ]

        result = await agent(Query(query="batch"))
        self.assertEqual(result.get("answer"), "merged")
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        joined = "\n".join(m.get("content") or "" for m in tool_messages)
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
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    a = await llm_query(prompt='first')\n"
                "    b = await llm_query(prompt='second')\n"
                "    print(a, b)\n"
                "asyncio.run(main())"
            )
        }
        turn2 = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': 'capped'})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
            # Only ONE sub-LM call should fire — the second is rejected.
            {"choices": [{"message": {"content": "first response"}}]},
            {"choices": [{"message": {"content": json.dumps(turn2)}}]},
        ]

        result = await agent(Query(query="overrun"))
        self.assertEqual(result.get("answer"), "capped")
        # Three completions total: 2 code-generator + 1 sub-LM.
        self.assertEqual(mock_completion.call_count, 3)
        tool_messages = [m for m in result.get("messages") if m.get("role") == "tool"]
        joined = "\n".join(m.get("content") or "" for m in tool_messages)
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
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    out = await llm_query(prompt='x')\n"
                "    print(out)\n"
                "asyncio.run(main())"
            )
        }
        per_call_turn2 = {
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    await submit(result={'answer': 'k'})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            # Call 1
            {"choices": [{"message": {"content": json.dumps(per_call_turn1)}}]},
            {"choices": [{"message": {"content": "first"}}]},
            {"choices": [{"message": {"content": json.dumps(per_call_turn2)}}]},
            # Call 2 — quota should be reset, so the sub-LM fires again.
            {"choices": [{"message": {"content": json.dumps(per_call_turn1)}}]},
            {"choices": [{"message": {"content": "second"}}]},
            {"choices": [{"message": {"content": json.dumps(per_call_turn2)}}]},
        ]

        await agent(Query(query="round 1"))
        result_2 = await agent(Query(query="round 2"))
        # Six completions total: 2 calls × (2 code-gen + 1 sub-LM).
        self.assertEqual(mock_completion.call_count, 6)
        tool_messages = [m for m in result_2.get("messages") if m.get("role") == "tool"]
        joined = "\n".join(m.get("content") or "" for m in tool_messages)
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
            "python_code": (
                "import asyncio\n"
                "async def main():\n"
                "    out = await llm_query(prompt='gist?')\n"
                "    await submit(result={'note': out['result']})\n"
                "asyncio.run(main())"
            )
        }

        mock_completion.side_effect = [
            {"choices": [{"message": {"content": json.dumps(turn1)}}]},
            {"choices": [{"message": {"content": "the gist"}}]},
        ]

        result = await agent(Query(query="hi"))
        result_json = result.get_json()
        self.assertIn("messages", result_json)
        self.assertNotIn("answer", result_json)
        assistant_msgs = [
            m for m in result.get("messages") if m.get("role") == "assistant"
        ]
        self.assertTrue(
            any(
                isinstance(m.get("content"), dict)
                and m["content"].get("note") == "the gist"
                for m in assistant_msgs
            ),
            f"submitted payload not appended; got: {assistant_msgs}",
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
