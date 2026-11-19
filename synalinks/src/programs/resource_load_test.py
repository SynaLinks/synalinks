# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Load tests: what a program opens does not pile up when it runs many times.

Each scenario runs once to warm up (imports, caches, the first sandbox's
one-off costs), then many more times, and checks after the last run that no
sandbox is left live, that no thread or file descriptor was left behind, and
that the process's resident memory grew by no more than a small bound: one
sandbox per run never released (its mount, its serving thread and loop) would
show in all of them.
"""

import gc
import json
import os
import sys
import threading
import types
import unittest
from unittest.mock import patch

import numpy as np

from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import DataModel
from synalinks.src.modules.agents.deep_agent import DeepAgent
from synalinks.src.modules.agents.rlm_agent import RecursiveLanguageModelAgent
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Program
from synalinks.src.sandboxes.sandbox import Sandbox
from synalinks.src.utils import keras_tuner_utils

RUNS = 20
# Resident memory a scenario may add over its runs, after its warm-up run.
RSS_BOUND_MB = 64


def _rss_mb():
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("no VmRSS")


def _answer(content):
    return {"choices": [{"message": {"content": content}}]}


def _submit(answer):
    code = "submit(result={'answer': %r})" % answer
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
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


class Query(DataModel):
    query: str


class Answer(DataModel):
    answer: str


def _messages():
    return ChatMessages(messages=[ChatMessage(role=ChatRole.USER, content="hi")])


@unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc/self/status")
class ResourceLoadTest(testing.TestCase):
    async def _under_load(self, run):
        """Run `run` once to warm up, then `RUNS` times; check nothing piled up."""
        await run()
        gc.collect()
        live, rss = set(Sandbox.live_sandboxes), _rss_mb()
        threads, fds = threading.active_count(), len(os.listdir("/proc/self/fd"))
        for _ in range(RUNS):
            await run()
        gc.collect()
        self.assertEqual(set(Sandbox.live_sandboxes) - live, set())
        self.assertLessEqual(threading.active_count(), threads)
        self.assertLessEqual(len(os.listdir("/proc/self/fd")), fds)
        grown = _rss_mb() - rss
        self.assertLess(
            grown, RSS_BOUND_MB, f"resident memory grew {grown:.0f} MB over {RUNS} runs"
        )

    @patch("litellm.acompletion")
    async def test_tuner_trials_of_a_program_holding_a_sandbox(self, mock_completion):
        # Each trial builds a program whose agent builds its own sandbox; the
        # tuner closes the program when the trial ends. The programs are kept,
        # as a sweep keeps its grid's: the garbage collector cannot stand in
        # for the close.
        mock_completion.side_effect = lambda *a, **k: _answer("done")
        kept = []

        async def build(hp):
            inputs = Input(data_model=ChatMessages)
            outputs = await DeepAgent(
                language_model=LanguageModel(model="ollama/mistral")
            )(inputs)
            program = Program(inputs=inputs, outputs=outputs)
            program.compile()
            kept.append(program)
            return program

        tuner = types.SimpleNamespace(
            hypermodel=types.SimpleNamespace(build=build),
            oracle=types.SimpleNamespace(
                objective=types.SimpleNamespace(name="reward", direction="max")
            ),
        )
        x = np.array([_messages()], dtype="object")

        async def trial():
            await keras_tuner_utils._run_trial_async(
                tuner, types.SimpleNamespace(hyperparameters=None), x=x, verbose=0
            )

        await self._under_load(trial)

    @patch("litellm.acompletion")
    async def test_calls_of_an_agent_that_builds_a_sandbox_per_call(
        self, mock_completion
    ):
        mock_completion.side_effect = lambda *a, **k: _submit("done")
        inputs = Input(data_model=Query)
        outputs = await RecursiveLanguageModelAgent(
            data_model=Answer,
            language_model=LanguageModel(model="ollama/mistral"),
            max_iterations=1,
        )(inputs)
        program = Program(inputs=inputs, outputs=outputs)

        async def call():
            self.assertEqual((await program(Query(query="hi"))).get("answer"), "done")

        await self._under_load(call)
        await program.close()

    @patch("litellm.acompletion")
    async def test_subagents_spawned_over_and_over(self, mock_completion):
        mock_completion.side_effect = lambda *a, **k: _answer("done")
        agent = DeepAgent(
            language_model=LanguageModel(model="ollama/mistral"),
            max_subagent_depth=1,
            name="spawner",
        )

        async def spawn():
            report = await agent.spawn_subagents(["a", "b", "c"])
            self.assertEqual(len(report["subagents"]), 3)

        await self._under_load(spawn)
        await agent.close()
