# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
import os
import time
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import DataModel
from synalinks.src.datasets.trace_dataset import TraceDataset
from synalinks.src.hooks.recorder import Recorder
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs.program import Program


class Query(DataModel):
    query: str


class Answer(DataModel):
    answer: str


async def calculate(expression: str):
    """Calculate the result of a mathematical expression.

    Args:
        expression (str): The expression to calculate, such as '2 + 2'.
    """
    return {"result": 30.0}


def _response(content, reasoning_content=None, tool_calls=None):
    message = {"content": content}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
            for i, (name, arguments) in enumerate(tool_calls)
        ]
    return {"choices": [{"message": message}]}


class TraceDatasetTest(testing.TestCase):
    async def _record(self, format):
        """Record a structured generator (twice) and a tool-calling agent."""
        root = os.path.join(self.get_temp_dir(), f"traces_{format}")
        language_model = LanguageModel(
            model="openai/gpt-4o-mini",
            hooks=[Recorder(base_dir=root, format=format)],
        )
        with patch("litellm.acompletion") as mock_completion:
            mock_completion.side_effect = [
                _response('{"answer": "Paris"}', reasoning_content="It is Paris."),
                _response('{"answer": "Paris, France"}'),
                _response(
                    "",
                    reasoning_content="I should use the calculator.",
                    tool_calls=[("calculate", {"expression": "10 + 20"})],
                ),
                _response("The result is 30.", reasoning_content="It returned 30."),
            ]
            inputs = Input(data_model=Query)
            outputs = await Generator(
                data_model=Answer, language_model=language_model, name="answerer"
            )(inputs)
            qa = Program(inputs=inputs, outputs=outputs, name="qa")
            await qa(Query(query="Capital of France?"))
            await qa(Query(query="Capital of France?"))

            inputs = Input(data_model=ChatMessages)
            outputs = await FunctionCallingAgent(
                language_model=language_model,
                tools=[Tool(calculate)],
                autonomous=True,
                max_iterations=3,
                name="calc_agent",
            )(inputs)
            agent = Program(inputs=inputs, outputs=outputs, name="agent")
            await agent(
                ChatMessages(messages=[ChatMessage(role="user", content="10 + 20?")])
            )
        return root

    def _examples(self, dataset):
        x, y = dataset.materialize()
        return [
            (inputs.get_json()["messages"], target.get_json())
            for inputs, target in zip(x, y)
        ]

    async def test_reads_every_recorded_call(self):
        root = await self._record("openai")

        examples = self._examples(TraceDataset(root))

        self.assertEqual(len(examples), 4)
        completions = [target for _, target in examples]
        self.assertIn(
            {
                "role": "assistant",
                "content": "The result is 30.",
                "reasoning_content": "It returned 30.",
            },
            completions,
        )
        for messages, target in examples:
            self.assertEqual(target["role"], "assistant")
            self.assertNotEqual(messages[-1]["role"], "assistant")

    async def test_filters_by_program_and_module(self):
        root = await self._record("openai")

        self.assertEqual(len(self._examples(TraceDataset(root, program="qa"))), 2)
        self.assertEqual(len(self._examples(TraceDataset(root, module="answerer"))), 2)
        self.assertEqual(
            len(
                self._examples(
                    TraceDataset(
                        root, module="calc_agent/tool_calls_generator_calc_agent"
                    )
                )
            ),
            2,
        )
        self.assertEqual(self._examples(TraceDataset(root, program="missing")), [])

    async def test_filters_by_metadata(self):
        root = await self._record("openai")
        later = time.time() + 60

        self.assertEqual(len(self._examples(TraceDataset(root, since=0))), 4)
        self.assertEqual(self._examples(TraceDataset(root, since=later)), [])
        self.assertEqual(len(self._examples(TraceDataset(root, until=later))), 4)
        self.assertEqual(self._examples(TraceDataset(root, session_id="other")), [])
        self.assertEqual(self._examples(TraceDataset(root, config_hash="0" * 64)), [])

    async def test_deduplicate_keeps_the_latest_of_identical_inputs(self):
        root = await self._record("openai")

        examples = self._examples(TraceDataset(root, program="qa", deduplicate=True))

        self.assertEqual(len(examples), 1)
        self.assertEqual(
            json.loads(examples[0][1]["content"]), {"answer": "Paris, France"}
        )

    async def test_drop_prefixes_keeps_the_whole_trajectory(self):
        root = await self._record("openai")

        examples = self._examples(TraceDataset(root, program="agent", drop_prefixes=True))

        (example,) = examples
        roles = [m["role"] for m in example[0]] + [example[1]["role"]]
        self.assertEqual(roles, ["system", "user", "assistant", "tool", "assistant"])

    async def test_tools_template_gives_each_example_its_tools(self):
        root = await self._record("openai")

        x, _ = TraceDataset(
            root, program="agent", tools_template="{{ tools | tojson }}"
        ).materialize()

        for inputs in x:
            (tool,) = inputs.get_json()["tools"]
            self.assertEqual(tool["function"]["name"], "calculate")

    async def test_templates_map_onto_program_data_models(self):
        root = await self._record("openai")

        x, y = TraceDataset(
            root,
            program="qa",
            input_data_model=Query,
            input_template='{"query": "Capital of France?"}',
            output_data_model=Answer,
            output_template="{{ outputs | tojson }}",
        ).materialize()

        self.assertEqual(x[0].get_json(), {"query": "Capital of France?"})
        self.assertEqual(y[0].get_json(), {"answer": "Paris"})

    async def test_tinker_and_customizer_read_back_as_openai(self):
        openai_root = await self._record("openai")
        expected = self._examples(TraceDataset(openai_root))
        expected_qa = self._examples(TraceDataset(openai_root, program="qa"))

        customizer = self._examples(TraceDataset(await self._record("customizer")))
        self.assertEqual(customizer, expected)

        tinker_root = await self._record("tinker")
        # Tinker cannot hold tool calls: only the generator's calls are examples.
        self.assertEqual(self._examples(TraceDataset(tinker_root)), expected_qa)
        self.assertEqual(
            self._examples(TraceDataset(tinker_root, include_unsupported=True)),
            expected,
        )

    def test_missing_root_raises(self):
        with self.assertRaises(FileNotFoundError):
            TraceDataset(os.path.join(self.get_temp_dir(), "nowhere"))
