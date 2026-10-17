# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Training and evaluating without targets.

A reward that needs no target (a judge grading the prediction on its own)
trains and evaluates on inputs alone: every path that takes a target must
take none, for every kind of program and every optimizer.
"""

import json
from unittest.mock import AsyncMock
from unittest.mock import patch

import numpy as np

from synalinks.src import modules
from synalinks.src import optimizers
from synalinks.src import programs
from synalinks.src import rewards
from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.modules.agents.function_calling_agent import FunctionCallingAgent
from synalinks.src.modules.embedding_models import EmbeddingModel
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.saving.object_registration import register_synalinks_serializable
from synalinks.src.testing.test_utils import AnswerWithRationale
from synalinks.src.testing.test_utils import Query
from synalinks.src.testing.test_utils import load_test_data

ANSWER = json.dumps({"rationale": "Known.", "answer": "Paris"})
PARIS = {"choices": [{"message": {"content": ANSWER}}]}
FINAL = {"choices": [{"message": {"content": "152648 + 485 = 153133."}}]}


class GradedAnswer(rewards.Reward):
    """A judge: grades the prediction alone, and is never given a target."""

    async def call(self, y_true, y_pred):
        assert y_true is None, f"a target reached the judge: {y_true}"
        return 1.0 if y_pred.get("answer") == "Paris" else 0.0


class GradedTurn(rewards.Reward):
    """A judge of an agent's turn: its last message."""

    async def call(self, y_true, y_pred):
        assert y_true is None, f"a target reached the judge: {y_true}"
        return 1.0 if "153133" in str(y_pred.get("messages")[-1].get("content")) else 0.0


@register_synalinks_serializable()
async def calculate(expression: str):
    """Calculate the result of a mathematical expression.

    Args:
        expression (str): The expression, such as '2 + 2'.
    """
    return {"result": eval(expression, {"__builtins__": {}})}


async def generator_program():
    x0 = modules.Input(data_model=Query)
    x1 = await modules.Generator(
        data_model=AnswerWithRationale,
        language_model=LanguageModel(model="ollama/mistral"),
    )(x0)
    return programs.Program(inputs=x0, outputs=x1, name="answer")


async def agent_program(autonomous=True):
    inputs = modules.Input(data_model=ChatMessages)
    outputs = await FunctionCallingAgent(
        language_model=LanguageModel(model="ollama/mistral"),
        tools=[modules.Tool(calculate)],
        autonomous=autonomous,
        max_iterations=2,
    )(inputs)
    return programs.Program(inputs=inputs, outputs=outputs, name="agent")


def questions(n):
    return np.array(
        [
            ChatMessages(messages=[ChatMessage(role="user", content="152648 + 485?")])
            for _ in range(n)
        ],
        dtype="object",
    )


def fixed_embeddings():
    async def embed(request):
        texts = request.get("texts") if hasattr(request, "get") else request.texts
        return {"embeddings": [[1.0, 0.0] for _ in texts]}

    return AsyncMock(spec=EmbeddingModel, side_effect=embed)


class NoTargetsTest(testing.TestCase):
    async def test_an_agent_program_is_built_at_construction(self):
        program = await agent_program()
        self.assertTrue(all(module.built for module in program._flatten_modules()))

    @patch("litellm.acompletion")
    async def test_evaluate_and_fit_an_agent(self, mock_completion):
        mock_completion.return_value = FINAL
        program = await agent_program()
        program.compile(optimizer=optimizers.RandomFewShot(), reward=GradedTurn())
        self.assertEqual((await program.evaluate(x=questions(3)))["reward"], 1.0)
        # The optimizer builds on the first sample it receives, not before.
        self.assertFalse(program.optimizer.built)
        history = await program.fit(x=questions(4), epochs=1, validation_split=0.5)
        self.assertTrue(program.optimizer.built)
        self.assertEqual(history.history["reward"], [1.0])
        self.assertEqual(history.history["val_reward"], [1.0])

    @patch("litellm.acompletion")
    async def test_evaluate_an_unbuilt_program(self, mock_completion):
        # Interactive, the agent's final generator is never traced: the
        # program is unbuilt, and evaluating builds it on a batch with no `y`.
        mock_completion.return_value = FINAL
        program = await agent_program(autonomous=False)
        self.assertFalse(all(module.built for module in program._flatten_modules()))
        program.compile(reward=GradedTurn())
        self.assertEqual((await program.evaluate(x=questions(2)))["reward"], 1.0)

    @patch("litellm.acompletion")
    async def test_train_and_test_on_a_batch(self, mock_completion):
        mock_completion.return_value = PARIS
        program = await generator_program()
        program.compile(optimizer=optimizers.RandomFewShot(), reward=GradedAnswer())
        (x_train, _), (x_test, _) = load_test_data()
        logs = await program.test_on_batch(x=x_test, return_dict=True)
        self.assertEqual(logs["reward"], 1.0)
        logs = await program.train_on_batch(
            step=0, x=x_train[:2], val_x=x_test[:2], return_dict=True
        )
        self.assertEqual(logs["reward"], 1.0)

    @patch("litellm.acompletion")
    async def test_every_optimizer_fits(self, mock_completion):
        mock_completion.return_value = PARIS
        (x_train, _), _ = load_test_data()
        for optimizer in (
            optimizers.RandomFewShot(),
            optimizers.OMEGA(
                language_model=LanguageModel(model="ollama/mistral"),
                embedding_model=fixed_embeddings(),
            ),
        ):
            with self.subTest(optimizer=type(optimizer).__name__):
                program = await generator_program()
                program.compile(optimizer=optimizer, reward=GradedAnswer())
                history = await program.fit(
                    x=x_train[:4], epochs=1, validation_split=0.5, minibatch_size=1
                )
                self.assertEqual(history.history["val_reward"], [1.0])
