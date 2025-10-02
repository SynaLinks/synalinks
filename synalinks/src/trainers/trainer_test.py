# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import metrics
from synalinks.src import modules
from synalinks.src import optimizers
from synalinks.src import programs
from synalinks.src import rewards
from synalinks.src import testing
from synalinks.src.backend import JsonDataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.testing.test_utils import AnswerWithRationale
from synalinks.src.testing.test_utils import Query
from synalinks.src.testing.test_utils import load_test_data
from synalinks.src.trainers.trainer import Trainer


class ExampleProgram(Trainer, modules.Generator):
    def __init__(self, data_model=None, language_model=None):
        modules.Generator.__init__(
            self,
            data_model=data_model,
            language_model=language_model,
        )
        Trainer.__init__(self)


async def program_test():
    x0 = modules.Input(data_model=Query)
    x1 = await modules.Generator(
        data_model=AnswerWithRationale,
        language_model=LanguageModel(
            model="ollama/mistral",
        ),
    )(x0)
    return programs.Program(
        inputs=x0,
        outputs=x1,
        name="chain_of_thought",
        description="Useful to answer step by step",
    )


class TestTrainer(testing.TestCase):
    def test_compiled_metrics(self):
        program = ExampleProgram(
            data_model=Query,
            language_model=LanguageModel(
                model="ollama/mistral",
            ),
        )

        program.compile(
            optimizer=optimizers.random_few_shot.RandomFewShot(),
            reward=rewards.ExactMatch(),
            metrics=[metrics.MeanMetricWrapper(rewards.exact_match)],
        )

        # The program should have 2 metrics: reward_tracker, compile_metrics,
        self.assertEqual(len(program.metrics), 2)
        self.assertEqual(program.metrics[0], program._reward_tracker)
        self.assertEqual(program.metrics[1], program._compile_metrics)

        # All metrics should have their variables created
        self.assertEqual(len(program._reward_tracker.variables), 1)

    @patch("litellm.acompletion")
    async def test_predict_on_batch(self, mock_completion):
        mock_answer = AnswerWithRationale(
            rationale="""The capital of France is well-known and is the seat of """
            """the French government.""",
            answer="Paris",
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_answer.get_json())}}]
        }

        program = await program_test()

        (x_train, y_train), (x_test, y_test) = load_test_data()

        y_pred = await program.predict_on_batch(x_train)

        self.assertEqual(len(y_pred), len(x_train))
        self.assertEqual(y_pred[0].get_json(), mock_answer.get_json())
        self.assertEqual(y_pred[1].get_json(), mock_answer.get_json())

    @patch("litellm.acompletion")
    async def test_test_on_batch(self, mock_completion):
        mock_answer = AnswerWithRationale(
            rationale="""The capital of France is well-known and is the seat of """
            """the French government.""",
            answer="Paris",
        )
        mock_answer = json.dumps(mock_answer.get_json())

        mock_completion.return_value = {
            "choices": [{"message": {"content": mock_answer}}]
        }

        program = await program_test()

        program.compile(
            optimizer=optimizers.random_few_shot.RandomFewShot(),
            reward=rewards.ExactMatch(in_mask=["answer"]),
            metrics=[metrics.MeanMetricWrapper(rewards.exact_match, in_mask=["answer"])],
        )

        (x_train, y_train), (x_test, y_test) = load_test_data()

        result_metrics = await program.test_on_batch(x_test, y_test, return_dict=False)
        self.assertEqual(len(result_metrics), 2)
        self.assertEqual(result_metrics[0], 0.10000000149011612)
        self.assertEqual(result_metrics[1], 0.10000000149011612)

    @patch("litellm.acompletion")
    async def test_evaluate(self, mock_completion):
        mock_answer = AnswerWithRationale(
            rationale="""The capital of France is well-known and is the seat of """
            """the French government.""",
            answer="Paris",
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_answer.get_json())}}]
        }

        program = await program_test()

        program.compile(
            optimizer=optimizers.random_few_shot.RandomFewShot(),
            reward=rewards.ExactMatch(in_mask=["answer"]),
            metrics=[
                metrics.MeanMetricWrapper(rewards.exact_match, in_mask=["answer"]),
            ],
        )

        (x_train, y_train), (x_test, y_test) = load_test_data()

        _ = await program.evaluate(
            x=x_test,
            y=y_test,
        )

    @patch("litellm.acompletion")
    async def test_predict(self, mock_completion):
        mock_answer = AnswerWithRationale(
            rationale="""The capital of France is well-known and is the seat of """
            """the French government.""",
            answer="Paris",
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_answer.get_json())}}]
        }

        program = await program_test()

        program.compile(
            optimizer=optimizers.random_few_shot.RandomFewShot(),
            reward=rewards.ExactMatch(in_mask=["answer"]),
            metrics=[
                metrics.MeanMetricWrapper(rewards.exact_match, in_mask=["answer"]),
            ],
        )

        (x_train, _), _ = load_test_data()

        y_data = await program.predict(x=x_train)

        self.assertEqual(len(y_data), len(x_train))
        self.assertIsInstance(y_data[0], JsonDataModel)
        self.assertIsInstance(y_data[1], JsonDataModel)
