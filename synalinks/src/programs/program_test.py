# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import optimizers
from synalinks.src import rewards
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Generator
from synalinks.src.modules import Input
from synalinks.src.modules import default_instructions
from synalinks.src.programs import Program
from synalinks.src.utils.nlp_utils import remove_numerical_suffix


class ProgramTest(testing.TestCase):
    async def test_get_state(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama/mistral")

        x0 = Input(data_model=Query)
        x1 = await Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought",
            description="Useful to answer in a step by step manner.",
        )

        data_model_fields = list(AnswerWithRationale.get_schema()["properties"].keys())
        instructions = default_instructions(data_model_fields)

        state_tree = program.get_state_tree()

        expected_tree = {
            "trainable_variables": {
                "generator": {
                    "state_generator": {
                        "instructions": instructions,
                        "examples": [],
                        "current_predictions": [],
                        "predictions": [],
                        "history": [],
                        "seed_candidates": [],
                        "candidates": [],
                        "best_candidates": [],
                        "nb_visit": 0,
                        "cumulative_reward": 0.0,
                    }
                }
            },
            "non_trainable_variables": {},
            "metrics_variables": {},
        }
        self.assertEqual(state_tree, expected_tree)

    async def test_recover_state(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama/mistral")

        x0 = Input(data_model=Query)
        x1 = await Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought",
            description="Useful to answer in a step by step manner.",
        )

        state_tree = program.get_state_tree()
        state_tree["trainable_variables"]["generator"]["state_generator"][
            "prompt_template"
        ] = "Dummy prompt template"
        program.set_state_tree(state_tree)
        self.assertEqual(state_tree, program.get_state_tree())

    async def test_get_state_after_compile(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(model="ollama/mistral")

        x0 = Input(data_model=Query)
        x1 = await Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought",
            description="Useful to answer in a step by step manner.",
        )

        program.compile(
            reward=rewards.ExactMatch(),
            optimizer=optimizers.random_few_shot.RandomFewShot(),
        )

        state_tree = program.get_state_tree()

        data_model_fields = list(AnswerWithRationale.get_schema()["properties"].keys())
        instructions = default_instructions(data_model_fields)

        expected_tree = {
            "trainable_variables": {
                "generator": {
                    "state_generator": {
                        "instructions": instructions,
                        "examples": [],
                        "current_predictions": [],
                        "predictions": [],
                        "history": [],
                        "seed_candidates": [],
                        "candidates": [],
                        "best_candidates": [],
                        "nb_visit": 0,
                        "cumulative_reward": 0.0,
                    }
                }
            },
            "non_trainable_variables": {},
            "optimizer_non_trainable_variables": {
                "random_few_shot": {
                    "iterations_random_few_shot": {
                        "iterations": 0,
                        "epochs": 0,
                    },
                },
            },
            "optimizer_trainable_variables": {},
            "metrics_variables": {
                "reward": {"total_with_count": {"total": 0.0, "count": 0}}
            },
        }
        self.maxDiff = None
        self.assertEqual(state_tree, expected_tree)

    async def test_recover_state_after_compile(self):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        x0 = Input(data_model=Query)
        x1 = await Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought",
            description="Useful to answer in a step by step manner.",
        )

        program.compile(
            reward=rewards.ExactMatch(),
            optimizer=optimizers.random_few_shot.RandomFewShot(),
        )

        state_tree = program.get_state_tree()
        state_tree["trainable_variables"]["generator"]["state_generator"][
            "prompt_template"
        ] = "Dummy prompt template"
        program.set_state_tree(state_tree)
        self.assertEqual(state_tree, program.get_state_tree())

    @patch("litellm.acompletion")
    async def test_get_state_after_training(self, mock_completion):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        x0 = Input(data_model=Query)
        x1 = await Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought",
            description="Useful to answer in a step by step manner.",
        )

        program.compile(
            reward=rewards.ExactMatch(in_mask=["answer"]),
            optimizer=optimizers.random_few_shot.RandomFewShot(),
        )

        (x_train, y_train), (x_test, y_test) = testing.test_utils.load_test_data()

        mock_completion.side_effect = testing.test_utils.mock_completion_data()

        _ = await program.fit(
            x=x_train,
            y=y_train,
            batch_size=32,
        )

        _ = program.get_state_tree()

    @patch("litellm.acompletion")
    async def test_recover_state_after_training(self, mock_completion):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        x0 = Input(data_model=Query)
        x1 = await Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought",
            description="Useful to answer in a step by step manner.",
        )

        program.compile(
            reward=rewards.ExactMatch(in_mask=["answer"]),
            optimizer=optimizers.random_few_shot.RandomFewShot(),
        )

        (x_train, y_train), (x_test, y_test) = testing.test_utils.load_test_data()

        mock_completion.side_effect = testing.test_utils.mock_completion_data()

        _ = await program.fit(
            x=x_train,
            y=y_train,
            batch_size=32,
        )

        state_tree = program.get_state_tree()
        program.set_state_tree(state_tree)
        new_state_tree = program.get_state_tree()
        self.assertEqual(state_tree, new_state_tree)

    @patch("litellm.acompletion")
    async def test_saving_after_training(self, mock_completion):
        class Query(DataModel):
            query: str

        class AnswerWithRationale(DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        x0 = Input(data_model=Query)
        x1 = await Generator(
            data_model=AnswerWithRationale,
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought",
            description="Useful to answer in a step by step manner.",
        )

        program.compile(
            reward=rewards.ExactMatch(in_mask=["answer"]),
            optimizer=optimizers.random_few_shot.RandomFewShot(),
        )

        (x_train, y_train), (x_test, y_test) = testing.test_utils.load_test_data()

        mock_completion.side_effect = testing.test_utils.mock_completion_data()

        _ = await program.fit(
            x=x_train,
            y=y_train,
            batch_size=32,
        )

        filepath = "/tmp/program.json"
        program.save("/tmp/program.json")
        cloned_program = Program.load(filepath)

        for var1 in program.variables:
            for var2 in cloned_program.variables:
                if remove_numerical_suffix(var1.path) == remove_numerical_suffix(
                    var2.path
                ):
                    self.assertEqual(var1.get_json(), var2.get_json())
