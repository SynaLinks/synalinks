# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.language_models import LanguageModel
from synalinks.src.modules import Generator
from synalinks.src.modules import Input
from synalinks.src.optimizers.random_few_shot import RandomFewShot
from synalinks.src.programs import Program
from synalinks.src.rewards.exact_match import ExactMatch
from synalinks.src.testing.test_utils import AnswerWithRationale
from synalinks.src.testing.test_utils import Query
from synalinks.src.testing.test_utils import load_test_data
from synalinks.src.testing.test_utils import mock_completion_data
from synalinks.src.testing.test_utils import mock_incorrect_completion_data


class RandomFewShotTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_random_few_shot_training(self, mock_completion):
        language_model = LanguageModel(
            model="ollama/mistral",
        )

        inputs = Input(data_model=Query)
        outputs = await Generator(
            language_model=language_model,
            data_model=AnswerWithRationale,
        )(inputs)

        program = Program(
            inputs=inputs,
            outputs=outputs,
            name="test_program",
            description="A test program",
        )

        program.compile(
            optimizer=RandomFewShot(
                nb_min_examples=1,
            ),
            reward=ExactMatch(in_mask=["answer"]),
        )

        (x_train, y_train), (x_test, y_test) = load_test_data()

        mock_responses = []
        mock_responses.extend(mock_incorrect_completion_data())
        mock_responses.extend(mock_completion_data())

        mock_completion.side_effect = mock_responses

        _ = await program.fit(
            x=x_train,
            y=y_train,
            epochs=2,
            batch_size=32,
        )

        program_vars = program.get_variable(index=0).get_json()
        self.assertTrue(len(program_vars["examples"]) > 0)
