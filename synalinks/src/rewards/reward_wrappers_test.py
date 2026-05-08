# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import warnings
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge
from synalinks.src.rewards.reward_wrappers import RewardFunctionWrapper
from synalinks.src.saving import object_registration


@object_registration.register_synalinks_serializable()
async def custom_reward_fn(y_true, y_pred, weight=1.0):
    """A simple custom reward function for testing."""
    if y_true.get_json() == y_pred.get_json():
        return 1.0 * weight
    return 0.0


class RewardFunctionWrapperTest(testing.TestCase):
    async def test_call(self):
        class Answer(DataModel):
            answer: str

        y_true = Answer(answer="hello")
        y_pred = Answer(answer="hello")

        wrapper = RewardFunctionWrapper(fn=custom_reward_fn)
        reward = await wrapper(y_true, y_pred)
        self.assertEqual(reward, 1.0)

    async def test_call_no_match(self):
        class Answer(DataModel):
            answer: str

        y_true = Answer(answer="hello")
        y_pred = Answer(answer="world")

        wrapper = RewardFunctionWrapper(fn=custom_reward_fn)
        reward = await wrapper(y_true, y_pred)
        self.assertEqual(reward, 0.0)

    async def test_call_with_kwargs(self):
        class Answer(DataModel):
            answer: str

        y_true = Answer(answer="hello")
        y_pred = Answer(answer="hello")

        wrapper = RewardFunctionWrapper(fn=custom_reward_fn, weight=0.5)
        reward = await wrapper(y_true, y_pred)
        self.assertEqual(reward, 0.5)

    def test_get_config(self):
        wrapper = RewardFunctionWrapper(fn=custom_reward_fn, weight=2.0)
        config = wrapper.get_config()
        self.assertIn("fn", config)
        self.assertIn("name", config)

    def test_repr(self):
        wrapper = RewardFunctionWrapper(fn=custom_reward_fn, weight=2.0)
        repr_str = repr(wrapper)
        self.assertIn("RewardFunctionWrapper", repr_str)

    def test_get_config_kwargs_namespaced(self):
        # Regression: previously _fn_kwargs was merged into the top-level
        # config dict and could overwrite base-class fields like `name` or
        # `reduction`. Kwargs must be stored under their own key.
        wrapper = RewardFunctionWrapper(
            fn=custom_reward_fn,
            name="my_wrapper",
            reduction="sum",
            weight=2.0,
        )
        config = wrapper.get_config()
        self.assertEqual(config["name"], "my_wrapper")
        self.assertEqual(config["reduction"], "sum")
        self.assertIn("fn", config)
        self.assertIn("fn_kwargs", config)


class ProgramAsJudgeTest(testing.TestCase):
    async def test_call(self):
        class Answer(DataModel):
            answer: str

        mock_program = AsyncMock()
        mock_program.return_value = {"reward": 0.75}

        judge = ProgramAsJudge(program=mock_program)
        y_true = Answer(answer="hello")
        y_pred = Answer(answer="world")

        reward = await judge(y_true, y_pred)
        self.assertEqual(reward, 0.75)

    async def test_call_missing_reward_key(self):
        class Answer(DataModel):
            answer: str

        mock_program = AsyncMock()
        mock_program.return_value = {"score": 0.5}

        judge = ProgramAsJudge(program=mock_program)
        y_true = Answer(answer="hello")
        y_pred = Answer(answer="world")

        reward = await judge(y_true, y_pred)
        self.assertEqual(reward, 0.0)

    async def test_call_program_returns_none(self):
        """Judge must return 0.0 and warn (not crash) when the program fails."""

        class Answer(DataModel):
            answer: str

        mock_program = AsyncMock()
        mock_program.return_value = None

        judge = ProgramAsJudge(program=mock_program)
        y_true = Answer(answer="hello")
        y_pred = Answer(answer="world")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reward = await judge(y_true, y_pred)

        self.assertEqual(reward, 0.0)
        judge_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        self.assertEqual(len(judge_warnings), 1)
        self.assertIn("returned None", str(judge_warnings[0].message))

    def test_get_config(self):
        mock_program = MagicMock()
        mock_program.get_config.return_value = {}
        judge = ProgramAsJudge(program=mock_program)
        config = judge.get_config()
        self.assertIn("program", config)

    def test_repr(self):
        mock_program = MagicMock()
        judge = ProgramAsJudge(program=mock_program)
        repr_str = repr(judge)
        self.assertIn("ProgramAsJudge", repr_str)

    async def test_get_config_serializes_program(self):
        # Regression: previously stored the raw Program object in the config
        # dict, which doesn't survive JSON round-trips. Should serialize
        # like RewardFunctionWrapper does for `fn`.
        from synalinks.src import modules
        from synalinks.src import programs
        from synalinks.src.modules.language_models import LanguageModel
        from synalinks.src.testing.test_utils import AnswerWithRationale
        from synalinks.src.testing.test_utils import Query

        x0 = modules.Input(data_model=Query)
        x1 = await modules.Generator(
            data_model=AnswerWithRationale,
            language_model=LanguageModel(model="ollama/mistral"),
        )(x0)
        program = programs.Program(inputs=x0, outputs=x1, name="judge")

        judge = ProgramAsJudge(program=program)
        config = judge.get_config()
        self.assertIn("program", config)
        # A serialized synalinks object is a dict, not a Program.
        self.assertIsInstance(config["program"], dict)
        self.assertIn("class_name", config["program"])
