# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import warnings
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from synalinks.src import ops
from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import DataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_chat_message
from synalinks.src.backend import is_chat_messages
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.modules import Input
from synalinks.src.modules import SelfCritique
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Program
from synalinks.src.rewards.lm_as_judge import LMAsJudgeProgram
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge
from synalinks.src.rewards.reward_wrappers import RewardFunctionWrapper
from synalinks.src.rewards.reward_wrappers import judge_inputs
from synalinks.src.saving import object_registration


@object_registration.register_synalinks_serializable()
async def custom_reward_fn(y_true, y_pred, weight=1.0):
    """A simple custom reward function for testing."""
    if y_true.get_json() == y_pred.get_json():
        return 1.0 * weight
    return 0.0


class RewardFunctionWrapperTest(testing.TestCase):
    def test_sync_fn_is_rejected(self):
        def sync_reward_fn(y_true, y_pred):
            return 1.0

        with self.assertRaisesRegex(TypeError, "sync_reward_fn"):
            RewardFunctionWrapper(fn=sync_reward_fn)

    def test_async_callable_object_is_accepted(self):
        class AsyncJudge:
            async def __call__(self, y_true, y_pred):
                return 0.5

        wrapper = RewardFunctionWrapper(fn=AsyncJudge(), name="judge")
        self.assertEqual(wrapper.name, "judge")

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

    async def test_from_config_round_trip(self):
        class Answer(DataModel):
            answer: str

        wrapper = RewardFunctionWrapper(
            fn=custom_reward_fn,
            name="my_wrapper",
            reduction="sum",
            in_mask=["answer"],
            weight=0.5,
        )
        restored = RewardFunctionWrapper.from_config(wrapper.get_config())
        self.assertEqual(restored.name, "my_wrapper")
        self.assertEqual(restored.reduction, "sum")
        self.assertEqual(restored.in_mask, ["answer"])
        # The function and its kwargs survive the round trip.
        reward = await restored(Answer(answer="hello"), Answer(answer="hello"))
        self.assertEqual(reward, 0.5)


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

    async def test_from_config_round_trip(self):
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

        judge = ProgramAsJudge(
            program=program, name="my_judge", reduction="sum", in_mask=["answer"]
        )
        restored = ProgramAsJudge.from_config(judge.get_config())
        self.assertIsInstance(restored, ProgramAsJudge)
        self.assertEqual(restored.name, "my_judge")
        self.assertEqual(restored.reduction, "sum")
        self.assertEqual(restored.in_mask, ["answer"])
        self.assertIsInstance(restored.program, programs.Program)
        self.assertEqual(restored.program.name, "judge")


class Query(DataModel):
    query: str


class Answer(DataModel):
    answer: str


class QueryAnswer(DataModel):
    query: str
    answer: str


class JudgeInputsTest(testing.TestCase):
    async def test_labels_input_output_gold_in_order(self):
        inputs = await judge_inputs(
            Answer(answer="Paris").to_json_data_model(),
            Answer(answer="Lyon").to_json_data_model(),
            x=Query(query="French capital?").to_json_data_model(),
        )
        self.assertEqual(
            inputs.get_json(),
            {
                "input_query": "French capital?",
                "output_answer": "Lyon",
                "gold_answer": "Paris",
            },
        )
        self.assertEqual(
            list(inputs.get_schema()["properties"]),
            ["input_query", "output_answer", "gold_answer"],
        )

    async def test_output_equal_to_its_input_is_shown(self):
        # A fixer given an already-correct text answers with that same text:
        # the judge must see it as the output, or it grades it as missing.
        class Text(DataModel):
            text: str

        class Fixed(DataModel):
            text: str
            explanation: str

        inputs = await judge_inputs(
            Text(text="Hello world."),
            Fixed(text="Hello world.", explanation="Already correct."),
            x=Text(text="Hello world."),
        )
        self.assertEqual(
            inputs.get_json(),
            {
                "input_text": "Hello world.",
                "output_text": "Hello world.",
                "output_explanation": "Already correct.",
                "gold_text": "Hello world.",
            },
        )

    async def test_multi_input_program(self):
        inputs = await judge_inputs(
            None,
            Answer(answer="Paris"),
            x=[Query(query="French capital?"), None, Answer(answer="a hint")],
        )
        self.assertEqual(
            inputs.get_json(),
            {
                "input_query": "French capital?",
                "input_answer": "a hint",
                "output_answer": "Paris",
            },
        )

    async def test_without_x_nor_gold(self):
        inputs = await judge_inputs(None, Answer(answer="Paris"))
        self.assertEqual(inputs.get_json(), {"output_answer": "Paris"})

    async def test_symbolic(self):
        # Symbolic traces carry no values to compare: nothing is deduplicated.
        inputs = await judge_inputs(
            SymbolicDataModel(data_model=Answer),
            SymbolicDataModel(data_model=QueryAnswer),
            x=SymbolicDataModel(data_model=Query),
        )
        self.assertTrue(is_symbolic_data_model(inputs))
        self.assertEqual(
            list(inputs.get_schema()["properties"]),
            ["input_query", "output_query", "output_answer", "gold_answer"],
        )

    async def test_chat_is_read_as_data(self):
        # Renamed `messages` keep judges (and agents) from continuing the
        # conversation they grade.
        x = ChatMessages(messages=[ChatMessage(role="user", content="Hi")])
        y_pred = await ops.concat(
            x.to_json_data_model(),
            ChatMessage(role="assistant", content="Hello!").to_json_data_model(),
        )
        inputs = await judge_inputs(
            ChatMessage(role="assistant", content="Hey"), y_pred, x=x
        )
        self.assertFalse(is_chat_messages(inputs))
        self.assertFalse(is_chat_message(inputs))
        self.assertEqual(
            list(inputs.get_json()),
            [
                "input_messages",
                "output_messages",
                "output_role",
                "output_content",
                "gold_role",
                "gold_content",
            ],
        )


class ProgramAsJudgeInputsTest(testing.TestCase):
    async def _functional_judge(self, with_x):
        language_model = LanguageModel(model="ollama/mistral")
        y_true = Input(data_model=Answer, optional=True)
        y_pred = Input(data_model=Answer)
        graded = await ops.logical_or(y_pred, y_true)
        inputs = [y_true, y_pred]
        if with_x:
            x = Input(data_model=Query, optional=True)
            graded = await ops.logical_or(x, graded)
            inputs.append(x)
        critique = await SelfCritique(language_model=language_model)(graded)
        return ProgramAsJudge(program=Program(inputs=inputs, outputs=critique))

    @patch("litellm.acompletion")
    async def test_three_input_program_gets_x_and_tolerates_none(self, mock_completion):
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"critique": "Ok", "reward": 1.0}'}}]
        }
        judge = await self._functional_judge(with_x=True)
        cases = [
            ({"x": Query(query="q")}, "{'query': 'q', 'answer': 'a', 'answer_1': 'g'}"),
            # Called standalone, without `x`, nor ground truth: optional inputs.
            ({}, "{'answer': 'a', 'answer_1': 'g'}"),
        ]
        for kwargs, prompt in cases:
            score = await judge(Answer(answer="g"), Answer(answer="a"), **kwargs)
            self.assertEqual(score, 1.0)
            self.assertEqual(
                mock_completion.call_args.kwargs["messages"][-1]["content"], prompt
            )
        await judge(None, Answer(answer="a"))
        self.assertEqual(
            mock_completion.call_args.kwargs["messages"][-1]["content"],
            "{'answer': 'a'}",
        )

    @patch("litellm.acompletion")
    async def test_two_input_program_does_not_get_x(self, mock_completion):
        # Existing judges are declared with two inputs: passing them a third
        # would break their input structure.
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"critique": "Ok", "reward": 1.0}'}}]
        }
        judge = await self._functional_judge(with_x=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            score = await judge(
                Answer(answer="g"), Answer(answer="a"), x=Query(query="q")
            )
        self.assertEqual(score, 1.0)
        self.assertNotIn(
            "query", mock_completion.call_args.kwargs["messages"][-1]["content"]
        )

    @patch("litellm.acompletion")
    async def test_one_input_program_gets_the_prediction(self, mock_completion):
        # A judge needing no ground truth declares only the prediction.
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"critique": "Ok", "reward": 1.0}'}}]
        }
        y_pred = Input(data_model=Answer)
        critique = await SelfCritique(
            language_model=LanguageModel(model="ollama/mistral")
        )(y_pred)
        judge = ProgramAsJudge(program=Program(inputs=y_pred, outputs=critique))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            score = await judge(
                Answer(answer="g"), Answer(answer="a"), x=Query(query="q")
            )
        self.assertEqual(score, 1.0)
        self.assertEqual(
            mock_completion.call_args.kwargs["messages"][-1]["content"], "{'answer': 'a'}"
        )

    async def test_subclassed_program_gets_x_when_its_call_takes_it(self):
        seen = []

        class Score(DataModel):
            reward: float

        class WithX(Program):
            async def call(self, inputs, x=None):
                seen.append(("with_x", len(inputs), x and x.get("query")))
                return Score(reward=1.0).to_json_data_model()

        class WithoutX(Program):
            async def call(self, inputs):
                seen.append(("without_x", len(inputs)))
                return Score(reward=1.0).to_json_data_model()

        for program in (WithX(), WithoutX()):
            judge = ProgramAsJudge(program=program)
            self.assertEqual(
                await judge(Answer(answer="a"), Answer(answer="a"), x=Query(query="q")),
                1.0,
            )
        self.assertEqual(seen, [("with_x", 2, "q"), ("without_x", 2)])

    @patch("litellm.acompletion")
    async def test_builtin_judge_program_gets_x_through_program_as_judge(
        self, mock_completion
    ):
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"critique": "Ok", "reward": 1.0}'}}]
        }
        program = LMAsJudgeProgram(language_model=LanguageModel(model="ollama/mistral"))
        await ProgramAsJudge(program=program)(
            None, Answer(answer="Paris"), x=Query(query="French capital?")
        )
        self.assertEqual(
            mock_completion.call_args.kwargs["messages"][-1]["content"],
            "{'input_query': 'French capital?', 'output_answer': 'Paris'}",
        )
