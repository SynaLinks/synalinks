# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import rewards
from synalinks.src import testing
from synalinks.src.backend import Rating20
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.rewards.rubric_rewards import Faithfulness
from synalinks.src.rewards.rubrics import get_rubric
from synalinks.src.rewards.rubrics import list_rubrics
from synalinks.src.rewards.rubrics_as_judge import Rubric
from synalinks.src.rewards.rubrics_as_judge import RubricsAsJudge
from synalinks.src.utils.naming import to_snake_case


class RubricRewardsTest(testing.TestCase):
    def test_preset_reward_is_direct_rubrics_as_judge_subclass(self):
        reward = Faithfulness(
            language_model=LanguageModel(model="ollama/mistral"),
        )
        self.assertIsInstance(reward, RubricsAsJudge)
        self.assertIs(reward.program.score_type, Rating20)
        self.assertEqual(
            [rubric.name for rubric in reward.program.rubrics],
            ["faithfulness"],
        )

    def test_preset_reward_config_round_trip(self):
        reward = Faithfulness(
            language_model=LanguageModel(model="ollama/mistral"),
        )
        restored = Faithfulness.from_config(reward.get_config())
        self.assertEqual(restored.name, reward.name)
        self.assertEqual(
            [rubric.name for rubric in restored.program.rubrics],
            ["faithfulness"],
        )

    def test_every_preset_reward(self):
        language_model = LanguageModel(model="ollama/mistral")
        presets = [
            cls
            for cls in rewards.ALL_OBJECTS
            if issubclass(cls, RubricsAsJudge) and cls is not RubricsAsJudge
        ]
        # One reward class per built-in rubric preset, and nothing else.
        self.assertEqual(
            sorted(to_snake_case(cls.__name__) for cls in presets), list_rubrics()
        )
        for cls in presets:
            preset = to_snake_case(cls.__name__)
            with self.subTest(preset=preset):
                reward = cls(language_model=language_model)
                self.assertEqual(reward.name, preset)
                self.assertEqual(
                    [rubric.name for rubric in reward.program.rubrics],
                    [Rubric.parse(rubric).name for rubric in get_rubric(preset)],
                )
                restored = rewards.deserialize(rewards.serialize(reward))
                self.assertIsInstance(restored, cls)
                self.assertIs(rewards.ALL_OBJECTS_DICT[preset], cls)
