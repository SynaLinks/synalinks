# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import numpy as np
import pytest

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.rewards.batch_reward import BatchReward
from synalinks.src.rewards.batch_reward import BatchRewardFunctionWrapper
from synalinks.src.saving import object_registration


@object_registration.register_synalinks_serializable()
async def batch_exact_match(y_true, y_pred):
    return [1.0 if t.get_json() == p.get_json() else 0.0 for t, p in zip(y_true, y_pred)]


@object_registration.register_synalinks_serializable()
async def batch_normalized(y_true, y_pred, temperature=1.0):
    raw = [float(p.get("score", 0.0)) for p in y_pred]
    total = sum(abs(r) for r in raw) + 1e-9
    return [r / total / temperature for r in raw]


class BatchRewardFunctionWrapperTest(testing.TestCase):
    async def test_per_sample_call(self):
        class Answer(DataModel):
            answer: str

        y_true = [Answer(answer="a"), Answer(answer="b"), Answer(answer="c")]
        y_pred = [Answer(answer="a"), Answer(answer="x"), Answer(answer="c")]

        wrapper = BatchRewardFunctionWrapper(fn=batch_exact_match, reduction="none")
        rewards = await wrapper.compute_batch(y_true, y_pred)
        self.assertEqual(rewards, [1.0, 0.0, 1.0])

    async def test_call_reduces(self):
        class Answer(DataModel):
            answer: str

        y_true = [Answer(answer="a"), Answer(answer="b"), Answer(answer="c")]
        y_pred = [Answer(answer="a"), Answer(answer="x"), Answer(answer="c")]

        wrapper = BatchRewardFunctionWrapper(fn=batch_exact_match, reduction="mean")
        reward = await wrapper(y_true, y_pred)
        self.assertAlmostEqual(reward, 2.0 / 3.0, places=6)

    async def test_call_with_kwargs(self):
        class Score(DataModel):
            score: float

        y_true = [Score(score=0.0), Score(score=0.0)]
        y_pred = [Score(score=2.0), Score(score=2.0)]

        wrapper = BatchRewardFunctionWrapper(
            fn=batch_normalized, reduction="none", temperature=2.0
        )
        rewards = await wrapper.compute_batch(y_true, y_pred)
        self.assertAlmostEqual(rewards[0], 0.25, places=4)
        self.assertAlmostEqual(rewards[1], 0.25, places=4)

    async def test_wrong_length_raises(self):
        class Answer(DataModel):
            answer: str

        async def bad_fn(y_true, y_pred):
            return [1.0]  # batch of 2 but returns 1

        y_true = [Answer(answer="a"), Answer(answer="b")]
        y_pred = [Answer(answer="a"), Answer(answer="b")]

        wrapper = BatchRewardFunctionWrapper(fn=bad_fn)
        with pytest.raises(ValueError, match="batch has 2 samples"):
            await wrapper.compute_batch(y_true, y_pred)

    async def test_non_list_return_raises(self):
        class Answer(DataModel):
            answer: str

        async def bad_fn(y_true, y_pred):
            return 0.5  # not a list

        wrapper = BatchRewardFunctionWrapper(fn=bad_fn)
        y_true = [Answer(answer="a")]
        y_pred = [Answer(answer="a")]
        with pytest.raises(TypeError, match="must return a list"):
            await wrapper.compute_batch(y_true, y_pred)

    async def test_masking(self):
        class Answer(DataModel):
            answer: str

        class AnswerWithText(DataModel):
            text: str
            answer: str

        y_true = [Answer(answer="Paris"), Answer(answer="Rome")]
        y_pred = [
            AnswerWithText(text="...", answer="Paris"),
            AnswerWithText(text="...", answer="Berlin"),
        ]

        wrapper = BatchRewardFunctionWrapper(
            fn=batch_exact_match, reduction="none", in_mask=["answer"]
        )
        rewards = await wrapper.compute_batch(y_true, y_pred)
        self.assertEqual(rewards, [1.0, 0.0])

    async def test_numpy_array_return_accepted(self):
        # Common idiom: batch normalization / group-relative rewards built
        # with numpy. The validator should accept any iterable of floats,
        # not strictly list/tuple.
        class Score(DataModel):
            score: float

        async def numpy_fn(y_true, y_pred):
            return np.array([1.0, 0.0, 0.5])

        wrapper = BatchRewardFunctionWrapper(fn=numpy_fn, reduction="none")
        y_true = [Score(score=0.0), Score(score=0.0), Score(score=0.0)]
        y_pred = [Score(score=0.0), Score(score=0.0), Score(score=0.0)]
        rewards = await wrapper.compute_batch(y_true, y_pred)
        self.assertEqual(rewards, [1.0, 0.0, 0.5])

    async def test_subclassing(self):
        class Answer(DataModel):
            answer: str

        class GroupRelative(BatchReward):
            async def call(self, y_true, y_pred):
                # Reward proportional to the count of matching siblings.
                matches = [
                    1.0 if t.get_json() == p.get_json() else 0.0
                    for t, p in zip(y_true, y_pred)
                ]
                total = sum(matches) or 1.0
                return [m / total for m in matches]

        y_true = [Answer(answer="a"), Answer(answer="b"), Answer(answer="c")]
        y_pred = [Answer(answer="a"), Answer(answer="b"), Answer(answer="x")]

        reward = GroupRelative(reduction="none")
        rewards = await reward.compute_batch(y_true, y_pred)
        self.assertEqual(rewards, [0.5, 0.5, 0.0])


class CompileRewardBatchDispatchTest(testing.TestCase):
    """End-to-end check that the trainer dispatch into compute_batch works."""

    async def test_compile_reward_batched_path(self):
        from synalinks.src.trainers.compile_utils import CompileReward

        class Answer(DataModel):
            answer: str

        reward = BatchRewardFunctionWrapper(fn=batch_exact_match, reduction="none")
        compile_reward = CompileReward(reward=reward)

        y_true = [Answer(answer="a"), Answer(answer="b"), Answer(answer="c")]
        y_pred = [Answer(answer="a"), Answer(answer="x"), Answer(answer="c")]

        compile_reward.build(y_true[0], y_pred[0])
        self.assertTrue(compile_reward.has_batch_rewards)

        rewards = await compile_reward.compute_batch(y_true, y_pred)
        self.assertEqual(rewards, [1.0, 0.0, 1.0])

    async def test_compile_reward_per_sample_path_unchanged(self):
        from synalinks.src.rewards.exact_match import ExactMatch
        from synalinks.src.trainers.compile_utils import CompileReward

        class Answer(DataModel):
            answer: str

        compile_reward = CompileReward(reward=ExactMatch())

        y_true = [Answer(answer="a"), Answer(answer="b")]
        y_pred = [Answer(answer="a"), Answer(answer="x")]

        compile_reward.build(y_true[0], y_pred[0])
        self.assertFalse(compile_reward.has_batch_rewards)
