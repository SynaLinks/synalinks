# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

import numpy as np

from synalinks.src import modules
from synalinks.src import programs
from synalinks.src import rewards
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.datasets.dataset import Dataset
from synalinks.src.metrics.agents_metrics import GapK
from synalinks.src.metrics.agents_metrics import PassAtK
from synalinks.src.metrics.agents_metrics import PassHatK
from synalinks.src.metrics.batch_metric import BatchMetric
from synalinks.src.modules.language_models import LanguageModel


class Answer(DataModel):
    answer: str


def _batch(n_correct, n_total, correct="correct", wrong="wrong"):
    """A batch = one problem's `n_total` samples, `n_correct` of them right.

    Returns `(y_true_batch, y_pred_batch)`: every target is the same problem's
    answer; predictions are the per-sample model outputs.
    """
    y_true = [Answer(answer=correct) for _ in range(n_total)]
    y_pred = [Answer(answer=correct) for _ in range(n_correct)]
    y_pred += [Answer(answer=wrong) for _ in range(n_total - n_correct)]
    return y_true, y_pred


class PassAtKTest(testing.TestCase):
    async def test_is_a_batch_metric(self):
        self.assertIsInstance(PassAtK(), BatchMetric)

    async def test_pass_at_1_unbiased(self):
        # n=4, c=1, k=1 -> 1 - C(3,1)/C(4,1) = 1 - 3/4 = 0.25
        metric = PassAtK(k=1)
        y_true, y_pred = _batch(1, 4)
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.25, places=6)

    async def test_pass_at_k_saturates_when_enough_correct(self):
        # n=4, c=1, k=4 -> C(3,4)=0 -> pass@4 = 1.0
        metric = PassAtK(k=4)
        y_true, y_pred = _batch(1, 4)
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, places=6)

    async def test_pass_at_k_zero_when_none_correct(self):
        metric = PassAtK(k=2)
        y_true, y_pred = _batch(0, 4)
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, places=6)

    async def test_k_clamped_to_batch_size(self):
        # n=2, c=1, k=5 -> clamps to k=2 -> 1 - C(1,2)/C(2,2) = 1 - 0 = 1.0
        metric = PassAtK(k=5)
        y_true, y_pred = _batch(1, 2)
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, places=6)

    async def test_single_sample_degenerates_to_pass_at_1(self):
        metric = PassAtK(k=1)
        hit = await metric(Answer(answer="correct"), Answer(answer="correct"))
        self.assertAlmostEqual(hit, 1.0, places=6)
        metric.reset_state()
        miss = await metric(Answer(answer="correct"), Answer(answer="wrong"))
        self.assertAlmostEqual(miss, 0.0, places=6)

    async def test_averages_over_problems(self):
        # Two problems (two batches): pass@1 = 0.25 and 0.75 -> mean 0.5
        metric = PassAtK(k=1)
        yt, yp = _batch(1, 4)
        await metric.update_state(yt, yp)
        yt, yp = _batch(3, 4)
        await metric.update_state(yt, yp)
        self.assertAlmostEqual(metric.result(), 0.5, places=6)

    async def test_direction_is_up(self):
        self.assertEqual(PassAtK().direction, "up")


class PassHatKTest(testing.TestCase):
    async def test_pass_hat_k_consistency(self):
        # n=4, c=2, k=2 -> C(2,2)/C(4,2) = 1/6
        metric = PassHatK(k=2)
        y_true, y_pred = _batch(2, 4)
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0 / 6.0, places=6)

    async def test_pass_hat_k_zero_when_fewer_than_k_correct(self):
        # c=1 < k=2 -> 0.0
        metric = PassHatK(k=2)
        y_true, y_pred = _batch(1, 4)
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 0.0, places=6)

    async def test_pass_hat_k_one_when_all_correct(self):
        # c=n=4, k=2 -> C(4,2)/C(4,2) = 1.0
        metric = PassHatK(k=2)
        y_true, y_pred = _batch(4, 4)
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 1.0, places=6)

    async def test_direction_is_up(self):
        self.assertEqual(PassHatK().direction, "up")


class GapKTest(testing.TestCase):
    async def test_gap_is_pass_at_k_minus_pass_hat_k(self):
        # n=4, c=2, k=2:
        #   pass@2  = 1 - C(2,2)/C(4,2) = 1 - 1/6 = 5/6
        #   pass^2  = C(2,2)/C(4,2)     = 1/6
        #   gap     = 5/6 - 1/6 = 4/6
        metric = GapK(k=2)
        y_true, y_pred = _batch(2, 4)
        score = await metric(y_true, y_pred)
        self.assertAlmostEqual(score, 4.0 / 6.0, places=6)

    async def test_gap_zero_when_all_or_nothing(self):
        metric = GapK(k=2)
        yt, yp = _batch(4, 4)
        self.assertAlmostEqual(await metric(yt, yp), 0.0, places=6)
        metric.reset_state()
        yt, yp = _batch(0, 4)
        self.assertAlmostEqual(await metric(yt, yp), 0.0, places=6)

    async def test_direction_is_down(self):
        self.assertEqual(GapK().direction, "down")


class SerializationTest(testing.TestCase):
    async def test_get_from_config_roundtrip(self):
        metric = PassAtK(k=3, pass_threshold=0.5)
        config = metric.get_config()
        restored = PassAtK.from_config(config)
        self.assertEqual(restored.k, 3)
        self.assertEqual(restored.pass_threshold, 0.5)
        # The reward round-trips into an ExactMatch instance.
        yt, yp = _batch(1, 3)  # n=3, c=1, k=3 -> 1.0
        self.assertAlmostEqual(await restored(yt, yp), 1.0, places=6)


class Question(DataModel):
    question: str


class _ListDataset(Dataset):
    """Tiny in-memory dataset over a list of `{q, a}` rows."""

    def __init__(self, rows, **kwargs):
        super().__init__(**kwargs)
        self._rows = rows

    def _iter_rows(self):
        yield from self._rows

    def __len__(self):
        return self._total_batches(len(self._rows))


def _answer_completion(answer):
    return {"choices": [{"message": {"content": json.dumps({"answer": answer})}}]}


class PassAtKEvaluateIntegrationTest(testing.TestCase):
    """Locks the dataset<->metric contract: `repeat == batch_size == k` makes
    each evaluate batch one problem's k samples, and a stochastic program is
    simulated by returning different completions across those k samples."""

    @patch("litellm.acompletion")
    async def test_evaluate_with_repeat_equals_batch_size(self, mock_completion):
        K = 4
        # Cross-batch order is deterministic (evaluate runs batch-by-batch);
        # within a batch, pass@k depends only on the *count* correct, not order.
        # pass@K is the "solved in at least one of K samples" semantics:
        #   Problem A "Paris":  0/4 correct -> pass@4 = 0.0
        #   Problem B "Berlin": 1/4 correct -> pass@4 = 1.0
        #   mean pass@4 = 0.5   (and per-sample reward = 1/8 = 0.125)
        mock_completion.side_effect = [
            _answer_completion("nope"),
            _answer_completion("nope"),
            _answer_completion("nope"),
            _answer_completion("nope"),
            _answer_completion("Berlin"),
            _answer_completion("nope"),
            _answer_completion("nope"),
            _answer_completion("nope"),
        ]

        x0 = modules.Input(data_model=Question)
        x1 = await modules.Generator(
            data_model=Answer,
            language_model=LanguageModel(model="ollama/mistral"),
        )(x0)
        program = programs.Program(inputs=x0, outputs=x1)

        program.compile(
            reward=rewards.ExactMatch(in_mask=["answer"]),
            metrics=[PassAtK(k=K, reward=rewards.ExactMatch(in_mask=["answer"]))],
        )

        dataset = _ListDataset(
            rows=[
                {"q": "Capital of France?", "a": "Paris"},
                {"q": "Capital of Germany?", "a": "Berlin"},
            ],
            input_data_model=Question,
            output_data_model=Answer,
            input_template='{"question": {{ q | tojson }}}',
            output_template='{"answer": {{ a | tojson }}}',
            batch_size=K,
            repeat=K,
        )

        results = await program.evaluate(x=dataset(), return_dict=True, verbose=0)
        # pass@4 = 0.5 (one problem solved in >=1 of 4), but per-sample reward
        # is only 0.125 -- the gap pass@k captures over plain accuracy.
        self.assertAlmostEqual(results["pass_at_k"], 0.5, places=6)
        self.assertAlmostEqual(results["reward"], 0.125, places=6)


class SampledRewardInputsTest(testing.TestCase):
    async def test_reward_gets_each_samples_inputs(self):
        seen = []

        class InputAware(rewards.Reward):
            async def call(self, y_true, y_pred, x=None):
                seen.append((x and x.get("answer"), y_pred.get("answer")))
                return 1.0

        y_true, y_pred = _batch(n_correct=1, n_total=2)
        metric = PassAtK(k=2, reward=InputAware())
        await metric.update_state(
            y_true, y_pred, x=[Answer(answer="x_correct"), Answer(answer="x_wrong")]
        )
        self.assertEqual(seen, [("x_correct", "correct"), ("x_wrong", "wrong")])
        # Without inputs (e.g. called directly), the reward gets `None`.
        seen.clear()
        await metric.update_state(y_true, y_pred)
        self.assertEqual([x for x, _ in seen], [None, None])

    async def test_reward_function_without_x_still_works(self):
        async def exact(y_true, y_pred):
            return float(y_true.get("answer") == y_pred.get("answer"))

        metric = PassHatK(k=2, reward=exact)
        y_true, y_pred = _batch(n_correct=2, n_total=2)
        await metric.update_state(y_true, y_pred, x=["x0", "x1"])
        self.assertEqual(metric.result(), 1.0)

    @patch("litellm.acompletion")
    async def test_judge_reward_grades_with_the_question_in_evaluate(
        self, mock_completion
    ):
        judge_prompts = []

        async def respond(*args, **kwargs):
            messages = kwargs["messages"]
            if "critique" in messages[0]["content"]:
                judge_prompts.append(messages[-1]["content"])
                content = '{"critique": "Correct.", "reward": 1.0}'
            else:
                content = '{"answer": "Paris"}'
            return {"choices": [{"message": {"content": content}}]}

        mock_completion.side_effect = respond
        language_model = LanguageModel(model="ollama/mistral")
        x0 = modules.Input(data_model=Question)
        x1 = await modules.Generator(data_model=Answer, language_model=language_model)(x0)
        program = programs.Program(inputs=x0, outputs=x1)
        program.compile(
            reward=rewards.ExactMatch(in_mask=["answer"]),
            metrics=[
                PassAtK(k=2, reward=rewards.LMAsJudge(language_model=language_model))
            ],
        )
        results = await program.evaluate(
            x=np.array([Question(question="Capital of France?")] * 2, dtype="object"),
            y=np.array([Answer(answer="Paris")] * 2, dtype="object"),
            batch_size=2,
            return_dict=True,
            verbose=0,
        )
        self.assertEqual(results["pass_at_k"], 1.0)
        self.assertEqual(
            judge_prompts,
            [
                "{'input_question': 'Capital of France?', 'output_answer': 'Paris', "
                "'gold_answer': 'Paris'}"
            ]
            * 2,
        )


class CompiledMetricOwnershipTest(testing.TestCase):
    """`compile` clones metrics but keeps their rewards, and binds by name."""

    async def _evaluate(self, metrics_list, respond):
        from synalinks.src.modules.language_models import LanguageModel

        with patch("litellm.acompletion", side_effect=respond):
            inputs = modules.Input(data_model=Question)
            outputs = await modules.Generator(
                data_model=Answer, language_model=LanguageModel(model="ollama/mistral")
            )(inputs)
            program = programs.Program(inputs=inputs, outputs=outputs)
            program.compile(reward=rewards.ExactMatch(), metrics=metrics_list)
            return await program.evaluate(
                x=np.array([Question(question="Q?")] * 2, dtype="object"),
                y=np.array([Answer(answer="Paris")] * 2, dtype="object"),
                batch_size=2,
                return_dict=True,
                verbose=0,
            )

    @staticmethod
    async def _respond(*args, **kwargs):
        class Response(dict):
            _hidden_params = {"response_cost": 0.01}

        judging = "critique" in kwargs["messages"][0]["content"]
        content = (
            '{"critique": "ok", "reward": 1.0}' if judging else '{"answer": "Paris"}'
        )
        return Response(
            choices=[{"message": {"content": content}}],
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    async def test_function_reward_compiles(self):
        async def exact(y_true, y_pred):
            return float(y_true.get_json() == y_pred.get_json())

        logs = await self._evaluate([PassAtK(k=2, reward=exact)], self._respond)
        self.assertEqual(logs["pass_at_k"], 1.0)

    async def test_judge_keeps_the_users_model_and_is_billed(self):
        from synalinks.src import metrics
        from synalinks.src.modules.language_models import LanguageModel

        judge_lm = LanguageModel(model="ollama/judge")
        logs = await self._evaluate(
            [
                PassAtK(k=2, reward=rewards.LMAsJudge(language_model=judge_lm)),
                metrics.ProgramCost(),
                metrics.RewardCost(),
            ],
            self._respond,
        )
        self.assertAlmostEqual(judge_lm.cumulated_cost, 0.02)
        self.assertAlmostEqual(logs["reward_cost"], 0.02)
        self.assertAlmostEqual(logs["program_cost"], 0.04)

    async def test_operational_metric_by_name_is_bound(self):
        logs = await self._evaluate(["ProgramCalls", "total_tokens"], self._respond)
        self.assertEqual(logs["ProgramCalls"], 2.0)
        self.assertEqual(logs["total_tokens"], 30.0)
