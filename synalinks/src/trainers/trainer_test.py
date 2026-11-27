# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import ast
import asyncio
import collections
import json
from unittest.mock import patch

import numpy as np

from synalinks.src import metrics
from synalinks.src import modules
from synalinks.src import optimizers
from synalinks.src import programs
from synalinks.src import rewards
from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import JsonDataModel
from synalinks.src.modules.language_models import LanguageModel
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
    async def test_fit_and_evaluate_a_source_of_unknown_length(self, mock_completion):
        """A source whose `__len__` raises, as a streaming dataset's does, is read to
        its end."""
        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"rationale": "Known.", "answer": "Paris"})
                    }
                }
            ]
        }

        class Streaming:
            def __init__(self, x, y):
                self.x, self.y = x, y

            def __iter__(self):
                for i in range(len(self.x)):
                    yield self.x[i : i + 1], self.y[i : i + 1]

            def __len__(self):
                raise NotImplementedError("Streaming datasets have unknown length.")

        program = await program_test()
        program.compile(
            optimizer=optimizers.random_few_shot.RandomFewShot(),
            reward=rewards.ExactMatch(in_mask=["answer"]),
        )
        (x_train, y_train), (x_test, y_test) = load_test_data()

        await program.fit(
            x=Streaming(x_train, y_train), epochs=1, validation_data=(x_test, y_test)
        )
        result = await program.evaluate(x=Streaming(x_test, y_test), return_dict=True)
        self.assertIn("reward", result)

    @patch("litellm.acompletion")
    async def test_evaluate_predicts_each_sample_once_when_unbuilt(self, mock_completion):
        """An unbuilt program must not be run twice on the batch it builds on.

        Building a program means calling it, and `evaluate` builds on the first
        batch — which is also the first batch it scores. Dropping that build's
        predictions and recomputing them costs a second full forward pass per
        sample in the batch, which for an agent program is a second agent run:
        every LM call and every tool call paid twice, for a result thrown away.

        The program here keeps a module its `call()` never reaches, which is
        what makes this permanent rather than a one-off. `built` is only set by
        calling a module, so a module on an untaken path never becomes built,
        `all(module.built ...)` never becomes true, and *every* `evaluate` pays
        the extra pass. Agents are full of such modules — a fallback generator
        that only runs when the loop ends without submitting, a tool the model
        happens not to call.
        """
        mock_answer = AnswerWithRationale(
            rationale="""The capital of France is well-known and is the seat of """
            """the French government.""",
            answer="Paris",
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_answer.get_json())}}]
        }

        language_model = LanguageModel(model="ollama/mistral")

        class ProgramWithAnUntakenPath(programs.Program):
            forward_passes = 0

            def __init__(self, language_model=None):
                super().__init__()
                self.answer = modules.Generator(
                    data_model=AnswerWithRationale,
                    language_model=language_model,
                )
                self.fallback = modules.Generator(
                    data_model=AnswerWithRationale,
                    language_model=language_model,
                )

            async def call(self, inputs, training=False):
                type(self).forward_passes += 1
                return await self.answer(inputs)

        program = ProgramWithAnUntakenPath(language_model=language_model)

        program.compile(
            optimizer=optimizers.random_few_shot.RandomFewShot(),
            reward=rewards.ExactMatch(in_mask=["answer"]),
            metrics=[
                metrics.MeanMetricWrapper(rewards.exact_match, in_mask=["answer"]),
            ],
        )

        (_, _), (x_test, y_test) = load_test_data()

        self.assertFalse(all(module.built for module in program._flatten_modules()))

        _ = await program.evaluate(x=x_test, y=y_test)

        # The build pass predicted the whole first batch, and those predictions
        # are what the first `test_on_batch` scores — so each sample is
        # predicted once, not twice. The one extra pass is paid once per
        # program, not once per sample, and is not what this reuse addresses.
        self.assertLessEqual(ProgramWithAnUntakenPath.forward_passes, len(x_test) + 1)

        # `fallback` is still unbuilt, so the next `evaluate` auto-builds again
        # — and must again cost exactly one pass per sample. This is what makes
        # the duplication permanent rather than a first-call toll.
        before = ProgramWithAnUntakenPath.forward_passes
        _ = await program.evaluate(x=x_test, y=y_test)
        self.assertFalse(all(module.built for module in program._flatten_modules()))
        self.assertEqual(ProgramWithAnUntakenPath.forward_passes - before, len(x_test))

    @patch("litellm.acompletion")
    async def test_fit_and_evaluate_without_targets(self, mock_completion):
        """A reward that needs no target (a judge) trains and evaluates on
        inputs alone, even when the program is unbuilt and `fit`/`evaluate`
        build it on a batch that has no `y`."""
        mock_answer = AnswerWithRationale(rationale="It is well known.", answer="Paris")
        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_answer.get_json())}}]
        }
        language_model = LanguageModel(model="ollama/mistral")

        class Graded(rewards.Reward):
            async def call(self, y_true, y_pred):
                assert y_true is None
                return 1.0 if y_pred.get("answer") == "Paris" else 0.0

        class ProgramWithAnUntakenPath(programs.Program):
            def __init__(self, language_model=None):
                super().__init__()
                self.answer = modules.Generator(
                    data_model=AnswerWithRationale, language_model=language_model
                )
                self.fallback = modules.Generator(
                    data_model=AnswerWithRationale, language_model=language_model
                )

            async def call(self, inputs, training=False):
                return await self.answer(inputs)

        program = ProgramWithAnUntakenPath(language_model=language_model)
        program.compile(
            optimizer=optimizers.random_few_shot.RandomFewShot(), reward=Graded()
        )
        (x_train, _), (x_test, _) = load_test_data()

        self.assertEqual((await program.evaluate(x=x_test))["reward"], 1.0)
        history = await program.fit(x=x_train, epochs=1, validation_split=0.5)
        self.assertEqual(history.history["reward"], [1.0])
        self.assertEqual(history.history["val_reward"], [1.0])

    @patch("litellm.acompletion")
    async def test_evaluate_reward_is_per_sample_mean_with_uneven_last_batch(
        self, mock_completion
    ):
        """The reported reward is the compiled reduction over every evaluated
        sample, whatever the batching: a short last batch must not weigh as much
        as a full one, and sum/min/max are not averaged across batches."""
        mock_answer = AnswerWithRationale(rationale="It is well known.", answer="Paris")
        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_answer.get_json())}}]
        }
        x = np.array([Query(query="q")] * 3, dtype="object")
        y = np.array(
            [
                AnswerWithRationale(rationale="r", answer=answer)
                for answer in ("Paris", "Paris", "Lyon")
            ],
            dtype="object",
        )
        expected = {"mean": 2 / 3, "none": 2 / 3, "sum": 2.0, "min": 0.0, "max": 1.0}

        for reduction, value in expected.items():
            program = ExampleProgram(
                data_model=AnswerWithRationale,
                language_model=LanguageModel(model="ollama/mistral"),
            )
            program.compile(
                reward=rewards.RewardFunctionWrapper(
                    rewards.exact_match, in_mask=["answer"], reduction=reduction
                )
            )
            for batch_size in (2, 3):
                result = await program.evaluate(
                    x=x, y=y, batch_size=batch_size, return_dict=True
                )
                self.assertAlmostEqual(result["reward"], value, msg=reduction)

    @patch("litellm.acompletion")
    async def test_train_reward_is_per_sample_mean_with_uneven_batches(
        self, mock_completion
    ):
        """The training reward is the mean over scored samples, not the mean of
        per-step means: a step scored on fewer samples weighs less."""
        mock_answer = AnswerWithRationale(rationale="It is well known.", answer="Paris")
        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_answer.get_json())}}]
        }
        program = ExampleProgram(
            data_model=AnswerWithRationale,
            language_model=LanguageModel(model="ollama/mistral"),
        )
        program.compile(
            optimizer=optimizers.random_few_shot.RandomFewShot(),
            reward=rewards.ExactMatch(in_mask=["answer"]),
        )
        x = np.array([Query(query="q")] * 3, dtype="object")
        y = np.array(
            [
                AnswerWithRationale(rationale="r", answer=answer)
                for answer in ("Paris", "Paris", "Lyon")
            ],
            dtype="object",
        )
        await program(x[0])
        program.reset_metrics()

        for step, batch in enumerate((slice(0, 2), slice(2, 3))):
            await program.train_on_batch(
                step=step, x=x[batch], y=y[batch], val_x=x[batch], val_y=y[batch]
            )

        self.assertAlmostEqual(program._reward_tracker.result(), 2 / 3)

    async def test_fit_one_shot_generator_with_many_epochs_raises(self):
        program = await program_test()
        program.compile(
            optimizer=optimizers.random_few_shot.RandomFewShot(),
            reward=rewards.ExactMatch(in_mask=["answer"]),
        )
        (x, y), (x_test, y_test) = load_test_data()
        batches = ((x[i : i + 1], y[i : i + 1]) for i in range(len(x)))
        with self.assertRaisesRegex(ValueError, "one-shot generator"):
            await program.fit(x=batches, epochs=2, validation_data=(x_test, y_test))

    @patch("litellm.acompletion")
    async def test_fit_shuffle_is_forwarded(self, mock_completion):
        mock_answer = AnswerWithRationale(rationale="r", answer="Paris")
        mock_completion.return_value = {
            "choices": [{"message": {"content": json.dumps(mock_answer.get_json())}}]
        }
        (x, y), _ = load_test_data()
        for shuffle in (False, True):
            program = await program_test()
            program.compile(
                optimizer=optimizers.random_few_shot.RandomFewShot(),
                reward=rewards.ExactMatch(in_mask=["answer"]),
            )
            seen = []
            train_on_batch = program.train_on_batch

            async def record(step, x, y=None, **kwargs):
                seen.append([q.query for q in x])
                return await train_on_batch(step, x, y, **kwargs)

            program.train_on_batch = record
            np.random.seed(0)
            await program.fit(x=x, y=y, batch_size=4, epochs=2, shuffle=shuffle)
            epochs = seen[: len(seen) // 2], seen[len(seen) // 2 :]
            self.assertEqual(epochs[0] == epochs[1], not shuffle)

    async def test_fit_without_validation_data_says_why(self):
        program = await program_test()
        program.compile(
            optimizer=optimizers.random_few_shot.RandomFewShot(),
            reward=rewards.ExactMatch(in_mask=["answer"]),
        )
        (x_train, y_train), _ = load_test_data()
        with self.assertRaisesRegex(ValueError, "needs validation data"):
            await program.fit(x=x_train, y=y_train, epochs=1, validation_split=0)

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


class TestCompiledMetricIsolation(testing.TestCase):
    """`compile()` takes ownership of its metrics: each program gets its OWN
    metric instances, so sharing one metrics list across the per-model
    programs of a sweep never lets state leak between programs. Covers both
    operational metrics (which bind to a program's LanguageModel and would
    otherwise all end up bound to the last program compiled) and ordinary
    metrics (whose accumulated state would otherwise carry across programs).
    """

    async def _program(self, model):
        x0 = modules.Input(data_model=Query)
        x1 = await modules.Generator(
            data_model=AnswerWithRationale,
            language_model=LanguageModel(model=model),
        )(x0)
        return programs.Program(inputs=x0, outputs=x1)

    def _bound_op_metric(self, program):
        """The operational metric `compile()` bound to this program."""
        from synalinks.src import tree

        for m in tree.flatten(program._compile_metrics._user_metrics):
            if hasattr(m, "bind_program"):
                return m
        raise AssertionError("no bound operational metric found")

    async def test_shared_operational_metric_binds_per_program(self):
        from synalinks.src.metrics.lm_metrics import Cost
        from synalinks.src.metrics.metrics_utils import language_models

        model_names = ["ollama/model-a", "ollama/model-b", "ollama/model-c"]
        costs = [0.01, 0.02, 0.03]

        # ONE metric instance reused across every program in the sweep.
        shared = Cost()
        progs = []
        for name in model_names:
            program = await self._program(name)
            program.compile(reward=rewards.ExactMatch(), metrics=[shared])
            progs.append(program)

        # Each model runs and accumulates its OWN provider cost.
        for program, cost in zip(progs, costs):
            language_models(program)[0].inference_cumulated_cost += cost

        bound = [self._bound_op_metric(p) for p in progs]

        # Each program must own a distinct bound metric...
        self.assertEqual(len({id(m) for m in bound}), len(progs))
        # ...reading only its own model's counters.
        for program, metric, cost in zip(progs, bound, costs):
            self.assertEqual(
                language_models(program),
                metric.language_models,
            )
            self.assertAlmostEqual(metric.result(), cost)

    async def test_compile_clones_metrics_so_programs_never_share(self):
        """A metric instance passed to `compile()` is cloned, not adopted:
        the caller's instance is never wired into any program, and each
        program owns a distinct clone -- for both ordinary and operational
        metrics."""
        from synalinks.src import metrics as metrics_module
        from synalinks.src import tree
        from synalinks.src.metrics.lm_metrics import Cost

        async def custom_metric(y_true, y_pred):
            return 0.0

        shared_ordinary = metrics_module.MeanMetricWrapper(custom_metric, name="custom")
        shared_ops = Cost()

        progs = []
        for name in ["ollama/model-a", "ollama/model-b"]:
            program = await self._program(name)
            program.compile(
                reward=rewards.ExactMatch(),
                metrics=[shared_ordinary, shared_ops],
            )
            progs.append(program)

        def owned(program):
            return list(tree.flatten(program._compile_metrics._user_metrics))

        for program in progs:
            instances = owned(program)
            # The caller's instances were cloned away, never adopted.
            self.assertNotIn(shared_ordinary, instances)
            self.assertNotIn(shared_ops, instances)
            # Clones keep the type, name, and (for the wrapper) the fn BY
            # REFERENCE -- the wrapped callable is shared, never deep-copied.
            by_name = {m.name: m for m in instances}
            self.assertIsInstance(by_name["custom"], metrics_module.MeanMetricWrapper)
            self.assertIs(by_name["custom"]._fn, custom_metric)

        # The two programs own distinct clones (nothing shared between them).
        a = {m.name: id(m) for m in owned(progs[0])}
        b = {m.name: id(m) for m in owned(progs[1])}
        for name in ("custom", "cost"):
            self.assertNotEqual(a[name], b[name])

        # The caller's shared instances stay pristine (never bound / built).
        self.assertEqual(shared_ops.language_models, [])


class TestCompileStringIdentifiers(testing.TestCase):
    """Keras-style: pass lowercase strings to `compile(...)` instead of
    instances. Lookup is case-insensitive: CamelCase still works but
    lowercase is the idiomatic form."""

    def _make_program(self):
        return ExampleProgram(
            data_model=Query,
            language_model=LanguageModel(model="ollama/mistral"),
        )

    def test_optimizer_string(self):
        program = self._make_program()
        program.compile(optimizer="randomfewshot", reward=rewards.ExactMatch())
        self.assertIsInstance(program.optimizer, optimizers.RandomFewShot)

    def test_optimizer_string_camelcase_also_works(self):
        program = self._make_program()
        program.compile(optimizer="RandomFewShot", reward=rewards.ExactMatch())
        self.assertIsInstance(program.optimizer, optimizers.RandomFewShot)

    def test_reward_string(self):
        program = self._make_program()
        program.compile(optimizer="randomfewshot", reward="exactmatch")
        self.assertIsInstance(program.reward, rewards.ExactMatch)

    def test_reward_string_camelcase_also_works(self):
        program = self._make_program()
        program.compile(optimizer="randomfewshot", reward="ExactMatch")
        self.assertIsInstance(program.reward, rewards.ExactMatch)

    def test_metrics_strings(self):
        program = self._make_program()
        program.compile(
            optimizer="randomfewshot",
            reward="exactmatch",
            metrics=["mean"],
        )
        # Metrics get wrapped in CompileMetrics; the underlying instance is
        # constructed via metrics.get(...).
        self.assertIsNotNone(program._compile_metrics)

    def test_full_keras_style_compile(self):
        program = self._make_program()
        program.compile(
            optimizer="randomfewshot",
            reward="exactmatch",
            metrics=["mean"],
        )
        self.assertIsInstance(program.optimizer, optimizers.RandomFewShot)
        self.assertIsInstance(program.reward, rewards.ExactMatch)
        self.assertTrue(program.compiled)

    def test_unknown_optimizer_string_raises(self):
        program = self._make_program()
        with self.assertRaises(ValueError):
            program.compile(
                optimizer="notarealoptimizer",
                reward=rewards.ExactMatch(),
            )

    def test_unknown_reward_string_raises(self):
        program = self._make_program()
        with self.assertRaises(ValueError):
            program.compile(
                optimizer="randomfewshot",
                reward="notarealreward",
            )

    def test_instance_pass_through_unchanged(self):
        program = self._make_program()
        opt = optimizers.RandomFewShot()
        rew = rewards.ExactMatch()
        program.compile(optimizer=opt, reward=rew)
        # Instances should be stored as-is (no re-instantiation).
        self.assertIs(program.optimizer, opt)
        self.assertIs(program.reward, rew)

    def test_dict_config_optimizer(self):
        program = self._make_program()
        program.compile(
            optimizer={"class_name": "randomfewshot", "config": {}},
            reward=rewards.ExactMatch(),
        )
        self.assertIsInstance(program.optimizer, optimizers.RandomFewShot)


class StratifiedMinibatchTest(testing.TestCase):
    def _targets(self, labels):
        return np.array(
            [
                JsonDataModel(json={"reward": r}, schema={"type": "object"})
                for r in labels
            ],
            dtype="object",
        )

    def test_minibatch_spans_the_target_classes(self):
        from synalinks.src.trainers.trainer import stratified_minibatch_indices

        y = self._targets([0.0] * 10 + [0.5] * 10 + [1.0] * 10)
        rng = np.random.default_rng(0)
        for _ in range(20):
            idx = stratified_minibatch_indices(y, 3, rng=rng)
            self.assertEqual(len(idx), 3)
            self.assertEqual(len(set(idx.tolist())), 3)
            self.assertEqual(sorted(y[i].get("reward") for i in idx), [0.0, 0.5, 1.0])

    def test_round_robin_when_size_exceeds_number_of_classes(self):
        from synalinks.src.trainers.trainer import stratified_minibatch_indices

        y = self._targets([0.0] * 10 + [1.0] * 10)
        idx = stratified_minibatch_indices(y, 4, rng=np.random.default_rng(1))
        labels = sorted(y[i].get("reward") for i in idx)
        self.assertEqual(labels, [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(len(set(idx.tolist())), 4)

    def test_falls_back_to_random_when_targets_do_not_repeat(self):
        from synalinks.src.trainers.trainer import stratified_minibatch_indices

        y = self._targets([i / 100 for i in range(30)])
        idx = stratified_minibatch_indices(y, 4, rng=np.random.default_rng(2))
        self.assertEqual(len(idx), 4)
        self.assertEqual(len(set(idx.tolist())), 4)

    def test_size_is_capped_at_population(self):
        from synalinks.src.trainers.trainer import stratified_minibatch_indices

        y = self._targets([0.0, 1.0, 0.0])
        idx = stratified_minibatch_indices(y, 10, rng=np.random.default_rng(3))
        self.assertEqual(sorted(idx.tolist()), [0, 1, 2])


class Question(DataModel):
    question: str


class Reply(DataModel):
    reply: str


class RecordingReward(rewards.Reward):
    """Records which (x, y_true, y_pred) triple each reward call sees."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen = []

    async def call(self, y_true, y_pred, x=None):
        self.seen.append((x.get("question"), y_true.get("reply"), y_pred.get("reply")))
        return 1.0


async def _reply_out_of_order(*args, **kwargs):
    # Later questions answer first, so a misaligned `gather` would show.
    question = ast.literal_eval(kwargs["messages"][-1]["content"])["question"]
    await asyncio.sleep(0.01 * (5 - int(question[-1])))
    return {
        "choices": [{"message": {"content": json.dumps({"reply": f"re {question}"})}}]
    }


class TestRewardInputs(testing.TestCase):
    @patch("litellm.acompletion", side_effect=_reply_out_of_order)
    async def test_each_reward_gets_its_own_sample_inputs(self, _):
        language_model = LanguageModel(model="ollama/mistral")
        x0 = modules.Input(data_model=Question)
        x1 = await modules.Generator(data_model=Reply, language_model=language_model)(x0)
        program = programs.Program(inputs=x0, outputs=x1)
        reward = RecordingReward()
        program.compile(reward=reward)

        questions = [f"q{i}" for i in range(5)]
        await program.evaluate(
            x=np.array([Question(question=q) for q in questions], dtype="object"),
            y=np.array([Reply(reply=f"re {q}") for q in questions], dtype="object"),
            batch_size=2,
        )
        self.assertEqual(
            sorted(reward.seen), [(q, f"re {q}", f"re {q}") for q in questions]
        )

    @patch("litellm.acompletion", side_effect=_reply_out_of_order)
    async def test_every_kind_of_reward_and_metric_gets_its_sample_inputs(self, _):
        seen = collections.defaultdict(list)

        async def fn_reward(y_true, y_pred, x=None):
            seen["function"].append((x.get("question"), y_pred.get("reply")))
            return 1.0

        async def kwargs_reward(y_true, y_pred, **kwargs):
            seen["kwargs"].append((kwargs["x"].get("question"), y_pred.get("reply")))
            return 1.0

        async def fn_metric(y_true, y_pred, x=None):
            seen["mean_metric"].append((x.get("question"), y_pred.get("reply")))
            return 1.0

        class InputMetric(metrics.Metric):
            async def update_state(self, y_true, y_pred, x=None):
                seen["metric"].append((x.get("question"), y_pred.get("reply")))

            def result(self):
                return 0.0

        language_model = LanguageModel(model="ollama/mistral")
        x0 = modules.Input(data_model=Question)
        x1 = await modules.Generator(data_model=Reply, language_model=language_model)(x0)
        program = programs.Program(inputs=x0, outputs=x1)
        nested = RecordingReward()
        program.compile(
            reward=rewards.ComposableReward([fn_reward, kwargs_reward, nested]),
            metrics=[
                InputMetric(name="input_metric"),
                metrics.MeanMetricWrapper(fn=fn_metric, name="mean_metric"),
            ],
        )

        questions = [f"q{i}" for i in range(5)]
        await program.evaluate(
            x=np.array([Question(question=q) for q in questions], dtype="object"),
            y=np.array([Reply(reply=f"re {q}") for q in questions], dtype="object"),
            batch_size=2,
        )
        expected = [(q, f"re {q}") for q in questions]
        for kind in ("function", "kwargs", "metric", "mean_metric"):
            self.assertEqual(sorted(seen[kind]), expected, kind)
        self.assertEqual(
            sorted(nested.seen), [(q, f"re {q}", f"re {q}") for q in questions]
        )


class TestOptimizerEpochEndMetrics(testing.TestCase):
    async def test_epoch_end_work_counts_in_the_optimizer_phase(self):
        """Model calls an optimizer makes at the end of an epoch (OMEGA embeds
        its candidates there) count in the optimizer phase and in that epoch's
        optimizer metrics, even with a validation pass in between."""
        from synalinks.src.backend import EmbeddingRequest
        from synalinks.src.modules.embedding_models import EmbeddingModel

        answer = json.dumps({"rationale": "r", "answer": "Paris"})
        completion = {"choices": [{"message": {"content": answer}}]}
        embedding = {
            "data": [{"embedding": [1.0, 0.0]}],
            "usage": {"prompt_tokens": 7, "total_tokens": 7},
        }

        class EmbeddingOptimizer(optimizers.random_few_shot.RandomFewShot):
            async def on_epoch_end(self, epoch, trainable_variables, **kwargs):
                await self.embedding_model(EmbeddingRequest(texts=["a"]))
                await super().on_epoch_end(epoch, trainable_variables, **kwargs)

        with (
            patch("litellm.acompletion", return_value=completion),
            patch("litellm.aembedding", return_value=embedding),
        ):
            embedding_model = EmbeddingModel(model="ollama/mxbai-embed-large")
            optimizer = EmbeddingOptimizer()
            optimizer.embedding_model = embedding_model
            inputs = modules.Input(data_model=Query)
            outputs = await modules.Generator(
                data_model=AnswerWithRationale,
                language_model=LanguageModel(model="ollama/mistral"),
            )(inputs)
            program = programs.Program(inputs=inputs, outputs=outputs)
            program.compile(
                optimizer=optimizer,
                reward=rewards.ExactMatch(in_mask=["answer"]),
                metrics=[metrics.OptimizerEmbeddingTokens()],
            )
            (x, y), (x_val, y_val) = load_test_data()
            history = await program.fit(
                x=x, y=y, validation_data=(x_val, y_val), epochs=2, verbose=0
            )
        self.assertEqual(history.history["optimizer_embedding_tokens"], [7.0, 7.0])
        self.assertEqual(embedding_model.optimizer_cumulated_tokens, 14)
