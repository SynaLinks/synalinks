# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
from unittest.mock import patch

from synalinks.src import metrics
from synalinks.src import modules
from synalinks.src import optimizers
from synalinks.src import programs
from synalinks.src import rewards
from synalinks.src import testing
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
        from synalinks.src.metrics.lm_metrics import _collect_language_models

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
            _collect_language_models(program)[0].inference_cumulated_cost += cost

        bound = [self._bound_op_metric(p) for p in progs]

        # Each program must own a distinct bound metric...
        self.assertEqual(len({id(m) for m in bound}), len(progs))
        # ...reading only its own model's counters.
        for program, metric, cost in zip(progs, bound, costs):
            self.assertEqual(
                _collect_language_models(program),
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

        shared_ordinary = metrics_module.MeanMetricWrapper(
            custom_metric, name="custom"
        )
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
            self.assertIsInstance(
                by_name["custom"], metrics_module.MeanMetricWrapper
            )
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
    instances. Lookup is case-insensitive — CamelCase still works but
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
