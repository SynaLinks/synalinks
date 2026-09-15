# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import warnings

from synalinks.src import testing
from synalinks.src.callbacks.budget_stopping import BudgetStopping
from synalinks.src.modules.embedding_models import EmbeddingModel
from synalinks.src.modules.language_models import LanguageModel


class _FakeModule:
    def __init__(self, lm=None, em=None):
        self.language_model = lm
        self.embedding_model = em
        self._modules = []


class _FakeProgram:
    """Mirrors the slice of `Program` the callback relies on: the stop
    flags and `_flatten_modules` to reach the models."""

    def __init__(self, modules):
        self.stop_training = False
        self.stop_evaluating = False
        self.stop_predicting = False
        self._modules = list(modules)

    def _flatten_modules(self, include_self=True, recursive=True):
        out = [self] if include_self else []
        out.extend(self._modules)
        return out


def _spend(model, cost=0.0, tokens=0):
    model.cumulated_cost += cost
    model.cumulated_tokens += tokens


def _setup(cb, *models):
    program = _FakeProgram([_FakeModule(lm=m) for m in models])
    cb.set_program(program)
    cb.on_train_begin()
    return program


class BudgetStoppingInitTest(testing.TestCase):
    def test_requires_a_budget(self):
        with self.assertRaisesRegex(ValueError, "at least one budget"):
            BudgetStopping()

    def test_rejects_non_positive_budgets(self):
        with self.assertRaisesRegex(ValueError, "max_cost"):
            BudgetStopping(max_cost=0)
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            BudgetStopping(max_tokens=-1)


class BudgetStoppingCostTest(testing.TestCase):
    def test_stops_when_cost_budget_reached(self):
        lm = LanguageModel(model="ollama/mistral")
        cb = BudgetStopping(max_cost=1.0)
        program = _setup(cb, lm)

        _spend(lm, cost=0.4, tokens=100)
        cb.on_train_batch_end(0)
        self.assertFalse(program.stop_training)

        _spend(lm, cost=0.6, tokens=100)
        cb.on_train_batch_end(1)
        self.assertTrue(program.stop_training)
        self.assertEqual(cb.stopped_epoch, 0)
        self.assertEqual(cb.stopped_batch, 1)
        self.assertIn("cost budget", cb.stop_reason)
        self.assertAlmostEqual(cb.spent_cost, 1.0)

    def test_only_counts_spend_since_train_begin(self):
        lm = LanguageModel(model="ollama/mistral")
        _spend(lm, cost=10.0, tokens=5000)
        cb = BudgetStopping(max_cost=1.0)
        program = _setup(cb, lm)

        self.assertEqual(cb.spent_cost, 0.0)
        _spend(lm, cost=0.5)
        cb.on_train_batch_end(0)
        self.assertFalse(program.stop_training)
        _spend(lm, cost=0.5)
        cb.on_train_batch_end(1)
        self.assertTrue(program.stop_training)

    def test_sums_over_language_and_embedding_models_and_fallbacks(self):
        fallback = LanguageModel(model="ollama/qwen3")
        lm = LanguageModel(model="ollama/mistral", fallback=fallback)
        em = EmbeddingModel(model="ollama/mxbai-embed-large")
        cb = BudgetStopping(max_cost=1.0)
        program = _FakeProgram([_FakeModule(lm=lm), _FakeModule(em=em)])
        cb.set_program(program)
        cb.on_train_begin()

        _spend(lm, cost=0.4)
        _spend(fallback, cost=0.3)
        _spend(em, cost=0.2)
        cb.on_train_batch_end(0)
        self.assertFalse(program.stop_training)
        self.assertAlmostEqual(cb.spent_cost, 0.9)

        _spend(em, cost=0.1)
        cb.on_train_batch_end(1)
        self.assertTrue(program.stop_training)

    def test_warns_once_when_provider_reports_no_cost(self):
        lm = LanguageModel(model="ollama/mistral")
        cb = BudgetStopping(max_cost=1.0)
        program = _setup(cb, lm)
        _spend(lm, tokens=200)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cb.on_epoch_end(0)
            cb.on_epoch_end(1)
        self.assertEqual(len(w), 1)
        self.assertIn("did not report any cost", str(w[0].message))
        self.assertFalse(program.stop_training)

    def test_no_warning_when_cost_is_reported(self):
        lm = LanguageModel(model="ollama/mistral")
        cb = BudgetStopping(max_cost=1.0)
        _setup(cb, lm)
        _spend(lm, cost=0.01, tokens=200)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            cb.on_epoch_end(0)
        self.assertEqual(len(w), 0)


class BudgetStoppingTokensTest(testing.TestCase):
    def test_stops_when_token_budget_reached(self):
        lm = LanguageModel(model="ollama/mistral")
        cb = BudgetStopping(max_tokens=1000)
        program = _setup(cb, lm)

        _spend(lm, tokens=600)
        cb.on_train_batch_end(0)
        self.assertFalse(program.stop_training)

        _spend(lm, tokens=400)
        cb.on_train_batch_end(1)
        self.assertTrue(program.stop_training)
        self.assertIn("token budget", cb.stop_reason)
        self.assertEqual(cb.spent_tokens, 1000)

    def test_checks_at_epoch_end(self):
        lm = LanguageModel(model="ollama/mistral")
        cb = BudgetStopping(max_tokens=100)
        program = _setup(cb, lm)
        cb.on_epoch_begin(2)
        _spend(lm, tokens=100)
        cb.on_epoch_end(2)
        self.assertTrue(program.stop_training)
        self.assertEqual(cb.stopped_epoch, 2)
        self.assertIsNone(cb.stopped_batch)

    def test_first_budget_reached_wins(self):
        lm = LanguageModel(model="ollama/mistral")
        cb = BudgetStopping(max_cost=100.0, max_tokens=10)
        program = _setup(cb, lm)
        _spend(lm, cost=0.01, tokens=10)
        cb.on_train_batch_end(0)
        self.assertTrue(program.stop_training)
        self.assertIn("token budget", cb.stop_reason)


class BudgetStoppingReuseTest(testing.TestCase):
    def test_reusable_across_runs(self):
        lm = LanguageModel(model="ollama/mistral")
        cb = BudgetStopping(max_tokens=10, verbose=1)
        program = _setup(cb, lm)
        _spend(lm, tokens=10)
        cb.on_train_batch_end(0)
        self.assertTrue(program.stop_training)
        cb.on_train_end()

        program.stop_training = False
        cb.on_train_begin()
        self.assertEqual(cb.spent_tokens, 0)
        self.assertIsNone(cb.stop_reason)
        _spend(lm, tokens=5)
        cb.on_train_batch_end(0)
        self.assertFalse(program.stop_training)


class BudgetStoppingEvaluateTest(testing.TestCase):
    def test_stops_standalone_evaluation(self):
        lm = LanguageModel(model="ollama/mistral")
        _spend(lm, cost=10.0, tokens=5000)
        cb = BudgetStopping(max_cost=1.0, verbose=1)
        program = _FakeProgram([_FakeModule(lm=lm)])
        cb.set_program(program)

        cb.on_test_begin()
        self.assertEqual(cb.spent_cost, 0.0)
        _spend(lm, cost=0.5)
        cb.on_test_batch_end(0)
        self.assertFalse(program.stop_evaluating)
        _spend(lm, cost=0.5)
        cb.on_test_batch_end(1)
        self.assertTrue(program.stop_evaluating)
        self.assertEqual(cb.stopped_batch, 1)
        cb.on_test_end()

    def test_stops_standalone_prediction(self):
        lm = LanguageModel(model="ollama/mistral")
        _spend(lm, tokens=5000)
        cb = BudgetStopping(max_tokens=100, verbose=1)
        program = _FakeProgram([_FakeModule(lm=lm)])
        cb.set_program(program)

        # `predict()` fires the predict hooks, not the test ones: the budget
        # baseline is taken at predict begin and earlier spend is ignored.
        cb.on_predict_begin()
        self.assertEqual(cb.spent_tokens, 0)
        _spend(lm, tokens=60)
        cb.on_predict_batch_end(0)
        self.assertFalse(program.stop_predicting)
        _spend(lm, tokens=40)
        cb.on_predict_batch_end(1)
        self.assertTrue(program.stop_predicting)
        self.assertIn("token budget", cb.stop_reason)
        self.assertEqual(cb._batch_kind, "prediction batch")
        cb.on_predict_end()

    def test_validation_inside_fit_shares_the_training_budget(self):
        lm = LanguageModel(model="ollama/mistral")
        cb = BudgetStopping(max_cost=1.0)
        program = _setup(cb, lm)

        _spend(lm, cost=0.7)
        cb.on_train_batch_end(0)
        self.assertFalse(program.stop_training)

        cb.on_test_begin()
        self.assertAlmostEqual(cb.spent_cost, 0.7)
        _spend(lm, cost=0.3)
        cb.on_test_batch_end(0)
        self.assertTrue(program.stop_evaluating)
        self.assertTrue(program.stop_training)
        cb.on_test_end()
        cb.on_epoch_end(0)
        cb.on_train_end()
        self.assertIn("cost budget", cb.stop_reason)
