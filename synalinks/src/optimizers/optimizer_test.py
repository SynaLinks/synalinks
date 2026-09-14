# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend.common.json_data_model import JsonDataModel
from synalinks.src.optimizers.optimizer import Optimizer


def _trainable(history=None, candidates=None, best_candidates=None, **extra):
    """Build a JsonDataModel shaped like a `Trainable` variable.

    Only the fields `on_epoch_end` touches are wired up, keeping the
    fixture minimal so each test reads as a contract assertion."""
    payload = {
        "candidates": candidates if candidates is not None else [],
        "best_candidates": best_candidates if best_candidates is not None else [],
        "history": history if history is not None else [],
        "examples": [],
        "current_predictions": [],
        "predictions": [],
        "seed_candidates": [],
        "nb_visit": 0,
        "cumulative_reward": 0.0,
        **extra,
    }
    schema = {
        "type": "object",
        "properties": {k: {} for k in payload},
    }
    return JsonDataModel(json=payload, schema=schema)


class OptimizerHistoryTest(testing.TestCase):
    async def test_on_epoch_end_appends_to_empty_history(self):
        optimizer = Optimizer(population_size=5)
        var = _trainable(
            candidates=[{"prompt": "a", "reward": 0.4}],
            best_candidates=[{"prompt": "b", "reward": 0.9}],
        )

        await optimizer.on_epoch_end(0, [var])

        history = var.get("history")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0], {"prompt": "b"})
        # `reward` must be stripped: history is a timeline of states,
        # not of scored candidates.
        self.assertNotIn("reward", history[0])

    async def test_on_epoch_end_appends_when_best_differs_from_last(self):
        optimizer = Optimizer(population_size=5)
        var = _trainable(
            history=[{"prompt": "old"}],
            candidates=[{"prompt": "new", "reward": 0.9}],
        )

        await optimizer.on_epoch_end(0, [var])

        history = var.get("history")
        self.assertEqual(history, [{"prompt": "old"}, {"prompt": "new"}])

    async def test_on_epoch_end_skips_when_best_equals_last(self):
        optimizer = Optimizer(population_size=5)
        var = _trainable(
            history=[{"prompt": "same"}],
            candidates=[{"prompt": "same", "reward": 0.9}],
        )

        await optimizer.on_epoch_end(0, [var])

        # Last entry already matches the new best → no duplicate appended.
        self.assertEqual(var.get("history"), [{"prompt": "same"}])

    async def test_on_epoch_end_grows_history_across_epochs(self):
        optimizer = Optimizer(population_size=5)
        var = _trainable()

        # Epoch 0: best = "v1"
        var.update({"candidates": [{"prompt": "v1", "reward": 0.5}]})
        await optimizer.on_epoch_end(0, [var])
        # Epoch 1: best = "v2" (added to candidates this round)
        var.update({"candidates": [{"prompt": "v2", "reward": 0.8}]})
        await optimizer.on_epoch_end(1, [var])
        # Epoch 2: same best as epoch 1; should NOT grow.
        var.update({"candidates": [{"prompt": "v2", "reward": 0.95}]})
        await optimizer.on_epoch_end(2, [var])
        # Epoch 3: new best "v3".
        var.update({"candidates": [{"prompt": "v3", "reward": 1.0}]})
        await optimizer.on_epoch_end(3, [var])

        self.assertEqual(
            var.get("history"),
            [{"prompt": "v1"}, {"prompt": "v2"}, {"prompt": "v3"}],
        )

    async def test_on_epoch_end_uses_best_across_candidates_and_best_candidates(self):
        optimizer = Optimizer(population_size=5)
        var = _trainable(
            candidates=[{"prompt": "from_candidates", "reward": 0.6}],
            best_candidates=[{"prompt": "from_best", "reward": 0.95}],
        )

        await optimizer.on_epoch_end(0, [var])

        # The single highest-reward entry across both pools wins,
        # regardless of which pool it came from.
        self.assertEqual(var.get("history"), [{"prompt": "from_best"}])

    async def test_on_epoch_end_populates_history_per_variable(self):
        optimizer = Optimizer(population_size=5)
        var_a = _trainable(candidates=[{"prompt": "alpha", "reward": 0.7}])
        var_b = _trainable(candidates=[{"prompt": "beta", "reward": 0.7}])

        await optimizer.on_epoch_end(0, [var_a, var_b])

        # Each variable maintains its own history; no cross-pollination.
        self.assertEqual(var_a.get("history"), [{"prompt": "alpha"}])
        self.assertEqual(var_b.get("history"), [{"prompt": "beta"}])


class OptimizerRewardAssignmentTest(testing.TestCase):
    async def test_counter_reflects_only_current_batch(self):
        """`nb_visit` / `cumulative_reward` are a per-batch struggle signal:
        `assign_reward_to_predictions` resets them to the current batch's
        scored predictions instead of accumulating across batches, while
        `predictions` keeps the full history. Regression test for the signal
        drifting away from the recorded predictions."""
        optimizer = Optimizer(population_size=5)
        var = _trainable(
            current_predictions=[{"reward": None}, {"reward": None}],
        )

        await optimizer.assign_reward_to_predictions([var], rewards=[1.0, 0.0])
        self.assertEqual(var.get("nb_visit"), 2)
        self.assertEqual(var.get("cumulative_reward"), 1.0)
        self.assertEqual(len(var.get("predictions")), 2)

        # A second batch RESETS the signal (it must be 3 / 3.0, not 5 / 4.0).
        var.update({"current_predictions": [{"reward": None}] * 3})
        await optimizer.assign_reward_to_predictions([var], rewards=[1.0, 1.0, 1.0])
        self.assertEqual(var.get("nb_visit"), 3)
        self.assertEqual(var.get("cumulative_reward"), 3.0)
        # The per-batch mean reflects only the current batch.
        self.assertEqual(var.get("cumulative_reward") / var.get("nb_visit"), 1.0)
        # ...but `predictions` still records every batch.
        self.assertEqual(len(var.get("predictions")), 5)

    async def test_empty_pass_preserves_batch_signal(self):
        """The validation assign pass has no `current_predictions`, so it must
        leave the train-batch signal untouched (not zero it out)."""
        optimizer = Optimizer(population_size=5)
        var = _trainable(
            current_predictions=[{"reward": None}, {"reward": None}],
        )
        await optimizer.assign_reward_to_predictions([var], rewards=[1.0, 0.0])
        # `current_predictions` was reset to [] by the call above; mimic the
        # validation pass that runs against the same variables.
        await optimizer.assign_reward_to_predictions([var], rewards=[0.5])
        self.assertEqual(var.get("nb_visit"), 2)
        self.assertEqual(var.get("cumulative_reward"), 1.0)


class _ConcreteOptimizer(Optimizer):
    async def propose_new_candidates(self, *args, **kwargs):
        return None


class MaybeAddCandidateTest(testing.TestCase):
    def _variable(self, best_candidates=None):
        return JsonDataModel(
            json={
                "instructions": "seed",
                "examples": [],
                "candidates": [],
                "best_candidates": best_candidates or [],
            },
            schema={
                "type": "object",
                "properties": {
                    "instructions": {"type": "string"},
                    "examples": {"type": "array"},
                    "candidates": {"type": "array"},
                    "best_candidates": {"type": "array"},
                },
            },
        )

    async def test_new_candidate_starts_with_one_measurement(self):
        optimizer = _ConcreteOptimizer()
        variable = self._variable()
        variable.update({"instructions": "v1"})
        await optimizer.maybe_add_candidate(0, variable, reward=0.2)
        (candidate,) = variable.get("candidates")
        self.assertEqual(candidate["instructions"], "v1")
        self.assertEqual(candidate["reward"], 0.2)
        self.assertEqual(candidate["reward_count"], 1)

    async def test_repeated_candidate_updates_running_mean(self):
        optimizer = _ConcreteOptimizer()
        variable = self._variable()
        variable.update({"instructions": "v1"})
        await optimizer.maybe_add_candidate(0, variable, reward=0.2)
        await optimizer.maybe_add_candidate(1, variable, reward=0.6)
        await optimizer.maybe_add_candidate(2, variable, reward=1.0)
        candidates = variable.get("candidates")
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(candidates[0]["reward"], 0.6)
        self.assertEqual(candidates[0]["reward_count"], 3)

    async def test_repeated_candidate_in_best_candidates_is_updated_in_place(self):
        optimizer = _ConcreteOptimizer()
        variable = self._variable(
            best_candidates=[
                {"instructions": "v1", "examples": [], "reward": 0.5, "reward_count": 1}
            ]
        )
        variable.update({"instructions": "v1"})
        await optimizer.maybe_add_candidate(0, variable, reward=0.9)
        self.assertEqual(variable.get("candidates"), [])
        (best,) = variable.get("best_candidates")
        self.assertAlmostEqual(best["reward"], 0.7)
        self.assertEqual(best["reward_count"], 2)

    async def test_metadata_never_leaks_into_the_variable(self):
        optimizer = _ConcreteOptimizer()
        variable = self._variable(
            best_candidates=[
                {"instructions": "best", "examples": [], "reward": 0.9, "reward_count": 3}
            ]
        )
        await optimizer.on_batch_end(0, 1, [variable])
        self.assertEqual(variable.get("instructions"), "best")
        self.assertNotIn("reward", variable.get_json())
        self.assertNotIn("reward_count", variable.get_json())


class CandidateScoreTest(testing.TestCase):
    def test_lower_confidence_bound(self):
        optimizer = _ConcreteOptimizer(reward_uncertainty=0.2)
        self.assertAlmostEqual(
            optimizer.candidate_score({"reward": 0.9, "reward_count": 4}), 0.9 - 0.1
        )
        self.assertAlmostEqual(
            optimizer.candidate_score({"reward": 0.9, "reward_count": 100}), 0.9 - 0.02
        )
        # A candidate without a count is treated as a single measurement.
        self.assertAlmostEqual(optimizer.candidate_score({"reward": 0.9}), 0.7)

    def test_zero_uncertainty_ranks_by_raw_reward(self):
        optimizer = _ConcreteOptimizer(reward_uncertainty=0.0)
        self.assertEqual(
            optimizer.candidate_score({"reward": 0.9, "reward_count": 1}), 0.9
        )

    async def test_promotion_prefers_well_measured_candidate_on_near_tie(self):
        """A fresh 0.95 on 4 samples must not beat a 0.90 measured on 60."""
        optimizer = _ConcreteOptimizer(reward_uncertainty=0.25)
        variable = JsonDataModel(
            json={
                "instructions": "seed",
                "examples": [],
                "candidates": [
                    {
                        "instructions": "lucky",
                        "examples": [],
                        "reward": 0.95,
                        "reward_count": 4,
                    },
                ],
                "best_candidates": [
                    {
                        "instructions": "solid",
                        "examples": [],
                        "reward": 0.90,
                        "reward_count": 60,
                    },
                ],
            },
            schema={
                "type": "object",
                "properties": {
                    "instructions": {"type": "string"},
                    "examples": {"type": "array"},
                    "candidates": {"type": "array"},
                    "best_candidates": {"type": "array"},
                },
            },
        )
        await optimizer.on_batch_end(0, 1, [variable])
        self.assertEqual(variable.get("instructions"), "solid")


class SampleWeightedRewardTest(testing.TestCase):
    def _variable(self, **extra):
        return JsonDataModel(
            json={
                "instructions": "seed",
                "examples": [],
                "candidates": [],
                "best_candidates": [],
                "history": [],
                **extra,
            },
            schema={
                "type": "object",
                "properties": {
                    "instructions": {"type": "string"},
                    "examples": {"type": "array"},
                    "candidates": {"type": "array"},
                    "best_candidates": {"type": "array"},
                    "history": {"type": "array"},
                },
            },
        )

    async def test_weight_is_the_number_of_samples(self):
        optimizer = _ConcreteOptimizer()
        variable = self._variable()
        variable.update({"instructions": "v1"})
        await optimizer.maybe_add_candidate(0, variable, reward=0.5, weight=4)
        await optimizer.maybe_add_candidate(1, variable, reward=1.0, weight=12)
        (candidate,) = variable.get("candidates")
        self.assertEqual(candidate["reward_count"], 16)
        self.assertAlmostEqual(candidate["reward"], (0.5 * 4 + 1.0 * 12) / 16)

    async def test_validation_reward_is_folded_into_the_promoted_candidate(self):
        optimizer = _ConcreteOptimizer()
        variable = self._variable(
            best_candidates=[
                {
                    "instructions": "promoted",
                    "examples": [],
                    "reward": 1.0,
                    "reward_count": 4,
                },
                {
                    "instructions": "other",
                    "examples": [],
                    "reward": 0.8,
                    "reward_count": 4,
                },
            ]
        )
        variable.update({"instructions": "promoted"})  # what the trainer validated
        optimizer.assign_validation_reward(
            [variable], logs={"reward": 0.7, "val_reward": 0.6}, val_size=60
        )
        promoted, other = variable.get("best_candidates")
        self.assertEqual(promoted["reward_count"], 64)
        self.assertAlmostEqual(promoted["reward"], (1.0 * 4 + 0.6 * 60) / 64)
        self.assertEqual(other["reward"], 0.8)
        self.assertEqual(other["reward_count"], 4)

    async def test_validation_feedback_is_a_no_op_without_val_reward(self):
        optimizer = _ConcreteOptimizer()
        variable = self._variable(
            best_candidates=[
                {
                    "instructions": "promoted",
                    "examples": [],
                    "reward": 1.0,
                    "reward_count": 4,
                }
            ]
        )
        variable.update({"instructions": "promoted"})
        optimizer.assign_validation_reward([variable], logs={"reward": 0.7}, val_size=60)
        optimizer.assign_validation_reward([variable], logs=None, val_size=60)
        (promoted,) = variable.get("best_candidates")
        self.assertEqual(promoted["reward"], 1.0)
        self.assertEqual(promoted["reward_count"], 4)

    async def test_on_epoch_end_folds_validation_then_ranks_by_confidence(self):
        """The lucky promoted candidate is corrected by the full validation and
        loses the top spot to the solidly measured one."""
        optimizer = _ConcreteOptimizer(reward_uncertainty=0.25, population_size=2)
        variable = self._variable(
            candidates=[
                {
                    "instructions": "lucky",
                    "examples": [],
                    "reward": 1.0,
                    "reward_count": 4,
                },
            ],
            best_candidates=[
                {
                    "instructions": "solid",
                    "examples": [],
                    "reward": 0.85,
                    "reward_count": 60,
                },
            ],
        )
        variable.update({"instructions": "lucky"})
        await optimizer.on_epoch_end(
            0, [variable], logs={"reward": 0.9, "val_reward": 0.7}, val_size=60
        )
        best = variable.get("best_candidates")
        self.assertEqual(best[0]["instructions"], "solid")
        self.assertEqual(variable.get("instructions"), "solid")
        lucky = next(c for c in best if c["instructions"] == "lucky")
        self.assertEqual(lucky["reward_count"], 64)
        self.assertAlmostEqual(lucky["reward"], (1.0 * 4 + 0.7 * 60) / 64)
