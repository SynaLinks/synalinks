# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import JsonDataModel
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.optimizers.evolutionary_optimizer import EvolutionaryOptimizer
from synalinks.src.optimizers.omega import OMEGA
from synalinks.src.optimizers.optimizer import Optimizer


class EvolutionaryOptimizerTest(testing.TestCase):
    def test_inheritance(self):
        """Test that EvolutionaryOptimizer inherits from Optimizer."""
        self.assertTrue(issubclass(EvolutionaryOptimizer, Optimizer))
        self.assertTrue(issubclass(OMEGA, EvolutionaryOptimizer))

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        optimizer = EvolutionaryOptimizer()

        self.assertIsNone(optimizer.language_model)
        self.assertEqual(optimizer.mutation_temperature, 0.3)
        self.assertEqual(optimizer.crossover_temperature, 0.3)
        self.assertEqual(optimizer.selection, "softmax")
        self.assertEqual(optimizer.selection_temperature, 0.3)
        self.assertEqual(optimizer.merging_rate, 0.02)
        self.assertEqual(optimizer.population_size, 10)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        lm = LanguageModel(model="ollama/mistral")

        optimizer = EvolutionaryOptimizer(
            language_model=lm,
            mutation_temperature=0.5,
            crossover_temperature=0.7,
            selection="best",
            selection_temperature=0.2,
            merging_rate=0.1,
            population_size=20,
            name="test_evo_optimizer",
            description="Test evolutionary optimizer",
        )

        self.assertIs(optimizer.language_model, lm)
        self.assertEqual(optimizer.mutation_temperature, 0.5)
        self.assertEqual(optimizer.crossover_temperature, 0.7)
        self.assertEqual(optimizer.selection, "best")
        self.assertEqual(optimizer.selection_temperature, 0.2)
        self.assertEqual(optimizer.merging_rate, 0.1)
        self.assertEqual(optimizer.population_size, 20)

    def test_invalid_selection(self):
        """Test that invalid selection raises ValueError."""
        with self.assertRaises(ValueError) as context:
            EvolutionaryOptimizer(selection="invalid")

        self.assertIn("selection", str(context.exception))

    def test_valid_selections(self):
        """Test that valid selections are accepted."""
        for selection in ["random", "best", "softmax"]:
            optimizer = EvolutionaryOptimizer(selection=selection)
            self.assertEqual(optimizer.selection, selection)

    def test_get_config(self):
        """Test that get_config returns all parameters."""
        optimizer = EvolutionaryOptimizer(
            mutation_temperature=0.4,
            crossover_temperature=0.6,
            selection="random",
            selection_temperature=0.25,
            merging_rate=0.05,
            population_size=15,
            name="config_test",
            description="Config test optimizer",
        )

        config = optimizer.get_config()

        self.assertEqual(config["mutation_temperature"], 0.4)
        self.assertEqual(config["crossover_temperature"], 0.6)
        self.assertEqual(config["selection"], "random")
        self.assertEqual(config["selection_temperature"], 0.25)
        self.assertEqual(config["merging_rate"], 0.05)
        self.assertEqual(config["population_size"], 15)

    async def test_on_batch_begin_epoch_zero(self):
        """Test on_batch_begin uses seed_candidates for epoch 0."""
        optimizer = EvolutionaryOptimizer(selection="random")

        seed_candidate = {"prompt": "seed_prompt"}
        trainable_variable = JsonDataModel(
            json={
                "seed_candidates": [seed_candidate],
                "best_candidates": [],
                "nb_visit": 5,
                "cumulative_reward": 1.0,
            },
            schema={
                "type": "object",
                "properties": {
                    "seed_candidates": {"type": "array"},
                    "best_candidates": {"type": "array"},
                    "nb_visit": {"type": "integer"},
                    "cumulative_reward": {"type": "number"},
                },
            },
        )

        await optimizer.on_batch_begin(0, 0, [trainable_variable])

        # Should reset nb_visit and cumulative_reward
        self.assertEqual(trainable_variable.get("nb_visit"), 0)
        self.assertEqual(trainable_variable.get("cumulative_reward"), 0.0)

    async def test_on_batch_begin_selection_best(self):
        """Test on_batch_begin with 'best' selection."""
        optimizer = EvolutionaryOptimizer(selection="best")

        best_candidates = [
            {"prompt": "low", "reward": 0.3},
            {"prompt": "high", "reward": 0.9},
            {"prompt": "mid", "reward": 0.6},
        ]

        trainable_variable = JsonDataModel(
            json={
                "seed_candidates": [],
                "best_candidates": best_candidates,
                "nb_visit": 0,
                "cumulative_reward": 0.0,
                "prompt": "initial",
            },
            schema={
                "type": "object",
                "properties": {
                    "seed_candidates": {"type": "array"},
                    "best_candidates": {"type": "array"},
                    "nb_visit": {"type": "integer"},
                    "cumulative_reward": {"type": "number"},
                    "prompt": {"type": "string"},
                },
            },
        )

        await optimizer.on_batch_begin(0, 1, [trainable_variable])

        # Should select the highest reward candidate
        self.assertEqual(trainable_variable.get("prompt"), "high")

    def test_select_candidate_empty(self):
        """Test select_candidate returns None for empty list."""
        optimizer = EvolutionaryOptimizer()
        result = optimizer.select_candidate([])
        self.assertIsNone(result)

    def test_select_candidate_random(self):
        """Test select_candidate with random selection."""
        optimizer = EvolutionaryOptimizer(selection="random")
        candidates = [
            {"prompt": "a", "reward": 0.5},
            {"prompt": "b", "reward": 0.8},
        ]
        result = optimizer.select_candidate(candidates)
        self.assertIn(result, candidates)

    def test_select_candidate_best(self):
        """Test select_candidate with best selection."""
        optimizer = EvolutionaryOptimizer(selection="best")
        candidates = [
            {"prompt": "low", "reward": 0.3},
            {"prompt": "high", "reward": 0.9},
            {"prompt": "mid", "reward": 0.6},
        ]
        result = optimizer.select_candidate(candidates)
        self.assertEqual(result["prompt"], "high")
        self.assertEqual(result["reward"], 0.9)

    def test_select_candidate_softmax(self):
        """Test select_candidate with softmax selection."""
        optimizer = EvolutionaryOptimizer(
            selection="softmax",
            selection_temperature=0.1,
        )
        candidates = [
            {"prompt": "low", "reward": 0.1},
            {"prompt": "high", "reward": 0.9},
        ]
        result = optimizer.select_candidate(candidates)
        self.assertIn(result, candidates)

    def test_sampling_temperature_default(self):
        """EvolutionaryOptimizer must inherit `sampling_temperature` from the
        base `Optimizer`, since `Optimizer.select_variable_name_to_update`
        reads it. Before the fix, accessing this attribute on an
        EvolutionaryOptimizer / OMEGA instance raised AttributeError."""
        optimizer = EvolutionaryOptimizer()
        self.assertEqual(optimizer.sampling_temperature, 0.3)

    async def test_propose_new_candidates_falls_back_to_seed_when_best_empty(
        self,
    ):
        """Regression test: on the very first training step `best_candidates`
        is still empty (it is populated later by `maybe_add_candidate`).
        `propose_new_candidates` used to pass the resulting `None` into
        `mutate_candidate`, which crashed downstream (e.g. OMEGA's
        `selected_candidate.items()` in `omega.py`). It must instead fall
        back to a random seed candidate, mirroring `on_batch_begin`.
        """
        seen = {}

        class _RecordingOptimizer(EvolutionaryOptimizer):
            async def mutate_candidate(
                self,
                step,
                trainable_variable,
                selected_candidate,
                x=None,
                y=None,
                y_pred=None,
                training=False,
            ):
                seen["selected_candidate"] = selected_candidate
                return None

            async def merge_candidate(
                self,
                step,
                trainable_variable,
                current_candidate,
                other_candidate,
                x=None,
                y=None,
                y_pred=None,
                training=False,
            ):
                return None

        # merging_rate=0 keeps the strategy on "mutation" regardless of epoch.
        optimizer = _RecordingOptimizer(selection="random", merging_rate=0.0)

        seed_candidate = {"prompt": "seed_prompt"}
        trainable_variable = JsonDataModel(
            json={
                "seed_candidates": [seed_candidate],
                "best_candidates": [],
                "nb_visit": 0,
                "cumulative_reward": 0.0,
                "prompt": "initial",
            },
            schema={
                "type": "object",
                "properties": {
                    "seed_candidates": {"type": "array"},
                    "best_candidates": {"type": "array"},
                    "nb_visit": {"type": "integer"},
                    "cumulative_reward": {"type": "number"},
                    "prompt": {"type": "string"},
                },
            },
            name="trainable_var",
        )

        await optimizer.propose_new_candidates(
            step=0,
            trainable_variables=[trainable_variable],
        )

        self.assertEqual(seen.get("selected_candidate"), seed_candidate)

    async def test_select_variable_name_to_update_does_not_raise(self):
        """Regression test: `select_variable_name_to_update` on an
        EvolutionaryOptimizer used to raise
        `AttributeError: 'EvolutionaryOptimizer' object has no attribute
        'sampling_temperature'`. It must now run and return a valid name."""
        optimizer = EvolutionaryOptimizer()

        class _Var:
            def __init__(self, name, nb_visit, cumulative_reward):
                self.name = name
                self._d = {
                    "nb_visit": nb_visit,
                    "cumulative_reward": cumulative_reward,
                }

            def get(self, key):
                return self._d[key]

        variables = [
            _Var("v0", nb_visit=1, cumulative_reward=0.5),
            _Var("v1", nb_visit=1, cumulative_reward=0.2),
        ]
        name = await optimizer.select_variable_name_to_update(variables)
        self.assertIn(name, {"v0", "v1"})
