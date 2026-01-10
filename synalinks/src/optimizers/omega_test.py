# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

from synalinks.src import testing
from synalinks.src.backend import JsonDataModel
from synalinks.src.optimizers.evolutionary_optimizer import EvolutionaryOptimizer
from synalinks.src.optimizers.omega import OMEGA
from synalinks.src.optimizers.omega import base_instructions
from synalinks.src.optimizers.omega import crossover_instructions
from synalinks.src.optimizers.omega import mutation_instructions
from synalinks.src.optimizers.omega import similarity_distance


class OMEGATest(testing.TestCase):
    def test_inheritance(self):
        """Test that OMEGA inherits from EvolutionaryOptimizer."""
        self.assertTrue(issubclass(OMEGA, EvolutionaryOptimizer))

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        optimizer = OMEGA()

        self.assertIsNone(optimizer.language_model)
        self.assertIsNone(optimizer.embedding_model)
        self.assertEqual(optimizer.mutation_temperature, 0.3)
        self.assertEqual(optimizer.crossover_temperature, 0.3)
        self.assertEqual(optimizer.k_nearest_fitter, 5)
        self.assertEqual(optimizer.algorithm, "dns")
        self.assertEqual(optimizer.selection, "softmax")
        self.assertEqual(optimizer.selection_temperature, 0.3)
        self.assertEqual(optimizer.merging_rate, 0.02)
        self.assertEqual(optimizer.population_size, 10)
        self.assertEqual(optimizer.instructions, "")

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        mock_lm = MagicMock()
        mock_em = MagicMock()

        optimizer = OMEGA(
            instructions="Test instructions",
            language_model=mock_lm,
            embedding_model=mock_em,
            mutation_temperature=0.5,
            crossover_temperature=0.7,
            k_nearest_fitter=10,
            algorithm="ga",
            selection="best",
            selection_temperature=0.2,
            merging_rate=0.1,
            population_size=20,
            name="test_omega",
            description="Test OMEGA optimizer",
        )

        self.assertEqual(optimizer.language_model, mock_lm)
        self.assertEqual(optimizer.embedding_model, mock_em)
        self.assertEqual(optimizer.mutation_temperature, 0.5)
        self.assertEqual(optimizer.crossover_temperature, 0.7)
        self.assertEqual(optimizer.k_nearest_fitter, 10)
        self.assertEqual(optimizer.algorithm, "ga")
        self.assertEqual(optimizer.selection, "best")
        self.assertEqual(optimizer.selection_temperature, 0.2)
        self.assertEqual(optimizer.merging_rate, 0.1)
        self.assertEqual(optimizer.population_size, 20)
        self.assertEqual(optimizer.instructions, "Test instructions")

    def test_invalid_algorithm(self):
        """Test that invalid algorithm raises ValueError."""
        with self.assertRaises(ValueError) as context:
            OMEGA(algorithm="invalid")

        self.assertIn("algorithm", str(context.exception))

    def test_valid_algorithms(self):
        """Test that valid algorithms are accepted."""
        for algorithm in ["ga", "dns"]:
            optimizer = OMEGA(algorithm=algorithm)
            self.assertEqual(optimizer.algorithm, algorithm)

    def test_get_config(self):
        """Test that get_config returns all parameters."""
        optimizer = OMEGA(
            instructions="Test",
            mutation_temperature=0.4,
            crossover_temperature=0.6,
            k_nearest_fitter=8,
            algorithm="ga",
            selection="random",
            selection_temperature=0.25,
            merging_rate=0.05,
            population_size=15,
            name="config_test",
            description="Config test optimizer",
        )

        config = optimizer.get_config()

        self.assertEqual(config["instructions"], "Test")
        self.assertEqual(config["mutation_temperature"], 0.4)
        self.assertEqual(config["crossover_temperature"], 0.6)
        self.assertEqual(config["k_nearest_fitter"], 8)
        self.assertEqual(config["algorithm"], "ga")
        self.assertEqual(config["selection"], "random")
        self.assertEqual(config["selection_temperature"], 0.25)
        self.assertEqual(config["merging_rate"], 0.05)
        self.assertEqual(config["population_size"], 15)

    async def test_competition_single_candidate(self):
        """Test competition with single candidate returns it unchanged."""
        optimizer = OMEGA()

        candidates = [{"prompt": "test", "reward": 0.8}]
        result = await optimizer.competition(candidates)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], candidates[0])

    async def test_competition_empty_candidates(self):
        """Test competition with empty candidates returns empty list."""
        optimizer = OMEGA()

        result = await optimizer.competition([])

        self.assertEqual(result, [])

    async def test_competition_filters_candidates(self):
        """Test that competition filters candidates based on DNS."""
        # Create mock embedding model
        mock_embedding_model = AsyncMock()
        mock_embedding_model.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        }

        optimizer = OMEGA(
            embedding_model=mock_embedding_model,
            k_nearest_fitter=2,
        )

        # Create candidates with different rewards
        candidates = [
            {"prompt": "test1", "reward": 0.9},
            {"prompt": "test2", "reward": 0.5},
            {"prompt": "test3", "reward": 0.7},
            {"prompt": "test4", "reward": 0.3},
        ]

        result = await optimizer.competition(candidates)

        # Should filter out some candidates based on DNS
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), len(candidates))

    async def test_on_epoch_end_sorts_and_selects_candidates_ga(self):
        """Test on_epoch_end sorts candidates and selects top ones (GA mode)."""
        optimizer = OMEGA(
            algorithm="ga",  # Skip DNS competition
            population_size=2,
        )

        candidates = [
            {"prompt": "c1", "reward": 0.3},
            {"prompt": "c2", "reward": 0.9},
        ]
        best_candidates = [
            {"prompt": "b1", "reward": 0.5},
        ]

        trainable_variable = JsonDataModel(
            json={
                "candidates": candidates,
                "best_candidates": best_candidates,
            },
            schema={
                "type": "object",
                "properties": {
                    "candidates": {"type": "array"},
                    "best_candidates": {"type": "array"},
                },
            },
        )

        await optimizer.on_epoch_end(0, [trainable_variable])

        # Should keep top 2 candidates by reward
        result = trainable_variable.get("best_candidates")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["reward"], 0.9)
        self.assertEqual(result[1]["reward"], 0.5)

    async def test_on_epoch_end_with_dns(self):
        """Test on_epoch_end applies DNS competition when algorithm='dns'."""
        # Create mock embedding model
        mock_embedding_model = AsyncMock()
        mock_embedding_model.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}

        optimizer = OMEGA(
            algorithm="dns",
            embedding_model=mock_embedding_model,
            population_size=3,
        )

        candidates = [
            {"prompt": "c1", "reward": 0.9},
            {"prompt": "c2", "reward": 0.5},
        ]
        best_candidates = [
            {"prompt": "b1", "reward": 0.7},
        ]

        trainable_variable = JsonDataModel(
            json={
                "candidates": candidates,
                "best_candidates": best_candidates,
            },
            schema={
                "type": "object",
                "properties": {
                    "candidates": {"type": "array"},
                    "best_candidates": {"type": "array"},
                },
            },
        )

        await optimizer.on_epoch_end(0, [trainable_variable])

        # Should have applied DNS and sorted
        result = trainable_variable.get("best_candidates")
        self.assertGreater(len(result), 0)
        self.assertLessEqual(len(result), 3)


class SimilarityDistanceTest(testing.TestCase):
    async def test_similarity_distance_identical_candidates(self):
        """Test similarity_distance returns 0 for identical candidates."""
        mock_embedding_model = AsyncMock()
        mock_embedding_model.return_value = {"embeddings": [[1.0, 0.0, 0.0]]}

        candidate = {"prompt": "test"}

        distance = await similarity_distance(
            candidate, candidate, embedding_model=mock_embedding_model
        )

        # Identical embeddings should have distance close to 0
        self.assertLessEqual(distance, 0.1)

    async def test_similarity_distance_different_candidates(self):
        """Test similarity_distance for different candidates."""
        call_count = [0]

        async def mock_embed(texts):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"embeddings": [[1.0, 0.0, 0.0]]}
            else:
                return {"embeddings": [[0.0, 1.0, 0.0]]}

        mock_embedding_model = AsyncMock(side_effect=mock_embed)

        candidate1 = {"prompt": "test1"}
        candidate2 = {"prompt": "test2"}

        distance = await similarity_distance(
            candidate1, candidate2, embedding_model=mock_embedding_model
        )

        # Orthogonal embeddings should have distance around 0.5
        self.assertGreater(distance, 0.3)
        self.assertLess(distance, 0.7)


class InstructionFunctionsTest(testing.TestCase):
    def test_base_instructions(self):
        """Test base_instructions returns non-empty string."""
        result = base_instructions()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        self.assertIn("optimization", result.lower())

    def test_mutation_instructions(self):
        """Test mutation_instructions includes variable keys."""
        keys = ["prompt", "rules", "examples"]
        result = mutation_instructions(keys)

        self.assertIsInstance(result, str)
        self.assertIn(str(keys), result)
        self.assertIn("enhance", result.lower())

    def test_crossover_instructions(self):
        """Test crossover_instructions includes variable keys."""
        keys = ["prompt", "rules"]
        result = crossover_instructions(keys)

        self.assertIsInstance(result, str)
        self.assertIn(str(keys), result)
        self.assertIn("combining", result.lower())
