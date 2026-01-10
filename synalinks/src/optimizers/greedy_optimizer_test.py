# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import testing
from synalinks.src.backend import JsonDataModel
from synalinks.src.optimizers.greedy_optimizer import GreedyOptimizer
from synalinks.src.optimizers.optimizer import Optimizer
from synalinks.src.optimizers.random_few_shot import RandomFewShot


class GreedyOptimizerTest(testing.TestCase):
    def test_inheritance(self):
        """Test that GreedyOptimizer inherits from Optimizer."""
        self.assertTrue(issubclass(GreedyOptimizer, Optimizer))
        self.assertTrue(issubclass(RandomFewShot, GreedyOptimizer))

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        optimizer = GreedyOptimizer()

        self.assertEqual(optimizer.nb_min_examples, 1)
        self.assertEqual(optimizer.nb_max_examples, 3)
        self.assertEqual(optimizer.sampling, "softmax")
        self.assertEqual(optimizer.sampling_temperature, 0.3)
        self.assertEqual(optimizer.population_size, 10)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        optimizer = GreedyOptimizer(
            nb_min_examples=2,
            nb_max_examples=5,
            sampling="best",
            sampling_temperature=0.5,
            population_size=20,
            name="test_optimizer",
            description="A test optimizer",
        )

        self.assertEqual(optimizer.nb_min_examples, 2)
        self.assertEqual(optimizer.nb_max_examples, 5)
        self.assertEqual(optimizer.sampling, "best")
        self.assertEqual(optimizer.sampling_temperature, 0.5)
        self.assertEqual(optimizer.population_size, 20)
        self.assertEqual(optimizer.name, "test_optimizer")
        self.assertEqual(optimizer.description, "A test optimizer")

    def test_invalid_sampling(self):
        """Test that invalid sampling raises ValueError."""
        with self.assertRaises(ValueError) as context:
            GreedyOptimizer(sampling="invalid")

        self.assertIn("sampling", str(context.exception))

    def test_valid_samplings(self):
        """Test that valid sampling values are accepted."""
        for sampling in ["random", "best", "softmax"]:
            optimizer = GreedyOptimizer(sampling=sampling)
            self.assertEqual(optimizer.sampling, sampling)

    def test_get_config(self):
        """Test that get_config returns all parameters."""
        optimizer = GreedyOptimizer(
            nb_min_examples=2,
            nb_max_examples=4,
            sampling="best",
            sampling_temperature=0.4,
            population_size=15,
            name="config_test",
            description="Config test optimizer",
        )

        config = optimizer.get_config()

        self.assertEqual(config["nb_min_examples"], 2)
        self.assertEqual(config["nb_max_examples"], 4)
        self.assertEqual(config["sampling"], "best")
        self.assertEqual(config["sampling_temperature"], 0.4)
        self.assertEqual(config["population_size"], 15)
        self.assertEqual(config["name"], "config_test")
        self.assertEqual(config["description"], "Config test optimizer")

    async def test_sample_best_predictions_empty(self):
        """Test sample_best_predictions with empty predictions."""
        optimizer = GreedyOptimizer(nb_min_examples=1, nb_max_examples=3)

        trainable_variable = JsonDataModel(
            json={"predictions": []},
            schema={
                "type": "object",
                "properties": {"predictions": {"type": "array"}},
            },
        )

        result = await optimizer.sample_best_predictions(trainable_variable)
        self.assertEqual(result, [])

    async def test_sample_best_predictions_few_predictions(self):
        """Test sample_best_predictions when fewer predictions than requested."""
        optimizer = GreedyOptimizer(nb_min_examples=3, nb_max_examples=5)

        predictions = [
            {"input": "q1", "output": "a1", "reward": 0.8},
            {"input": "q2", "output": "a2", "reward": 0.9},
        ]

        trainable_variable = JsonDataModel(
            json={"predictions": predictions},
            schema={
                "type": "object",
                "properties": {"predictions": {"type": "array"}},
            },
        )

        result = await optimizer.sample_best_predictions(trainable_variable)
        # Should return all predictions when fewer than requested
        self.assertEqual(len(result), 2)

    async def test_sample_best_predictions_softmax(self):
        """Test sample_best_predictions with softmax sampling."""
        optimizer = GreedyOptimizer(
            nb_min_examples=1,
            nb_max_examples=3,
            sampling="softmax",
            sampling_temperature=0.1,  # Low temp biases toward high reward
        )

        predictions = [
            {"input": f"q{i}", "output": f"a{i}", "reward": i / 10.0} for i in range(10)
        ]

        trainable_variable = JsonDataModel(
            json={"predictions": predictions},
            schema={
                "type": "object",
                "properties": {"predictions": {"type": "array"}},
            },
        )

        result = await optimizer.sample_best_predictions(trainable_variable)
        self.assertGreaterEqual(len(result), 1)
        self.assertLessEqual(len(result), 3)

    async def test_sample_best_predictions_random(self):
        """Test sample_best_predictions with random sampling."""
        optimizer = GreedyOptimizer(
            nb_min_examples=2,
            nb_max_examples=2,
            sampling="random",
        )

        predictions = [
            {"input": f"q{i}", "output": f"a{i}", "reward": i / 10.0} for i in range(10)
        ]

        trainable_variable = JsonDataModel(
            json={"predictions": predictions},
            schema={
                "type": "object",
                "properties": {"predictions": {"type": "array"}},
            },
        )

        result = await optimizer.sample_best_predictions(trainable_variable)
        self.assertEqual(len(result), 2)

    async def test_sample_best_predictions_best(self):
        """Test sample_best_predictions with best sampling selects highest rewards."""
        optimizer = GreedyOptimizer(
            nb_min_examples=3,
            nb_max_examples=3,
            sampling="best",
        )

        predictions = [
            {"input": "q0", "output": "a0", "reward": 0.1},
            {"input": "q1", "output": "a1", "reward": 0.9},
            {"input": "q2", "output": "a2", "reward": 0.5},
            {"input": "q3", "output": "a3", "reward": 0.8},
            {"input": "q4", "output": "a4", "reward": 0.3},
        ]

        trainable_variable = JsonDataModel(
            json={"predictions": predictions},
            schema={
                "type": "object",
                "properties": {"predictions": {"type": "array"}},
            },
        )

        result = await optimizer.sample_best_predictions(trainable_variable)
        self.assertEqual(len(result), 3)
        # Should select the top 3 by reward: 0.9, 0.8, 0.5
        rewards = [p["reward"] for p in result]
        self.assertEqual(rewards, [0.9, 0.8, 0.5])
