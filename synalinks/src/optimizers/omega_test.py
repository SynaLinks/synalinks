# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import json
import os
from types import SimpleNamespace
from typing import Literal
from unittest.mock import AsyncMock
from unittest.mock import patch

import numpy as np

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.modules import Input
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.decision_models import DecisionModel
from synalinks.src.modules.embedding_models import EmbeddingModel
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.optimizers.evolutionary_optimizer import EvolutionaryOptimizer
from synalinks.src.optimizers.omega import OMEGA
from synalinks.src.optimizers.omega import base_instructions
from synalinks.src.optimizers.omega import crossover_instructions
from synalinks.src.optimizers.omega import mutation_instructions
from synalinks.src.optimizers.omega import similarity_distance
from synalinks.src.programs import Program
from synalinks.src.rewards.exact_match import ExactMatch
from synalinks.src.testing.test_utils import mock_decision_model


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
        self.assertEqual(optimizer.merging_rate, 0.05)
        self.assertEqual(optimizer.population_size, 10)
        self.assertEqual(optimizer.instructions, "")

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        lm = LanguageModel(model="ollama/mistral")
        em = EmbeddingModel(model="ollama/mxbai-embed-large")

        optimizer = OMEGA(
            instructions="Test instructions",
            language_model=lm,
            embedding_model=em,
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

        self.assertIs(optimizer.language_model, lm)
        self.assertIs(optimizer.embedding_model, em)
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

    @staticmethod
    def _embedding_by_prompt(vectors):
        """An embedding mock that maps each content text to a fixed vector."""

        async def embed(request):
            texts = request.get("texts") if hasattr(request, "get") else request.texts
            return {"embeddings": [vectors[text] for text in texts]}

        return AsyncMock(spec=EmbeddingModel, side_effect=embed)

    # Four candidates on three orthogonal descriptors (distance 1 between axes):
    #   c1 0.9 @e1 -> no fitter candidate                  -> +inf
    #   c2 0.5 @e1 -> fitter c1 (0), c3 (1)                -> k=1: 0.0 | k=2: 0.5
    #   c3 0.7 @e2 -> fitter c1 (1)                        -> 1.0
    #   c4 0.3 @e3 -> fitter c1 (1), c2 (1), c3 (1)        -> 1.0  (alone in its region)
    _DNS_CANDIDATES = [
        {"prompt": "c1", "reward": 0.9},
        {"prompt": "c2", "reward": 0.5},
        {"prompt": "c3", "reward": 0.7},
        {"prompt": "c4", "reward": 0.3},
    ]
    _DNS_VECTORS = {
        "c1": [1.0, 0.0, 0.0],
        "c2": [1.0, 0.0, 0.0],
        "c3": [0.0, 1.0, 0.0],
        "c4": [0.0, 0.0, 1.0],
    }

    async def test_competition_ranks_by_dns_competition_fitness(self):
        optimizer = OMEGA(
            embedding_model=self._embedding_by_prompt(self._DNS_VECTORS),
            k_nearest_fitter=2,
        )
        result = await optimizer.competition(list(self._DNS_CANDIDATES))
        # Nothing is removed: ranking only.
        self.assertEqual(len(result), 4)
        # inf (c1) > 1.0 (c3, c4: tie broken by reward) > 0.5 (c2)
        self.assertEqual([c["prompt"] for c in result], ["c1", "c3", "c4", "c2"])
        fitness = await optimizer.competition_fitness(list(self._DNS_CANDIDATES))
        self.assertEqual(fitness[0], float("inf"))
        self.assertAlmostEqual(fitness[1], 0.5)
        self.assertAlmostEqual(fitness[2], 1.0)
        self.assertAlmostEqual(fitness[3], 1.0)

    async def test_competition_penalizes_clone_of_the_best_over_a_novel_worse_one(self):
        """A worse candidate in its own region outranks a better clone of the best."""
        optimizer = OMEGA(
            embedding_model=self._embedding_by_prompt(self._DNS_VECTORS),
            k_nearest_fitter=2,
        )
        result = await optimizer.competition(list(self._DNS_CANDIDATES))
        ranks = {c["prompt"]: i for i, c in enumerate(result)}
        self.assertLess(ranks["c4"], ranks["c2"])  # c4 (0.3, novel) beats c2 (0.5, clone)

    async def test_competition_uses_k_nearest_fitter(self):
        """c2 has two fitter neighbours at 0 and 1: k=1 averages the nearest
        only (0.0), k=2 averages both (0.5), k>2 cannot use more than exist."""
        for k, expected in ((1, 0.0), (2, 0.5), (5, 0.5)):
            optimizer = OMEGA(
                embedding_model=self._embedding_by_prompt(self._DNS_VECTORS),
                k_nearest_fitter=k,
            )
            fitness = await optimizer.competition_fitness(list(self._DNS_CANDIDATES))
            self.assertAlmostEqual(fitness[1], expected, msg=f"k={k}")

    async def test_competition_has_no_distance_threshold(self):
        """Scaling every distance leaves the ranking unchanged."""
        for scale in (1e-3, 1.0, 1e3):

            async def scaled_distance(c1, c2, embedding_model=None, **kwargs):
                return scale * abs(c1["reward"] - c2["reward"])

            optimizer = OMEGA(k_nearest_fitter=2, distance_function=scaled_distance)
            result = await optimizer.competition(list(self._DNS_CANDIDATES))
            # c1 inf; c2 and c4 tie at 0.3*scale (reward breaks the tie); c3 at 0.2*scale
            self.assertEqual(
                [c["prompt"] for c in result], ["c1", "c2", "c4", "c3"], f"scale={scale}"
            )

    async def test_competition_computes_each_pair_once(self):
        calls = []

        async def counting_distance(c1, c2, embedding_model=None, **kwargs):
            calls.append((c1["prompt"], c2["prompt"]))
            return 1.0

        optimizer = OMEGA(k_nearest_fitter=2, distance_function=counting_distance)
        await optimizer.competition(list(self._DNS_CANDIDATES))
        self.assertEqual(len(calls), len(set(tuple(sorted(c)) for c in calls)))

    def _make_trainable_variable(self):
        """A minimal trainable variable for mutation/crossover tests."""
        trainable_variable = JsonDataModel(
            json={"instructions": "do the thing"},
            schema={
                "type": "object",
                "properties": {"instructions": {"type": "string"}},
            },
        )
        trainable_variable.description = "the variable to optimize"
        return trainable_variable

    def _make_batch(self):
        """Minimal x / y / y_pred batches whose items expose `get_json`."""
        x = [JsonDataModel(json={"q": "hi"}, schema={"type": "object"})]
        y = [JsonDataModel(json={"a": "bye"}, schema={"type": "object"})]
        y_pred = [JsonDataModel(json={"a": "yo"}, schema={"type": "object"})]
        return x, y, y_pred

    async def test_mutate_candidate_survives_lm_failure(self):
        """A failing mutation LM call is skipped (returns None), not fatal."""
        optimizer = OMEGA()
        optimizer.set_program(SimpleNamespace(description="A test program"))

        trainable_variable = self._make_trainable_variable()
        schema_id = id(trainable_variable.get_schema())
        optimizer.mutation_programs[schema_id] = AsyncMock(
            side_effect=RuntimeError("litellm timeout after 600s"),
        )

        x, y, y_pred = self._make_batch()

        with self.assertWarns(UserWarning):
            result = await optimizer.mutate_candidate(
                3,
                trainable_variable,
                {"instructions": "do the thing"},
                x=x,
                y=y,
                y_pred=y_pred,
                training=True,
            )

        self.assertIsNone(result)
        optimizer.mutation_programs[schema_id].assert_awaited_once()

    async def test_merge_candidate_survives_lm_failure(self):
        """A failing crossover LM call is skipped (returns None), not fatal."""
        optimizer = OMEGA()
        optimizer.set_program(SimpleNamespace(description="A test program"))

        trainable_variable = self._make_trainable_variable()
        schema_id = id(trainable_variable.get_schema())
        optimizer.crossover_programs[schema_id] = AsyncMock(
            side_effect=RuntimeError("litellm timeout after 600s"),
        )

        x, y, y_pred = self._make_batch()

        with self.assertWarns(UserWarning):
            result = await optimizer.merge_candidate(
                3,
                trainable_variable,
                {"instructions": "current"},
                {"instructions": "other"},
                x=x,
                y=y,
                y_pred=y_pred,
                training=True,
            )

        self.assertIsNone(result)
        optimizer.crossover_programs[schema_id].assert_awaited_once()

    async def test_mutate_candidate_returns_program_output_on_success(self):
        """On success the mutation program's output is returned unchanged."""
        optimizer = OMEGA()
        optimizer.set_program(SimpleNamespace(description="A test program"))

        trainable_variable = self._make_trainable_variable()
        schema_id = id(trainable_variable.get_schema())
        expected = JsonDataModel(
            json={"instructions": "improved"},
            schema={"type": "object"},
        )
        optimizer.mutation_programs[schema_id] = AsyncMock(return_value=expected)

        x, y, y_pred = self._make_batch()

        result = await optimizer.mutate_candidate(
            3,
            trainable_variable,
            {"instructions": "do the thing"},
            x=x,
            y=y,
            y_pred=y_pred,
            training=True,
        )

        self.assertEqual(result.get_json(), {"instructions": "improved"})

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
                "prompt": "seed",
                "candidates": candidates,
                "best_candidates": best_candidates,
                "history": [],
            },
            schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "candidates": {"type": "array"},
                    "best_candidates": {"type": "array"},
                    "history": {"type": "array"},
                },
            },
        )

        await optimizer.on_epoch_end(0, [trainable_variable])

        # Should keep top 2 candidates by reward
        result = trainable_variable.get("best_candidates")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["reward"], 0.9)
        self.assertEqual(result[1]["reward"], 0.5)
        self.assertEqual(trainable_variable.get("prompt"), "c2")
        self.assertEqual(trainable_variable.get("candidates"), [])
        self.assertEqual(optimizer.epochs, 1)

    async def test_on_epoch_end_with_dns(self):
        """DNS keeps the top `population_size` by competition fitness, then
        sorts the survivors by reward and records the best in the variable."""
        optimizer = OMEGA(
            algorithm="dns",
            embedding_model=self._embedding_by_prompt(self._DNS_VECTORS),
            k_nearest_fitter=2,
            population_size=3,
        )
        candidates = [dict(c) for c in self._DNS_CANDIDATES[:2]]  # c1, c2
        best_candidates = [dict(c) for c in self._DNS_CANDIDATES[2:]]  # c3, c4
        trainable_variable = JsonDataModel(
            json={
                "prompt": "seed",
                "candidates": candidates,
                "best_candidates": best_candidates,
                "history": [],
            },
            schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "candidates": {"type": "array"},
                    "best_candidates": {"type": "array"},
                    "history": {"type": "array"},
                },
            },
        )

        await optimizer.on_epoch_end(0, [trainable_variable])

        survivors = trainable_variable.get("best_candidates")
        # c2, the clone of the best with a lower reward, is the one dropped.
        self.assertEqual([c["prompt"] for c in survivors], ["c1", "c3", "c4"])
        self.assertEqual(trainable_variable.get("candidates"), [])
        # The base bookkeeping ran: best content written back, history kept,
        # epoch counter advanced.
        self.assertEqual(trainable_variable.get("prompt"), "c1")
        self.assertEqual(trainable_variable.get("history"), [{"prompt": "c1"}])
        self.assertEqual(optimizer.epochs, 1)


class SimilarityDistanceTest(testing.TestCase):
    async def test_similarity_distance_identical_candidates(self):
        """Identical content is at distance exactly 0."""
        mock_embedding_model = AsyncMock(spec=EmbeddingModel)
        mock_embedding_model.return_value = {"embeddings": [[1.0, 0.0, 0.0]]}

        candidate = {"prompt": "test"}

        distance = await similarity_distance(
            candidate, candidate, embedding_model=mock_embedding_model
        )

        self.assertAlmostEqual(distance, 0.0)

    async def test_similarity_distance_identical_multi_field_candidates(self):
        """Several fields with different embeddings still give 0 for a clone:
        the mean of the unit vectors is renormalized."""
        mock_embedding_model = AsyncMock(spec=EmbeddingModel)
        mock_embedding_model.return_value = {
            "embeddings": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        }

        candidate = {"prompt": "a", "hint": "b"}

        distance = await similarity_distance(
            candidate, dict(candidate), embedding_model=mock_embedding_model
        )

        self.assertAlmostEqual(distance, 0.0)

    async def test_similarity_distance_orthogonal_candidates(self):
        call_count = [0]

        async def mock_embed(texts):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"embeddings": [[1.0, 0.0, 0.0]]}
            else:
                return {"embeddings": [[0.0, 1.0, 0.0]]}

        mock_embedding_model = AsyncMock(side_effect=mock_embed)

        distance = await similarity_distance(
            {"prompt": "test1"}, {"prompt": "test2"}, embedding_model=mock_embedding_model
        )

        self.assertAlmostEqual(distance, 1.0)

    async def test_similarity_distance_ignores_candidate_metadata(self):
        """`reward` / `reward_count` are not content: they are neither embedded
        nor allowed to make two identical candidates look different."""
        seen = []

        async def mock_embed(request):
            seen.extend(request.get("texts"))
            return {"embeddings": [[1.0, 0.0, 0.0] for _ in request.get("texts")]}

        mock_embedding_model = AsyncMock(side_effect=mock_embed)

        distance = await similarity_distance(
            {"prompt": "same", "reward": 0.1, "reward_count": 1},
            {"prompt": "same", "reward": 0.9, "reward_count": 7},
            embedding_model=mock_embedding_model,
        )

        self.assertAlmostEqual(distance, 0.0)
        self.assertEqual(seen, ["same", "same"])

    async def test_similarity_distance_failed_embedding_is_maximal(self):
        mock_embedding_model = AsyncMock(spec=EmbeddingModel)
        mock_embedding_model.return_value = None

        distance = await similarity_distance(
            {"prompt": "a"}, {"prompt": "b"}, embedding_model=mock_embedding_model
        )

        self.assertEqual(distance, 1.0)


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


class ConfidenceAwareDNSTest(testing.TestCase):
    async def test_fitter_uses_the_confidence_bound(self):
        """With uncertainty on, a 0.95 measured on 4 samples (bound 0.825) is not
        'fitter' than a 0.90 measured on 100 (bound 0.875), so the solid
        candidate has no fitter neighbour and gets infinite fitness."""

        async def unit_distance(c1, c2, embedding_model=None, **kwargs):
            return 1.0

        candidates = [
            {"prompt": "lucky", "reward": 0.95, "reward_count": 4},
            {"prompt": "solid", "reward": 0.90, "reward_count": 100},
        ]
        optimizer = OMEGA(distance_function=unit_distance, reward_uncertainty=0.25)
        fitness = await optimizer.competition_fitness(candidates)
        self.assertEqual(fitness[1], float("inf"))
        self.assertAlmostEqual(fitness[0], 1.0)

        optimizer = OMEGA(distance_function=unit_distance, reward_uncertainty=0.0)
        fitness = await optimizer.competition_fitness(candidates)
        self.assertEqual(fitness[0], float("inf"))
        self.assertAlmostEqual(fitness[1], 1.0)

    async def test_on_epoch_end_folds_validation_before_competition(self):
        async def unit_distance(c1, c2, embedding_model=None, **kwargs):
            return 1.0

        optimizer = OMEGA(
            algorithm="dns",
            distance_function=unit_distance,
            reward_uncertainty=0.0,
            population_size=1,
        )
        trainable_variable = JsonDataModel(
            json={
                "prompt": "lucky",
                "candidates": [{"prompt": "lucky", "reward": 1.0, "reward_count": 4}],
                "best_candidates": [
                    {"prompt": "solid", "reward": 0.8, "reward_count": 60}
                ],
                "history": [],
            },
            schema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "candidates": {"type": "array"},
                    "best_candidates": {"type": "array"},
                    "history": {"type": "array"},
                },
            },
        )
        # Full validation says the promoted "lucky" candidate is really a 0.5.
        await optimizer.on_epoch_end(
            0, [trainable_variable], logs={"val_reward": 0.5}, val_size=60
        )
        (survivor,) = trainable_variable.get("best_candidates")
        self.assertEqual(survivor["prompt"], "solid")
        self.assertEqual(trainable_variable.get("prompt"), "solid")

    def test_reward_uncertainty_in_config(self):
        optimizer = OMEGA(reward_uncertainty=0.1)
        self.assertEqual(optimizer.get_config()["reward_uncertainty"], 0.1)


class LengthGuidelineTest(testing.TestCase):
    """Mutation and crossover prompts push back on instruction growth, which
    otherwise inflates candidates until judge prompts overflow their context."""

    def test_mutation_instructions_constrain_length(self):
        text = mutation_instructions(["instructions"]).lower()
        self.assertIn("about the same length", text)
        self.assertIn("do not add text", text)
        self.assertIn("longer is not better", text)

    def test_crossover_instructions_constrain_length(self):
        text = crossover_instructions(["instructions"]).lower()
        self.assertIn("must not be longer than the longer of the two inputs", text)
        self.assertIn("instead of concatenating", text)
        self.assertIn("longer is not better", text)


class GoodBadPredictionsTest(testing.TestCase):
    def _batch(self, rewards):
        x = [
            JsonDataModel(json={"q": f"q{i}"}, schema={"type": "object"})
            for i in range(len(rewards))
        ]
        y = [
            JsonDataModel(json={"a": f"gt{i}"}, schema={"type": "object"})
            for i in range(len(rewards))
        ]
        y_pred = [
            JsonDataModel(json={"a": f"pred{i}"}, schema={"type": "object"})
            for i in range(len(rewards))
        ]
        return x, y, y_pred

    def test_split_selects_best_and_worst_without_overlap(self):
        optimizer = OMEGA(nb_best_predictions=1, nb_worst_predictions=2)
        x, y, y_pred = self._batch([0.5, 1.0, 0.0, 0.25])
        good, bad = optimizer.split_predictions(
            x=x, y=y, y_pred=y_pred, rewards=[0.5, 1.0, 0.0, 0.25]
        )
        self.assertEqual([g["inputs"]["q"] for g in good], ["q1"])
        self.assertEqual([b["inputs"]["q"] for b in bad], ["q2", "q3"])  # worst first
        self.assertEqual(good[0]["reward"], 1.0)
        self.assertEqual(good[0]["predicted_output"], {"a": "pred1"})
        self.assertEqual(good[0]["ground_truth"], {"a": "gt1"})
        self.assertEqual([b["reward"] for b in bad], [0.0, 0.25])

    def test_split_small_batch_never_duplicates_a_sample(self):
        optimizer = OMEGA(nb_best_predictions=2, nb_worst_predictions=3)
        x, y, y_pred = self._batch([0.2, 0.9, 0.4])
        good, bad = optimizer.split_predictions(
            x=x, y=y, y_pred=y_pred, rewards=[0.2, 0.9, 0.4]
        )
        self.assertEqual([g["inputs"]["q"] for g in good], ["q1", "q2"])
        self.assertEqual([b["inputs"]["q"] for b in bad], ["q0"])

    def test_split_without_rewards_puts_everything_in_bad(self):
        optimizer = OMEGA(nb_best_predictions=1, nb_worst_predictions=3)
        x, y, y_pred = self._batch([None, None])
        good, bad = optimizer.split_predictions(x=x, y=y, y_pred=y_pred, rewards=None)
        self.assertEqual(good, [])
        self.assertEqual(len(bad), 2)
        self.assertIsNone(bad[0]["reward"])

    def test_split_missing_reward_ranks_as_worst(self):
        optimizer = OMEGA(nb_best_predictions=1, nb_worst_predictions=1)
        x, y, y_pred = self._batch([0.3, None, 0.9])
        good, bad = optimizer.split_predictions(
            x=x, y=y, y_pred=y_pred, rewards=[0.3, None, 0.9]
        )
        self.assertEqual(good[0]["inputs"]["q"], "q2")
        self.assertEqual(bad[0]["inputs"]["q"], "q1")

    async def test_mutation_program_receives_the_two_groups(self):
        optimizer = OMEGA(nb_best_predictions=1, nb_worst_predictions=2)
        optimizer.set_program(SimpleNamespace(description="A test program"))
        trainable_variable = JsonDataModel(
            json={"instructions": "do the thing"},
            schema={"type": "object", "properties": {"instructions": {"type": "string"}}},
        )
        trainable_variable.description = "the variable"
        schema_id = id(trainable_variable.get_schema())
        program = AsyncMock(return_value=None)
        optimizer.mutation_programs[schema_id] = program
        x, y, y_pred = self._batch([0.5, 1.0, 0.0])
        await optimizer.mutate_candidate(
            0,
            trainable_variable,
            {"instructions": "do the thing"},
            x=x,
            y=y,
            y_pred=y_pred,
            rewards=[0.5, 1.0, 0.0],
            training=True,
        )
        (inputs,), _ = program.await_args
        payload = inputs.get_json()
        self.assertEqual([g["inputs"]["q"] for g in payload["good_predictions"]], ["q1"])
        self.assertEqual(
            [b["inputs"]["q"] for b in payload["bad_predictions"]], ["q2", "q0"]
        )
        self.assertNotIn("program_inputs", payload)

    def test_group_sizes_in_config_and_validated(self):
        optimizer = OMEGA(nb_best_predictions=2, nb_worst_predictions=5)
        config = optimizer.get_config()
        self.assertEqual(config["nb_best_predictions"], 2)
        self.assertEqual(config["nb_worst_predictions"], 5)
        with self.assertRaises(ValueError):
            OMEGA(nb_worst_predictions=-1)

    def test_instructions_explain_the_groups(self):
        self.assertIn("bad_predictions", mutation_instructions(["instructions"]))
        self.assertIn("good_predictions", mutation_instructions(["instructions"]))
        self.assertIn("bad_predictions", crossover_instructions(["instructions"]))


class EchoedInputsTest(testing.TestCase):
    def test_predicted_output_drops_fields_identical_to_the_input(self):
        optimizer = OMEGA(nb_best_predictions=0, nb_worst_predictions=1)
        x = [
            JsonDataModel(
                json={"steps": [1, 2], "session_id": "s"}, schema={"type": "object"}
            )
        ]
        y_pred = [
            JsonDataModel(
                json={
                    "steps": [1, 2],
                    "session_id": "s",
                    "critique": "meh",
                    "reward": 0.3,
                },
                schema={"type": "object"},
            )
        ]
        _good, bad = optimizer.split_predictions(
            x=x, y=None, y_pred=y_pred, rewards=[0.4]
        )
        self.assertEqual(bad[0]["inputs"], {"steps": [1, 2], "session_id": "s"})
        self.assertEqual(bad[0]["predicted_output"], {"critique": "meh", "reward": 0.3})

    def test_predicted_output_keeps_fields_that_differ_from_the_input(self):
        optimizer = OMEGA(nb_best_predictions=0, nb_worst_predictions=1)
        x = [JsonDataModel(json={"answer": "draft"}, schema={"type": "object"})]
        y_pred = [JsonDataModel(json={"answer": "final"}, schema={"type": "object"})]
        _good, bad = optimizer.split_predictions(
            x=x, y=None, y_pred=y_pred, rewards=[0.4]
        )
        self.assertEqual(bad[0]["predicted_output"], {"answer": "final"})

    def test_non_dict_predictions_pass_through(self):
        optimizer = OMEGA(nb_best_predictions=0, nb_worst_predictions=1)
        _good, bad = optimizer.split_predictions(
            x=["raw"], y=None, y_pred=[None], rewards=[0.1]
        )
        self.assertEqual(bad[0]["inputs"], "raw")
        self.assertIsNone(bad[0]["predicted_output"])


class HardExampleMemoryTest(testing.TestCase):
    def _batch(self, names, rewards):
        x = [JsonDataModel(json={"q": n}, schema={"type": "object"}) for n in names]
        y = [
            JsonDataModel(json={"a": f"gt-{n}"}, schema={"type": "object"}) for n in names
        ]
        y_pred = [
            JsonDataModel(json={"q": n, "a": f"pred-{n}"}, schema={"type": "object"})
            for n in names
        ]
        return x, y, y_pred, rewards

    def test_recurring_failures_are_appended_to_bad_predictions(self):
        optimizer = OMEGA(
            nb_best_predictions=1,
            nb_worst_predictions=1,
            nb_hard_examples=1,
            hard_example_min_observations=2,
        )
        # "hard" is judged twice with low reward; "easy" twice with high reward.
        optimizer.observe_training_batch(*self._batch(["hard", "easy"], [0.1, 0.9]))
        optimizer.observe_training_batch(*self._batch(["hard", "easy"], [0.2, 1.0]))
        # A new batch that does not contain "hard".
        x, y, y_pred, rewards = self._batch(["b1", "b2"], [0.8, 0.6])
        good, bad = optimizer.split_predictions(x=x, y=y, y_pred=y_pred, rewards=rewards)
        self.assertEqual([g["inputs"]["q"] for g in good], ["b1"])
        self.assertEqual([b["inputs"]["q"] for b in bad], ["b2", "hard"])
        hard = bad[1]
        self.assertAlmostEqual(hard["reward"], 0.15)
        self.assertEqual(hard["nb_observations"], 2)
        self.assertEqual(hard["ground_truth"], {"a": "gt-hard"})
        # the echoed input field is stripped from the remembered output
        self.assertEqual(hard["predicted_output"], {"a": "pred-hard"})
        self.assertIsNone(bad[0]["nb_observations"])

    def test_hard_examples_skip_inputs_already_in_the_batch(self):
        optimizer = OMEGA(
            nb_best_predictions=0, nb_worst_predictions=1, nb_hard_examples=2
        )
        optimizer.observe_training_batch(*self._batch(["hard", "other"], [0.0, 0.1]))
        optimizer.observe_training_batch(*self._batch(["hard", "other"], [0.0, 0.1]))
        x, y, y_pred, rewards = self._batch(["hard"], [0.0])
        _good, bad = optimizer.split_predictions(x=x, y=y, y_pred=y_pred, rewards=rewards)
        self.assertEqual([b["inputs"]["q"] for b in bad], ["hard", "other"])
        self.assertEqual([b["nb_observations"] for b in bad], [None, 2])

    def test_min_observations_gates_the_memory(self):
        optimizer = OMEGA(
            nb_worst_predictions=0, nb_hard_examples=3, hard_example_min_observations=3
        )
        optimizer.observe_training_batch(*self._batch(["h"], [0.0]))
        optimizer.observe_training_batch(*self._batch(["h"], [0.0]))
        self.assertEqual(optimizer.hard_examples(), [])
        optimizer.observe_training_batch(*self._batch(["h"], [0.0]))
        self.assertEqual(len(optimizer.hard_examples()), 1)

    def test_memory_disabled_with_zero_hard_examples(self):
        optimizer = OMEGA(nb_hard_examples=0)
        optimizer.observe_training_batch(*self._batch(["h"], [0.0]))
        optimizer.observe_training_batch(*self._batch(["h"], [0.0]))
        self.assertEqual(optimizer.hard_examples(), [])
        self.assertEqual(optimizer._difficulty, {})

    def test_hard_example_params_in_config_and_validated(self):
        optimizer = OMEGA(nb_hard_examples=2, hard_example_min_observations=4)
        config = optimizer.get_config()
        self.assertEqual(config["nb_hard_examples"], 2)
        self.assertEqual(config["hard_example_min_observations"], 4)
        with self.assertRaises(ValueError):
            OMEGA(hard_example_min_observations=0)


class _Ticket(DataModel):
    message: str = Field(description="The customer message")


class _Team(DataModel):
    team: Literal["billing", "technical"] = Field(
        description="Which team should handle the ticket?"
    )


EVOLVED_INSTRUCTIONS = "Route billing questions to billing, crashes to technical."


@patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"})
class OMEGAWithDecisionModelTest(testing.TestCase):
    @patch("litellm.aembedding")
    @patch("litellm.acompletion")
    async def test_evolves_the_instructions_of_a_decision_model(
        self, mock_completion, mock_embedding
    ):
        """OMEGA's language model writes the candidate instructions, and the
        decision model answers with each candidate in its state."""

        def answers(payload):
            # The decision model only gets it right with the evolved
            # instructions in its system message.
            system = payload["state"][0]["content"]
            user = payload["state"][-1]["content"]
            if EVOLVED_INSTRUCTIONS in system and "crash" in user:
                team, other = "technical", "billing"
            else:
                team, other = "billing", "technical"
            return {
                "team": {
                    "type": "choice",
                    "choice": team,
                    "probabilities": {team: 0.8, other: 0.2},
                    "confidence": 0.7,
                }
            }

        async def completion(*args, **kwargs):
            schema = kwargs["response_format"]["json_schema"]["schema"]
            candidate = {}
            for key, value in schema.get("properties", {}).items():
                if key == "instructions":
                    candidate[key] = EVOLVED_INSTRUCTIONS
                elif value.get("type") == "string":
                    candidate[key] = "Separate billing from technical issues."
                else:
                    candidate[key] = []
            return {"choices": [{"message": {"content": json.dumps(candidate)}}]}

        async def embedding(*args, **kwargs):
            inputs = kwargs.get("input") or args[1]
            inputs = inputs if isinstance(inputs, list) else [inputs]
            return {"data": [{"embedding": [0.1, 0.2, 0.3]} for _ in inputs]}

        mock_completion.side_effect = completion
        mock_embedding.side_effect = embedding
        decision_model = DecisionModel(model="typesafe/jev-latest")
        payloads = mock_decision_model(decision_model, answers)

        x0 = Input(data_model=_Ticket)
        x1 = await Generator(
            data_model=_Team,
            decision_model=decision_model,
            instructions="Triage.",
        )(x0)
        program = Program(inputs=x0, outputs=x1)
        program.compile(
            reward=ExactMatch(),
            optimizer=OMEGA(
                language_model=LanguageModel(model="openai/gpt-4o-mini"),
                embedding_model=EmbeddingModel(model="openai/text-embedding-3-small"),
            ),
        )
        # Each ticket's team follows from its text, so a reward is earned only
        # by answering from the input (validation included).
        x = np.array(
            [_Ticket(message="I was charged twice"), _Ticket(message="The app crashes")]
            * 2,
            dtype="object",
        )
        y = np.array(
            [_Team(team="billing"), _Team(team="technical")] * 2,
            dtype="object",
        )

        history = await program.fit(
            x=x, y=y, epochs=2, batch_size=2, verbose=0, validation_data=(x, y)
        )

        self.assertEqual(
            program.trainable_variables[0].get("instructions"), EVOLVED_INSTRUCTIONS
        )
        self.assertIn(EVOLVED_INSTRUCTIONS, payloads[-1]["state"][0]["content"])
        self.assertEqual(history.history["val_reward"][-1], 1.0)
