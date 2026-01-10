# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import random

import numpy

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import out_mask_json
from synalinks.src.optimizers.optimizer import Optimizer
from synalinks.src.saving import serialization_lib


@synalinks_export("synalinks.optimizers.EvolutionaryOptimizer")
class EvolutionaryOptimizer(Optimizer):
    """Abstract base class for evolutionary LLM-based optimizers.

    Evolutionary optimizers use LLMs for mutation and crossover operations,
    with configurable selection strategies for choosing candidates to evolve.

    Subclasses must implement:
        - `build()`: Build mutation and crossover programs
        - `mutate_candidate()`: Apply mutation to generate a new candidate
        - `merge_candidate()`: Apply crossover to merge two candidates

    Args:
        language_model (LanguageModel): The language model to use for
            mutation and crossover.
        mutation_temperature (float): The temperature for the LM calls of
            the mutation programs.
        crossover_temperature (float): The temperature for the LM calls of
            the crossover programs.
        selection (str): The method to select the candidate to evolve at the
            beginning of a batch between ['random', 'best', 'softmax'].
            (Default to 'softmax').
        selection_temperature (float): The temperature for softmax selection.
            Used only when `selection='softmax'`. Lower values concentrate
            selection on high-reward candidates, higher values make selection
            more uniform (Default 0.3).
        merging_rate (float): Rate at which crossover vs mutation is selected.
            (Default to 0.02).
        population_size (int): The maximum number of best candidates to keep
            during the optimization process.
        name (str): Optional name for the optimizer instance.
        description (str): Optional description of the optimizer instance.
    """

    def __init__(
        self,
        language_model=None,
        mutation_temperature=0.3,
        crossover_temperature=0.3,
        selection="softmax",
        selection_temperature=0.3,
        merging_rate=0.02,
        population_size=10,
        name=None,
        description=None,
        **kwargs,
    ):
        super().__init__(
            population_size=population_size,
            name=name,
            description=description,
        )
        self.merging_rate = merging_rate
        self.language_model = language_model
        self.mutation_temperature = mutation_temperature
        self.crossover_temperature = crossover_temperature

        self.kwargs = kwargs

        selections = ["best", "random", "softmax"]
        if selection not in selections:
            raise ValueError(f"Parameter `selection` should be between {selections}")
        self.selection = selection
        self.selection_temperature = selection_temperature

        self.mutation_programs = {}
        self.crossover_programs = {}

    def select_candidate(self, candidates):
        """Select a candidate from the list based on the selection strategy.

        Args:
            candidates (list): List of candidate dictionaries with 'reward' key

        Returns:
            The selected candidate, or None if the list is empty
        """
        if not candidates:
            return None

        if self.selection == "random":
            return random.choice(candidates)
        elif self.selection == "best":
            return sorted(
                candidates,
                key=lambda x: x.get("reward", 0),
                reverse=True,
            )[0]
        elif self.selection == "softmax":
            rewards = numpy.array(
                [candidate.get("reward", 0) for candidate in candidates]
            )
            scaled_rewards = rewards / self.selection_temperature
            exp_rewards = numpy.exp(scaled_rewards - numpy.max(scaled_rewards))
            probabilities = exp_rewards / numpy.sum(exp_rewards)
            return numpy.random.choice(
                candidates,
                size=1,
                replace=False,
                p=probabilities,
            ).tolist()[0]

    async def on_batch_begin(
        self,
        step,
        epoch,
        trainable_variables,
    ):
        """Called at the beginning of a batch.

        Implements selection strategies for choosing which candidate to evolve.

        Args:
            step (int): The batch number
            epoch (int): The epoch number
            trainable_variables (list): The list of trainable variables
        """
        for trainable_variable in trainable_variables:
            best_candidates = trainable_variable.get("best_candidates")
            if epoch == 0:
                seed_candidates = trainable_variable.get("seed_candidates")
                if len(seed_candidates) > 0:
                    seed_candidate = random.choice(seed_candidates)
                    trainable_variable.update(
                        {
                            **seed_candidate,
                        },
                    )
            else:
                if len(best_candidates) > 0:
                    best_candidate = self.select_candidate(best_candidates)
                    best_candidate = out_mask_json(
                        best_candidate,
                        mask=["reward"],
                    )
                    trainable_variable.update(
                        {
                            **best_candidate,
                        },
                    )
                else:
                    seed_candidates = trainable_variable.get("seed_candidates")
                    if len(seed_candidates) > 0:
                        seed_candidate = random.choice(seed_candidates)
                        trainable_variable.update(
                            {
                                **seed_candidate,
                            },
                        )
            trainable_variable.update(
                {
                    "nb_visit": 0,
                    "cumulative_reward": 0.0,
                },
            )

    async def propose_new_candidates(
        self,
        step,
        trainable_variables,
        x=None,
        y=None,
        y_pred=None,
        training=False,
    ):
        """Generate new candidates using mutation or crossover strategy.

        This method selects which variable to update, chooses between mutation
        and crossover based on merging_rate, and applies the appropriate operation.

        Args:
            step (int): The current training step
            trainable_variables (list): List of trainable variables to optimize
            x: Input data batch
            y: Ground truth data batch
            y_pred: Predicted outputs from the current model
            training (bool): Whether in training mode
        """
        variable_name_to_update = await self.select_variable_name_to_update(
            trainable_variables,
        )

        strategy = await self.select_evolving_strategy()

        for trainable_variable in trainable_variables:
            if trainable_variable.name == variable_name_to_update:
                best_candidates = trainable_variable.get("best_candidates")
                selected_candidate = self.select_candidate(best_candidates)

                if strategy == "mutation":
                    new_candidate = await self.mutate_candidate(
                        step,
                        trainable_variable,
                        selected_candidate,
                        x=x,
                        y=y,
                        y_pred=y_pred,
                        training=training,
                    )
                elif strategy == "crossover":
                    if len(best_candidates) >= 2:
                        # Select another candidate for crossover
                        other_candidate = await self.select_candidate_to_merge(
                            step,
                            trainable_variable,
                        )
                        new_candidate = await self.merge_candidate(
                            step,
                            trainable_variable,
                            selected_candidate,
                            other_candidate,
                            x=x,
                            y=y,
                            y_pred=y_pred,
                            training=training,
                        )
                    else:
                        # Fallback to mutation if not enough candidates to merge
                        new_candidate = await self.mutate_candidate(
                            step,
                            trainable_variable,
                            selected_candidate,
                            x=x,
                            y=y,
                            y_pred=y_pred,
                            training=training,
                        )

                await self.assign_candidate(
                    trainable_variable,
                    new_candidate=new_candidate,
                )

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
        """Apply mutation to generate a new candidate.

        Subclasses must implement this method to define their mutation logic.

        Args:
            step (int): The current training step
            trainable_variable: The trainable variable (for metadata access)
            selected_candidate: The selected candidate to mutate
            x: Input data batch
            y: Ground truth data batch
            y_pred: Predicted outputs from the current model
            training (bool): Whether in training mode

        Returns:
            The mutated candidate
        """
        raise NotImplementedError(
            "EvolutionaryOptimizer subclasses must implement `mutate_candidate()`."
        )

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
        """Apply crossover to merge two selected candidates.

        Subclasses must implement this method to define their crossover logic.

        Args:
            step (int): The current training step
            trainable_variable: The trainable variable (for metadata access)
            current_candidate: First selected candidate to merge
            other_candidate: Second selected candidate to merge
            x: Input data batch
            y: Ground truth data batch
            y_pred: Predicted outputs from the current model
            training (bool): Whether in training mode

        Returns:
            The merged candidate
        """
        raise NotImplementedError(
            "EvolutionaryOptimizer subclasses must implement `merge_candidate()`."
        )

    async def select_evolving_strategy(self):
        """Select between mutation and crossover based on merging_rate.

        Returns:
            str: Either "mutation" or "crossover"
        """
        rand = random.random()
        if rand > (self.merging_rate * self.epochs):
            return "mutation"
        else:
            return "crossover"

    def get_config(self):
        base_config = super().get_config()
        config = {
            "merging_rate": self.merging_rate,
            "mutation_temperature": self.mutation_temperature,
            "crossover_temperature": self.crossover_temperature,
            "selection": self.selection,
            "selection_temperature": self.selection_temperature,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model,
            )
        }
        return {**base_config, **config, **language_model_config}

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model"),
        )
        return cls(language_model=language_model, **config)
