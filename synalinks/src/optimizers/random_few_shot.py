# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import math

import numpy as np

from synalinks.src.api_export import synalinks_export
from synalinks.src.optimizers.optimizer import Optimizer


@synalinks_export("synalinks.optimizers.RandomFewShot")
class RandomFewShot(Optimizer):
    """Sample randomly among the best examples to populate the LM's prompt to make it
        learn using Few Shot Learning. Additionaly use an evolutionary method to merge the examples
        from the best candidates over time.

    Example:

    ```python
    import synalinks
    import asyncio

    async def main():
        # ... your program definition

        program.compile(
            reward=synalinks.rewards.ExactMatch(),
            optimizer=synalinks.optimizers.RandomFewShot(
                nb_min_examples=1,
                nb_max_examples=3,
                sampling_temperature=1.0,
                merging_rate=0.02,
            ),
        )

        history = await program.fit(...)
    ```

    References:
        - [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/pdf/2310.03714)
        - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

    Args:
        nb_min_examples (int): The min number of examples for few-shot learning (Default to 1).
        nb_max_examples (int): The max number of examples for few-shot learning (Default to 3).
        sampling_temperature (float): The sampling_temperature for softmax sampling of the few-shot
            learning examples. Lower values concentrate sampling on high-reward predictions,
            higher values make sampling more uniform (Default 1.0).
        merging_rate (float): Rate at which crossover vs mutation is selected. (Default to 0.02).
        population_size (int): The maximum number of best candidates to keep
            during the optimization process.
        name (str): Optional name for the optimizer instance.
        description (str): Optional description of the optimizer instance.
    """

    def __init__(
        self,
        nb_min_examples=1,
        nb_max_examples=3,
        sampling_temperature=1.0,
        merging_rate=0.02,
        population_size=10,
        name=None,
        description=None,
    ):
        super().__init__(
            merging_rate=merging_rate,
            population_size=population_size,
            name=name,
            description=description,
        )
        self.nb_min_examples = nb_min_examples
        self.nb_max_examples = nb_max_examples
        self.sampling_temperature = sampling_temperature

    async def build(self, _):
        self.built = True

    async def propose_new_candidates(
        self,
        step,
        trainable_variables,
        x=None,
        y=None,
        y_pred=None,
        training=False,
    ):
        variable_name_to_update = await self.select_variable_name_to_update(
            trainable_variables,
        )

        strategy = await self.select_evolving_strategy()

        for trainable_variable in trainable_variables:
            if trainable_variable.name == variable_name_to_update:
                if strategy == "mutation":
                    examples = await self.sample_best_predictions(
                        trainable_variable,
                    )
                elif strategy == "crossover":
                    candidate_to_merge = await self.select_candidate_to_merge(
                        step,
                        trainable_variable,
                    )
                    if candidate_to_merge:
                        examples = await self.merge_examples(
                            trainable_variable.get("examples"),
                            candidate_to_merge.get("examples"),
                        )
                    else:
                        examples = await self.sample_best_predictions(
                            trainable_variable,
                        )
                await self.assign_candidate(
                    trainable_variable,
                    examples=examples,
                )

    async def merge_examples(
        self,
        examples1,
        examples2,
    ):
        nb_examples = math.floor((len(examples1) + len(examples2)) / 2.0)
        all_examples = examples1 + examples2
        if len(all_examples) > nb_examples:
            rewards = np.array([ex.get("reward", 0) for ex in all_examples])
            scaled_rewards = rewards / self.sampling_temperature
            exp_rewards = np.exp(scaled_rewards - np.max(scaled_rewards))
            probabilities = exp_rewards / np.sum(exp_rewards)
            examples = np.random.choice(
                all_examples,
                size=nb_examples,
                replace=False,
                p=probabilities,
            ).tolist()
        else:
            examples = all_examples
        return examples

    async def sample_best_predictions(
        self,
        trainable_variable,
    ):
        predictions = trainable_variable.get("predictions")
        nb_examples = np.random.randint(self.nb_min_examples, self.nb_max_examples + 1)
        selected_predictions = []
        if nb_examples != 0:
            if len(predictions) > nb_examples:
                rewards = np.array([pred.get("reward", 0) for pred in predictions])
                scaled_rewards = rewards / self.sampling_temperature
                exp_rewards = np.exp(scaled_rewards - np.max(scaled_rewards))
                probabilities = exp_rewards / np.sum(exp_rewards)
                selected_predictions = np.random.choice(
                    predictions,
                    size=nb_examples,
                    replace=False,
                    p=probabilities,
                ).tolist()
            else:
                selected_predictions = predictions
        return selected_predictions

    def get_config(self):
        return {
            "nb_min_examples": self.nb_min_examples,
            "nb_max_examples": self.nb_max_examples,
            "sampling_temperature": self.sampling_temperature,
            "merging_rate": self.merging_rate,
            "population_size": self.population_size,
            "name": self.name,
            "description": self.description,
        }
