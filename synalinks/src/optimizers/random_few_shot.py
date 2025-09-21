# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import copy
import math
import random

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
                temperature=1.0,
                merging_rate=0.02,
            ),
        )

        history = await program.fit(...)
    ```

    References:
        - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

    Args:
        nb_min_examples (int): The min number of examples for few-shot learning (Default to 1).
        nb_max_examples (int): The max number of examples for few-shot learning (Default to 3).
        temperature (float): The temperature for softmax sampling of the few-shot
            learning examples. Lower values concentrate sampling on high-reward predictions,
            higher values make sampling more uniform (Default 1.0).
        merging_rate (float): Rate at which crossover vs mutation is selected. (Default to 0.02).
        nb_max_best_candidates (int): The maximum number of best candidates to keep
            during the optimization process.
        name (str): Optional name for the optimizer instance.
        description (str): Optional description of the optimizer instance.
    """

    def __init__(
        self,
        nb_min_examples=1,
        nb_max_examples=3,
        temperature=1.0,
        merging_rate=0.02,
        nb_max_best_candidates=5,
        name=None,
        description=None,
    ):
        super().__init__(
            nb_max_best_candidates=nb_max_best_candidates,
            name=name,
            description=description,
        )
        self.nb_min_examples = nb_min_examples
        self.nb_max_examples = nb_max_examples
        self.temperature = temperature
        self.merging_rate = merging_rate

    async def build(self, _):
        self.built = True

    async def propose_new_candidates(
        self,
        step,
        trainable_variables,
        x=None,
        y=None,
        y_pred=None,
    ):
        variable_name_to_update = self.select_variable_name_to_update(
            trainable_variables,
        )

        strategy = self.select_evolving_strategy()

        for trainable_variable in trainable_variables:
            if trainable_variable.name == variable_name_to_update:
                if strategy == "mutation":
                    examples = self.sample_best_predictions(
                        trainable_variable,
                    )
                elif strategy == "crossover":
                    candidate_to_merge = self.select_candidate_to_merge(
                        step,
                        trainable_variable,
                    )
                    examples = self.merge_examples(
                        trainable_variable.get("examples"),
                        candidate_to_merge.get("examples"),
                    )
                self.assign_candidate(
                    trainable_variable,
                    examples=examples,
                )

    def select_variable_name_to_update(self, trainable_variables):
        rewards = []
        for trainable_variable in trainable_variables:
            nb_visit = trainable_variable.get("nb_visit")
            cumulative_reward = trainable_variable.get("cumulative_reward")
            if nb_visit == 0:
                variable_reward = 10000 # when inverted, this will result in a very low prob to be selected
            else:
                variable_reward = cumulative_reward / nb_visit
            rewards.append(variable_reward)
        rewards = np.array(rewards)
        inverted_rewards = -rewards
        scaled_rewards = inverted_rewards / self.temperature
        exp_rewards = np.exp(scaled_rewards - np.max(scaled_rewards))
        probabilities = exp_rewards / np.sum(exp_rewards)
        selected_variable = np.random.choice(
            trainable_variables,
            size=1,
            replace=False,
            p=probabilities,
        ).tolist()[0]
        return selected_variable.name

    def select_evolving_strategy(self):
        rand = random.random()
        if rand > (self.merging_rate * self.epochs):
            return "mutation"
        else:
            return "crossover"

    def select_candidate_to_merge(
        self,
        step,
        trainable_variable,
    ):
        best_candidates = trainable_variable.get("best_candidates")
        best_candidates = copy.deepcopy(best_candidates)
        del best_candidates[step]
        selected_candidate = random.choice(best_candidates)
        return selected_candidate

    def merge_examples(
        self,
        examples1,
        examples2,
    ):
        nb_examples = math.floor((len(examples1) + len(examples2)) / 2.0)
        all_examples = examples1 + examples2
        if len(all_examples) > nb_examples:
            rewards = np.array([ex.get("reward", 0) for ex in all_examples])
            scaled_rewards = rewards / self.temperature
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

    def sample_best_predictions(
        self,
        trainable_variable,
    ):
        predictions = trainable_variable.get("predictions")
        nb_examples = np.random.randint(self.nb_min_examples, self.nb_max_examples + 1)
        selected_predictions = []
        if nb_examples != 0:
            if len(predictions) > nb_examples:
                rewards = np.array([pred.get("reward", 0) for pred in predictions])
                scaled_rewards = rewards / self.temperature
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
            "temperature": self.temperature,
            "merging_rate": self.merging_rate,
            "nb_max_best_candidates": self.nb_max_best_candidates,
            "name": self.name,
            "description": self.description,
        }
