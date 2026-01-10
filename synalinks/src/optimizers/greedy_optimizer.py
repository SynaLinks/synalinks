# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import numpy as np

from synalinks.src.api_export import synalinks_export
from synalinks.src.optimizers.optimizer import Optimizer


@synalinks_export("synalinks.optimizers.GreedyOptimizer")
class GreedyOptimizer(Optimizer):
    """Abstract base class for greedy sampling-based optimizers.

    Greedy optimizers focus on sampling from best predictions.
    They use configurable sampling strategies to select high-reward
    predictions as few-shot examples.

    Subclasses must implement:
        - `propose_new_candidates()`: Generate new candidates using sampling

    Args:
        nb_min_examples (int): The min number of examples for few-shot
            learning (Default to 1).
        nb_max_examples (int): The max number of examples for few-shot
            learning (Default to 3).
        sampling (str): The method to sample predictions between
            ['random', 'best', 'softmax']. (Default to 'softmax').
        sampling_temperature (float): The temperature for softmax sampling.
            Used only when `sampling='softmax'`. Lower values concentrate
            sampling on high-reward predictions, higher values make sampling
            more uniform (Default 0.3).
        population_size (int): The maximum number of best candidates to keep
            during the optimization process.
        name (str): Optional name for the optimizer instance.
        description (str): Optional description of the optimizer instance.
    """

    def __init__(
        self,
        nb_min_examples=1,
        nb_max_examples=3,
        sampling="softmax",
        sampling_temperature=0.3,
        population_size=10,
        name=None,
        description=None,
        **kwargs,
    ):
        super().__init__(
            population_size=population_size,
            name=name,
            description=description,
            **kwargs,
        )
        self.nb_min_examples = nb_min_examples
        self.nb_max_examples = nb_max_examples

        samplings = ["random", "best", "softmax"]
        if sampling not in samplings:
            raise ValueError(f"Parameter `sampling` should be between {samplings}")
        self.sampling = sampling
        self.sampling_temperature = sampling_temperature

    async def sample_best_predictions(
        self,
        trainable_variable,
    ):
        """Sample predictions based on the configured sampling strategy.

        Samples between `nb_min_examples` and `nb_max_examples` predictions
        from the trainable variable using the configured sampling method:
        - 'random': uniformly random selection
        - 'best': select highest reward predictions
        - 'softmax': softmax sampling biased toward higher rewards

        Args:
            trainable_variable: The trainable variable containing predictions.

        Returns:
            List of selected predictions.
        """
        predictions = trainable_variable.get("predictions")
        nb_examples = np.random.randint(self.nb_min_examples, self.nb_max_examples + 1)
        selected_predictions = []

        if nb_examples != 0:
            if len(predictions) > nb_examples:
                if self.sampling == "random":
                    selected_predictions = np.random.choice(
                        predictions,
                        size=nb_examples,
                        replace=False,
                    ).tolist()
                elif self.sampling == "best":
                    sorted_predictions = sorted(
                        predictions,
                        key=lambda x: x.get("reward", 0),
                        reverse=True,
                    )
                    selected_predictions = sorted_predictions[:nb_examples]
                elif self.sampling == "softmax":
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
        base_config = super().get_config()
        base_config.update(
            {
                "nb_min_examples": self.nb_min_examples,
                "nb_max_examples": self.nb_max_examples,
                "sampling": self.sampling,
                "sampling_temperature": self.sampling_temperature,
            }
        )
        return base_config
