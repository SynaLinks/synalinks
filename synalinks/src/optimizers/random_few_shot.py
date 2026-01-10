# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.optimizers.greedy_optimizer import GreedyOptimizer


@synalinks_export("synalinks.optimizers.RandomFewShot")
class RandomFewShot(GreedyOptimizer):
    """Sample among the best predictions to populate the LM's prompt.

    Makes the model learn using Few Shot Learning by selecting predictions
    as examples based on their rewards using the configured sampling strategy.

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
                sampling="softmax",
                sampling_temperature=1.0,
            ),
        )

        history = await program.fit(...)
    ```

    References:
        - DSPy: Compiling Declarative Language Model Calls into
          Self-Improving Pipelines (https://arxiv.org/pdf/2310.03714)
        - Language Models are Few-Shot Learners
          (https://arxiv.org/abs/2005.14165)

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
    ):
        super().__init__(
            nb_min_examples=nb_min_examples,
            nb_max_examples=nb_max_examples,
            sampling=sampling,
            sampling_temperature=sampling_temperature,
            population_size=population_size,
            name=name,
            description=description,
        )

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

        for trainable_variable in trainable_variables:
            if trainable_variable.name == variable_name_to_update:
                examples = await self.sample_best_predictions(
                    trainable_variable,
                )
                await self.assign_candidate(
                    trainable_variable,
                    examples=examples,
                )
