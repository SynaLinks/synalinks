# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from typing import Any
from typing import List
from typing import Optional

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import Trainable
from synalinks.src.backend import out_mask_json
from synalinks.src.backend.common import numpy as np
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.ttc.chain_of_thought import ChainOfThought
from synalinks.src.optimizers.random_few_shot import RandomFewShot
from synalinks.src.programs.program import Program
from synalinks.src.rewards.reward import squeeze_or_expand_to_same_rank
from synalinks.src.saving import serialization_lib


def base_instructions():
    """
    Base instructions that define the context for all optimization programs.
    These instructions explain that the system optimizes JSON variables in a computation graph.
    """
    return """
You are an integral part of an optimization system designed to improve JSON variables within a computation graph (i.e. the program).
Each module in the graph performs specific computations, with JSON variables serving as the state.
These variables can represent prompts, code, plans, rules, or any other JSON-compatible data.
""".strip()


def mutation_instructions(variables_keys):
    """
    Instructions for the mutation program that optimizes variables.

    Args:
        variables_keys (list): List of keys that the variable should contain
    """
    return f"""
Your primary task is to creatively enhance the provided variable so that the predicted output aligns as closely as possible with the ground truth.
Pay close attention to the variable's description, its intended use, and the broader context of the computation graph.

Guidelines:
- Ensure the new variable is generalizable and performs well across various inputs of the same kind.
- If you have to optimize a variable containing code, don't hardcode solutions but use a generalizable code.
- Include all specified keys: {variables_keys}.
- Justify each change with clear reasoning, referencing the variable's purpose and the desired output.
- If no ground truth is provided, the goal is to critically enhance the predicted output.
""".strip()


class MutationInputs(DataModel):
    program_description: str = Field(
        description="The program description",
    )
    program_inputs: List[Any] = Field(
        description="The inputs of the program",
    )
    program_predicted_outputs: List[Any] = Field(
        description="The program's predicted outputs",
    )
    program_ground_truth: List[Optional[Any]] = Field(
        description="The program's ground truth",
    )
    variable_description: str = Field(
        description="The description of the variable to optimize within that program"
    )
    current_variable: Any = Field(
        description="The variable to optimize",
    )


def crossover_instructions(variables_keys):
    """
    Instructions for the crossover program that optimizes variables.

    Args:
        variables_keys (list): List of keys that the variable should contain
    """
    return f"""
Your responsibility is to create a new, optimized variable by strategically combining features from the current variable and a high-performing candidate.
The new variable should improve the alignment of the predicted output with the ground truth.

Guidelines:
- Analyze both the current variable and the other high-performing variable, identifying their respective strengths and weaknesses.
- Pay close attention to the variable's description, its intended use, and the broader context of the computation graph.
- Ensure the new variable is generalizable and performs well across various inputs of the same kind.
- If you have to optimize a variable containing code, don't hardcode solutions but use a generalizable code.
- Include all specified keys: {variables_keys}.
- Justify each feature you incorporate, explaining how it contributes to better performance or alignment with the ground truth.
- If no ground truth is provided, the goal is to critically enhance the predicted output. 
""".strip()


class CrossoverInputs(DataModel):
    program_description: str = Field(
        description="The program description",
    )
    program_inputs: List[Any] = Field(
        description="The inputs of the program",
    )
    program_predicted_outputs: List[Any] = Field(
        description="The program's predicted outputs",
    )
    program_ground_truth: List[Optional[Any]] = Field(
        description="The program's ground truth",
    )
    variable_description: str = Field(
        description="The description of the variable to optimize within that program",
    )
    other_variable: Any = Field(
        description="other high performing variable to merge",
    )
    current_variable: Any = Field(
        description="current high performing variable to merge",
    )


async def similarity_distance(candidate1, candidate2, embedding_model=None, axis=-1):
    """The default cosine similarity distance used by Dominated Novelty Search

    If the candidates have multiple fields, they are combined by averaging the
    vectors before calculating the distance allowing candidates to have
    variable/dynamic number of fields.

    Args:
        candidate1 (dict): The first variable candidate
        candidate2 (dict): The second variable candidate
        embedding_model (EmbeddingModel): The embedding model to use
        axis (int): The axis along which compute the similarity (default -1)
    """
    embeddings1 = await embedding_model(tree.flatten(candidate1))
    embeddings2 = await embedding_model(tree.flatten(candidate2))
    embeddings1 = embeddings1["embeddings"]
    embeddings2 = embeddings2["embeddings"]
    embeddings1 = np.convert_to_tensor(embeddings1)
    embeddings2 = np.convert_to_tensor(embeddings2)
    embeddings1, embeddings2 = squeeze_or_expand_to_same_rank(embeddings1, embeddings2)
    embeddings1 = np.normalize(embeddings1, axis=axis)
    embeddings2 = np.normalize(embeddings2, axis=axis)
    embeddings1 = np.mean(embeddings1, axis=0)
    embeddings2 = np.mean(embeddings2, axis=0)
    similarity = (np.sum(embeddings1 * embeddings2, axis=axis) + 1) / 2
    return 1 - similarity


@synalinks_export("synalinks.optimizers.OMEGA")
class OMEGA(RandomFewShot):
    """OMEGA: OptiMizEr as Genetic Algorithm - A genetic optimizer with dominated novelty search.

    This optimizer is **unique to Synalinks** and the result of our research effort on advancing neuro-symbolic AI.

    Dominated Novelty Search (DNS), is a SOTA Quality-Diversity optimization method that implements a competition function in a classic genetic algorithm.

    The key insight behind Dominated Novelty Search is that candidates should be eliminated from the population if they are both:

    - Inferior in reward/fitness
    - Similar to existing candidates/solutions

    This algorithm creates an evolutionary pressure to focus on high performing candidates **Or** candidates that explore other approaches.

    This approach only add one step to the traditional genetic algorithm and *outperform* MAP-Elites, Threshold-Elites and Cluster-Elites.

    This allow the system to explore the search space more quickly by eliminating non-promising candidates while preserving diversity to avoid local optimum.

    At Synalinks, we adapted this algorithm for LM-based optimization, to do so we use an embedding model to compute the candidate's descriptor and a cosine distance between solutions.

    **Note**: In Synalinks, unlike other In-Context learning frameworks, a variable (the module's state to optimize) is a JSON object not a simple string.
    Which has multiple implications, we maintain a 100% correct structure through constrained JSON decoding, and we allow the state to have variable/dynamic
    number of fields, which is handled by this approach by embedding each field and averaging them before computing the distance required by DNS.

    Example:
    ```
    import synalinks
    import asyncio

    async def main():
        # ... your program definition

        program.compile(
            reward=synalinks.rewards.ExactMatch(),
            optimizer=synalinks.optimizers.OMEGA(
                language_model=language_model,
                embedding_model=embedding_model,
            )
        )

        history = await program.fit(...)
    ```

    Concerning the inspirations for this optimizer:
        - Dominated Novelty Search for their elegant Quality-Diversity algorithm that outperform many other evolutionary strategies.
        - DSPY's GEPA for feeding the optimizer program with the raw training data and for formalizing the evolutionary optimization strategy (**not** the MAP-Elites method used).
        - DeepMind's AlphaEvolve have been a huge inspiration, more on the motivational side as they didn't released the code.

    References:
        - [Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity](https://arxiv.org/html/2502.00593v1)
        - [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/pdf/2507.19457)
        - [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/pdf/2506.13131)

    Args:
        language_model (LanguageModel): The language model to use.
        embedding_model (EmbeddingModel): The embedding model to use to compute candidates
            descriptors according to Dominated Novelty Search.
        k_nearest_fitter (int): The K nearest fitter used by Dominated Novelty Search.
        distance_function (callable): Optional. The distance function to use by Dominated Novelty Search.
            If no function is provided, use the default cosine distance.
        mutation_temperature (float): The temperature for the LM calls of the mutation programs.
        crossover_temperature (float): The temperature for the LM calls of the crossover programs.
        few_shot_learning (bool): If `True` enable the selection of examples using
            the same method than the `RandomFewShot` optimizer.
        nb_min_examples (int): The min number of examples for few-shot learning (Default to 1).
        nb_max_examples (int): The max number of examples for few-shot learning (Default to 3).
        sampling_temperature (float): The temperature for softmax sampling of the few-shot
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
        language_model=None,
        embedding_model=None,
        k_nearest_fitter=5,
        distance_function=None,
        mutation_temperature=1.0,
        crossover_temperature=1.0,
        merging_rate=0.02,
        few_shot_learning=False,
        nb_min_examples=1,
        nb_max_examples=3,
        sampling_temperature=1.0,
        population_size=10,
        name=None,
        description=None,
        **kwargs,
    ):
        super().__init__(
            nb_min_examples=nb_min_examples,
            nb_max_examples=nb_max_examples,
            sampling_temperature=sampling_temperature,
            merging_rate=merging_rate,
            population_size=population_size,
            name=name,
            description=description,
        )
        self.language_model = language_model
        self.embedding_model = embedding_model
        self.mutation_temperature = mutation_temperature
        self.crossover_temperature = crossover_temperature
        self.k_nearest_fitter = k_nearest_fitter
        self.few_shot_learning = few_shot_learning

        if not distance_function:
            self.distance_function = similarity_distance
        else:
            self.distance_function = distance_function

        self.kwargs = kwargs

        self.mutation_programs = {}
        self.crossover_programs = {}

    async def build(self, trainable_variables):
        """
        Build the optimizer programs based on the trainable variables.

        Args:
            trainable_variables (list): List of variables that will be optimized
        """
        for trainable_variable in trainable_variables:
            schema_id = id(trainable_variable.get_schema())
            mask = list(Trainable.keys())
            symbolic_variable = trainable_variable.to_symbolic_data_model().out_mask(
                mask=mask
            )

            if schema_id not in self.mutation_programs:
                inputs = Input(data_model=MutationInputs)
                outputs = await ChainOfThought(
                    data_model=symbolic_variable,
                    language_model=self.language_model,
                    temperature=self.mutation_temperature,
                    instructions="\n".join(
                        [
                            base_instructions(),
                            mutation_instructions(list(symbolic_variable.keys())),
                        ]
                    ),
                )(inputs)
                outputs = outputs.in_mask(mask=list(symbolic_variable.keys()))
                program = Program(
                    inputs=inputs,
                    outputs=outputs,
                    name=f"{trainable_variable.name}_mutation",
                    description="The mutation program that fix/optimize variables",
                )
                self.mutation_programs[schema_id] = program

            if schema_id not in self.crossover_programs:
                inputs = Input(data_model=CrossoverInputs)
                outputs = await ChainOfThought(
                    data_model=symbolic_variable,
                    language_model=self.language_model,
                    temperature=self.crossover_temperature,
                    instructions="\n".join(
                        [
                            base_instructions(),
                            crossover_instructions(list(symbolic_variable.keys())),
                        ]
                    ),
                )(inputs)
                outputs = outputs.in_mask(mask=list(symbolic_variable.keys()))
                program = Program(
                    inputs=inputs,
                    outputs=outputs,
                    name=f"{trainable_variable.name}_crossover",
                    description="The crossover program that combine high performing variables",
                )
                self.crossover_programs[schema_id] = program

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
                mask = list(Trainable.keys())
                schema_id = id(trainable_variable.get_schema())
                if strategy == "mutation":
                    masked_variable = out_mask_json(
                        trainable_variable.get_json(),
                        mask=mask,
                    )
                    inputs = MutationInputs(
                        program_description=self.program.description,
                        program_inputs=[inp.get_json() for inp in x],
                        program_predicted_outputs=[
                            pred.get_json() if pred else None for pred in y_pred
                        ],
                        program_ground_truth=(
                            [gt.get_json() for gt in y] if y is not None else []
                        ),
                        variable_description=trainable_variable.description,
                        current_variable=masked_variable,
                    )
                    program = self.mutation_programs[schema_id]
                    new_candidate = await program(inputs, training=training)
                    if self.few_shot_learning:
                        examples = await self.sample_best_predictions(
                            trainable_variable,
                        )
                    else:
                        examples = None
                elif strategy == "crossover":
                    candidate_to_merge = await self.select_candidate_to_merge(
                        step,
                        trainable_variable,
                    )
                    if candidate_to_merge:
                        current_variable = out_mask_json(
                            trainable_variable.get_json(),
                            mask=mask,
                        )
                        other_variable = out_mask_json(
                            candidate_to_merge,
                            mask=mask,
                        )
                        inputs = CrossoverInputs(
                            program_description=self.program.description,
                            program_inputs=[inp.get_json() for inp in x],
                            program_predicted_outputs=[
                                pred.get_json() if pred else None for pred in y_pred
                            ],
                            program_ground_truth=(
                                [gt.get_json() for gt in y] if y is not None else []
                            ),
                            variable_description=trainable_variable.description,
                            other_variable=other_variable,
                            current_variable=current_variable,
                        )
                        program = self.crossover_programs[schema_id]
                        new_candidate = await program(inputs, training=training)
                        if self.few_shot_learning:
                            examples = self.merge_examples(
                                trainable_variable.get("examples"),
                                candidate_to_merge.get("examples"),
                            )
                        else:
                            examples = None
                    else:
                        masked_variable = out_mask_json(
                            trainable_variable.get_json(),
                            mask=mask,
                        )
                        inputs = MutationInputs(
                            program_description=self.program.description,
                            program_inputs=[inp.get_json() for inp in x],
                            program_predicted_outputs=[
                                pred.get_json() if pred else None for pred in y_pred
                            ],
                            program_ground_truth=(
                                [gt.get_json() for gt in y] if y is not None else []
                            ),
                            variable_description=trainable_variable.description,
                            current_variable=masked_variable,
                        )
                        program = self.mutation_programs[schema_id]
                        new_candidate = await program(inputs, training=training)
                        if self.few_shot_learning:
                            examples = await self.sample_best_predictions(
                                trainable_variable,
                            )
                        else:
                            examples = None

                await self.assign_candidate(
                    trainable_variable,
                    new_candidate=new_candidate,
                    examples=examples,
                )

    async def competition(
        self,
        candidates,
    ):
        """
        This function implement Dominated Novelty Search.

        Args:
            candidates (list): The list of candidates to evaluate.

        Returns:
            (list): The selected candidates.
        """
        if len(candidates) <= 1:
            return candidates

        mask = list(Trainable.keys())
        mask.append("reward")

        N = len(candidates)
        fitness_values = [c.get("reward", 0.0) for c in candidates]
        competition_scores = [0.0] * N

        for i in range(N):
            fi = fitness_values[i]

            fitter_indices = [j for j in range(N) if j != i and fitness_values[j] > fi]

            if not fitter_indices:
                competition_scores[i] = 1.0
            else:
                distances = []
                for j in fitter_indices:
                    distance = await self.distance_function(
                        out_mask_json(candidates[i], mask=mask),
                        out_mask_json(candidates[j], mask=mask),
                        embedding_model=self.embedding_model,
                        **self.kwargs,
                    )
                    distances.append((j, distance))

                distances.sort(key=lambda x: x[1])
                k = min(self.k_nearest_fitter, len(distances))
                k_nearest_distances = [d[1] for d in distances[:k]]

                competition_scores[i] = float(
                    np.sum(k_nearest_distances) / k if k > 0 else 0.0
                )
        median_score = np.median(competition_scores)
        selected_candidates = []
        for i, candidate in enumerate(candidates):
            if competition_scores[i] >= median_score:
                selected_candidates.append(candidate)
        return selected_candidates

    async def on_epoch_end(
        self,
        epoch,
        trainable_variables,
    ):
        """Called at the end of an epoch

        Args:
            epoch (int): The epoch number
            trainable_variables (list): The list of trainable variables
        """
        for trainable_variable in trainable_variables:
            candidates = trainable_variable.get("candidates")
            best_candidates = trainable_variable.get("best_candidates")
            all_candidates = candidates + best_candidates
            sorted_candidates = sorted(
                all_candidates,
                key=lambda x: x.get("reward"),
                reverse=True,
            )
            sorted_candidates = await self.competition(sorted_candidates)
            selected_candidates = sorted_candidates[: self.population_size]
            trainable_variable.update(
                {
                    "best_candidates": selected_candidates,
                }
            )
        self.increment_epochs()

    def get_config(self):
        config = {
            "k_nearest_fitter": self.k_nearest_fitter,
            "mutation_temperature": self.mutation_temperature,
            "crossover_temperature": self.crossover_temperature,
            "few_shot_learning": self.few_shot_learning,
            "nb_min_examples": self.nb_min_examples,
            "nb_max_examples": self.nb_max_examples,
            "sampling_temperature": self.sampling_temperature,
            "merging_rate": self.merging_rate,
            "population_size": self.population_size,
            "name": self.name,
            "description": self.description,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model,
            )
        }
        embedding_model_config = {
            "embedding_model": serialization_lib.serialize_synalinks_object(
                self.embedding_model,
            )
        }
        return {**config, **language_model_config, **embedding_model_config}

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model"),
        )
        embedding_model = serialization_lib.deserialize_synalinks_object(
            config.pop("embedding_model"),
        )
        return cls(
            language_model=language_model, embedding_model=embedding_model, **config
        )
