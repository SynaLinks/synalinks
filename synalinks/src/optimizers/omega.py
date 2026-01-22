# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

if TYPE_CHECKING:
    from synalinks.src.backend.common.variables import Variable
    from synalinks.src.embedding_models.embedding_model import EmbeddingModel


from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import Trainable
from synalinks.src.backend import out_mask_json
from synalinks.src.backend.common import numpy as np
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.ttc.chain_of_thought import ChainOfThought
from synalinks.src.optimizers.evolutionary_optimizer import EvolutionaryOptimizer
from synalinks.src.programs.program import Program
from synalinks.src.rewards.reward import squeeze_or_expand_to_same_rank
from synalinks.src.saving import serialization_lib


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


def base_instructions():
    """Base instructions that define the context for all optimization programs.

    These instructions explain that the system optimizes JSON variables
    in a computation graph.
    """
    return """
You are an integral part of an optimization system designed to improve
JSON variables within a computation graph (i.e. the program).
Each module in the graph performs specific computations, with JSON variables
serving as the state.
These variables can represent prompts, code, plans, rules, or any other
JSON-compatible data.
""".strip()


def mutation_instructions(variables_keys):
    """Instructions for the mutation program that optimizes variables.

    Args:
        variables_keys (list): List of keys that the variable should contain
    """
    return f"""
Your primary task is to creatively enhance the provided variable so that the
predicted output aligns as closely as possible with the ground truth.
Pay close attention to the variable's description, its intended use, and the
broader context of the computation graph.

Guidelines:
- Ensure the new variable is generalizable and performs well across various
  inputs of the same kind.
- Include all specified keys: {variables_keys}.
- Justify each change with clear reasoning, referencing the variable's purpose
  and the desired output.
- If no ground truth is provided, the goal is to critically enhance the
  predicted output.
- If you have to optimize a variable containing code, provide a generalizable
  algorithm.
- Always focus on ONLY one aspect at the time.
- If the instructions/prompt contains general information, keep it.
""".strip()


def crossover_instructions(variables_keys):
    """Instructions for the crossover program that optimizes variables.

    Args:
        variables_keys (list): List of keys that the variable should contain
    """
    return f"""
Your responsibility is to create a new, optimized variable by strategically
combining features from the current variable and a high-performing candidate.
The new variable should improve the alignment of the predicted output with
the ground truth.

Guidelines:
- Analyze both the current variable and the other high-performing variable,
  identifying their respective strengths and weaknesses.
- Pay close attention to the variable's description, its intended use, and the
  broader context of the computation graph.
- Ensure the new variable is generalizable and performs well across various
  inputs of the same kind.
- Include all specified keys: {variables_keys}.
- Justify each feature you incorporate, explaining how it contributes to
  better performance or alignment with the ground truth.
- If no ground truth is provided, the goal is to critically enhance the
  predicted output.
- If you have to optimize a variable containing code, provide a generalizable
  algorithm.
- Always focus on ONLY one aspect at the time.
- If the instructions/prompt contains general information, keep it.
""".strip()


async def similarity_distance(
    candidate1: Dict[str, Any],
    candidate2: Dict[str, Any],
    embedding_model: Optional["EmbeddingModel"] = None,
    axis: int = -1,
) -> float:
    """Compute distance between two candidates using embeddings.

    This function computes the cosine distance between the mean embeddings
    of two candidate JSON objects. Each field of the JSON is embedded
    separately, normalized to unit length, then averaged.

    Args:
        candidate1 (dict): First candidate (dict or JSON-serializable object)
        candidate2 (dict): Second candidate (dict or JSON-serializable object)
        embedding_model (EmbeddingModel): The embedding model for computing embeddings
        axis (int): The axis along which to compute the similarity (default: -1)

    Returns:
        float: Cosine distance between candidates (0 = identical, 1 = orthogonal)
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
class OMEGA(EvolutionaryOptimizer):
    """OMEGA: OptiMizEr as Genetic Algorithm.

    A genetic optimizer with dominated novelty search.

    This optimizer is **unique to Synalinks** and the result of our research
    effort on advancing neuro-symbolic AI.

    Dominated Novelty Search (DNS), is a SOTA Quality-Diversity optimization
    method that implements a competition function in a classic genetic
    algorithm.

    The key insight behind Dominated Novelty Search is that candidates should
    be eliminated from the population if they are both:

    - Inferior in reward/fitness
    - Similar to existing candidates/solutions

    This algorithm creates an evolutionary pressure to focus on high performing
    candidates **Or** candidates that explore other approaches.

    This approach only add one step to the traditional genetic algorithm and
    *outperform* MAP-Elites, Threshold-Elites and Cluster-Elites.

    This allow the system to explore the search space more quickly by
    eliminating non-promising candidates while preserving diversity to avoid
    local optimum.

    At Synalinks, we adapted this algorithm for LM-based optimization, to do
    so we use an embedding model to compute the candidate's descriptor and a
    cosine distance between solutions.

    **Note**: In Synalinks, unlike other In-Context learning frameworks, a
    variable (the module's state to optimize) is a JSON object not a simple
    string. Which has multiple implications, we maintain a 100% correct
    structure through constrained JSON decoding, and we allow the state to
    have variable/dynamic number of fields, which is handled by this approach
    by embedding each field and averaging them before computing the distance
    required by DNS.

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
        - Dominated Novelty Search for their elegant Quality-Diversity
          algorithm that outperform many other evolutionary strategies.
        - DSPY's GEPA for feeding the optimizer program with the raw training
          data and for formalizing the evolutionary optimization strategy
          (**NOT** the MAP-Elites method used).
        - DeepMind's AlphaEvolve have been a huge inspiration, more on the
          motivational side as they didn't released the code.

    References:
        - Dominated Novelty Search: Rethinking Local Competition in
          Quality-Diversity (https://arxiv.org/html/2502.00593v1)
        - GEPA: Reflective Prompt Evolution Can Outperform Reinforcement
          Learning (https://arxiv.org/pdf/2507.19457)
        - AlphaEvolve: A coding agent for scientific and algorithmic
          discovery (https://arxiv.org/pdf/2506.13131)

    Args:
        instructions (str): Additional instructions about the task for the
            optimizer.
        language_model (LanguageModel): The language model to use.
        embedding_model (EmbeddingModel): The embedding model to use to
            compute candidates descriptors according to Dominated Novelty
            Search.
        k_nearest_fitter (int): The K nearest fitter used by Dominated
            Novelty Search.
        distance_function (callable): Optional. The distance function to use
            by Dominated Novelty Search. If no function is provided, use
            the default cosine distance.
        mutation_temperature (float): The temperature for the LM calls of
            the mutation programs.
        crossover_temperature (float): The temperature for the LM calls of
            the crossover programs.
        reasoning_effort (string): Optional. The reasoning effort for the LM call
            between ['minimal', 'low', 'medium', 'high', 'disable', 'none', None].
            Default to None (no reasoning).
        algorithm (str): The mechanism to use for the genetic algorithm
            between ['ga', 'dns']. This parameter is provided for ablation
            studies and shouldn't be modified. (Default to 'dns').
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
        instructions=None,
        language_model=None,
        embedding_model=None,
        k_nearest_fitter=5,
        distance_function=None,
        mutation_temperature=0.3,
        crossover_temperature=0.3,
        reasoning_effort=None,
        merging_rate=0.02,
        algorithm="dns",
        selection="softmax",
        selection_temperature=0.3,
        population_size=10,
        name=None,
        description=None,
        **kwargs,
    ):
        super().__init__(
            language_model=language_model,
            mutation_temperature=mutation_temperature,
            crossover_temperature=crossover_temperature,
            selection=selection,
            selection_temperature=selection_temperature,
            merging_rate=merging_rate,
            population_size=population_size,
            name=name,
            description=description,
            **kwargs,
        )
        if not instructions:
            instructions = ""
        self.instructions = instructions
        self.reasoning_effort = reasoning_effort

        # DNS-specific parameters
        self.embedding_model = embedding_model
        self.k_nearest_fitter = k_nearest_fitter
        self.distance_function = distance_function

        algorithms = ["ga", "dns"]
        if algorithm not in algorithms:
            raise ValueError(f"Parameter `algorithm` should be between {algorithms}")
        self.algorithm = algorithm

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
                    reasoning_effort=self.reasoning_effort,
                    instructions=(
                        "\n".join(
                            [
                                base_instructions(),
                                mutation_instructions(list(symbolic_variable.keys())),
                            ]
                        )
                        if not self.instructions
                        else "\n".join(
                            [
                                self.instructions,
                                base_instructions(),
                                mutation_instructions(list(symbolic_variable.keys())),
                            ]
                        )
                    ),
                    name=f"mutation_cot_{schema_id}_" + self.name,
                )(inputs)
                outputs = outputs.in_mask(mask=list(symbolic_variable.keys()))
                program = Program(
                    inputs=inputs,
                    outputs=outputs,
                    name=f"mutation_{schema_id}_" + self.name,
                    description="The mutation program that fix/optimize variables",
                )
                self.mutation_programs[schema_id] = program

            if schema_id not in self.crossover_programs:
                inputs = Input(data_model=CrossoverInputs)
                outputs = await ChainOfThought(
                    data_model=symbolic_variable,
                    language_model=self.language_model,
                    temperature=self.crossover_temperature,
                    reasoning_effort=self.reasoning_effort,
                    instructions=(
                        "\n".join(
                            [
                                base_instructions(),
                                crossover_instructions(list(symbolic_variable.keys())),
                            ]
                        )
                        if not self.instructions
                        else "\n".join(
                            [
                                self.instructions,
                                base_instructions(),
                                crossover_instructions(list(symbolic_variable.keys())),
                            ]
                        )
                    ),
                    name=f"crossover_cot_{schema_id}_" + self.name,
                )(inputs)
                outputs = outputs.in_mask(mask=list(symbolic_variable.keys()))
                program = Program(
                    inputs=inputs,
                    outputs=outputs,
                    name=f"crossover_{schema_id}_" + self.name,
                    description="Crossover program combining high performing variables",
                )
                self.crossover_programs[schema_id] = program

        self.built = True

    async def mutate_candidate(
        self,
        step: int,
        trainable_variable: "Variable",
        selected_candidate: Dict[str, Any],
        x: Optional[List[Any]] = None,
        y: Optional[List[Any]] = None,
        y_pred: Optional[List[Any]] = None,
        training: bool = False,
    ) -> Dict[str, Any]:
        """Apply mutation to generate a new candidate using LLM.

        Creates mutation inputs from the selected candidate and training data,
        then calls the mutation program to generate an optimized variant.

        Args:
            step (int): The current training step
            trainable_variable (Variable): The trainable variable (for metadata access)
            selected_candidate (dict): The selected candidate to mutate
            x (list): Input data batch
            y (list): Ground truth data batch
            y_pred (list): Predicted outputs from the current model
            training (bool): Whether in training mode

        Returns:
            dict: The mutated candidate from the mutation program
        """
        mask = list(Trainable.keys())
        schema_id = id(trainable_variable.get_schema())
        masked_variable = out_mask_json(
            selected_candidate,
            mask=mask,
        )
        inputs = MutationInputs(
            program_description=self.program.description,
            program_inputs=[inp.get_json() for inp in x],
            program_predicted_outputs=[
                pred.get_json() if pred else None for pred in y_pred
            ],
            program_ground_truth=([gt.get_json() for gt in y] if y is not None else []),
            variable_description=trainable_variable.description,
            current_variable=masked_variable,
        )
        program = self.mutation_programs[schema_id]
        return await program(inputs, training=training)

    async def merge_candidate(
        self,
        step: int,
        trainable_variable: "Variable",
        current_candidate: Dict[str, Any],
        other_candidate: Dict[str, Any],
        x: Optional[List[Any]] = None,
        y: Optional[List[Any]] = None,
        y_pred: Optional[List[Any]] = None,
        training: bool = False,
    ) -> Dict[str, Any]:
        """Apply crossover to merge two selected candidates.

        Creates crossover inputs combining two high-performing candidates,
        then calls the crossover program to generate a merged variant.

        Args:
            step (int): The current training step
            trainable_variable (Variable): The trainable variable (for metadata access)
            current_candidate (dict): First selected candidate to merge
            other_candidate (dict): Second selected candidate to merge
            x (list): Input data batch
            y (list): Ground truth data batch
            y_pred (list): Predicted outputs from the current model
            training (bool): Whether in training mode

        Returns:
            dict: The merged candidate from the crossover program
        """
        mask = list(Trainable.keys())
        schema_id = id(trainable_variable.get_schema())
        current_variable = out_mask_json(
            current_candidate,
            mask=mask,
        )
        other_variable = out_mask_json(
            other_candidate,
            mask=mask,
        )
        inputs = CrossoverInputs(
            program_description=self.program.description,
            program_inputs=[inp.get_json() for inp in x],
            program_predicted_outputs=[
                pred.get_json() if pred else None for pred in y_pred
            ],
            program_ground_truth=([gt.get_json() for gt in y] if y is not None else []),
            variable_description=trainable_variable.description,
            other_variable=other_variable,
            current_variable=current_variable,
        )
        program = self.crossover_programs[schema_id]
        return await program(inputs, training=training)

    async def competition(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Dominated Novelty Search (DNS) competition.

        DNS filters candidates by removing those that are both:
        - Inferior in reward (dominated)
        - Similar to existing candidates (not novel)

        This maintains diversity while focusing on high-performing candidates.

        Args:
            candidates (list): List of candidate dictionaries with 'reward' key

        Returns:
            list: Filtered list of candidates that passed the DNS competition
        """
        if len(candidates) <= 1:
            return candidates

        distance_function = (
            self.distance_function if self.distance_function else similarity_distance
        )

        selected_candidates = []
        for candidate in candidates:
            is_dominated = False
            for other in candidates:
                if other is candidate:
                    continue
                distance = await distance_function(
                    candidate,
                    other,
                    embedding_model=self.embedding_model,
                )
                # Check if within k-nearest neighborhood
                if distance < 1.0 / self.k_nearest_fitter:
                    # Check if dominated (lower reward)
                    if candidate.get("reward", 0) < other.get("reward", 0):
                        is_dominated = True
                        break
            if not is_dominated:
                selected_candidates.append(candidate)

        return selected_candidates if selected_candidates else [candidates[0]]

    async def on_epoch_end(self, epoch, trainable_variables):
        """Called at the end of each epoch.

        Applies DNS competition (if algorithm='dns') to filter candidates,
        then selects the top candidates based on population_size.

        Args:
            epoch (int): The epoch number
            trainable_variables (list): The list of trainable variables
        """
        for trainable_variable in trainable_variables:
            candidates = trainable_variable.get("candidates")
            best_candidates = trainable_variable.get("best_candidates")

            # Combine current candidates with best candidates
            all_candidates = candidates + best_candidates

            # Apply DNS competition if enabled
            if self.algorithm == "dns" and len(all_candidates) > 1:
                all_candidates = await self.competition(all_candidates)

            # Sort by reward and keep top population_size candidates
            all_candidates = sorted(
                all_candidates,
                key=lambda x: x.get("reward", 0),
                reverse=True,
            )
            trainable_variable.update(
                {
                    "candidates": [],
                    "best_candidates": all_candidates[: self.population_size],
                }
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "instructions": self.instructions,
                "reasoning_effort": self.reasoning_effort,
                "k_nearest_fitter": self.k_nearest_fitter,
                "algorithm": self.algorithm,
            }
        )
        if self.embedding_model:
            config["embedding_model"] = serialization_lib.serialize_synalinks_object(
                self.embedding_model
            )
        return config

    @classmethod
    def from_config(cls, config):
        embedding_model = None
        if "embedding_model" in config:
            embedding_model = serialization_lib.deserialize_synalinks_object(
                config.pop("embedding_model")
            )
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(
            language_model=language_model,
            embedding_model=embedding_model,
            **config,
        )
