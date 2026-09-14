# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import hashlib
import json
import warnings
from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

if TYPE_CHECKING:
    from synalinks.src.backend.common.variables import Variable
    from synalinks.src.modules.embedding_models.embedding_model import EmbeddingModel

import numpy as _np

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import EmbeddingRequest
from synalinks.src.backend import Field
from synalinks.src.backend import Trainable
from synalinks.src.backend import out_mask_json
from synalinks.src.backend.common import numpy as np
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.embedding_models import get as _get_em
from synalinks.src.modules.ttc.chain_of_thought import ChainOfThought
from synalinks.src.optimizers.evolutionary_optimizer import EvolutionaryOptimizer
from synalinks.src.optimizers.optimizer import CANDIDATE_METADATA_KEYS
from synalinks.src.saving import serialization_lib


class ScoredPrediction(DataModel):
    inputs: Any = Field(
        description="The program input",
    )
    predicted_output: Optional[Any] = Field(
        description="What the program predicted for this input with the current variable",
        default=None,
    )
    ground_truth: Optional[Any] = Field(
        description="The expected output, when known",
        default=None,
    )
    reward: Optional[float] = Field(
        description="The reward of the prediction (higher is better), when known",
        default=None,
    )
    nb_observations: Optional[int] = Field(
        description=(
            "For recurring hard examples: how many times this input was judged so "
            "far; `reward` is then the mean over those judgements"
        ),
        default=None,
    )


class MutationInputs(DataModel):
    program_description: str = Field(
        description="The program description",
    )
    good_predictions: List[ScoredPrediction] = Field(
        description=(
            "The predictions of the batch with the highest rewards: what the current "
            "variable already handles well and must keep working"
        ),
    )
    bad_predictions: List[ScoredPrediction] = Field(
        description=(
            "The predictions of the batch with the lowest rewards, followed by the "
            "inputs the program keeps getting wrong across batches (recurring hard "
            "examples): what the current variable handles worst and should be fixed"
        ),
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
    good_predictions: List[ScoredPrediction] = Field(
        description=(
            "The predictions of the batch with the highest rewards under the current "
            "variable: what must keep working"
        ),
    )
    bad_predictions: List[ScoredPrediction] = Field(
        description=(
            "The predictions of the batch with the lowest rewards under the current "
            "variable: what the merge should fix"
        ),
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
- `bad_predictions` are the inputs the current variable handles worst: work
  out what the variable gets wrong on them and fix that. `good_predictions`
  are handled well: preserve whatever makes them work, do not regress them.
- If you have to optimize a variable containing code, provide a generalizable
  algorithm.
- Always focus on ONLY one aspect at the time.
- If the instructions/prompt contains general information, keep it.
- Keep the variable about the same length as the current one: do not add text
  without removing or merging something. Prefer rewording an existing rule
  over adding a new one, and drop rules that proved redundant or unhelpful.
  Longer is not better; a compact variable generalizes better and costs less.
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
- `bad_predictions` are the inputs the current variable handles worst: take
  from the other variable what would fix them. `good_predictions` are handled
  well: keep whatever makes them work.
- If you have to optimize a variable containing code, provide a generalizable
  algorithm.
- Always focus on ONLY one aspect at the time.
- If the instructions/prompt contains general information, keep it.
- The combined variable must not be longer than the longer of the two inputs:
  merge overlapping rules into one instead of concatenating them, and keep
  only the strongest formulation of each idea. Longer is not better.
""".strip()


async def similarity_distance(
    candidate1: Dict[str, Any],
    candidate2: Dict[str, Any],
    embedding_model: Optional["EmbeddingModel"] = None,
    axis: int = -1,
) -> float:
    """Cosine distance between two candidates, from their content only.

    Each trainable field of a candidate is embedded separately; the unit
    vectors are averaged and the mean is renormalized, so two candidates with
    the same content are at distance 0 whatever their number of fields.
    Candidate metadata (`reward`, `reward_count`) is not part of the content
    and is ignored.

    Args:
        candidate1 (dict): First candidate (dict or JSON-serializable object)
        candidate2 (dict): Second candidate (dict or JSON-serializable object)
        embedding_model (EmbeddingModel): The embedding model for computing embeddings
        axis (int): The axis along which to compute the similarity (default: -1)

    Returns:
        float: Cosine distance between candidates (0 = identical, 1 = orthogonal).
            The maximal distance 1.0 is returned when an embedding call fails.
    """
    vector1 = await _candidate_vector(candidate1, embedding_model, axis=axis)
    vector2 = await _candidate_vector(candidate2, embedding_model, axis=axis)
    if vector1 is None or vector2 is None:
        return 1.0
    similarity = float(_np.sum(vector1 * vector2))
    return max(0.0, 1.0 - similarity)


def candidate_content_texts(candidate: Dict[str, Any]) -> List[str]:
    """The string leaves of a candidate's trainable content, metadata excluded."""
    content = out_mask_json(candidate, mask=CANDIDATE_METADATA_KEYS)
    return [str(leaf) for leaf in tree.flatten(content)]


async def _candidate_vector(candidate, embedding_model, axis=-1):
    texts = candidate_content_texts(candidate)
    if not texts:
        texts = [""]
    result = await embedding_model(EmbeddingRequest(texts=texts))
    if result is None:
        return None
    embeddings = result["embeddings"]
    if embeddings is None or len(embeddings) == 0:
        return None
    vectors = _np.asarray(embeddings, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors[None, :]
    vectors = np.normalize(vectors, axis=axis)
    mean = _np.mean(vectors, axis=0)
    norm = _np.linalg.norm(mean)
    if norm == 0:
        return mean
    return mean / norm


def _without_echoed_inputs(predicted_output, inputs):
    """Drop from a prediction the fields that merely repeat the input.

    Modules that return their inputs concatenated with their outputs (e.g.
    `SelfCritique`, `Generator(return_inputs=True)`) would otherwise show every
    sample twice to the mutation/crossover programs. Only fields whose value is
    identical to the input's are removed, so genuine outputs are kept even when
    they share a key name with an input field.
    """
    if not isinstance(predicted_output, dict) or not isinstance(inputs, dict):
        return predicted_output
    return {
        key: value
        for key, value in predicted_output.items()
        if not (key in inputs and inputs[key] == value)
    }


@synalinks_export(
    [
        "synalinks.OMEGA",
        "synalinks.optimizers.OMEGA",
    ]
)
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
        nb_best_predictions (int): How many of the batch's highest-reward
            predictions are shown to the mutation/crossover programs as
            `good_predictions` (what must keep working). Default 1.
        nb_worst_predictions (int): How many of the batch's lowest-reward
            predictions are shown as `bad_predictions` (what to fix). Default 3.
            Predictions without a reward count as worst.
        nb_hard_examples (int): How many recurring hard examples are appended
            to `bad_predictions`: inputs judged at least
            `hard_example_min_observations` times during training whose mean
            reward is the lowest, excluding inputs already in the batch. They
            carry their last judged output and `nb_observations`. Default 1;
            0 disables the memory.
        hard_example_min_observations (int): Minimum number of judgements
            before an input can be a recurring hard example. Default 2.
        distance_function (callable): Optional. The distance function to use
            by Dominated Novelty Search. If no function is provided, use
            the default cosine distance.
        mutation_temperature (float): The temperature for the LM calls of
            the mutation programs.
        crossover_temperature (float): The temperature for the LM calls of
            the crossover programs.
        reasoning_effort (string): Optional. The reasoning effort for the LM call
            between ['minimal', 'low', 'medium', 'high', 'xhigh', 'disable',
            'none', None].
            Default to None (no reasoning).
        use_chain_of_thought (bool): Whether the mutation/crossover programs use
            a `ChainOfThought` (which first writes a prompt-driven `thinking`
            field, then the variable) or a plain `Generator` that emits the
            variable directly. Disabling it gives a tighter, more bounded
            generation (fewer tokens, no runaway reasoning) at the cost of the
            explicit reasoning step -- useful for smaller/local models that ramble
            under thinking. (Default to True).
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
        merging_rate (float): Probability that a proposal is a crossover rather
            than a mutation, constant over training.
            Default 0.05: about one proposal in twenty is a crossover of two
            candidates once the population holds at least two; mutation otherwise.
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
        use_chain_of_thought=True,
        merging_rate=0.05,
        algorithm="dns",
        selection="softmax",
        selection_temperature=0.3,
        population_size=10,
        reward_uncertainty=0.25,
        nb_best_predictions=1,
        nb_worst_predictions=3,
        nb_hard_examples=1,
        hard_example_min_observations=2,
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
            reward_uncertainty=reward_uncertainty,
            name=name,
            description=description,
            **kwargs,
        )
        if not instructions:
            instructions = ""
        self.instructions = instructions
        self.reasoning_effort = reasoning_effort
        self.use_chain_of_thought = use_chain_of_thought
        if int(nb_best_predictions) < 0 or int(nb_worst_predictions) < 0:
            raise ValueError(
                "`nb_best_predictions` and `nb_worst_predictions` must be >= 0"
            )
        self.nb_best_predictions = int(nb_best_predictions)
        self.nb_worst_predictions = int(nb_worst_predictions)
        if int(nb_hard_examples) < 0 or int(hard_example_min_observations) < 1:
            raise ValueError(
                "`nb_hard_examples` must be >= 0 and "
                "`hard_example_min_observations` must be >= 1"
            )
        self.nb_hard_examples = int(nb_hard_examples)
        self.hard_example_min_observations = int(hard_example_min_observations)
        # Per-input difficulty memory, fed by `observe_training_batch`:
        # key -> {"inputs", "ground_truth", "last_output", "rewards"}.
        self._difficulty = {}

        # DNS-specific parameters
        self.embedding_model = _get_em(embedding_model)
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
        # Lazy import: `Program` -> `Trainer` -> `synalinks.rewards` triggers
        # a cycle if `omega` loads before `programs.program` finishes.
        from synalinks.src.programs.program import Program

        # `ChainOfThought` prepends a prompt-driven `thinking` field; a plain
        # `Generator` emits the variable directly. The `in_mask` below keeps only
        # the variable fields either way, so they are drop-in interchangeable.
        module_cls = ChainOfThought if self.use_chain_of_thought else Generator

        for trainable_variable in trainable_variables:
            schema_id = id(trainable_variable.get_schema())
            mask = list(Trainable.keys())
            symbolic_variable = trainable_variable.to_symbolic_data_model().out_mask(
                mask=mask
            )

            if schema_id not in self.mutation_programs:
                inputs = Input(data_model=MutationInputs)
                outputs = await module_cls(
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
                    name=f"mutation_module_{schema_id}_" + self.name,
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
                outputs = await module_cls(
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
                    name=f"crossover_module_{schema_id}_" + self.name,
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

    @staticmethod
    def _input_key(inputs):
        try:
            payload = json.dumps(inputs, sort_keys=True, default=str)
        except TypeError:
            payload = repr(inputs)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def observe_training_batch(self, x=None, y=None, y_pred=None, rewards=None):
        """Record each judged training sample in the difficulty memory."""
        if self.nb_hard_examples <= 0 or x is None or rewards is None:
            return
        x = list(x)
        y = list(y) if y is not None else [None] * len(x)
        y_pred = list(y_pred) if y_pred is not None else [None] * len(x)
        rewards = list(rewards)
        for i, inp in enumerate(x):
            if i >= len(rewards) or rewards[i] is None:
                continue
            inputs = inp.get_json() if hasattr(inp, "get_json") else inp
            predicted = y_pred[i] if i < len(y_pred) else None
            predicted = (
                predicted.get_json() if hasattr(predicted, "get_json") else predicted
            )
            target = y[i] if i < len(y) else None
            target = target.get_json() if hasattr(target, "get_json") else target
            entry = self._difficulty.setdefault(
                self._input_key(inputs),
                {
                    "inputs": inputs,
                    "ground_truth": target,
                    "last_output": None,
                    "rewards": [],
                },
            )
            entry["last_output"] = _without_echoed_inputs(predicted, inputs)
            entry["ground_truth"] = target
            entry["rewards"].append(float(rewards[i]))

    def hard_examples(self, exclude_keys=()):
        """The recurring hard examples: lowest mean reward first.

        Args:
            exclude_keys (iterable): Input keys to skip (the current batch).

        Returns:
            (list): Up to `nb_hard_examples` `ScoredPrediction` dicts with
                `nb_observations` set.
        """
        if self.nb_hard_examples <= 0:
            return []
        exclude = set(exclude_keys)
        eligible = [
            (sum(e["rewards"]) / len(e["rewards"]), -len(e["rewards"]), key, e)
            for key, e in self._difficulty.items()
            if key not in exclude
            and len(e["rewards"]) >= self.hard_example_min_observations
        ]
        eligible.sort(key=lambda t: (t[0], t[1]))
        return [
            {
                "inputs": e["inputs"],
                "predicted_output": e["last_output"],
                "ground_truth": e["ground_truth"],
                "reward": mean,
                "nb_observations": len(e["rewards"]),
            }
            for mean, _neg_count, _key, e in eligible[: self.nb_hard_examples]
        ]

    def split_predictions(self, x=None, y=None, y_pred=None, rewards=None):
        """Split a batch into the best and worst predictions by reward.

        Args:
            x (list): The batch inputs.
            y (list): The batch ground truth (optional).
            y_pred (list): The predictions for the batch (optional).
            rewards (list): The per-sample rewards (optional). A missing reward
                ranks the sample as worst.

        Returns:
            (tuple): `(good, bad)`, two lists of `ScoredPrediction` dicts. `good`
                holds the `nb_best_predictions` highest rewards, `bad` the
                `nb_worst_predictions` lowest among the remaining samples, so a
                sample is never in both, followed by up to `nb_hard_examples`
                recurring hard examples from earlier batches. Worst-first in
                `bad`, best-first in `good`.
        """
        x = list(x) if x is not None else []
        y = list(y) if y is not None else [None] * len(x)
        y_pred = list(y_pred) if y_pred is not None else [None] * len(x)
        rewards = list(rewards) if rewards is not None else [None] * len(x)

        def as_json(value):
            return value.get_json() if hasattr(value, "get_json") else value

        scored = []
        for i, inp in enumerate(x):
            reward = rewards[i] if i < len(rewards) else None
            inputs = as_json(inp)
            predicted = as_json(y_pred[i]) if i < len(y_pred) else None
            scored.append(
                {
                    "inputs": inputs,
                    "predicted_output": _without_echoed_inputs(predicted, inputs),
                    "ground_truth": as_json(y[i]) if i < len(y) else None,
                    "reward": None if reward is None else float(reward),
                    "nb_observations": None,
                }
            )
        # Unknown rewards rank as worst; ties keep batch order.
        ordered = sorted(
            scored,
            key=lambda p: (p["reward"] is None, -(p["reward"] or 0.0)),
        )
        good = [p for p in ordered[: self.nb_best_predictions] if p["reward"] is not None]
        rest = ordered[len(good) :]
        bad = list(reversed(rest))[: self.nb_worst_predictions]
        batch_keys = [self._input_key(p["inputs"]) for p in scored]
        bad = bad + self.hard_examples(exclude_keys=batch_keys)
        return good, bad

    async def mutate_candidate(
        self,
        step: int,
        trainable_variable: "Variable",
        selected_candidate: Dict[str, Any],
        x: Optional[List[Any]] = None,
        y: Optional[List[Any]] = None,
        y_pred: Optional[List[Any]] = None,
        rewards: Optional[List[float]] = None,
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
            rewards (list): Per-sample rewards of `y_pred`, used to split the
                batch into `good_predictions` / `bad_predictions`
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
        good, bad = self.split_predictions(x=x, y=y, y_pred=y_pred, rewards=rewards)
        inputs = MutationInputs(
            program_description=self.program.description,
            good_predictions=good,
            bad_predictions=bad,
            variable_description=trainable_variable.description,
            current_variable=masked_variable,
        )
        program = self.mutation_programs[schema_id]
        # A failing LM call (timeout, rate limit, transient API error) must not
        # kill fit(): warn and return None so the current best candidate is kept.
        # assign_candidate / maybe_add_candidate treat a None candidate as a no-op.
        try:
            return await program(inputs, training=training)
        except Exception as e:
            warnings.warn(
                f"OMEGA mutation at step {step} failed and was skipped "
                f"(keeping the current best candidate): {type(e).__name__}: {e}",
                stacklevel=2,
            )
            return None

    async def merge_candidate(
        self,
        step: int,
        trainable_variable: "Variable",
        current_candidate: Dict[str, Any],
        other_candidate: Dict[str, Any],
        x: Optional[List[Any]] = None,
        y: Optional[List[Any]] = None,
        y_pred: Optional[List[Any]] = None,
        rewards: Optional[List[float]] = None,
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
            rewards (list): Per-sample rewards of `y_pred`, used to split the
                batch into `good_predictions` / `bad_predictions`
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
        good, bad = self.split_predictions(x=x, y=y, y_pred=y_pred, rewards=rewards)
        inputs = CrossoverInputs(
            program_description=self.program.description,
            good_predictions=good,
            bad_predictions=bad,
            variable_description=trainable_variable.description,
            other_variable=other_variable,
            current_variable=current_variable,
        )
        program = self.crossover_programs[schema_id]
        # A failing LM call (timeout, rate limit, transient API error) must not
        # kill fit(): warn and return None so the current best candidate is kept.
        # assign_candidate / maybe_add_candidate treat a None candidate as a no-op.
        try:
            return await program(inputs, training=training)
        except Exception as e:
            warnings.warn(
                f"OMEGA crossover at step {step} failed and was skipped "
                f"(keeping the current best candidate): {type(e).__name__}: {e}",
                stacklevel=2,
            )
            return None

    async def competition_fitness(self, candidates: List[Dict[str, Any]]) -> List[float]:
        """Dominated Novelty Search competition fitness of each candidate.

        Following Bahlous-Boldi et al. (2025), a candidate's competition
        fitness is the mean distance to its `k_nearest_fitter` nearest
        candidates with a strictly higher reward, or infinity when no
        candidate is fitter. A candidate is penalized when it sits close to
        better ones, whatever its own reward; the best candidate and the
        candidates alone in their region score highest.

        Args:
            candidates (list): List of candidate dictionaries with 'reward' key

        Returns:
            list: One competition fitness per candidate, in input order.
        """
        distance_function = (
            self.distance_function if self.distance_function else similarity_distance
        )
        rewards = [self.candidate_score(c) for c in candidates]
        distances = {}

        async def distance(i, j):
            key = (min(i, j), max(i, j))
            if key not in distances:
                distances[key] = await distance_function(
                    candidates[key[0]],
                    candidates[key[1]],
                    embedding_model=self.embedding_model,
                )
            return distances[key]

        k = max(1, int(self.k_nearest_fitter))
        fitness = []
        for i in range(len(candidates)):
            fitter = [j for j, r in enumerate(rewards) if r > rewards[i]]
            if not fitter:
                fitness.append(float("inf"))
                continue
            nearest = sorted([await distance(i, j) for j in fitter])[:k]
            fitness.append(sum(nearest) / len(nearest))
        return fitness

    async def competition(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank candidates by DNS competition fitness.

        There is no distance threshold: the population is truncated by rank
        on the competition fitness (see `on_epoch_end`), so the outcome does
        not depend on the scale of the distance function.

        Args:
            candidates (list): List of candidate dictionaries with 'reward' key

        Returns:
            list: The same candidates ranked by decreasing competition fitness,
                ties broken by decreasing reward. No candidate is removed.
        """
        if len(candidates) <= 1:
            return list(candidates)
        fitness = await self.competition_fitness(candidates)
        rewards = [self.candidate_score(c) for c in candidates]
        order = sorted(
            range(len(candidates)),
            key=lambda i: (fitness[i], rewards[i]),
            reverse=True,
        )
        return [candidates[i] for i in order]

    async def on_epoch_end(self, epoch, trainable_variables, logs=None, val_size=None):
        """Called at the end of each epoch.

        With `algorithm="dns"`, the candidates of the epoch and the current
        best candidates are ranked by DNS competition fitness and the top
        `population_size` survive. With `algorithm="ga"`, the top
        `population_size` by reward survive. Before that, the epoch-end
        validation reward is folded into the promoted candidate. The base class
        then writes the best survivor into the variable and records the history.

        Args:
            epoch (int): The epoch number
            trainable_variables (list): The list of trainable variables
        """
        self.assign_validation_reward(trainable_variables, logs=logs, val_size=val_size)
        for trainable_variable in trainable_variables:
            candidates = trainable_variable.get("candidates")
            best_candidates = trainable_variable.get("best_candidates")
            all_candidates = candidates + best_candidates
            if not all_candidates:
                continue
            if self.algorithm == "dns":
                ranked = await self.competition(all_candidates)
                survivors = ranked[: self.population_size]
            else:
                survivors = sorted(
                    all_candidates,
                    key=self.candidate_score,
                    reverse=True,
                )[: self.population_size]
            trainable_variable.update(
                {
                    "candidates": [],
                    "best_candidates": sorted(
                        survivors,
                        key=self.candidate_score,
                        reverse=True,
                    ),
                }
            )
        await super().on_epoch_end(epoch, trainable_variables)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "instructions": self.instructions,
                "reasoning_effort": self.reasoning_effort,
                "use_chain_of_thought": self.use_chain_of_thought,
                "k_nearest_fitter": self.k_nearest_fitter,
                "algorithm": self.algorithm,
                "nb_best_predictions": self.nb_best_predictions,
                "nb_worst_predictions": self.nb_worst_predictions,
                "nb_hard_examples": self.nb_hard_examples,
                "hard_example_min_observations": self.hard_example_min_observations,
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
