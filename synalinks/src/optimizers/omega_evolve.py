from typing import Any
from typing import List
from typing import Optional

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import Trainable
from synalinks.src.backend import out_mask_json
from synalinks.src.modules.core.input_module import Input
from synalinks.src.modules.ttc.chain_of_thought import ChainOfThought
from synalinks.src.optimizers.random_few_shot import RandomFewShot
from synalinks.src.programs.program import Program
from synalinks.src.saving import serialization_lib


def base_instructions():
    """
    Base instructions that define the context for all optimization programs.
    These instructions explain that the system optimizes JSON variables in a computation graph.
    """
    return """
You are part of an optimization system that improve JSON variables.
The system to optimize is a computation graph (i.e the program) composed of modules performing computation (with JSON variables as state).
The module's variables are JSON objects that can represent prompts, code, plans, rules or any other JSON-based variable.
""".strip()


def mutation_instructions(variables_keys):
    """
    Instructions for the mutation program that optimizes variables.

    Args:
        variables_keys (list): List of keys that the variable should contain
    """
    return f"""
Your only responsability is to identify how to enhance the variable so that the predicted output match the ground truth.
Pay attention to the description of the variable, and the context in which it is used.
You have to come up with a new variable composed of the following keys: {variables_keys}.
The variables may contains general instructions that you must retain.
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
Your only responsability is to identify how to enhance the variable so that the predicted output match the ground truth.
The current variable is provided along with another high performing variable candidate to take inspiration from. 
Pay attention to the description of the variable, and the context in which it is used.
You have to come up with a new variable composed of the following keys: {variables_keys}.
The variables may contains general instructions that you must retain.
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
    


@synalinks_export("synalinks.optimizers.OMEGA")
class OMEGAEvolve(RandomFewShot):
    """An evolutionary optimizer that can enhance/optimize **ANY** trainable
        variable of the computation graph (Synalink's DAG).
    
    This Optimizer is still experimental and subject to changes, but these 
    changes should not impact you if you use it, so don't hesitate.
    
    This SOTA optimizer have unique features:
        
    - First the variable to update is selected based on its performance/number of
        visits during the training batch and the temperature parameter (softmax weighted).
        In order to avoid optimizing blindly modules that wasn't called during
        a batch and focus on worst performing ones.
    - It has a merging rate that grow over time to first explore mutations before
        merging high performing variables, balancing exploration/exploitation.
    - If a program has modules that wasn't called during a batch, which might be the case 
        if you used the `Branch` module, then it won't be updated because the optimizer focus
        only on visited Modules it has information about.
    - It uses the **entire** training batch as feedback, if your `batch_size` > 1
        then it will have access to all the batch samples, which makes possible to repeat
        the training data (disable LM caching in that case!) to learn in a grouped fashion like GRPO.
    - The examples evolve also, which makes possible to combine the variables evolutions + 
        examples. Which is powerfull when you deal with tasks that benefit from examples 
        (classification, QA, Math etc).
    - Combine **constrained JSON decoding** with variable optimization, **a variable is a JSON object** 
        not just a string in Synalinks! So you can use Synalinks to optimize
        protein chains, logic rules, code etc. while ensuring a robust structure!
    - The optimizer has access to the training batch data, for mutation *And* crossover.
            
    If you want to explore the future of neuro-symbolic self-evolving systems, contact us.
    While these systems are not "hard" to code thanks to Synalinks, they requires 
    technical knowledge and a deep understanding of multiple AI paradigm.
    
    For now, only 2 modules features self-evolving trainable variables:
    
    - The `Generator` module that has self-evolving instructions.
    - The `PythonSynthesis` module that has self-evolving python scripts.
        
    More will be added in the future.
        
    References:
        - [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)

    Args:
        language_model (LanguageModel): The language model to use.
        few_shot_learning (bool): If `True` enable the selection of examples using
            the same method than the `RandomFewShot` optimizer.
        nb_min_examples (int): The min number of examples for few-shot learning (Default to 1).
        nb_max_examples (int): The max number of examples for few-shot learning (Default to 3).
        temperature (float): The temperature for softmax sampling of the few-shot
            learning examples. Lower values concentrate sampling on high-reward predictions,
            higher values make sampling more uniform (Default 1.0).
        merging_rate (float): Rate at which crossover vs mutation is selected. (Default to 0.02)
        name (str): Optional name for the optimizer instance.
        description (str): Optional description of the optimizer instance.
    """

    def __init__(
        self,
        language_model=None,
        few_shot_learning=False,
        nb_min_examples=1,
        nb_max_examples=3,
        temperature=1.0,
        merging_rate=0.02,
        name=None,
        description=None,
    ):
        super().__init__(
            nb_min_examples=nb_min_examples,
            nb_max_examples=nb_max_examples,
            temperature=temperature,
            merging_rate=merging_rate,
            name=name,
            description=description,
        )
        self.language_model = language_model
        self.few_shot_learning = few_shot_learning

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
    ):
        variable_name_to_update = self.select_variable_name_to_update(
            trainable_variables,
        )

        strategy = self.select_evolving_strategy()

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
                    new_candidate = await program(inputs)
                    if self.few_shot_learning:
                        examples = self.sample_best_predictions(
                            trainable_variable,
                        )
                    else:
                        examples = None
                elif strategy == "crossover":
                    candidate_to_merge = self.select_candidate_to_merge(
                        step,
                        trainable_variable,
                    )
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
                    new_candidate = await program(inputs)
                    if self.few_shot_learning:
                        examples = self.merge_examples(
                            trainable_variable.get("examples"),
                            candidate_to_merge.get("examples"),
                        )
                    else:
                        examples = None

                self.assign_candidate(
                    trainable_variable,
                    new_candidate=new_candidate,
                    examples=examples,
                )

    def get_config(self):
        config = {
            "few_shot_learning": self.few_shot_learning,
            "nb_min_examples": self.nb_min_examples,
            "nb_max_examples": self.nb_max_examples,
            "temperature": self.temperature,
            "merging_rate": self.merging_rate,
            "name": self.name,
            "description": self.description,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model,
            )
        }
        full_config = {**config, **language_model_config}
        return full_config

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model"),
        )
        return cls(language_model=language_model, **config)
