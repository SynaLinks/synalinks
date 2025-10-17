# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from typing import List

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import Trainable
from synalinks.src.modules.module import Module
from synalinks.src.modules.ttc.chain_of_thought import ChainOfThought
from synalinks.src.saving import serialization_lib


class Step(DataModel):
    """The individual step to execute"""
    step: str = Field(
        description="The step to execute",
    )


class SequentialPlan(Trainable):
    """The sequential step by step plan to achieve the task"""
    steps: List[str] = Field(
        description="The list of steps",
    )


class SequentialPlanSynthesis(Module):
    """A module that executes a sequential plan of steps.

    This module features a sequential plan as a trainable variable, allowing optimizers
    to refine the plan during the training loop based on iterative feedback.
    
    Basically learning to plan based on iterative feedback and automatic selection of the best plan.

    The module executes each step in the plan sequentially, passing the output of each
    step as input to the next step. The runner is responsible for executing
    each individual step. The most common runners are usually a `FunctionCallingAgent`, 
    `ChainOfThought` or `Generator` module, but you can use any Module or Program.
    
    This module start by defaut without any plan, so it is equivalent to a single runner call.
    
    This module works **ONLY** with advanced optimizers (**NOT** the `RandomFewShot` optimizer).
    
    **Note**: The inputs are forwarded to the runner each time by concatenating the inputs with
    the previous steps outputs. So **ensure that the runner doesn't returns the inputs**, use 
    `return_inputs=False` or `return_inputs_with_trajectory=False` when configuring your runner.

    Example:
    
    ```python
    import synalinks
    import asyncio
    
    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )
        
    class FinalReport(synalinks.DataModel):
        report: str = synalinks.Field(
            description="The final report",
        )
    
    class TaskSummary(synalinks.DataModel):
        summary: str = synalinks.Field(
            description="The summary of the executed task",
        )
    
    async def main():
        tools = # ... tools definition (see `FunctionCallingAgent`)
        
        inputs = synalinks.Input(data_model=Query)
        outputs = await synalinks.SequentialPlanSynthesis(
            data_model=FinalReport,
            language_model=language_model,
            runner=synalinks.FunctionCallingAgent(
                data_model=TaskSummary,
                language_model=language_model,
                tools=tools,
                return_inputs_with_trajectory=False,
            ),
        )(inputs)
        
        program = synalinks.Program(
            inputs=inputs,
            outputs=outputs,
            name="planner_agent",
            description="An agent that learn a step by step plan to achieve a task",
        )
        
    ```
    
    Args:
        schema (dict): The target JSON schema.
            If not provided use the `data_model` to infer it.
        data_model (DataModel | SymbolicDataModel | JsonDataModel): The target data
            model for structured output.
        language_model (LanguageModel): The language model to use.
        steps (list): Optional. The default list of steps being a list of strings.
        seed_steps (list): Optional. A list of steps to use as seed for the
            optimization. If not provided, use the default steps as seed.
        runner (Module | Program): Required. The runner that executes each step.
        return_inputs (bool): Optional. Whether or not to concatenate the inputs to
            the outputs (Default to False).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.    
    """
    
    def __init__(
        self,
        schema=None,
        data_model=None,
        language_model=None,
        steps=None,
        seed_steps=None,
        runner=None,
        return_inputs=True,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema
        
        if not steps:
            steps = []
        self.steps = steps
            
        if not seed_steps:
            seed_steps = [[]]
        
        self.seed_steps = seed_steps
        if not runner:
            raise ValueError("The `runner` parameter is required.")
        if not isinstance(runner, Module):
            raise ValueError("The `runner` parameter should be a `Module` or `Program`.")
        
        self.language_model = language_model
        self.runner = runner
        self.return_inputs = return_inputs
        
        self.state = self.add_variable(
            initializer=SequentialPlan(
                steps=self.steps,
                seed_candidates=self.seed_steps,
            ).get_json(),
            data_model=SequentialPlan,
            name="state"+self.name,
        )
        
        self.final_generator = ChainOfThought(
            schema=self.schema,
            language_model=self.language_model,
            return_inputs=self.return_inputs,
            name="final_generator_"+self.name,
        )
        
    async def call(self, inputs, training=False):
        if not inputs:
            return None
        steps = self.state.get("steps")
        previous_steps = None
        if steps:
            for i, step in enumerate(steps):
                step_result = await self.runner(inputs, training=training)
                if not previous_steps:
                    previous_steps = step_result
                else:
                    previous_steps = await ops.concat(
                        previous_steps,
                        step_result,
                        name=+f"step_{i}_with_inputs"+self.name,
                    )
                inputs = await ops.concat(
                    inputs,
                    await ops.concat(
                        previous_steps,
                        Step(step=step),
                        name=f"step_{i}_"+self.name,
                    ),
                    name=f"step_{i}_with_inputs_"+self.name,
                )
        else:
            result = await self.runner(inputs, training=training)
            inputs = await ops.concat(
                inputs,
                result,
                name="with_inputs_"+self.name,
            )
        return await self.final_generator(inputs, training=training)
        
    async def compute_output_spec(self, inputs, training=False):
        _ = await self.runner(inputs)
        return await self.final_generator(inputs)
        
    def get_config(self):
        config = {
            "schema": self.schema,
            "steps": self.steps,
            "seed_steps": self.seed_steps,
            "return_inputs": self.return_inputs,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model,
            )
        }
        runner_config = {
            "runner": serialization_lib.serialize_synalinks_object(
                self.runner,
            )
        }
        return {
            **config,
            **language_model_config,
            **runner_config,
        }

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model"),
        )
        runner = serialization_lib.deserialize_synalinks_object(
            config.pop("runner"),
        )
        return cls(
            language_model=language_model,
            runner=runner,
            **config,
        )