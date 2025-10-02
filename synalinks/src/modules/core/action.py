# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import GenericIO
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib
from synalinks.src.utils import tool_utils


class GenericAction(DataModel):
    """A generic action with inputs/outputs"""

    action: GenericIO = Field(description="An action already performed")


@synalinks_export(
    [
        "synalinks.modules.Action",
        "synalinks.Action",
    ]
)
class Action(Module):
    """Use a `LanguageModel` to perform a function call given the input datamodel.

    This module use structured output to call a given Python function.
    This module can be used in agents or traditional workflows seamlessly,
    it use the input data model to infer the function parameters.

    The output of this module contains the inputs infered by the language model
    as well as the outputs of the function call.

    Note: The function **MUST** return a JSON object dict and be asynchronous.

    Example:

    ```python
    import synalinks
    import asyncio

    async def main():

        class Query(synalinks.DataModel):
            query: str

        async def calculate(expression: str):
            \"""Calculate the result of a mathematical expression.

            Args:
                expression (str): The mathematical expression to calculate, such as
                    '2 + 2'. The expression can contain numbers, operators (+, -, *, /),
                    parentheses, and spaces.
            \"""
            if not all(char in "0123456789+-*/(). " for char in expression):
                return {
                    "result": None,
                    "log": "Error: invalid characters in expression",
                }
            try:
                # Evaluate the mathematical expression safely
                result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
                return {
                    "result": result,
                    "log": "Successfully executed",
                }
            except Exception as e:
                return {
                    "result": None,
                    "log": f"Error: {e}",
                }

        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )

        x0 = synalinks.Input(data_model=Query)
        x1 = await synalinks.Action(
            fn=calculate,
            language_model=language_model,
        )(x0)

        program = synalinks.Program(
            inputs=x0,
            outputs=x1,
            name="calculator",
            description="This program perform the calculation of an expression",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    Args:
        fn (Callable): The function to call.
        language_model (LanguageModel): The language model to use.
        prompt_template (str): The default jinja2 prompt template
            to use (see `Generator`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        instructions (list): The default instructions to use (see `Generator`).
        seed_instructions (list): Optional. A list of instructions to use as seed for the
            optimization. If not provided, use the default instructions as seed.
        temperature (float): Optional. The temperature for the LM call.
        use_inputs_schema (bool): Optional. Whether or not use the inputs schema in
            the prompt (Default to False) (see `Generator`).
        use_outputs_schema (bool): Optional. Whether or not use the outputs schema in
            the prompt (Default to False) (see `Generator`).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        fn,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        seed_instructions=None,
        temperature=0.0,
        use_inputs_schema=False,
        use_outputs_schema=False,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.fn = fn
        schema = tool_utils.Tool(fn).get_tool_schema()
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.action = Generator(
            schema=schema,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            seed_instructions=self.seed_instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            name=self.name + "_generator",
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        fn_inputs = await self.action(inputs, training=training)
        try:
            fn_outputs = await self.fn(**fn_inputs.get_json())
        except Exception as e:
            fn_outputs = {"error": str(e)}
        generic_io = GenericIO(inputs=fn_inputs.get_json(), outputs=fn_outputs)
        return JsonDataModel(
            json=GenericAction(action=generic_io.get_json()).get_json(),
            schema=GenericAction.get_schema(),
            name=self.name,
        )

    async def compute_output_spec(self, inputs, training=False):
        _ = await self.action(inputs)
        return SymbolicDataModel(schema=GenericAction.get_schema(), name=self.name)

    def get_config(self):
        config = {
            "fn": self.fn,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "seed_instructions": self.seed_instructions,
            "temperature": self.temperature,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model
            )
        }
        return {**config, **language_model_config}

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(language_model=language_model, **config)
