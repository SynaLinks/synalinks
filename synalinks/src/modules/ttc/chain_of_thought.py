# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


class Thinking(DataModel):
    thinking: str = Field(
        description="Your step by step thinking",
    )


@synalinks_export(
    [
        "synalinks.modules.ChainOfThought",
        "synalinks.ChainOfThought",
    ]
)
class ChainOfThought(Module):
    """Useful to answer in a step by step manner.

    This component concatenate thinking fields to your data model/schema and generate
    a prediction allowing the LM to think step by step before answering.

    The parameter K specify the number of thinking fields to add (Default to 1).

    Example:

    ```python
    import synalink
    import asyncio

    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    class Answer(synalinks.DataModel):
        answer: str = synalinks.Field(
            description="The correct answer",
        )

    async def main():

        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )

        x0 = synalinks.Input(data_model=Query)
        x1 = await synalinks.ChainOfThought(
            data_model=Answer,
            language_model=language_model,
            k=3,
        )(x0)

        program = synalinks.Program(
            inputs=x0,
            outputs=x1,
            name="answer_with_chain_of_thought",
            description="Useful to answer step by step",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    References:
        - [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)

    Args:
        schema (dict): The target JSON schema.
            If not provided use the `data_model` to infer it.
        data_model (DataModel | SymbolicDataModel | JsonDataModel): The target data model.
        language_model (LanguageModel): The language model to use.
        prompt_template (str): The jinja2 prompt template (see `Generator`).
        examples (list): The default list of examples, the examples
            are a list of tuples containing input/output JSON pairs.
        instructions (str): The default instructions being a string containing
            instructions for the language model.
        seed_instructions (list): Optional. A list of instructions to use as seed for the
            optimization. If not provided, use the default instructions as seed.
        temperature (float): Optional. The temperature for the LM call.
        use_inputs_schema (bool): Optional. Whether or not use the inputs schema in
            the prompt (Default to False) (see `Generator`).
        use_outputs_schema (bool): Optional. Whether or not use the outputs schema in
            the prompt (Default to False) (see `Generator`).
        k (int): The number of thinking fields to add.
        return_inputs (bool): Optional. Whether or not to concatenate the inputs to
            the outputs (Default to False) (see `Generator`).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        schema=None,
        data_model=None,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        seed_instructions=None,
        temperature=0.0,
        use_inputs_schema=False,
        use_outputs_schema=False,
        k=1,
        return_inputs=False,
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
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.return_inputs = return_inputs
        self.k = k

        thinking_data_model = Thinking
        if k > 1:
            for _ in range(k - 1):
                thinking_data_model += Thinking

        final_data_model = thinking_data_model + SymbolicDataModel(schema=self.schema)

        self.generator = Generator(
            data_model=final_data_model,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            seed_instructions=self.seed_instructions,
            temperature=self.temperature,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            return_inputs=self.return_inputs,
            name=self.name + "_generator",
        )

    async def call(self, inputs, training=False):
        return await self.generator(inputs, training=training)

    def get_config(self):
        config = {
            "schema": self.schema,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "seed_instructions": self.seed_instructions,
            "temperature": self.temperature,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "return_inputs": self.return_inputs,
            "k": self.k,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model,
            )
        }
        return {
            **config,
            **language_model_config,
        }

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model"),
        )
        return cls(
            language_model=language_model,
            **config,
        )
