# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from enum import Enum

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


# This is a trick to ensure a reward between 0.0 and 1.0
# for structured output.
# (as most providers don't support pydantic constraints nor regexes)
class Score(float, Enum):
    VERY_BAD = 0.0
    POOR = 0.1
    BELOW_AVERAGE = 0.2
    LOW_AVERAGE = 0.3
    MEDIUM_LOW = 0.4
    MEDIUM = 0.5
    MEDIUM_HIGH = 0.6
    ABOVE_AVERAGE = 0.7
    HIGH_AVERAGE = 0.8
    GOOD = 0.9
    VERY_GOOD = 1.0


class Critique(DataModel):
    critique: str = Field(
        description="The elaborated critique of the provided inputs",
    )


class CritiqueWithReward(DataModel):
    critique: str = Field(
        description="The elaborated critique of the provided inputs",
    )
    reward: Score = Field(
        description=(
            "The reward value corresponding to the critique"
            "  (a float between 0.0 and 1.0)"
            " 0.0 being very bad and 1.0 very good"
        ),
    )


@synalinks_export(
    [
        "synalinks.modules.SelfCritique",
        "synalinks.SelfCritique",
    ]
)
class SelfCritique(Module):
    """Useful to critique the given inputs.

    This component critique the inputs given and eventually generate
    an intermediate reward between 0.0 and 1.0.

    You can enable or disable the intermediate reward computation by
    using the `return_reward` flag (default to True).

    To have more accurate results, ensure that the inputs are provided along
    with the output to evaluate using `return_inputs` in your modules.

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
            return_inputs=True,
        )(x0)
        x2 = await synalinks.SelfCritique(
            language_model=language_model,
        )(x1)

        program = synalinks.Program(
            inputs=x0,
            outputs=x2,
            name="answer_with_cot_and_self_critique",
            description="Useful to answer accurately",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    Args:
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
        return_reward (bool): Optional. Whether or not to compute an intermediate reward.
        return_inputs (bool): Optional. Whether or not to concatenate the inputs to
            the outputs (Default to True) (see `Generator`).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        seed_instructions=None,
        temperature=0.0,
        use_inputs_schema=False,
        use_outputs_schema=False,
        return_reward=True,
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
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.return_reward = return_reward
        self.return_inputs = return_inputs

        if self.return_reward:
            schema = CritiqueWithReward.get_schema()
        else:
            schema = Critique.get_schema()

        self.generator = Generator(
            schema=schema,
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
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "seed_instructions": self.seed_instructions,
            "temperature": self.temperature,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "return_reward": self.return_reward,
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
