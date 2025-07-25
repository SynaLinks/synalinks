# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)


from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import dynamic_enum
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


class Question(DataModel):
    question: str = Field(description="The question to ask yourself.")


class DecisionAnswer(DataModel):
    thinking: str = Field(
        description="Your step by step thinking to choose the correct label."
    )
    choice: str = Field(description="The chosen label.")


@synalinks_export(["synalinks.modules.Decision", "synalinks.Decision"])
class Decision(Module):
    """Perform a decision on the given input based on a question and a list of labels.

    This module dynamically create an `Enum` schema based on the given labels and
    use it to generate a possible answer using structured output.

    This ensure that the LM answer is **always** one of the provided labels.

    Example:

    ```python
    import synalinks
    import asyncio

    async def main():

        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )

        x0 = synalinks.Input(data_model=synalinks.ChatMessages)
        x1 = await synalinks.Decision(
            question="What is the danger level of the discussion?",
            labels=["low", "medium", "high"],
            language_model=language_model,
        )(x0)

        program = synalinks.Program(
            inputs=x0,
            outputs=x1,
            name="discussion_danger_assessment",
            description="This program assesses the level of danger in a discussion.",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    You can view this module, as performing a single label classification on the input.

    Args:
        question (str): The question to ask.
        labels (list): The list of labels to choose from (strings).
        language_model (LanguageModel): The language model to use.
        prompt_template (str): The default jinja2 prompt template
            to use (see `Generator`).
        static_system_prompt (str): A static system prompt that **do not** evolve
            during training. This prompt allow the user to provide additional
            information that won't be changed during training. Allowing to cache
            it and reduce inference costs (see `Generator`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        instructions (list): The default instructions to use (see `Generator`).
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
        question=None,
        labels=None,
        language_model=None,
        prompt_template=None,
        static_system_prompt=None,
        examples=None,
        instructions=None,
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
        if not question:
            raise ValueError("The `question` argument must be provided.")
        if not labels:
            raise ValueError("The `labels` argument must be provided.")
        if not isinstance(labels, list):
            raise ValueError("The `labels` parameter must be a list of string.")
        schema = dynamic_enum(DecisionAnswer.get_schema(), "choice", labels)
        self.schema = schema
        self.question = question
        self.labels = labels
        self.language_model = language_model
        self.static_system_prompt = static_system_prompt
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.decision = Generator(
            schema=self.schema,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            static_system_prompt=self.static_system_prompt,
            examples=self.examples,
            instructions=self.instructions,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            name=self.name + "_generator",
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        inputs = await ops.concat(
            inputs,
            Question(question=self.question),
            name=self.name + "_inputs_with_question",
        )
        result = await self.decision(inputs, training=training)
        return result

    def get_config(self):
        config = {
            "question": self.question,
            "labels": self.labels,
            "static_system_prompt": self.static_system_prompt,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
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
