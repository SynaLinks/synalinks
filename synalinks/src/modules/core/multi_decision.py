# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)


from typing import List

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import dynamic_enum_array
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


class Question(DataModel):
    question: str = Field(description="The question to ask yourself.")


class MultiDecisionAnswer(DataModel):
    thinking: str = Field(
        description="Your step by step thinking to choose the correct labels."
    )
    choices: List[str] = Field(description="The chosen labels (one or more).")


def default_multi_decision_instructions(labels):
    """The multi-decision default instructions"""
    return f"""
You will be given a question, your task is to answer step-by-step to choose
one or more of the following labels: {labels}
""".strip()


@synalinks_export(["synalinks.modules.MultiDecision", "synalinks.MultiDecision"])
class MultiDecision(Module):
    """Perform a multi-label selection on the given input.

    This module dynamically creates an array-of-enum schema based on
    the given labels and uses it to generate a structured answer.

    This ensures that the LM answer **always** contains only values
    from the provided labels — no hallucinated entries.

    Mirrors `Decision` but allows **multiple** choices instead
    of exactly one.

    Example:

    ```python
    import synalinks
    import asyncio

    async def main():

        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )

        x0 = synalinks.Input(data_model=synalinks.ChatMessages)
        x1 = await synalinks.MultiDecision(
            question="Which topics does this article cover?",
            labels=["science", "politics", "sports", "technology"],
            language_model=language_model,
        )(x0)

        program = synalinks.Program(
            inputs=x0,
            outputs=x1,
            name="article_topic_classifier",
            description="Multi-label topic classifier.",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    The output contains ``thinking`` (chain-of-thought) and ``choices``
    (a list constrained to the provided labels).

    Args:
        question (str): The question to ask.
        labels (list): The list of labels to choose from (strings).
        language_model (LanguageModel): The language model to use.
        inline (bool): When True the enum is placed directly in the
            array items (no ``$defs`` / ``$ref`` indirection).
            Defaults to True for simpler schemas.
        prompt_template (str): The default jinja2 prompt template
            to use (see ``Generator``).
        examples (list): The default examples to use in the prompt
            (see ``Generator``).
        instructions (list): The default instructions to use
            (see ``Generator``).
        seed_instructions (list): Optional. A list of instructions to
            use as seed for the optimization. If not provided, use the
            default instructions as seed.
        temperature (float): Optional. The temperature for the LM call.
        reasoning_effort (string): Optional. The reasoning effort for the
            LM call between
            ['minimal', 'low', 'medium', 'high', 'disable', 'none', None].
            Default to None (no reasoning).
        use_inputs_schema (bool): Optional. Whether or not use the inputs
            schema in the prompt (Default to False) (see ``Generator``).
        use_outputs_schema (bool): Optional. Whether or not use the outputs
            schema in the prompt (Default to False) (see ``Generator``).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be
            trainable.
    """

    def __init__(
        self,
        *,
        question=None,
        labels=None,
        language_model=None,
        inline=True,
        prompt_template=None,
        examples=None,
        instructions=None,
        seed_instructions=None,
        temperature=0.0,
        reasoning_effort=None,
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
        schema = dynamic_enum_array(
            MultiDecisionAnswer.get_schema(),
            "choices",
            labels,
            inline=inline,
        )
        self.schema = schema
        self.question = question
        self.labels = labels
        self.inline = inline
        self.language_model = _get_lm(language_model)
        self.prompt_template = prompt_template
        self.examples = examples
        if not instructions:
            instructions = default_multi_decision_instructions(self.labels)
        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.decision = Generator(
            schema=self.schema,
            language_model=self.language_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
            use_inputs_schema=self.use_inputs_schema,
            use_outputs_schema=self.use_outputs_schema,
            name="generator_" + self.name,
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        inputs = await ops.concat(
            inputs,
            Question(question=self.question),
            name="inputs_with_question_" + self.name,
        )
        result = await self.decision(inputs, training=training)
        return result

    def get_config(self):
        config = {
            "question": self.question,
            "labels": self.labels,
            "inline": self.inline,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "seed_instructions": self.seed_instructions,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
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
