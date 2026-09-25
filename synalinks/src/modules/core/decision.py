# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import dynamic_enum
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.decision_models import resolve_decision_model
from synalinks.src.modules.decision_models.decision_model import choice_schema
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


class Question(DataModel):
    question: str = Field(description="The question to ask yourself.")


class DecisionAnswer(DataModel):
    thinking: str = Field(
        description="Your step by step thinking to choose the correct label."
    )
    choice: str = Field(description="The chosen label.")


class DecisionModelAnswer(DataModel):
    choice: str = Field(description="The chosen label.")


def decision_model_schema(labels):
    """Return the `Decision` output schema when a `DecisionModel` decides.

    Decision models answer typed questions without reasoning step by step, so
    there is no `thinking` field: the output is only the `choice`.
    """
    return dynamic_enum(DecisionModelAnswer.get_schema(), "choice", labels)


def decision_model_question_schema(question, labels):
    """Return the schema the `DecisionModel` answers for a `Decision`: the
    question, asked as is, as a choice over the labels."""
    return {
        "type": "object",
        "properties": {"decision": choice_schema(question, labels)},
        "required": ["decision"],
    }


def default_decision_instructions(labels):
    """The decision default instructions"""
    return f"""
You will be given a question, your task is to answer step-by-step to choose
one the following labels: {labels}
""".strip()


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

    Pass a `decision_model` to decide with a `DecisionModel` instead of the
    language model: faster and cheaper, with calibrated answers, but without
    step by step reasoning. The question is asked to the decision model as
    is, and the output has no `thinking` field, only the `choice`. With
    `min_confidence`, a decision the decision model is not sure enough about
    is not taken: the module returns `None`, so a `Branch` selects no branch.

    ```python
    x1 = await synalinks.Decision(
        question="What is the danger level of the discussion?",
        labels=["low", "medium", "high"],
        decision_model=synalinks.DecisionModel(model="typesafe/jev-latest"),
    )(x0)
    ```

    You can view this module, as performing a single label classification on the input.

    Args:
        question (str): The question to ask.
        labels (list): The list of labels to choose from (strings).
        language_model (LanguageModel): The language model to use.
        prompt_template (str): The default jinja2 prompt template
            to use (see `Generator`).
        examples (list): The default examples to use in the prompt
            (see `Generator`).
        instructions (list): The default instructions to use (see `Generator`).
        seed_instructions (list): Optional. A list of instructions to use as seed for the
            optimization. If not provided, use the default instructions as seed.
        temperature (float): Optional. The temperature for the LM call.
        max_tokens (int): Optional. Default None (model's own default). Caps the
            generation length.
        top_p (float): Optional. Default None (model's own default). Nucleus
            sampling probability.
        top_k (int): Optional. Default None (model's own default). Top-k sampling
            cutoff.
        reasoning_effort (string): Optional. The reasoning effort for the LM call
            between ['minimal', 'low', 'medium', 'high', 'disable', 'none', None].
            Default to None (no reasoning).
        use_inputs_schema (bool): Optional. Whether or not use the inputs schema in
            the prompt (Default to False) (see `Generator`).
        use_outputs_schema (bool): Optional. Whether or not use the outputs schema in
            the prompt (Default to False) (see `Generator`).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
        decision_model (DecisionModel): Optional. A decision model to decide
            with instead of the language model: the question is asked as is,
            over the labels, and the output has no `thinking` field.
        min_confidence (float): Optional. With a decision model, the confidence
            (from 0 to 1) under which no decision is taken: the module then
            returns `None`. Default to None (always decide).
    """

    def __init__(
        self,
        *,
        question=None,
        labels=None,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        seed_instructions=None,
        temperature=None,
        max_tokens=None,
        top_p=None,
        top_k=None,
        reasoning_effort=None,
        use_inputs_schema=False,
        use_outputs_schema=False,
        name=None,
        description=None,
        trainable=True,
        decision_model=None,
        min_confidence=None,
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
        self.question = question
        self.labels = labels
        self.language_model = _get_lm(language_model)
        self.decision_model = resolve_decision_model(decision_model, language_model)
        if min_confidence is not None and self.decision_model is None:
            raise ValueError(
                "`min_confidence` requires a `decision_model`: a language model "
                "gives no confidence to compare it to."
            )
        self.min_confidence = min_confidence
        if self.decision_model is not None:
            self.schema = decision_model_schema(labels)
            generator_schema = decision_model_question_schema(question, labels)
        else:
            self.schema = dynamic_enum(DecisionAnswer.get_schema(), "choice", labels)
            generator_schema = self.schema
        self.prompt_template = prompt_template
        self.examples = examples
        if not instructions:
            instructions = default_decision_instructions(self.labels)
        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.reasoning_effort = reasoning_effort
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.decision = Generator(
            schema=generator_schema,
            language_model=self.language_model,
            decision_model=self.decision_model,
            prompt_template=self.prompt_template,
            examples=self.examples,
            instructions=self.instructions,
            seed_instructions=self.seed_instructions,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
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
        if result is None or self.decision_model is None:
            return result
        answer = result.get("decision")
        if self.min_confidence is not None and answer["confidence"] < self.min_confidence:
            # Not sure enough: abstain, like a failed decision.
            return None
        return JsonDataModel(
            json={"choice": answer["choice"]},
            schema=self.schema,
            name=result.name,
        )

    async def compute_output_spec(self, inputs, training=False):
        if self.decision_model is None:
            return await super().compute_output_spec(inputs, training=training)
        return SymbolicDataModel(schema=self.schema, name=self.name)

    def get_config(self):
        config = {
            "question": self.question,
            "labels": self.labels,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "seed_instructions": self.seed_instructions,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "reasoning_effort": self.reasoning_effort,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "min_confidence": self.min_confidence,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        language_model_config = {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model
            )
        }
        if self.decision_model is not None:
            language_model_config["decision_model"] = (
                serialization_lib.serialize_synalinks_object(self.decision_model)
            )
        return {**config, **language_model_config}

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        if "decision_model" in config:
            config["decision_model"] = serialization_lib.deserialize_synalinks_object(
                config.pop("decision_model"),
            )
        return cls(language_model=language_model, **config)
