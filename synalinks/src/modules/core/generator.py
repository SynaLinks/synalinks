# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import re

import jinja2

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import Instructions
from synalinks.src.backend import Prediction
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_chat_message
from synalinks.src.backend import is_chat_messages
from synalinks.src.backend import is_strictly_chat_message
from synalinks.src.backend import is_strictly_chat_messages
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.language_models.language_model import StreamingIterator
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib

ROLES = [ChatRole.SYSTEM, ChatRole.USER, ChatRole.ASSISTANT, ChatRole.TOOL, ChatRole.FUNCTION]

_SYSTEM_PROMPT_ROLES = {ChatRole.SYSTEM.value, ChatRole.DEVELOPER.value}


def _has_system_prompt(messages):
    return any(m.get("role") in _SYSTEM_PROMPT_ROLES for m in messages)


XML_TAGS_REGEX = re.compile(
    r"<(" + "|".join(ROLES) + r")\s*(?:[^>]*)>\s*([\s\S]*?)\s*</\1>",
    re.MULTILINE,
)


@synalinks_export("synalinks.default_prompt_template")
def default_prompt_template():
    """Returns the default prompt template.

    Returns:
        (str): The default prompt template.
    """
    return """
# Instructions
{{ instructions }}
{% if inputs_schema %}
# Input Schema
{{ inputs_schema }}
{% endif %}{% if outputs_schema %}
# Output schema
{{ outputs_schema }}
{% endif %}{% if examples %}
# Examples
{% for example in examples %}
## Input:
{{ example[0] }}
## Output:
{{ example[1] }}
{% endfor %}
{% endif %}
""".strip()


def default_instructions(data_model_fields):
    return f"""
Your task is to answer with a JSON containing the following keys: {data_model_fields}
""".strip()


@synalinks_export(["synalinks.modules.Generator", "synalinks.Generator"])
class Generator(Module):
    """
    Use a `LanguageModel` to generate a data model from an arbitrary input data model.

    Example:

    ```python
    import synalinks
    import asyncio

    async def main():

        class Query(DataModel):
            query: str = synalinks.Field(
                description="The user query",
            )

        class AnswerWithCritique(synalinks.DataModel):
            thinking: str = synalinks.Field(
                description="Your step by step thinking",
            )
            critique: str = synalinks.Field(
                description="The critique of the above thinking",
            )
            answer: str = synalinks.Field(
                description="The correct answer",
            )

        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )

        x0 = synalinks.Input(data_model=Query)
        x1 = await synalinks.Generator(
            data_model=AnswerWithCritique,
            language_model=language_model,
        )(x0)

        program = synalinks.Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought_with_critique",
            description="Useful to answer step by step and evaluate your answer",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    Args:
        schema (dict): The target JSON schema.
            If not provided use the `data_model` to infer it.
        data_model (DataModel | SymbolicDataModel | JsonDataModel): The target data
            model for structured output.
        language_model (LanguageModel): The language model to use.
        prompt_template (str): The jinja2 prompt template.
        examples (list): The default list of examples, the examples
            are a list of tuples containing input/output JSON pairs.
        instructions (str): The default instructions being a string containing
            instructions for the language model.
        seed_instructions (list): Optional. A list of instructions to use as seed for the
            optimization. If not provided, use the default instructions as seed.
        use_inputs_schema (bool): Optional. Whether or not use the inputs schema in
            the prompt (Default to False).
        use_outputs_schema (bool): Optional. Whether or not use the outputs schema in
            the prompt (Default to False).
        return_inputs (bool): Optional. Whether or not to concatenate the inputs to
            the outputs (Default to False).
        temperature (float): Optional. The temperature for the LM call.
        reasoning_effort (string): Optional. The reasoning effort for the LM call
            between ['minimal', 'low', 'medium', 'high', 'disable', 'none', None].
            Default to None (no reasoning).
        streaming (str): Optional. If true stream the LM response, enabled only if
            `schema` is `None` and only during inference (not during training).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
    """

    def __init__(
        self,
        *,
        schema=None,
        data_model=None,
        language_model=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        seed_instructions=None,
        use_inputs_schema=False,
        use_outputs_schema=False,
        return_inputs=False,
        temperature=0.0,
        reasoning_effort=None,
        streaming=False,
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
        # `language_model` may be None; `ops.predict` resolves the default
        # at call time (or raises if none is set).
        self.language_model = _get_lm(language_model)
        if not prompt_template:
            prompt_template = default_prompt_template()
        self.prompt_template = prompt_template
        if not examples:
            examples = []
        self.examples = examples
        if not instructions and self.schema:
            data_model_keys = list(self.schema["properties"].keys())
            instructions = default_instructions(data_model_keys)
        self.instructions = instructions
        self.return_inputs = return_inputs
        self.temperature = temperature
        efforts = ["minimal", "low", "medium", "high", "disable", "none", None]
        if reasoning_effort not in efforts:
            raise ValueError(
                f"The reasoning effort parameter should be one of: {efforts}"
            )
        self.reasoning_effort = reasoning_effort
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        if schema and streaming:
            streaming = False
        self.streaming = streaming

        predictions = [
            Prediction(
                inputs=example[0],
                outputs=example[1],
                reward=None,
            ).get_json()
            for example in examples
        ]

        if not seed_instructions:
            seed_instructions = []
        self.seed_instructions = seed_instructions

        seed_candidates = [
            {
                "instructions": seed_instruction,
            }
            for seed_instruction in self.seed_instructions
        ]

        self.state = self.add_variable(
            initializer=Instructions(
                instructions=instructions,
                examples=predictions,
                seed_candidates=seed_candidates,
            ).get_json(),
            data_model=Instructions,
            name="state_" + self.name,
        )

    async def call(self, inputs, tools=None, training=False):
        if not inputs:
            return None
        msgs = self.format_messages(inputs)
        if self.streaming and not training:
            streaming = True
        else:
            streaming = False
        value = await self.language_model(
            msgs,
            schema=self.schema,
            tools=tools,
            streaming=streaming,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )
        if isinstance(value, StreamingIterator):
            result = value
        elif not value:
            result = None
        else:
            result = value.clone(name="prediction_" + self.name)
        if streaming:
            return result
        if result:
            if training:
                predictions = self.state.get("current_predictions")
                predictions.append(
                    {
                        "inputs": inputs.get_json(),
                        "outputs": result.get_json(),
                        "reward": None,
                    }
                )
            if self.return_inputs:
                return await ops.concat(
                    inputs,
                    result,
                    name="with_inputs_" + self.name,
                )
            else:
                return result
        return None

    async def compute_output_spec(self, inputs, tools=None, training=False):
        if self.schema:
            if self.return_inputs:
                return await ops.concat(
                    inputs,
                    SymbolicDataModel(
                        schema=self.schema,
                        name=self.name,
                    ),
                    name="with_inputs_" + self.name,
                )
            else:
                return SymbolicDataModel(
                    schema=self.schema,
                    name=self.name,
                )
        else:
            if self.return_inputs:
                return await ops.concat(
                    inputs,
                    SymbolicDataModel(
                        schema=ChatMessage.get_schema(),
                        name=self.name,
                    ),
                    name="with_inputs_" + self.name,
                )
            else:
                return SymbolicDataModel(
                    schema=ChatMessage.get_schema(),
                    name=self.name,
                )

    def format_messages(self, inputs=None):
        # Strict chat inputs: skip Jinja2 rendering when the caller already
        # supplies their own system/developer prompt; otherwise inject ours.
        if is_strictly_chat_messages(inputs):
            msgs = inputs.get("messages")
            if _has_system_prompt(msgs):
                return ChatMessages(messages=msgs)
            system_message = self._render_system_message(inputs)
            return ChatMessages(messages=[system_message.get_json(), *msgs])
        if is_strictly_chat_message(inputs):
            msg = inputs.get_json()
            if _has_system_prompt([msg]):
                return ChatMessages(messages=[msg])
            system_message = self._render_system_message(inputs)
            return ChatMessages(messages=[system_message.get_json(), msg])

        system_message = self._render_system_message(inputs)
        if is_chat_messages(inputs):
            data = inputs.get_json()
            msgs = data.get("messages")
            # Fields declared before `messages` are the inputs; fields after it
            # are outputs (e.g. concatenated by an upstream generator), so only
            # the leading fields are rendered as the input turn.
            keys = list(data.keys())
            inputs_fields = {k: data[k] for k in keys[: keys.index("messages")]}
            messages = [system_message]
            if inputs_fields:
                messages.append(
                    ChatMessage(
                        role="user",
                        content=f"## Input:\n{inputs_fields}\n## Output:\n",
                    )
                )
            messages.extend(msgs)
            return ChatMessages(messages=messages)
        if is_chat_message(inputs):
            return ChatMessages(messages=[system_message, inputs.get_json()])
        user_message = ChatMessage(
            role="user", content=f"## Input:\n{inputs.get_json()}\n## Output:\n"
        )
        return ChatMessages(messages=[system_message, user_message])

    def _render_system_message(self, inputs):
        template = jinja2.Template(self.prompt_template)
        rendered_prompt = template.render(
            inputs_schema=inputs.get_schema() if self.use_inputs_schema else None,
            outputs_schema=self.schema if self.use_outputs_schema else None,
            examples=[
                (pred.get("inputs"), pred.get("outputs"))
                for pred in self.state.get("examples")
            ],
            instructions=self.state.get("instructions"),
        )
        return ChatMessage(role="system", content=rendered_prompt)

    def get_config(self):
        config = {
            "schema": self.schema,
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "seed_instructions": self.seed_instructions,
            "use_inputs_schema": self.use_inputs_schema,
            "use_outputs_schema": self.use_outputs_schema,
            "return_inputs": self.return_inputs,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "streaming": self.streaming,
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
