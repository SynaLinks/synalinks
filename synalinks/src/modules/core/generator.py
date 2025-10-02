# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

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
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib

ROLES = [ChatRole.SYSTEM, ChatRole.USER, ChatRole.ASSISTANT, ChatRole.TOOL]

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
<system>
<INSTRUCTIONS>
{{ instructions }}
</INSTRUCTIONS>{% if inputs_schema %}
<INPUT_SCHEMA>
{{ inputs_schema }}
<INPUT_SCHEMA>
{% endif %}{% if outputs_schema %}
<OUTPUT_SCHEMA>
{{ outputs_schema }}
</OUTPUT_SCHEMA>
{% endif %}{% if examples %}
<EXAMPLES>{% for example in examples %}
<EXAMPLE>
Input:
{{ example[0] }}
Output:
{{ example[1] }}
</EXAMPLE>{% endfor %}
</EXAMPLES>
{% endif %}
</system>
{% if inputs %}
<user>
Input:
{{ inputs }}
Output:
</user>
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
        streaming (str): Optional. If true stream the LM response, enabled only if
            `schema` is `None` and only during inference (not during training).
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
        use_inputs_schema=False,
        use_outputs_schema=False,
        return_inputs=False,
        temperature=0.0,
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
        if not language_model:
            raise ValueError("You should provide `language_model` parameter.")
        self.language_model = language_model
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
            name=self.name + "_state",
        )

    async def call(self, inputs, training=False):
        if not inputs:
            return None
        msgs = self.format_messages(inputs)
        if self.streaming and not training:
            streaming = True
        else:
            streaming = False
        result = await ops.predict(
            msgs,
            schema=self.schema,
            language_model=self.language_model,
            streaming=streaming,
            name=self.name + "_prediction",
            temperature=self.temperature,
        )
        if streaming:
            return result
        if result:
            if training:
                predictions = self.state.get("predictions")
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
                    name=self.name + "_with_inputs",
                )
            else:
                return result
        return None

    async def compute_output_spec(self, inputs, training=False):
        if self.schema:
            if self.return_inputs:
                return await ops.concat(
                    inputs,
                    SymbolicDataModel(
                        schema=self.schema,
                        name=self.name,
                    ),
                    name=self.name + "_with_inputs",
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
                    name=self.name + "_with_inputs",
                )
            else:
                return SymbolicDataModel(
                    schema=ChatMessage.get_schema(),
                    name=self.name,
                )

    def format_messages(self, inputs=None):
        template = jinja2.Template(self.prompt_template)
        rendered_prompt = template.render(
            inputs_schema=inputs.get_schema() if self.use_inputs_schema else None,
            outputs_schema=self.schema if self.use_outputs_schema else None,
            examples=[
                (pred.get("inputs"), pred.get("outputs"))
                for pred in self.state.get("examples")
            ],
            instructions=self.state.get("instructions"),
            inputs=inputs.get_json() if inputs else None,
        )
        matches = XML_TAGS_REGEX.findall(rendered_prompt)
        extracted_tags = [(match[0], match[1].strip()) for match in matches]
        msgs = ChatMessages()
        for message in extracted_tags:
            role, content = message
            if content:
                msgs.messages.append(ChatMessage(role=role, content=content))
        return msgs

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
