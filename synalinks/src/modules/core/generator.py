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
from synalinks.src.backend import is_chat_messages
from synalinks.src.backend import is_strictly_chat_message
from synalinks.src.backend import is_strictly_chat_messages
from synalinks.src.backend.common.op_scope import current_op_scope
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.language_models.language_model import StreamingIterator
from synalinks.src.modules.language_models.language_model import _tool_to_wire
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib

ROLES = [
    ChatRole.SYSTEM,
    ChatRole.USER,
    ChatRole.ASSISTANT,
    ChatRole.TOOL,
    ChatRole.FUNCTION,
]

_SYSTEM_PROMPT_ROLES = {ChatRole.SYSTEM.value, ChatRole.DEVELOPER.value}


def _as_tool_list(tools):
    """Normalize `tools` (a list, a `{name: Tool}` mapping, or None) to a list."""
    if not tools:
        return []
    if isinstance(tools, dict):
        return list(tools.values())
    return list(tools)


def _tools_to_schemas(tools):
    """Convert live `Tool` objects to OpenAI wire-format declaration dicts."""
    return [_tool_to_wire(tool) for tool in _as_tool_list(tools)]


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
<instructions>
{{ instructions }}
</instructions>
{% if inputs_schema %}
<input_schema>
{{ inputs_schema }}
</input_schema>
{% endif %}{% if outputs_schema %}
<output_schema>
{{ outputs_schema }}
</output_schema>
{% endif %}{% if examples %}
<examples>
{% for example in examples %}
<example>
<input>
{{ example[0] }}
</input>
<output>
{{ example[1] }}
</output>
</example>
{% endfor %}
</examples>
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
        temperature (float): Optional. The sampling temperature for the LM call.
            Default to None — when None it is NOT sent, so the model's own
            generation defaults apply (e.g. a vLLM-served model uses its
            `generation_config.json`). Set a float to override.
        max_tokens (int): Optional. Cap on the number of tokens generated. Default
            to None (not sent → the provider/model default). Set it to bound
            runaway / looping generations.
        top_p (float): Optional. Nucleus-sampling top-p. Default None (not sent →
            model default).
        top_k (int): Optional. Top-k sampling. Default None (not sent → model
            default).
        reasoning_effort (string): Optional. The reasoning effort for the LM call
            between ['minimal', 'low', 'medium', 'high', 'disable', 'none', None].
            Default to None (no reasoning).
        streaming (str): Optional. If true stream the LM response, enabled only if
            `schema` is `None`. Honored in every phase (inference, reward,
            optimizer) and in training: in a batched loop (predict / evaluate /
            the optimizer forward pass) or during training the stream is drained
            into a concrete prediction so it stays scorable while still recording
            time-to-first / time-to-last token (the optimizer phase included, so
            its TTFT is measured). Only an interactive single call (no active
            op_scope and not training) hands the live stream back to the caller.
        tools (list): Optional. Live `synalinks.modules.Tool` objects (or a
            `{name: Tool}` mapping) the generator always exposes, merged with any
            `tools` passed to `call`. Serialized as `tool_schemas` (their wire
            form) in the config, the way `data_model` is stored as `schema`.
        tool_schemas (list): Optional. Already-wire-formatted tool declaration
            dicts (OpenAI `{"type": "function", ...}` shape) the generator always
            exposes. Merged with any `tool_schemas` passed to `call`. Being plain
            JSON, they serialize with the config (unlike per-call `tools`).
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
        temperature=None,
        max_tokens=None,
        top_p=None,
        top_k=None,
        reasoning_effort=None,
        streaming=False,
        tools=None,
        tool_schemas=None,
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
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
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
        # Live `Tool` objects the generator always exposes; merged with any
        # passed per-call. They are not JSON, so `get_config` serializes them as
        # wire-format `tool_schemas` (the way `data_model` is stored as `schema`).
        self.tools = _as_tool_list(tools)
        # Already wire-format (OpenAI `{"type": "function", ...}`) tool
        # declarations the generator always exposes. Plain JSON dicts, so they
        # serialize as-is and are merged with any per-call `tool_schemas`.
        self.tool_schemas = tool_schemas

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

    async def call(self, inputs, tools=None, tool_schemas=None, training=False):
        if not inputs:
            return None
        # Merge the always-on, constructor-level tools/schemas with any passed
        # per-call.
        tools = self.tools + _as_tool_list(tools) or None
        tool_schemas = list(self.tool_schemas or []) + list(tool_schemas or []) or None
        msgs = self.format_messages(inputs)
        # Streaming is honored in every phase (inference, reward, optimizer) and
        # in training. When it is drained below (any batched loop or training
        # pass) the prediction stays concrete and scorable while still recording
        # TTFT / TTLT, so there is no reason to disable it -- including during
        # training and single-sample (batch=1) runs. Keeping it on in the
        # optimizer phase is deliberate: its time-to-first-token must be measured
        # too.
        streaming = bool(self.streaming)
        # Only forward sampling params that were explicitly set, so an unset
        # (None) value falls through to the model's own generation defaults
        # (e.g. a vLLM-served model applies its generation_config.json) instead
        # of being sent as `null`. litellm.drop_params drops *unsupported* keys,
        # not None-valued ones, so the filtering must happen here.
        sampling = {}
        if self.temperature is not None:
            sampling["temperature"] = self.temperature
        if self.max_tokens is not None:
            sampling["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            sampling["top_p"] = self.top_p
        if self.top_k is not None:
            sampling["top_k"] = self.top_k
        value = await self.language_model(
            msgs,
            schema=self.schema,
            tools=tools,
            tool_schemas=tool_schemas,
            streaming=streaming,
            reasoning_effort=self.reasoning_effort,
            **sampling,
        )
        if isinstance(value, StreamingIterator):
            # A purely interactive call (no op_scope, not training) hands the
            # lazy stream to the caller to consume. Inside a batched loop
            # (predict / evaluate run under op_scope) or during training the
            # prediction must be concrete -- to be scored and recorded into the
            # optimizer's prediction state -- so drain the stream into a
            # ChatMessage; time-to-first / time-to-last token are still recorded
            # as it is consumed.
            if current_op_scope() is None and not training:
                return value
            value = await value.aconsume(
                name=f"{self.language_model.name}_response"
            )
        if not value:
            result = None
        else:
            result = value.clone(name="prediction_" + self.name)
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

    async def compute_output_spec(
        self, inputs, tools=None, tool_schemas=None, training=False
    ):
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
                        content=f"<input>\n{inputs_fields}\n</input>\n<output>\n",
                    )
                )
            messages.extend(msgs)
            return ChatMessages(messages=messages)
        # NB: a strictly-chat-message input is already handled by the
        # is_strictly_chat_message early-return above. Anything that only
        # *contains* a chat message here also carries extra fields (a reward's
        # `gold_`-prefixed reference, inputs concatenated by an upstream
        # generator, ...). Splatting the whole dict into a single ChatMessage
        # would trip its `extra="forbid"`, so it falls through to be rendered as
        # input data below.
        user_message = ChatMessage(
            role="user", 
            content=f"<input>\n{inputs.get_json()}\n</input>\n<output>\n",
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
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "reasoning_effort": self.reasoning_effort,
            "streaming": self.streaming,
            # Live `tools` are converted to their wire form and stored alongside
            # `tool_schemas` (the way `data_model` is stored as `schema`).
            "tool_schemas": (
                list(self.tool_schemas or []) + _tools_to_schemas(self.tools)
            )
            or None,
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
