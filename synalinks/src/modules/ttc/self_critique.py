# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)


import copy

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend import FineScore
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.backend.pydantic.metrics import get_score_type
from synalinks.src.backend.pydantic.metrics import normalize_score
from synalinks.src.backend.pydantic.metrics import score_type_description
from synalinks.src.backend.pydantic.metrics import score_type_json_type
from synalinks.src.backend.pydantic.metrics import serialize_score_type
from synalinks.src.modules.core.generator import Generator
from synalinks.src.modules.decision_models import resolve_decision_model
from synalinks.src.modules.decision_models.decision_model import UnsupportedSchemaError
from synalinks.src.modules.decision_models.decision_model import score_schema
from synalinks.src.modules.language_models import get as _get_lm
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


class Critique(DataModel):
    critique: str = Field(
        description="The elaborated critique of the provided inputs",
    )


class CritiqueWithReward(DataModel):
    critique: str = Field(
        description="The elaborated critique of the provided inputs",
    )
    reward: FineScore = Field(
        description=(
            "The reward value corresponding to the critique"
            "  (a float between 0.0 and 1.0)"
            " 0.0 being very bad and 1.0 very good"
        ),
    )


_NORMALIZED_REWARD_SCHEMA = {
    "title": "Reward",
    "type": "number",
    "minimum": 0.0,
    "maximum": 1.0,
    "description": (
        "The reward value corresponding to the critique, "
        "normalized between 0.0 (very bad) and 1.0 (very good)"
    ),
}


def critique_with_reward_schema(score_type):
    """Build the `CritiqueWithReward` schema for a given score type.

    The `reward` property is replaced by an inline enum of the members of
    `score_type` (e.g. `[1, 2, 3, 4, 5]` for `Rating`) so the language
    model picks a value on that scale; `$defs` from the default `Score`
    field are dropped so the schema stays self-contained.

    Args:
        score_type (type | str): The score scale (see `get_score_type`).

    Returns:
        (dict): The JSON schema handed to the underlying `Generator`.
    """
    score_type = get_score_type(score_type)
    schema = copy.deepcopy(CritiqueWithReward.get_schema())
    schema.pop("$defs", None)
    schema["properties"]["reward"] = {
        "title": "Reward",
        "type": score_type_json_type(score_type),
        "enum": [member.value for member in score_type],
        "description": (
            "The reward value corresponding to the critique "
            f"({score_type_description(score_type)})"
        ),
    }
    return schema


# The levels a decision model grades the inputs on, from the lowest reward to
# the highest.
DECISION_MODEL_REWARD_LEVELS = [
    "Very bad.",
    "Bad.",
    "Acceptable.",
    "Good.",
    "Very good.",
]


def decision_model_reward_schema():
    """Return the `SelfCritique` schema a `DecisionModel` answers.

    Decision models do not write text, so there is no `critique`: the
    `reward` is a score question over `DECISION_MODEL_REWARD_LEVELS`.
    """
    return {
        "title": "Reward",
        "type": "object",
        "properties": {
            "reward": score_schema(
                "How good are the provided inputs?",
                DECISION_MODEL_REWARD_LEVELS,
            ),
        },
        "required": ["reward"],
        "additionalProperties": False,
    }


def default_critique_instructions(score_type=None, return_reward=True):
    """Return the default instructions of `SelfCritique` for a score scale.

    The instructions spell out the grading scale in plain words (e.g. "an
    integer between 1 and 5, 1 being very bad and 5 very good") so the
    language model knows what the `reward` means even when the output schema
    is not included in the prompt (`use_outputs_schema=False`, the default).

    Args:
        score_type (type | str): The score scale (see `get_score_type`).
        return_reward (bool): Whether a `reward` is requested alongside the
            critique; without it the instructions only ask for the critique.

    Returns:
        (str): The instructions string.
    """
    if not return_reward:
        return (
            "Your task is to carefully examine the provided inputs and write "
            "an elaborated critique of them."
        )
    return (
        "Your task is to carefully examine the provided inputs and write an "
        "elaborated critique of them, then grade them with a reward. The reward "
        f"is {score_type_description(score_type)}."
    )


def _normalized_reward_schema(schema):
    """Return a copy of `schema` whose `reward` property is a 0..1 float."""
    schema = copy.deepcopy(schema)
    schema.setdefault("properties", {})["reward"] = dict(_NORMALIZED_REWARD_SCHEMA)
    # Drop the default `FineScore` definition if nothing references it anymore.
    defs = schema.get("$defs")
    if defs and "FineScore" in defs:
        defs.pop("FineScore")
        if not defs:
            schema.pop("$defs")
    return schema


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

    The scale the language model picks the reward from is controlled by
    `score_type`: `synalinks.FineScore` (default, 21 float levels), a
    coarser `synalinks.Score` (11 levels), or an integer Likert-style
    `synalinks.Rating` (1 to 5), `synalinks.Rating10` (1 to 10) or
    `synalinks.Rating20` (1 to 20). Whatever the scale, the `reward` in
    the module's output is automatically normalized to a float between
    0.0 (lowest level) and 1.0 (highest level), so downstream consumers
    such as `ProgramAsJudge` / `LMAsJudge` always see a 0..1 reward.

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
            instructions for the language model. If not provided, defaults to
            `default_critique_instructions(score_type, return_reward)`, which
            spells out the grading scale of `score_type`.
        seed_instructions (list): Optional. A list of instructions to use as seed for the
            optimization. If not provided, use the default instructions as seed.
        temperature (float): Optional. The temperature for the LM call.
        max_tokens (int): Optional. Maximum number of tokens to generate. Default
            None (the model's own default; caps generation length when set).
        top_p (float): Optional. The nucleus sampling probability for the LM call.
            Default None (the model's own default).
        top_k (int): Optional. The top-k sampling cutoff for the LM call.
            Default None (the model's own default).
        reasoning_effort (string): Optional. The reasoning effort for the LM call
            between ['minimal', 'low', 'medium', 'high', 'xhigh', 'disable',
            'none', None].
            Default to None (no reasoning).
        use_inputs_schema (bool): Optional. Whether or not use the inputs schema in
            the prompt (Default to False) (see `Generator`).
        use_outputs_schema (bool): Optional. Whether or not use the outputs schema in
            the prompt (Default to False) (see `Generator`).
        return_reward (bool): Optional. Whether or not to compute an intermediate reward.
        score_type (type | str): Optional. Language model only. The scale the
            language model picks the reward from: `synalinks.FineScore`
            (default), `synalinks.Score`, `synalinks.Rating`, `synalinks.Rating10`,
            `synalinks.Rating20`, any `Enum` whose members are `int` or `float`,
            or the name of one of them.
            The output `reward` is always normalized to a float between 0.0 and 1.0.
        return_inputs (bool): Optional. Whether or not to concatenate the inputs to
            the outputs (Default to True) (see `Generator`).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
        decision_model (DecisionModel): Optional. A decision model to grade
            with instead of the language model. It does not write a critique:
            the output has no `critique` field, and `return_reward` must be
            True.
    """

    def __init__(
        self,
        *,
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
        return_reward=True,
        score_type=None,
        return_inputs=True,
        name=None,
        description=None,
        trainable=True,
        decision_model=None,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        self.language_model = _get_lm(language_model)
        self.decision_model = resolve_decision_model(decision_model, language_model)
        if decision_model is None and not return_reward:
            # A decision model does not write the critique: the default one
            # does not apply.
            self.decision_model = None
        self.score_type = get_score_type(score_type or FineScore)
        self.prompt_template = prompt_template
        self.examples = examples
        use_decision_model = self.decision_model is not None
        if use_decision_model and not return_reward:
            raise UnsupportedSchemaError(
                "A `DecisionModel` does not write a critique: use it with "
                "`return_reward=True`, or use a `LanguageModel`."
            )
        if instructions is None and use_decision_model:
            instructions = (
                "Your task is to carefully examine the provided inputs and grade them."
            )
        elif instructions is None:
            instructions = default_critique_instructions(
                self.score_type, return_reward=return_reward
            )
        self.instructions = instructions
        self.seed_instructions = seed_instructions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.reasoning_effort = reasoning_effort
        self.use_inputs_schema = use_inputs_schema
        self.use_outputs_schema = use_outputs_schema
        self.return_reward = return_reward
        self.return_inputs = return_inputs

        if use_decision_model:
            schema = decision_model_reward_schema()
        elif self.return_reward:
            schema = critique_with_reward_schema(self.score_type)
        else:
            schema = Critique.get_schema()

        self.generator = Generator(
            schema=schema,
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
            return_inputs=self.return_inputs,
            name="generator_" + self.name,
        )

    async def call(self, inputs, training=False):
        outputs = await self.generator(inputs, training=training)
        if outputs is None or not self.return_reward:
            return outputs
        return self._normalize_reward(outputs)

    def _normalize_reward(self, outputs):
        """Rewrite `reward` from the native `score_type` scale to a 0..1 float.

        The underlying generator keeps working in the native scale (so its
        schema, prompt and recorded training predictions stay consistent);
        only the module's output is normalized.
        """
        schema = _normalized_reward_schema(outputs.get_schema())
        if is_symbolic_data_model(outputs):
            return SymbolicDataModel(schema=schema, name=outputs.name)
        json = dict(outputs.get_json())
        reward = json.get("reward")
        if isinstance(reward, dict):
            # A decision model's score answer: its level index, out of the top.
            json["reward"] = reward["score"] / (len(reward["legend"]) - 1)
        elif reward is not None:
            json["reward"] = normalize_score(reward, self.score_type)
        return JsonDataModel(json=json, schema=schema, name=outputs.name)

    def get_config(self):
        config = {
            "score_type": serialize_score_type(self.score_type),
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
        if self.decision_model is not None:
            language_model_config["decision_model"] = (
                serialization_lib.serialize_synalinks_object(self.decision_model)
            )
        return {
            **config,
            **language_model_config,
        }

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model"),
        )
        if "decision_model" in config:
            config["decision_model"] = serialization_lib.deserialize_synalinks_object(
                config.pop("decision_model"),
            )
        return cls(
            language_model=language_model,
            **config,
        )
