# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import copy
import re

from synalinks.src import ops
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.backend.pydantic.metrics import Rating20
from synalinks.src.backend.pydantic.metrics import get_score_type
from synalinks.src.backend.pydantic.metrics import normalize_score
from synalinks.src.backend.pydantic.metrics import score_type_description
from synalinks.src.backend.pydantic.metrics import score_type_json_type
from synalinks.src.backend.pydantic.metrics import serialize_score_type
from synalinks.src.modules import Generator
from synalinks.src.modules import Module
from synalinks.src.modules.ttc.self_critique import CritiqueWithReward
from synalinks.src.rewards.reward_wrappers import ProgramAsJudge
from synalinks.src.rewards.rubrics import get_rubric
from synalinks.src.saving import serialization_lib
from synalinks.src.utils.naming import to_snake_case

RUBRIC_NAME_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")

DEFAULT_INSTRUCTIONS = (
    "Your task is to grade the given answer against each criterion in "
    "<rubrics>, independently. Score every criterion on its own merits, "
    "using the scale in <score_scale>, then write a critique naming the "
    "criteria that lost points and why. Grade only these criteria."
)


def default_rubrics_prompt_template():
    """Return the default prompt template used by `RubricsAsJudge`."""
    return """
<instructions>
{{ instructions }}
</instructions>
{% if rubrics %}
<rubrics>
{% for rubric in rubrics %}
<rubric>
<name>
{{ rubric.name }}
</name>
<description>
{{ rubric.description }}
</description>{% if rubric.weight != 1.0 %}
<weight>
{{ "%g"|format(rubric.weight) }}
</weight>{% endif %}
</rubric>
{% endfor %}
</rubrics>
{% endif %}{% if score_scale %}
<score_scale>
{{ score_scale }}
</score_scale>
{% endif %}{% if inputs_schema %}
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


def normalize_rubric_name(name):
    """Normalize a rubric name into a JSON property name."""
    name = to_snake_case(str(name).strip())
    if not name or name[0].isdigit():
        name = f"criterion_{name}" if name else "criterion"
    return name


class Rubric:
    """One rubric criterion for `RubricsAsJudge`.

    Args:
        name (str): Short identifier used as the JSON property name. It is
            normalized to snake_case.
        description (str): Description of what the judge should evaluate.
        weight (float): Positive relative weight. Defaults to 1.0.
    """

    def __init__(self, name, description, weight=1.0):
        weight = float(weight)
        if weight <= 0:
            raise ValueError(
                f"Rubric {name!r} has weight {weight}; weights must be positive."
            )
        self.name = normalize_rubric_name(name)
        if not RUBRIC_NAME_REGEX.match(self.name):
            raise ValueError(
                f"Rubric name {name!r} normalizes to {self.name!r}, "
                "which is not usable as a JSON property."
            )
        self.description = str(description)
        self.weight = weight

    @classmethod
    def parse(cls, rubric):
        if isinstance(rubric, cls):
            return rubric
        if isinstance(rubric, dict):
            missing = {"name", "description"} - set(rubric)
            if missing:
                raise ValueError(f"Rubric {rubric!r} is missing {sorted(missing)}.")
            return cls(
                name=rubric["name"],
                description=rubric["description"],
                weight=rubric.get("weight", 1.0),
            )
        if isinstance(rubric, str):
            return cls(name=normalize_rubric_name(rubric)[:40], description=rubric)
        raise TypeError(f"Cannot interpret {rubric!r} as a rubric.")

    def get_config(self):
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def rubrics_schema(rubrics, score_type):
    """Return the constrained grade-sheet schema for a list of rubrics."""
    score_type = get_score_type(score_type)
    properties = {
        "critique": {
            "title": "Critique",
            "type": "string",
            "description": "Why each criterion was scored the way it was.",
        }
    }
    for rubric in rubrics:
        properties[rubric.name] = {
            "title": rubric.name,
            "type": score_type_json_type(score_type),
            "enum": [member.value for member in score_type],
            "description": rubric.description,
        }
    return {
        "title": "RubricGrades",
        "type": "object",
        "properties": properties,
        "required": ["critique", *[rubric.name for rubric in rubrics]],
    }


def rubrics_schema_with_reward(schema):
    """Return a copy of the grade-sheet schema with the computed reward field."""
    schema = copy.deepcopy(schema)
    schema.setdefault("properties", {})["reward"] = {
        "title": "Reward",
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "The weighted mean of the normalized criterion scores.",
    }
    required = list(schema.get("required", []))
    if "reward" not in required:
        required.append("reward")
    schema["required"] = required
    return schema


class RubricsAsJudgeProgram(Module):
    """Score each criterion with a language model, then combine in Python."""

    def __init__(
        self,
        language_model=None,
        rubrics=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        name=None,
        description=None,
        trainable=True,
    ):
        super().__init__(name=name, description=description, trainable=trainable)
        if isinstance(rubrics, str):
            rubrics = get_rubric(rubrics)
        if not rubrics:
            raise ValueError("`rubrics` is required and must not be empty.")
        self.rubrics = [Rubric.parse(rubric) for rubric in rubrics]
        names = [rubric.name for rubric in self.rubrics]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate rubric names after normalization: {names}.")

        self.score_type = get_score_type(score_type or Rating20)
        self.language_model = language_model
        self.prompt_template = prompt_template
        self.examples = examples
        self.instructions = instructions or DEFAULT_INSTRUCTIONS
        self.schema = rubrics_schema(self.rubrics, self.score_type)
        self.output_schema = rubrics_schema_with_reward(self.schema)
        self.generator = Generator(
            schema=self.schema,
            language_model=language_model,
            prompt_template=prompt_template or default_rubrics_prompt_template(),
            prompt_variables={
                "rubrics": [rubric.get_config() for rubric in self.rubrics],
                "score_scale": score_type_description(self.score_type),
            },
            examples=examples,
            instructions=self.instructions,
            name="generator_" + self.name,
        )

    async def call(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("The inputs should be a list or tuple.")
        if len(inputs) != 2:
            raise ValueError("The inputs of the program should have a length of 2.")
        y_true, y_pred = inputs
        if not y_pred:
            return CritiqueWithReward(
                critique="Empty prediction: nothing to evaluate.", reward=0.0
            ).to_json_data_model()

        if y_true:
            y_true = await ops.prefix(y_true, prefix="gold", name="gold_y_true")
            inputs = await ops.concat(y_true, y_pred, name="y_true_with_y_pred")
        else:
            inputs = y_pred
        grades = await self.generator(inputs)
        if grades is None:
            return CritiqueWithReward(
                critique="The judge produced no grade sheet.", reward=0.0
            ).to_json_data_model()
        return self.combine(grades)

    def combine(self, grades):
        if is_symbolic_data_model(grades):
            return SymbolicDataModel(schema=self.output_schema, name=grades.name)

        json = dict(grades.get_json())
        total = 0.0
        weights = 0.0
        missing = []
        for rubric in self.rubrics:
            value = json.get(rubric.name)
            if value is None:
                missing.append(rubric.name)
                continue
            total += normalize_score(value, self.score_type) * rubric.weight
            weights += rubric.weight
        if missing:
            return CritiqueWithReward(
                critique=(
                    "The judge skipped "
                    f"{', '.join(missing)}; the grade sheet is incomplete."
                ),
                reward=0.0,
            ).to_json_data_model()
        json["reward"] = total / weights if weights else 0.0
        return JsonDataModel(json=json, schema=self.output_schema, name=grades.name)

    def get_config(self):
        config = {
            "rubrics": [rubric.get_config() for rubric in self.rubrics],
            "prompt_template": self.prompt_template,
            "examples": self.examples,
            "instructions": self.instructions,
            "score_type": serialize_score_type(self.score_type),
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        return {
            "language_model": serialization_lib.serialize_synalinks_object(
                self.language_model
            ),
            **config,
        }

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        return cls(language_model=language_model, **config)


@synalinks_export(
    [
        "synalinks.RubricsAsJudge",
        "synalinks.rewards.RubricsAsJudge",
    ]
)
class RubricsAsJudge(ProgramAsJudge):
    """Grade an answer against named criteria and combine them by weight.

    Args:
        language_model (LanguageModel): The language model that grades.
        rubrics (list | str): Criteria. Each item can be a `Rubric`, a dict with
            `name`, `description` and optional `weight`, or a bare string. A string
            selects a built-in rubric preset (see `synalinks.rewards.list_rubrics`).
        prompt_template (str): Optional Jinja2 prompt template.
        examples (list): Optional examples for the prompt.
        instructions (str): Optional judge instructions.
        score_type (type | str): Per-criterion score scale. Defaults to `Rating20`.
        reduction (str): Reward reduction. Defaults to `"mean"`.
        name (str): Optional reward name.
        in_mask (list): Optional fields to keep before judging.
        out_mask (list): Optional fields to remove before judging.
        in_mask_pattern (str): Optional regex of fields to keep.
        out_mask_pattern (str): Optional regex of fields to remove.
    """

    def __init__(
        self,
        language_model=None,
        rubrics=None,
        prompt_template=None,
        examples=None,
        instructions=None,
        score_type=None,
        reduction="mean",
        name="rubrics_as_judge",
        in_mask=None,
        out_mask=None,
        in_mask_pattern=None,
        out_mask_pattern=None,
    ):
        program = RubricsAsJudgeProgram(
            language_model=language_model,
            rubrics=rubrics,
            prompt_template=prompt_template,
            examples=examples,
            instructions=instructions,
            score_type=score_type,
        )
        super().__init__(
            program=program,
            reduction=reduction,
            name=name,
            in_mask=in_mask,
            out_mask=out_mask,
            in_mask_pattern=in_mask_pattern,
            out_mask_pattern=out_mask_pattern,
        )

    @classmethod
    def from_config(cls, config):
        program = serialization_lib.deserialize_synalinks_object(config.pop("program"))
        kwargs = {
            "language_model": program.language_model,
            "prompt_template": program.prompt_template,
            "examples": program.examples,
            "instructions": program.instructions,
            "score_type": program.score_type,
            **config,
        }
        if cls is RubricsAsJudge:
            kwargs["rubrics"] = [rubric.get_config() for rubric in program.rubrics]
        return cls(**kwargs)
