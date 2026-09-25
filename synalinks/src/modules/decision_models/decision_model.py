# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import contextvars
import copy
import json
import logging
import os
import time
import warnings

import httpx2
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import retry_if_exception
from tenacity import stop_after_attempt

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend.common.op_scope import current_op_scope
from synalinks.src.modules.language_models.language_model import _message_to_wire
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib
from synalinks.src.utils.file_cache import FileCache
from synalinks.src.utils.retry_utils import rate_limit_aware_wait

SUPPORTED_PROVIDERS = ("typesafe",)

QUESTION_TYPES = ("noul", "choice", "score")

# API limits (see https://docs.typesafe.ai/api).
MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"
TYPESAFE_BASE_URL_ENV = "TYPESAFE_BASE_URL"
DEFAULT_TYPESAFE_API_BASE = "https://api.typesafe.ai"

# USD per input token, keyed by versioned model ID prefix. Output tokens are
# free. The response reports the versioned ID even when an alias such as
# `jev-latest` was requested, so the price follows the model that answered.
INPUT_COST_PER_TOKEN = {
    "jev-1.13": 0.042 / 1_000_000,
}

# Statuses an identical retry can fix: rate limiting (429), overload (529) and
# transient server errors. Validation (422) and auth (401/403) errors are raised
# straight through instead of consuming the retry budget.
RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


@synalinks_export("synalinks.decision_models.UnsupportedSchemaError")
class UnsupportedSchemaError(ValueError):
    """Raised when a schema asks for more than a decision model can answer.

    Decision models only answer typed questions (yes/no, choice, score): a
    field they would have to generate, such as a free-form string, is not
    supported. Use a `LanguageModel` for such schemas.
    """


def validate_questions(questions):
    """Check a question map against the API limits before sending it.

    Args:
        questions (dict): A map of question ID to question dict, each with a
            `type` (`"noul"`, `"choice"` or `"score"`), `instructions`, and
            the `criteria` of that type.

    Raises:
        ValueError: If a question is malformed.
    """
    if not isinstance(questions, dict) or not questions:
        raise ValueError(
            "`questions` must be a non-empty dict of question ID to question."
        )
    for question_id, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError(
                f"Question {question_id!r} must be a dict, got {question!r}."
            )
        question_type = question.get("type")
        if question_type not in QUESTION_TYPES:
            raise ValueError(
                f"Question {question_id!r} has type {question_type!r}; "
                f"expected one of {QUESTION_TYPES}."
            )
        if not question.get("instructions"):
            raise ValueError(f"Question {question_id!r} is missing `instructions`.")
        criteria = question.get("criteria")
        if question_type == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError(
                    f"Choice question {question_id!r} needs `criteria` as a "
                    "non-empty dict of option to description (or None)."
                )
            if len(criteria) > MAX_CHOICE_OPTIONS:
                raise ValueError(
                    f"Choice question {question_id!r} has {len(criteria)} options; "
                    f"the maximum is {MAX_CHOICE_OPTIONS}."
                )
        elif question_type == "score":
            if not isinstance(criteria, list) or not (
                MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS
            ):
                raise ValueError(
                    f"Score question {question_id!r} needs `criteria` as a list of "
                    f"{MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} ordered levels."
                )
        elif criteria is not None:
            if not isinstance(criteria, dict) or not set(criteria) <= {"true", "false"}:
                raise ValueError(
                    f"Noul question {question_id!r} accepts `criteria` only as a "
                    "dict with `true` and/or `false` descriptions."
                )


# Probabilities off [0, 1] by no more than this are float noise and clipped;
# anything further out is a malformed answer.
PROBABILITY_TOLERANCE = 1e-6


def probability_schema(title, description=None):
    """Return the JSON schema of a probability: a number in [0, 1]."""
    schema = {"title": title, "type": "number", "minimum": 0.0, "maximum": 1.0}
    if description:
        schema["description"] = description
    return schema


def probabilities_schema(keys, description):
    """Return the JSON schema of a probability per key (option, label or level)."""
    keys = [str(key) for key in keys]
    return {
        "title": "Probabilities",
        "type": "object",
        "description": description,
        "properties": {key: probability_schema(key) for key in keys},
        "required": keys,
        "additionalProperties": False,
    }


def confidence_schema():
    """Return the JSON schema of a `confidence` field."""
    return probability_schema(
        "Confidence", "How certain the decision model is, from 0 to 1."
    )


def _instructions_description(instructions):
    """Return `instructions` as a field description (JSON text for a dict)."""
    if isinstance(instructions, str):
        return instructions
    return json.dumps(instructions)


@synalinks_export("synalinks.decision_models.noul_schema")
def noul_schema(instructions):
    """Return the schema of a field answered by a yes/no (noul) question.

    The answer is `{"noul": p}`, the probability that the answer is yes.

    Args:
        instructions (str | dict): The question, used as the field description.

    Returns:
        (dict): The JSON schema of the field.
    """
    return {
        "title": "NoulAnswer",
        "type": "object",
        "description": _instructions_description(instructions),
        "properties": {
            "noul": probability_schema("Noul", "The probability that the answer is yes."),
        },
        "required": ["noul"],
        "additionalProperties": False,
    }


@synalinks_export("synalinks.decision_models.choice_schema")
def choice_schema(instructions, options):
    """Return the schema of a field answered by a choice question.

    The answer is `{"choice", "probabilities", "confidence"}`.

    Args:
        instructions (str | dict): The question, used as the field description.
        options (list | dict): The options, or a dict mapping each option to
            its description (or `None`).

    Returns:
        (dict): The JSON schema of the field.
    """
    if not isinstance(options, dict):
        options = {option: None for option in options}
    keys = [str(option) for option in options]
    probabilities = probabilities_schema(keys, "The probability of each option.")
    for key, description in zip(keys, options.values()):
        if description is not None:
            probabilities["properties"][key]["description"] = description
    return {
        "title": "ChoiceAnswer",
        "type": "object",
        "description": _instructions_description(instructions),
        "properties": {
            "choice": {
                "title": "Choice",
                "type": "string",
                "enum": keys,
                "description": "The most probable option.",
            },
            "probabilities": probabilities,
            "confidence": confidence_schema(),
        },
        "required": ["choice", "probabilities", "confidence"],
        "additionalProperties": False,
    }


@synalinks_export("synalinks.decision_models.score_schema")
def score_schema(instructions, levels):
    """Return the schema of a field answered by a score question.

    The answer is `{"score", "legend", "probabilities", "confidence"}`, where
    `score` is the probability-weighted level index.

    Args:
        instructions (str | dict): The question, used as the field description.
        levels (list): The ordered level descriptions (2 to 10).

    Returns:
        (dict): The JSON schema of the field.
    """
    keys = [str(i) for i in range(len(levels))]
    return {
        "title": "ScoreAnswer",
        "type": "object",
        "description": _instructions_description(instructions),
        "properties": {
            "score": {
                "title": "Score",
                "type": "number",
                "minimum": 0.0,
                "maximum": float(len(keys) - 1),
                "description": "The probability-weighted level index.",
            },
            "legend": {
                "title": "Legend",
                "type": "object",
                "description": "The description of each level, keyed by level index.",
                "properties": {
                    key: {"title": key, "type": "string", "description": level}
                    for key, level in zip(keys, levels)
                },
                "required": keys,
                "additionalProperties": False,
            },
            "probabilities": probabilities_schema(
                keys, "The probability of each level, keyed by level index."
            ),
            "confidence": confidence_schema(),
        },
        "required": ["score", "legend", "probabilities", "confidence"],
        "additionalProperties": False,
    }


def _resolve_ref(schema, node):
    """Follow a local `$ref` (e.g. `#/$defs/Team`) of a pydantic schema."""
    while isinstance(node, dict) and "$ref" in node:
        target = schema
        for part in node["$ref"].lstrip("#/").split("/"):
            target = target.get(part, {})
        node = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
    return node


def _field_instructions(key, description):
    if not description:
        raise UnsupportedSchemaError(
            f"Field {key!r} needs a `description`: it is the question asked "
            "to the decision model."
        )
    # A description holding a JSON object is sent as structured instructions.
    if description.lstrip().startswith("{"):
        try:
            return json.loads(description)
        except json.JSONDecodeError:
            pass
    return description


def questions_from_schema(schema):
    """Infer the decision model questions from an output schema.

    Each top-level field is one question, keyed by its name and asked with its
    `description`. The question type follows the field type:

    - `boolean`: a noul question, answered with `True` when p >= 0.5.
    - A string `enum` (`Literal` or `Enum`): a choice question over the
        values, answered with the most probable one.
    - A `noul_schema`, `choice_schema` or `score_schema` object: the matching
        question, answered with the full answer (probabilities, confidence...).

    Args:
        schema (dict): The output JSON schema.

    Returns:
        (tuple): The dict of questions, and the set of fields answered with a
            plain value (`boolean` or `enum`) rather than the full answer.

    Raises:
        UnsupportedSchemaError: If a field cannot be answered by a decision
            model.
    """
    properties = (schema or {}).get("properties")
    if not properties:
        raise UnsupportedSchemaError(
            "The decision model `schema` must be an object schema with at least "
            f"one field, got {schema!r}."
        )
    questions = {}
    plain = set()
    for key, prop in properties.items():
        node = _resolve_ref(schema, prop)
        instructions = _field_instructions(key, node.get("description"))
        fields = node.get("properties") or {}
        if node.get("type") == "boolean":
            questions[key] = {"type": "noul", "instructions": instructions}
            plain.add(key)
        elif "enum" in node:
            questions[key] = {
                "type": "choice",
                "instructions": instructions,
                "criteria": {str(option): None for option in node["enum"]},
            }
            plain.add(key)
        elif "noul" in fields:
            questions[key] = {"type": "noul", "instructions": instructions}
        elif "choice" in fields:
            options = _resolve_ref(schema, fields["choice"]).get("enum") or []
            probabilities = _resolve_ref(schema, fields.get("probabilities") or {})
            descriptions = probabilities.get("properties") or {}
            questions[key] = {
                "type": "choice",
                "instructions": instructions,
                "criteria": {
                    str(option): (descriptions.get(str(option)) or {}).get("description")
                    for option in options
                },
            }
        elif "score" in fields and "legend" in fields:
            legend = _resolve_ref(schema, fields["legend"]).get("properties") or {}
            questions[key] = {
                "type": "score",
                "instructions": instructions,
                "criteria": [
                    (legend.get(str(i)) or {}).get("description")
                    for i in range(len(legend))
                ],
            }
        else:
            field_type = node.get("type") or "untyped"
            raise UnsupportedSchemaError(
                f"Field {key!r} ({field_type}) cannot be answered by a decision "
                "model: decision models do not generate text or values, they "
                "only answer typed questions. Use a `bool`, a string enum "
                "(`Literal` or `Enum`), or a `noul_schema`, `choice_schema` or "
                "`score_schema` object, or a `LanguageModel` to generate it."
            )
    try:
        validate_questions(questions)
    except ValueError as e:
        raise UnsupportedSchemaError(str(e)) from e
    return questions, plain


def outputs_from_answers(questions, plain, answers):
    """Shape validated answers into the values of the output schema."""
    outputs = {}
    for key, answer in answers.items():
        if key in plain:
            if questions[key]["type"] == "noul":
                outputs[key] = answer["noul"] >= 0.5
            else:
                outputs[key] = answer["choice"]
        else:
            outputs[key] = {k: v for k, v in answer.items() if k != "type"}
    return outputs


def _check_probability(value, where):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{where} is not a number: {value!r}.")
    if not -PROBABILITY_TOLERANCE <= value <= 1 + PROBABILITY_TOLERANCE:
        raise ValueError(f"{where} is outside [0, 1]: {value!r}.")
    return min(max(float(value), 0.0), 1.0)


def _check_probabilities(probabilities, keys, where):
    if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
        raise ValueError(f"{where} probabilities do not match {list(keys)}.")
    return {
        key: _check_probability(probabilities[key], f"{where} probability of {key!r}")
        for key in keys
    }


def validate_answers(questions, answers):
    """Check the answers against their questions and normalize them.

    Every question must have an answer of its type, choices must be one of the
    options, and probabilities, nouls and confidences must be in [0, 1]
    (values off by float noise are clipped).

    Args:
        questions (dict): The questions that were asked.
        answers (dict): The answers returned by the API.

    Returns:
        (dict): The answers, restricted to the asked questions and clipped.

    Raises:
        ValueError: If an answer is missing or malformed.
    """
    if not isinstance(answers, dict):
        raise ValueError(f"Expected answers as a dict, got {answers!r}.")
    checked = {}
    for question_id, question in questions.items():
        answer = answers.get(question_id)
        where = f"Answer {question_id!r}"
        question_type = question["type"]
        if not isinstance(answer, dict) or answer.get("type") != question_type:
            raise ValueError(f"{where} is missing or is not a {question_type} answer.")
        if question_type == "noul":
            checked[question_id] = {
                "type": "noul",
                "noul": _check_probability(answer.get("noul"), f"{where} noul"),
            }
            continue
        if question_type == "choice":
            keys = [str(option) for option in question["criteria"]]
            if answer.get("choice") not in keys:
                raise ValueError(
                    f"{where} chose {answer.get('choice')!r}, not an option."
                )
            checked[question_id] = {
                "type": "choice",
                "choice": answer["choice"],
                "probabilities": _check_probabilities(
                    answer.get("probabilities"), keys, where
                ),
                "confidence": _check_probability(
                    answer.get("confidence"), f"{where} confidence"
                ),
            }
            continue
        keys = [str(i) for i in range(len(question["criteria"]))]
        score = answer.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f"{where} score is not a number: {score!r}.")
        top = len(keys) - 1
        if not -PROBABILITY_TOLERANCE <= score <= top + PROBABILITY_TOLERANCE:
            raise ValueError(f"{where} score is outside [0, {top}]: {score!r}.")
        legend = answer.get("legend")
        if not isinstance(legend, dict) or set(legend) != set(keys):
            raise ValueError(f"{where} legend does not match the levels.")
        checked[question_id] = {
            "type": "score",
            "score": min(max(float(score), 0.0), float(top)),
            "legend": {key: legend[key] for key in keys},
            "probabilities": _check_probabilities(
                answer.get("probabilities"), keys, where
            ),
            "confidence": _check_probability(
                answer.get("confidence"), f"{where} confidence"
            ),
        }
    return checked


# Usage of the decision model call running in the current task, read by hooks
# (the MLflow monitor) when the call ends, like `LanguageModel` does.
_CURRENT_CALL_USAGE = contextvars.ContextVar("synalinks_dm_call_usage", default=None)


def current_call_usage():
    """Return the usage of the decision model call in the current task.

    Returns:
        (dict): `input_tokens`, `output_tokens`, `total_tokens`, `cost`,
            `elapsed_s`, the `model` that answered and its raw `answers` (with
            probabilities and confidence) for an API call; `cache_hit` when
            the answers came from the cache; `error` (and `fallback` when a
            fallback answered) when every attempt failed. `None` before the
            call.
    """
    return _CURRENT_CALL_USAGE.get()


def _is_retryable(exc):
    if isinstance(exc, httpx2.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUSES
    return isinstance(exc, httpx2.TransportError)


def _input_cost_per_token(model):
    for prefix, cost in INPUT_COST_PER_TOKEN.items():
        if model == prefix or model.startswith(prefix + "."):
            return cost
    return None


def _accumulate(obj, phase_prefix, increments):
    for suffix, delta in increments.items():
        attr = f"{phase_prefix}cumulated_{suffix}"
        setattr(obj, attr, getattr(obj, attr) + delta)


_COUNTER_SUFFIXES = (
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "tokens",
    "elapsed_s",
    "cost",
    "failed_calls",
    "fallback_activations",
    "cache_hits",
)


@synalinks_export(
    [
        "synalinks.DecisionModel",
        "synalinks.decision_models.DecisionModel",
    ]
)
class DecisionModel(Module):
    """A decision model API wrapper.

    Decision models (System One models) evaluate a state against typed
    questions and return calibrated probabilities instead of generated text.
    They are fast and cheap, and their answers are always one of the options
    you gave, which makes them a good fit for routing, classification and
    grading. They do not reason step by step, generate text or read images.

    A decision model is called like a `LanguageModel`, with chat messages and
    the target output `schema`, and the request is inferred from them: the
    chat messages are sent as the `state`, so the system message (instructions
    and few-shot examples) is part of the context the questions are answered
    in, and each field of the output schema is one question, asked with the
    field's `description`. The output follows the schema. The field type sets
    the question type:

    - `bool`: a yes/no (noul) question. The field is `True` when the
        probability of yes is at least 0.5.
    - A string enum (`Literal` or `Enum`): pick one option, up to 255. The
        field is the most probable option.
    - `noul_schema(...)`: a yes/no question answered with `{"noul": p}`, the
        probability that the answer is yes.
    - `choice_schema(...)`: pick one option (optionally described), answered
        with `{"choice", "probabilities", "confidence"}`.
    - `score_schema(...)`: rate along 2 to 10 ordered levels, answered with
        `{"score", "legend", "probabilities", "confidence"}`, where `score` is
        the probability-weighted level index.

    Decision models do not generate: any other field (a free-form `str`, a
    number, a list...) raises an `UnsupportedSchemaError`. Use a
    `LanguageModel` for those, or `check_schema()` to check a schema first.

    Answers are validated before they are returned: a missing answer, a choice
    outside the options or a probability outside [0, 1] fails the call.

    Refer to what the messages hold in backticks, e.g.
    ``"Does `message` ask for a refund?"``. A description holding a JSON
    object is sent as structured instructions. All questions of a call see the
    same state and are answered independently and in parallel, so ask several
    at once rather than making several calls.

    The modules that can use a decision model take it as their
    `decision_model` (never as their `language_model`): `Generator`,
    `Decision`, `MultiDecision`, `Branch`, `SelfCritique` and `RubricsAsJudge`
    (and the rubric rewards). `Decision`, `MultiDecision`, `SelfCritique` and
    `RubricsAsJudge` switch to a data model made of such questions (without
    `thinking` or `critique`) when given one.

    **Using TypeSafe models**

    ```python
    import synalinks
    import os
    from typing import Literal

    os.environ["TYPESAFE_API_KEY"] = "your-api-key"

    decision_model = synalinks.DecisionModel(
        model="typesafe/jev-latest",
    )

    messages = synalinks.ChatMessages(
        messages=[
            synalinks.ChatMessage(
                role="system",
                content="You triage the support tickets of an online shop.",
            ),
            synalinks.ChatMessage(
                role="user",
                content="I was charged twice. Please fix this ASAP.",
            ),
        ]
    )

    class Triage(synalinks.DataModel):
        is_billing: bool = synalinks.Field(
            description="Is the ticket about billing?",
        )
        team: Literal["billing", "technical"] = synalinks.Field(
            description="Which team should handle the ticket?",
        )

    triage = await decision_model(messages, schema=Triage.get_schema())
    print(triage.get("is_billing"), triage.get("team"))

    urgency = await decision_model(
        messages,
        schema={
            "type": "object",
            "properties": {
                "urgency": synalinks.decision_models.score_schema(
                    "How urgent is the ticket?",
                    ["Can wait", "This week", "Today"],
                ),
            },
        },
    )
    print(urgency.get("urgency")["probabilities"])
    ```

    **Routing with a `Branch`**

    A decision model picks the branch: the question is asked as is, over the
    labels, without step by step reasoning.

    ```python
    import synalinks
    import asyncio

    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    class Answer(synalinks.DataModel):
        answer: str = synalinks.Field(
            description="The correct answer",
        )

    class AnswerWithThinking(synalinks.DataModel):
        thinking: str = synalinks.Field(
            description="Your step by step thinking",
        )
        answer: str = synalinks.Field(
            description="The correct answer",
        )

    async def main():
        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )
        decision_model = synalinks.DecisionModel(
            model="typesafe/jev-latest",
        )

        x0 = synalinks.Input(data_model=Query)
        (x1, x2) = await synalinks.Branch(
            question="What is the difficulty level of the above query?",
            labels=["easy", "difficult"],
            branches=[
                synalinks.Generator(
                    data_model=Answer,
                    language_model=language_model,
                ),
                synalinks.Generator(
                    data_model=AnswerWithThinking,
                    language_model=language_model,
                ),
            ],
            decision_model=decision_model,
        )(x0)
        x3 = x1 | x2

        program = synalinks.Program(
            inputs=x0,
            outputs=x3,
            name="conditional_reasoning",
            description="Think step by step only when the query needs it",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    **Grading with rubrics (compile + fit)**

    A decision model grades every rubric in a single call, which makes it a
    fast and cheap reward to train a program with.

    ```python
    import synalinks
    import asyncio

    async def main():
        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )
        decision_model = synalinks.DecisionModel(
            model="typesafe/jev-latest",
        )

        x0 = synalinks.Input(
            data_model=synalinks.datasets.gsm8k.get_input_data_model(),
        )
        x1 = await synalinks.Generator(
            data_model=synalinks.datasets.gsm8k.get_output_data_model(),
            language_model=language_model,
        )(x0)

        program = synalinks.Program(
            inputs=x0,
            outputs=x1,
        )

        program.compile(
            reward=synalinks.rewards.RubricsAsJudge(
                decision_model=decision_model,
                rubrics=[
                    {
                        "name": "correct",
                        "description": "The answer matches the reference.",
                        "weight": 2.0,
                    },
                    {
                        "name": "sound_reasoning",
                        "description": "Every step of the thinking is valid.",
                    },
                ],
            ),
            optimizer=synalinks.optimizers.RandomFewShot(),
        )

        (x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

        history = await program.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_test, y_test),
            epochs=2,
            batch_size=32,
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    The API key is read from `TYPESAFE_API_KEY` on every call (it is never
    stored in the config). Without it, a call fails like any failed call: it
    warns and returns `None`, or asks the `fallback` model. Set
    `TYPESAFE_BASE_URL` (or `api_base`) to use another endpoint.

    **Note**: Use an `.env` file and `.gitignore` to keep your API keys out
    of the code and out of any config file you push to a repository.

    Args:
        model (str): The model to use, prefixed by its provider
            (e.g. `"typesafe/jev-latest"`). Pin a versioned ID
            (e.g. `"typesafe/jev-1.13.0"`) when you have tuned thresholds on it.
        api_base (str): Optional. The endpoint to use.
        timeout (float): Optional. The timeout in seconds of each HTTP request
            (default to 30).
        retry (int): Optional. The number of attempts (default to 5).
        retry_max_wait (int): Optional. Max seconds to wait between retries when a
            rate-limit `Retry-After` header is honored (default to 60).
        fallback (DecisionModel): Optional. The decision model to fallback to
            if anything is wrong.
        cache_dir (str): Optional. Directory for a persistent on-disk cache.
            When set, every successful response is saved as a JSON file keyed
            by the full request (model, state and questions), and identical
            requests are answered from disk. (Default to None, disabled).
        cost_per_token (float): Optional. USD per input token, overriding the
            built-in price table (e.g. for a custom plan).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        hooks (list): Optional. Hooks to attach to this module's calls.
    """

    def __init__(
        self,
        *,
        model=None,
        api_base=None,
        timeout=30.0,
        retry=5,
        retry_max_wait=60,
        fallback=None,
        cache_dir=None,
        cost_per_token=None,
        name=None,
        description=None,
        hooks=None,
    ):
        super().__init__(
            trainable=False,
            name=name,
            description=description,
            hooks=hooks,
        )
        if model is None:
            raise ValueError("You need to set the `model` argument for any DecisionModel")
        provider, _, model_id = model.partition("/")
        if provider not in SUPPORTED_PROVIDERS or not model_id:
            raise ValueError(
                f"Unsupported decision model {model!r}. Use `<provider>/<model>` "
                f"with a provider in {list(SUPPORTED_PROVIDERS)}, "
                "e.g. `typesafe/jev-latest`."
            )
        self.model = model
        self.provider = provider
        self.model_id = model_id
        self.api_base = api_base
        self.timeout = timeout
        self.retry = retry
        self.retry_max_wait = retry_max_wait
        if fallback is not None:
            # Lazy import: `get` lives in the package __init__ which imports
            # this file at load time.
            from synalinks.src.modules.decision_models import get as _get_dm

            fallback = _get_dm(fallback)
        self.fallback = fallback
        self.cache_dir = cache_dir
        self._file_cache = FileCache(cache_dir) if cache_dir else None
        self.cost_per_token = cost_per_token
        # Test seam: an `httpx2` transport (e.g. `httpx2.MockTransport`).
        self._transport = None
        # All-time counters across every call (training + inference), then
        # phase-scoped ones populated based on the `synalinks_op_scope` set by
        # the trainer: "inference", "reward" and "optimizer".
        for prefix in ("", "inference_", "reward_", "optimizer_"):
            for suffix in _COUNTER_SUFFIXES:
                default = 0.0 if suffix in ("elapsed_s", "cost") else 0
                setattr(self, f"{prefix}cumulated_{suffix}", default)
        self.last_call_prompt_tokens = 0
        self.last_call_completion_tokens = 0
        self.last_call_tokens = 0
        self.last_call_elapsed_s = 0.0
        self.last_call_cost = 0.0
        self.last_call_model = None
        # No state depends on the input shape, so mark built up-front and
        # skip Module's auto-build path (which would try to trace `call`).
        self.built = True

    def _record(self, increments):
        """Bump the all-time and the active phase's counters."""
        _accumulate(self, "", increments)
        op = current_op_scope()
        if op is not None:
            _accumulate(self, f"{op}_", increments)

    async def call(
        self,
        messages,
        schema=None,
        tools=None,
        tool_schemas=None,
        streaming=False,
        **kwargs,
    ):
        """Answer the questions inferred from `schema` about the chat messages.

        Same interface as `LanguageModel.call()`, so a decision model can be
        used wherever a language model is (e.g. in a `Generator`), as long as
        the schema only asks questions it can answer.

        Args:
            messages (ChatMessages): The chat messages to evaluate, sent as the
                `state`. Like with a `LanguageModel`, the system message
                carries the instructions and examples, and the user message
                the inputs.
            schema (dict): The output JSON schema. Each field is one question
                (see the class docstring).
            tools (list): Not supported: decision models do not call tools.
            tool_schemas (list): Not supported: decision models do not call
                tools.
            streaming (bool): Ignored: the answers are not generated.
            **kwargs (keyword arguments): Ignored sampling arguments
                (e.g. `temperature`), for interface compatibility.

        Returns:
            (JsonDataModel): The answers, following `schema`, or `None` if
                every attempt failed and no fallback answered.

        Raises:
            UnsupportedSchemaError: If the schema asks for something a decision
                model cannot answer.
            ValueError: If `tools` or `tool_schemas` are given.
        """
        if tools or tool_schemas:
            raise ValueError(f"{self} does not call tools.")
        if not messages:
            return None
        questions, plain = questions_from_schema(schema)
        _CURRENT_CALL_USAGE.set(None)
        typed_messages = (
            messages
            if hasattr(messages, "messages")
            else ChatMessages(messages=messages.get("messages", []))
        )
        state = [_message_to_wire(message) for message in typed_messages.messages]
        cache_key = None
        if self._file_cache is not None:
            cache_key = self._file_cache.make_key(
                {
                    "model": self.model,
                    "api_base": self.api_base,
                    "state": state,
                    "questions": questions,
                }
            )
            if cache_key is not None:
                cached_json = self._file_cache.get(cache_key)
                if cached_json is not None:
                    self._record({"cache_hits": 1})
                    _CURRENT_CALL_USAGE.set({"cache_hit": True})
                    return JsonDataModel(
                        json=cached_json,
                        schema=schema,
                        name=f"{self.name}_response",
                    )
        try:
            answers = await self._call_with_retry(state, questions)
        except Exception as e:
            warnings.warn(f"All retries failed for {self}: {e}")
            self._record({"failed_calls": 1})
            if self.fallback:
                self._record({"fallback_activations": 1})
                result = await self.fallback(messages, schema=schema)
                # The fallback's usage is reported on its own span: this call
                # only reports its failure.
                _CURRENT_CALL_USAGE.set({"error": str(e), "fallback": True})
                return result
            _CURRENT_CALL_USAGE.set({"error": str(e)})
            return None
        json_outputs = outputs_from_answers(questions, plain, answers)
        if cache_key is not None:
            self._file_cache.set(cache_key, json_outputs)
        return JsonDataModel(
            json=json_outputs,
            schema=schema,
            name=f"{self.name}_response",
        )

    async def compute_output_spec(
        self,
        messages,
        schema=None,
        tools=None,
        tool_schemas=None,
        streaming=False,
        **kwargs,
    ):
        self.check_schema(schema)
        return SymbolicDataModel(schema=schema, name=f"{self.name}_response")

    def check_schema(self, schema):
        """Check that a decision model can answer an output schema.

        Every field must be a question a decision model answers: a `bool`, a
        string enum (`Literal` or `Enum`), or a `noul_schema`,
        `choice_schema` or `score_schema` object, each with a description.

        Args:
            schema (dict): The output JSON schema to check.

        Raises:
            UnsupportedSchemaError: If a field cannot be answered by a
                decision model.
        """
        questions_from_schema(schema)

    def _endpoint(self):
        api_base = (
            self.api_base
            or os.environ.get(TYPESAFE_BASE_URL_ENV)
            or DEFAULT_TYPESAFE_API_BASE
        )
        return f"{api_base.rstrip('/')}/v1/systemone"

    async def _call_with_retry(self, state, questions):
        """Perform the API call with tenacity retry logic.

        Returns:
            (dict): The validated answers, keyed by question ID.
        """
        logger = logging.getLogger(__name__)
        api_key = os.environ.get(TYPESAFE_API_KEY_ENV)
        if not api_key:
            raise ValueError(
                f"`{TYPESAFE_API_KEY_ENV}` is not set; it is required to call {self}."
            )
        payload = {
            "state": state,
            "model": self.model_id,
            "questions": copy.deepcopy(questions),
        }

        @retry(
            stop=stop_after_attempt(self.retry),
            # Honor a rate-limit `Retry-After` header; fall back to
            # exponential backoff for other retryable errors.
            wait=rate_limit_aware_wait(max_wait=self.retry_max_wait),
            retry=retry_if_exception(_is_retryable),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _do_call():
            t0 = time.perf_counter()
            async with httpx2.AsyncClient(
                transport=self._transport,
                timeout=self.timeout,
            ) as client:
                response = await client.post(
                    self._endpoint(),
                    json=payload,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                response.raise_for_status()
            elapsed_s = time.perf_counter() - t0
            body = response.json()
            # A malformed answer is a `ValueError`: not retried, it fails the
            # call (fallback or `None`) instead of reaching downstream code.
            answers = validate_answers(questions, body.get("answers"))
            answered_by = body.get("model") or self.model_id
            usage = body.get("usage") or {}
            prompt_tokens = int(usage.get("input_tokens") or 0)
            completion_tokens = int(usage.get("output_tokens") or 0)
            cost_per_token = (
                self.cost_per_token
                if self.cost_per_token is not None
                else _input_cost_per_token(answered_by)
            )
            increments = {
                "calls": 1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tokens": prompt_tokens + completion_tokens,
                "elapsed_s": elapsed_s,
            }
            if cost_per_token is not None:
                increments["cost"] = prompt_tokens * cost_per_token
            self.last_call_prompt_tokens = prompt_tokens
            self.last_call_completion_tokens = completion_tokens
            self.last_call_tokens = prompt_tokens + completion_tokens
            self.last_call_elapsed_s = elapsed_s
            self.last_call_cost = increments.get("cost", 0.0)
            self.last_call_model = answered_by
            _CURRENT_CALL_USAGE.set(
                {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "cost": increments.get("cost"),
                    "elapsed_s": elapsed_s,
                    "finish_reason": None,
                    "model": f"{self.provider}/{answered_by}",
                    "answers": answers,
                }
            )
            self._record(increments)
            return answers

        return await _do_call()

    @classmethod
    def supported_providers(cls):
        """Returns the supported decision model provider prefixes.

        These are the values accepted before the `/` in `model`, e.g.
        `"typesafe"` in `"typesafe/jev-latest"`.

        ```python
        import synalinks

        print(synalinks.DecisionModel.supported_providers())
        ```

        Returns:
            (list): The sorted list of supported provider prefixes.
        """
        return list(SUPPORTED_PROVIDERS)

    def _obj_type(self):
        return "DecisionModel"

    def get_config(self):
        config = {
            "model": self.model,
            "api_base": self.api_base,
            "timeout": self.timeout,
            "retry": self.retry,
            "retry_max_wait": self.retry_max_wait,
            "cache_dir": self.cache_dir,
            "cost_per_token": self.cost_per_token,
            "name": self.name,
            "description": self.description,
        }
        if self.fallback:
            fallback_config = {
                "fallback": serialization_lib.serialize_synalinks_object(
                    self.fallback,
                )
            }
            return {**fallback_config, **config}
        return config

    @classmethod
    def from_config(cls, config):
        if "fallback" in config:
            fallback = serialization_lib.deserialize_synalinks_object(
                config.pop("fallback")
            )
            return cls(fallback=fallback, **config)
        return cls(**config)

    def __repr__(self):
        api_base = f" api_base={self.api_base}" if self.api_base else ""
        return f"<DecisionModel model={self.model}{api_base}>"
