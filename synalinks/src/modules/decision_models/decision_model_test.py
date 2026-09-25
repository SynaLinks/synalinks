# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import os
from enum import Enum
from typing import Literal
from unittest.mock import patch

from absl.testing import parameterized

import synalinks
from synalinks.src import testing
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import DataModel
from synalinks.src.backend import Field
from synalinks.src.backend.common.op_scope import _OP_SCOPE
from synalinks.src.modules.decision_models import deserialize
from synalinks.src.modules.decision_models import serialize
from synalinks.src.modules.decision_models.decision_model import DecisionModel
from synalinks.src.modules.decision_models.decision_model import UnsupportedSchemaError
from synalinks.src.modules.decision_models.decision_model import choice_schema
from synalinks.src.modules.decision_models.decision_model import noul_schema
from synalinks.src.modules.decision_models.decision_model import questions_from_schema
from synalinks.src.modules.decision_models.decision_model import score_schema
from synalinks.src.modules.decision_models.decision_model import validate_questions
from synalinks.src.testing.test_utils import mock_decision_model


def ticket(message):
    return ChatMessages(messages=[ChatMessage(role="user", content=message)])


class Billing(DataModel):
    is_billing: bool = Field(description="Is `message` about billing?")


class Team(str, Enum):
    billing = "billing"
    technical = "technical"


class Triage(DataModel):
    is_billing: bool = Field(description="Is `message` about billing?")
    team: Literal["billing", "technical"] = Field(
        description="Which team should handle `message`?"
    )
    enum_team: Team = Field(description="Which team should handle `message`?")


QUESTIONS = {
    "is_billing": {"type": "noul", "instructions": "Is `message` about billing?"},
}

ANSWERS = {"is_billing": {"type": "noul", "noul": 0.95}}

MIXED_SCHEMA = {
    "type": "object",
    "properties": {
        "billing": noul_schema("Is `message` about billing?"),
        "team": choice_schema(
            "Which team should handle `message`?",
            {"billing": "Payments and refunds.", "technical": None},
        ),
        "urgency": score_schema(
            "How urgent is `message`?", ["Can wait", "This week", "Today"]
        ),
    },
    "required": ["billing", "team", "urgency"],
}

MIXED_QUESTIONS = {
    "billing": {"type": "noul", "instructions": "Is `message` about billing?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle `message`?",
        "criteria": {"billing": "Payments and refunds.", "technical": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is `message`?",
        "criteria": ["Can wait", "This week", "Today"],
    },
}

MIXED_ANSWERS = {
    "billing": {"type": "noul", "noul": 1.0000001},
    "team": {
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.9, "technical": 0.1},
        "confidence": 0.8,
    },
    "urgency": {
        "type": "score",
        "score": 1.2,
        "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
        "probabilities": {"0": 0.0, "1": 0.8, "2": 0.2},
        "confidence": 0.7,
    },
}


@patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"})
class DecisionModelTest(testing.TestCase):
    async def test_call_sends_state_and_questions(self):
        decision_model = DecisionModel(model="typesafe/jev-latest")
        payloads = mock_decision_model(decision_model, ANSWERS)

        result = await decision_model(
            ticket("I was charged twice."),
            schema=Billing.get_schema(),
        )

        self.assertEqual(
            payloads,
            [
                {
                    "state": [{"role": "user", "content": "I was charged twice."}],
                    "model": "jev-latest",
                    "questions": QUESTIONS,
                }
            ],
        )
        self.assertEqual(result.get_json(), {"is_billing": True})
        self.assertEqual(result.get_schema(), Billing.get_schema())
        self.assertEqual(decision_model.last_call_model, "jev-1.13.0")

    async def test_usage_and_cost_counters(self):
        decision_model = DecisionModel(model="typesafe/jev-latest")
        mock_decision_model(decision_model, ANSWERS)

        token = _OP_SCOPE.set("reward")
        try:
            await decision_model(ticket("Hi"), schema=Billing.get_schema())
        finally:
            _OP_SCOPE.reset(token)

        self.assertEqual(decision_model.cumulated_calls, 1)
        self.assertEqual(decision_model.cumulated_prompt_tokens, 100)
        self.assertEqual(decision_model.cumulated_completion_tokens, 10)
        self.assertEqual(decision_model.cumulated_tokens, 110)
        # Output tokens are free: 100 input tokens at $0.042 / Mtok.
        self.assertAlmostEqual(decision_model.cumulated_cost, 100 * 0.042e-6)
        self.assertEqual(decision_model.reward_cumulated_calls, 1)
        self.assertEqual(decision_model.inference_cumulated_calls, 0)

    async def test_cost_per_token_override(self):
        decision_model = DecisionModel(model="typesafe/jev-latest", cost_per_token=1e-3)
        mock_decision_model(decision_model, ANSWERS)
        await decision_model(ticket("Hi"), schema=Billing.get_schema())
        self.assertAlmostEqual(decision_model.cumulated_cost, 0.1)

    async def test_retry_on_rate_limit(self):
        decision_model = DecisionModel(model="typesafe/jev-latest", retry=3)
        payloads = mock_decision_model(decision_model, 429, 529, ANSWERS)

        result = await decision_model(ticket("Hi"), schema=Billing.get_schema())

        self.assertEqual(len(payloads), 3)
        self.assertEqual(result.get_json(), {"is_billing": True})

    async def test_validation_error_is_not_retried(self):
        decision_model = DecisionModel(model="typesafe/jev-latest", retry=3)
        payloads = mock_decision_model(decision_model, 422)

        with self.assertWarns(UserWarning):
            result = await decision_model(ticket("Hi"), schema=Billing.get_schema())

        self.assertIsNone(result)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(decision_model.cumulated_failed_calls, 1)

    async def test_retry_exhausted_uses_fallback(self):
        fallback = DecisionModel(model="typesafe/jev-1.13.0")
        decision_model = DecisionModel(
            model="typesafe/jev-latest", retry=2, fallback=fallback
        )
        mock_decision_model(decision_model, 503)
        fallback_payloads = mock_decision_model(fallback, ANSWERS)

        with self.assertWarns(UserWarning):
            result = await decision_model(ticket("Hi"), schema=Billing.get_schema())

        self.assertEqual(result.get_json(), {"is_billing": True})
        self.assertEqual(fallback_payloads[0]["model"], "jev-1.13.0")
        self.assertEqual(decision_model.cumulated_fallback_activations, 1)

    async def test_missing_api_key_returns_none(self):
        decision_model = DecisionModel(model="typesafe/jev-latest")
        payloads = mock_decision_model(decision_model, ANSWERS)

        with patch.dict(os.environ, {}, clear=True):
            with self.assertWarns(UserWarning):
                result = await decision_model(ticket("Hi"), schema=Billing.get_schema())

        self.assertIsNone(result)
        self.assertEqual(payloads, [])

    async def test_file_cache(self):
        decision_model = DecisionModel(
            model="typesafe/jev-latest", cache_dir=self.get_temp_dir()
        )
        payloads = mock_decision_model(decision_model, ANSWERS)

        first = await decision_model(ticket("Hi"), schema=Billing.get_schema())
        second = await decision_model(ticket("Hi"), schema=Billing.get_schema())

        self.assertEqual(len(payloads), 1)
        self.assertEqual(first.get_json(), second.get_json())
        self.assertEqual(decision_model.cumulated_cache_hits, 1)

    async def test_symbolic_call_has_output_schema(self):
        decision_model = DecisionModel(model="typesafe/jev-latest")
        x0 = synalinks.Input(data_model=ChatMessages)
        x1 = await decision_model(x0, schema=MIXED_SCHEMA)
        self.assertEqual(x1.get_schema(), MIXED_SCHEMA)

    async def test_plain_fields(self):
        decision_model = DecisionModel(model="typesafe/jev-latest")
        choice = {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.3, "technical": 0.7},
            "confidence": 0.6,
        }
        payloads = mock_decision_model(
            decision_model,
            {
                "is_billing": {"type": "noul", "noul": 0.2},
                "team": choice,
                "enum_team": choice,
            },
        )

        result = await decision_model(ticket("Hi"), schema=Triage.get_schema())

        self.assertEqual(
            result.get_json(),
            {"is_billing": False, "team": "technical", "enum_team": "technical"},
        )
        team_question = {
            "type": "choice",
            "instructions": "Which team should handle `message`?",
            "criteria": {"billing": None, "technical": None},
        }
        self.assertEqual(
            payloads[0]["questions"],
            {
                **QUESTIONS,
                "team": team_question,
                "enum_team": team_question,
            },
        )

    def test_questions_from_schema(self):
        questions, plain = questions_from_schema(MIXED_SCHEMA)
        self.assertEqual(questions, MIXED_QUESTIONS)
        self.assertEqual(plain, {})
        structured = noul_schema({"question": "q", "guidance": "g"})
        questions, _ = questions_from_schema(
            {"type": "object", "properties": {"a": structured}}
        )
        self.assertEqual(
            questions["a"]["instructions"], {"question": "q", "guidance": "g"}
        )

    def test_invalid_schema(self):
        class NoDescription(DataModel):
            is_billing: bool

        class Unsupported(DataModel):
            summary: str = Field(description="Summarize `message`.")

        for schema in [
            None,
            {"type": "object", "properties": {}},
            NoDescription.get_schema(),
            Unsupported.get_schema(),
        ]:
            with self.assertRaises(ValueError):
                questions_from_schema(schema)

    async def test_generation_field_raises(self):
        class Summary(DataModel):
            summary: str = Field(description="Summarize the ticket.")

        decision_model = DecisionModel(model="typesafe/jev-latest")
        payloads = mock_decision_model(decision_model, ANSWERS)

        with self.assertRaisesRegex(ValueError, "do not generate"):
            await decision_model(ticket("Hi"), schema=Summary.get_schema())
        with self.assertRaisesRegex(ValueError, "do not generate"):
            await decision_model(
                synalinks.Input(data_model=ChatMessages), schema=Summary.get_schema()
            )
        self.assertEqual(payloads, [])

    async def test_valid_answers_are_clipped_and_typed(self):
        decision_model = DecisionModel(model="typesafe/jev-latest")
        mock_decision_model(decision_model, MIXED_ANSWERS)

        result = await decision_model(ticket("Hi"), schema=MIXED_SCHEMA)

        answers = result.get_json()
        # Float noise above 1 is clipped.
        self.assertEqual(answers["billing"], {"noul": 1.0})
        self.assertEqual(answers["team"]["choice"], "billing")
        self.assertEqual(answers["urgency"]["score"], 1.2)
        self.assertEqual(
            answers["urgency"]["legend"],
            {"0": "Can wait", "1": "This week", "2": "Today"},
        )

    @parameterized.named_parameters(
        ("noul_out_of_range", "billing", {"type": "noul", "noul": 1.5}),
        ("wrong_type", "billing", {"type": "choice", "noul": 0.5}),
        (
            "unknown_choice",
            "team",
            {
                "type": "choice",
                "choice": "sales",
                "probabilities": {"billing": 0.5, "technical": 0.5},
                "confidence": 0.0,
            },
        ),
        (
            "missing_probability",
            "team",
            {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 1.0},
                "confidence": 1.0,
            },
        ),
        (
            "negative_confidence",
            "team",
            {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 1.0, "technical": 0.0},
                "confidence": -0.2,
            },
        ),
        (
            "score_out_of_range",
            "urgency",
            {
                "type": "score",
                "score": 3.0,
                "legend": {"0": "a", "1": "b", "2": "c"},
                "probabilities": {"0": 0.0, "1": 0.0, "2": 1.0},
                "confidence": 1.0,
            },
        ),
        ("missing_answer", "urgency", None),
    )
    async def test_malformed_answers_fail_the_call(self, question_id, answer):
        decision_model = DecisionModel(model="typesafe/jev-latest", retry=3)
        answers = dict(MIXED_ANSWERS)
        if answer is None:
            del answers[question_id]
        else:
            answers[question_id] = answer
        payloads = mock_decision_model(decision_model, answers)

        with self.assertWarns(UserWarning):
            result = await decision_model(ticket("Hi"), schema=MIXED_SCHEMA)

        self.assertIsNone(result)
        # A malformed answer is not retried.
        self.assertEqual(len(payloads), 1)

    async def test_tools_raise(self):
        decision_model = DecisionModel(model="typesafe/jev-latest")
        with self.assertRaisesRegex(ValueError, "does not call tools"):
            await decision_model(
                ticket("Hi"),
                tool_schemas=[{"type": "function", "function": {"name": "f"}}],
            )

    async def test_generator_with_decision_model(self):
        class Query(DataModel):
            query: str

        class Summary(DataModel):
            summary: str = Field(description="Summarize the query.")

        decision_model = DecisionModel(model="typesafe/jev-latest")
        payloads = mock_decision_model(decision_model, ANSWERS)
        generator = synalinks.Generator(
            data_model=Billing,
            decision_model=decision_model,
            instructions="Triage the support tickets.",
        )

        result = await generator(Query(query="I was charged twice."))

        self.assertEqual(result.get_json(), {"is_billing": True})
        system, user = payloads[0]["state"]
        self.assertIn("Triage the support tickets.", system["content"])
        self.assertEqual(user["content"], "{'query': 'I was charged twice.'}")
        self.assertEqual(payloads[0]["questions"], QUESTIONS)

        restored = synalinks.Generator.from_config(generator.get_config())
        self.assertIsInstance(restored.decision_model, DecisionModel)

        # A schema the decision model cannot answer fails at construction.
        with self.assertRaises(UnsupportedSchemaError):
            synalinks.Generator(data_model=Summary, decision_model=decision_model)

    def test_not_a_language_model(self):
        from synalinks.src.modules import language_models

        decision_model = DecisionModel(model="typesafe/jev-latest")
        with self.assertRaisesRegex(ValueError, "not a `LanguageModel`"):
            language_models.get(decision_model)
        # Modules take a decision model only through `decision_model`.
        with self.assertRaisesRegex(ValueError, "not a `LanguageModel`"):
            synalinks.Generator(data_model=Billing, language_model=decision_model)
        with self.assertRaisesRegex(ValueError, "not a `LanguageModel`"):
            synalinks.ChainOfThought(data_model=Billing, language_model=decision_model)

    async def test_score_types_are_score_questions(self):
        class Review(DataModel):
            quality: synalinks.Rating = Field(
                description="How good is the answer, from 1 (worst) to 5 (best)?"
            )
            precision: synalinks.FineScore = Field(
                description="How precise is the answer, from 0 to 1?"
            )

        questions, _ = questions_from_schema(Review.get_schema())
        self.assertEqual(
            questions["quality"],
            {
                "type": "score",
                "instructions": "How good is the answer, from 1 (worst) to 5 (best)?",
                "criteria": ["1", "2", "3", "4", "5"],
            },
        )
        # 21 values: asked over the 10 levels the API allows, spread over 0..1.
        self.assertEqual(len(questions["precision"]["criteria"]), 10)
        self.assertEqual(questions["precision"]["criteria"][0], "0")
        self.assertEqual(questions["precision"]["criteria"][-1], "1")

        def score(value, levels):
            keys = [str(i) for i in range(levels)]
            return {
                "type": "score",
                "score": value,
                "legend": {key: key for key in keys},
                "probabilities": {key: 1.0 / levels for key in keys},
                "confidence": 0.5,
            }

        decision_model = DecisionModel(model="typesafe/jev-latest")
        mock_decision_model(
            decision_model,
            # 3.4 on levels 0..4 -> 4.4 on 1..5 -> 4; 4.5 on 0..9 -> 0.5 -> 0.5.
            {"quality": score(3.4, 5), "precision": score(4.5, 10)},
        )
        result = await decision_model(ticket("Hi"), schema=Review.get_schema())
        self.assertEqual(result.get_json(), {"quality": 4, "precision": 0.5})
        Review(**result.get_json())  # the values belong to their scales

    def test_check_schema(self):
        decision_model = DecisionModel(model="typesafe/jev-latest")
        decision_model.check_schema(MIXED_SCHEMA)
        decision_model.check_schema(Triage.get_schema())
        with self.assertRaises(UnsupportedSchemaError):
            decision_model.check_schema({"type": "object", "properties": {}})

    def test_schema_helpers(self):
        self.assertEqual(
            questions_from_schema(
                {
                    "type": "object",
                    "properties": {"team": choice_schema("Which team?", ["a", "b"])},
                }
            )[0]["team"]["criteria"],
            {"a": None, "b": None},
        )
        # A description that only looks like JSON is sent as plain text.
        questions, _ = questions_from_schema(
            {"type": "object", "properties": {"a": noul_schema("{not json")}}
        )
        self.assertEqual(questions["a"]["instructions"], "{not json")

    def test_api_limits_are_schema_errors(self):
        for schema in [
            {
                "type": "object",
                "properties": {"a": choice_schema("q", [str(i) for i in range(256)])},
            },
            {"type": "object", "properties": {"a": score_schema("q", ["only"])}},
            {
                "type": "object",
                "properties": {"a": score_schema("q", list("abcdefghijk"))},
            },
        ]:
            with self.assertRaises(UnsupportedSchemaError):
                questions_from_schema(schema)

    @parameterized.named_parameters(
        ("not_a_dict", ["billing"]),
        (
            "noul_not_a_number",
            {**MIXED_ANSWERS, "billing": {"type": "noul", "noul": "x"}},
        ),
        (
            "score_not_a_number",
            {
                **MIXED_ANSWERS,
                "urgency": {**MIXED_ANSWERS["urgency"], "score": None},
            },
        ),
        (
            "legend_mismatch",
            {
                **MIXED_ANSWERS,
                "urgency": {**MIXED_ANSWERS["urgency"], "legend": {"0": "Can wait"}},
            },
        ),
    )
    async def test_more_malformed_answers_fail_the_call(self, answers):
        decision_model = DecisionModel(model="typesafe/jev-latest")
        mock_decision_model(decision_model, lambda payload: answers)

        with self.assertWarns(UserWarning):
            result = await decision_model(ticket("Hi"), schema=MIXED_SCHEMA)

        self.assertIsNone(result)

    async def test_unknown_price_counts_no_cost(self):
        decision_model = DecisionModel(model="typesafe/jev-latest")
        mock_decision_model(decision_model, ANSWERS, model="jev-9.0.0")
        await decision_model(ticket("Hi"), schema=Billing.get_schema())
        self.assertEqual(decision_model.cumulated_cost, 0.0)
        self.assertEqual(decision_model.cumulated_prompt_tokens, 100)
        self.assertEqual(decision_model.last_call_model, "jev-9.0.0")

    async def test_empty_messages_return_none(self):
        decision_model = DecisionModel(model="typesafe/jev-latest")
        payloads = mock_decision_model(decision_model, ANSWERS)
        self.assertIsNone(await decision_model(None, schema=Billing.get_schema()))
        self.assertEqual(payloads, [])

    def test_model_is_required(self):
        with self.assertRaisesRegex(ValueError, "model"):
            DecisionModel()

    def test_unsupported_provider(self):
        with self.assertRaisesRegex(ValueError, "Unsupported decision model"):
            DecisionModel(model="openai/gpt-4o-mini")
        with self.assertRaisesRegex(ValueError, "Unsupported decision model"):
            DecisionModel(model="jev-latest")

    def test_validate_questions(self):
        validate_questions(
            {
                "a": {"type": "noul", "instructions": "q", "criteria": {"true": "t"}},
                "b": {"type": "choice", "instructions": "q", "criteria": {"x": None}},
                "c": {"type": "score", "instructions": "q", "criteria": ["lo", "hi"]},
            }
        )
        invalid = [
            {},
            {"a": {"type": "vote", "instructions": "q"}},
            {"a": {"type": "noul"}},
            {"a": {"type": "choice", "instructions": "q", "criteria": ["x"]}},
            {"a": {"type": "score", "instructions": "q", "criteria": ["only"]}},
            {
                "a": {
                    "type": "score",
                    "instructions": "q",
                    "criteria": list("abcdefghijk"),
                }
            },
            {"a": {"type": "noul", "instructions": "q", "criteria": {"maybe": "m"}}},
        ]
        for questions in invalid:
            with self.assertRaises(ValueError):
                validate_questions(questions)

    def test_serialization(self):
        decision_model = DecisionModel(
            model="typesafe/jev-latest",
            fallback="typesafe/jev-1.13.0",
            cost_per_token=1e-6,
        )
        restored = deserialize(serialize(decision_model))
        self.assertEqual(restored.get_config(), decision_model.get_config())
        self.assertEqual(restored.fallback.model, "typesafe/jev-1.13.0")

    def test_default_decision_model(self):
        from synalinks.src.backend import config
        from synalinks.src.modules.decision_models import get

        self.assertIsNone(get(None))
        decision_model = DecisionModel(model="typesafe/jev-latest")
        with patch.object(config, "_persist_config"):
            config.set_default_decision_model(decision_model)
        try:
            self.assertIs(get(None), decision_model)
        finally:
            with patch.object(config, "_persist_config"):
                config.set_default_decision_model(None)


class DefaultDecisionModelTest(testing.TestCase):
    """Modules that accept a decision model use the default decision model
    instead of the default language model."""

    def setUp(self):
        super().setUp()
        from synalinks.src.backend import config

        self.default = DecisionModel(model="typesafe/jev-latest")
        self.language_model = synalinks.LanguageModel(model="ollama/mistral")
        with patch.object(config, "_persist_config"):
            config.set_default_decision_model(self.default)

        def clear():
            with patch.object(config, "_persist_config"):
                config.set_default_decision_model(None)

        self.addCleanup(clear)

    def test_decision_modules_use_the_default(self):
        decision = synalinks.Decision(question="Easy?", labels=["easy", "hard"])
        self.assertIs(decision.decision_model, self.default)
        self.assertNotIn("thinking", decision.schema["properties"])
        self.assertIs(decision.decision.decision_model, self.default)

        multi_decision = synalinks.MultiDecision(question="Which?", labels=["a", "b"])
        self.assertIs(multi_decision.decision_model, self.default)

        branch = synalinks.Branch(
            question="Easy?",
            labels=["easy", "hard"],
            branches=[synalinks.Identity(), synalinks.Identity()],
        )
        self.assertIs(branch.decision_model, self.default)
        self.assertIs(branch.decision.decision_model, self.default)

        self_critique = synalinks.SelfCritique()
        self.assertIs(self_critique.decision_model, self.default)

        rubrics = synalinks.rewards.RubricsAsJudge(rubrics="faithfulness")
        self.assertIs(rubrics.program.decision_model, self.default)

    def test_explicit_models_win(self):
        other = DecisionModel(model="typesafe/jev-1.13.0")
        decision = synalinks.Decision(
            question="Easy?", labels=["easy", "hard"], decision_model=other
        )
        self.assertIs(decision.decision_model, other)
        # An explicit language model is used, not the default decision model.
        decision = synalinks.Decision(
            question="Easy?", labels=["easy", "hard"], language_model=self.language_model
        )
        self.assertIsNone(decision.decision_model)
        self.assertIn("thinking", decision.schema["properties"])
        branch = synalinks.Branch(
            question="Easy?",
            labels=["easy", "hard"],
            branches=[synalinks.Identity(), synalinks.Identity()],
            language_model=self.language_model,
        )
        self.assertIsNone(branch.decision_model)

    def test_default_wins_over_the_default_language_model(self):
        from synalinks.src.backend import config

        with patch.object(config, "_persist_config"):
            config.set_default_language_model(self.language_model)

        def clear():
            with patch.object(config, "_persist_config"):
                config.set_default_language_model(None)

        self.addCleanup(clear)
        # Modules pass their resolved (default) language model down: it must
        # not count as a language model chosen over the default decision model.
        generator = synalinks.Generator(
            data_model=Billing, language_model=synalinks.default_language_model()
        )
        self.assertIs(generator.decision_model, self.default)
        decision = synalinks.Decision(question="Easy?", labels=["easy", "hard"])
        self.assertIs(decision.decision_model, self.default)
        # Another language model is a choice: it is used.
        other = synalinks.LanguageModel(model="ollama/qwen3")
        self.assertIsNone(
            synalinks.Generator(data_model=Billing, language_model=other).decision_model
        )

    def test_default_only_where_it_applies(self):
        # A Generator uses the default decision model only for a schema it
        # can answer.
        self.assertIs(
            synalinks.Generator(data_model=Billing).decision_model, self.default
        )

        class Summary(DataModel):
            summary: str = Field(description="Summarize the query.")

        self.assertIsNone(synalinks.Generator(data_model=Summary).decision_model)
        # Its `thinking` field needs generation.
        self.assertIsNone(
            synalinks.ChainOfThought(data_model=Billing).generator.decision_model
        )
        # No critique from a decision model, no score scale for one.
        self.assertIsNone(synalinks.SelfCritique(return_reward=False).decision_model)
        rubrics = synalinks.rewards.RubricsAsJudge(
            rubrics="faithfulness", score_type="Rating"
        )
        self.assertIsNone(rubrics.program.decision_model)
