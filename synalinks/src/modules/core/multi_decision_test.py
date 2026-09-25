# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json
import os
from unittest.mock import patch

from synalinks.src import testing
from synalinks.src.backend import DataModel
from synalinks.src.backend import dynamic_enum_array
from synalinks.src.modules import Input
from synalinks.src.modules.core.multi_decision import MultiDecision
from synalinks.src.modules.core.multi_decision import MultiDecisionAnswer
from synalinks.src.modules.decision_models import DecisionModel
from synalinks.src.modules.language_models import LanguageModel
from synalinks.src.programs import Program
from synalinks.src.testing.test_utils import mock_decision_model


class MultiDecisionTest(testing.TestCase):
    @patch("litellm.acompletion")
    async def test_basic_multi_decision(self, mock_completion):
        class Query(DataModel):
            query: str

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        expected_string = json.dumps(
            {
                "thinking": "The article discusses CRISPR gene editing "
                "applied to cancer, which covers science and health.",
                "choices": ["science", "health"],
            }
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        x0 = Input(data_model=Query)
        x1 = await MultiDecision(
            question="Which topics does this article cover?",
            labels=["science", "politics", "sports", "health"],
            language_model=language_model,
        )(x0)

        program = Program(
            inputs=x0,
            outputs=x1,
        )

        result = await program(
            Query(
                query="Researchers have developed a new CRISPR technique "
                "that could revolutionize cancer treatment."
            )
        )

        self.assertEqual(result.get_json(), json.loads(expected_string))

    @patch("litellm.acompletion")
    async def test_multi_decision_single_choice(self, mock_completion):
        """MultiDecision should work when only one label is selected."""

        class Query(DataModel):
            query: str

        language_model = LanguageModel(
            model="ollama/mistral",
        )

        expected_string = json.dumps(
            {
                "thinking": "This is clearly about sports.",
                "choices": ["sports"],
            }
        )

        mock_completion.return_value = {
            "choices": [{"message": {"content": expected_string}}]
        }

        x0 = Input(data_model=Query)
        x1 = await MultiDecision(
            question="Which topics?",
            labels=["science", "sports"],
            language_model=language_model,
        )(x0)

        program = Program(inputs=x0, outputs=x1)

        result = await program(Query(query="Olympic athletes set new records."))

        self.assertEqual(result.get("choices"), ["sports"])

    def test_schema_uses_inline_by_default(self):
        """Default inline=True should produce items with direct enum."""
        labels = ["a", "b", "c"]
        schema = dynamic_enum_array(
            MultiDecisionAnswer.get_schema(),
            "choices",
            labels,
            inline=True,
        )
        items = schema["properties"]["choices"]["items"]
        self.assertEqual(items, {"type": "string", "enum": labels})
        self.assertNotIn("Choices", schema.get("$defs", {}))

    def test_schema_ref_mode(self):
        """inline=False should produce $ref items."""
        labels = ["x", "y"]
        schema = dynamic_enum_array(
            MultiDecisionAnswer.get_schema(),
            "choices",
            labels,
            inline=False,
        )
        items = schema["properties"]["choices"]["items"]
        self.assertEqual(items, {"$ref": "#/$defs/Choices"})
        self.assertIn("Choices", schema["$defs"])

    def test_missing_question_raises(self):
        with self.assertRaises(ValueError):
            MultiDecision(
                labels=["a", "b"],
                language_model=LanguageModel(model="ollama/mistral"),
            )

    def test_missing_labels_raises(self):
        with self.assertRaises(ValueError):
            MultiDecision(
                question="Pick topics",
                language_model=LanguageModel(model="ollama/mistral"),
            )

    def test_serialization_round_trip(self):
        """get_config / from_config must survive a round trip."""
        lm = LanguageModel(model="ollama/mistral")
        md = MultiDecision(
            question="Pick topics",
            labels=["a", "b", "c"],
            language_model=lm,
            inline=True,
        )
        config = md.get_config()
        self.assertEqual(config["question"], "Pick topics")
        self.assertEqual(config["labels"], ["a", "b", "c"])
        self.assertTrue(config["inline"])


class MultiDecisionConfigTest(testing.TestCase):
    def test_seed_instructions_reach_the_generator(self):
        multi_decision = MultiDecision(
            question="Which topics?",
            labels=["science", "finance"],
            seed_instructions=["Pick every topic the query covers."],
            language_model=LanguageModel(model="ollama/mistral"),
        )
        restored = MultiDecision.from_config(multi_decision.get_config())
        self.assertEqual(
            restored.decision.seed_instructions, ["Pick every topic the query covers."]
        )


@patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"})
class MultiDecisionWithDecisionModelTest(testing.TestCase):
    async def test_one_noul_per_label(self):
        class Query(DataModel):
            query: str

        decision_model = DecisionModel(model="typesafe/jev-latest")
        payloads = mock_decision_model(
            decision_model,
            {
                "label_0": {"type": "noul", "noul": 0.9},
                "label_1": {"type": "noul", "noul": 0.2},
                "label_2": {"type": "noul", "noul": 0.6},
            },
        )

        x0 = Input(data_model=Query)
        x1 = await MultiDecision(
            question="Which topics does the query cover?",
            labels=["science", "finance", "sports"],
            decision_model=decision_model,
        )(x0)
        program = Program(inputs=x0, outputs=x1)

        result = await program(Query(query="Biotech startup raises funds"))

        self.assertEqual(result.get_json(), {"choices": ["science", "sports"]})
        self.assertEqual(list(x1.get_schema()["properties"]), ["choices"])
        module = program.get_module(index=1)
        restored = MultiDecision.from_config(module.get_config())
        self.assertIsInstance(restored.decision_model, DecisionModel)
        self.assertEqual(restored.decision.schema, module.decision.schema)
        self.assertEqual(
            payloads[0]["questions"]["label_1"],
            {
                "type": "noul",
                "instructions": "Which topics does the query cover? "
                "Does the label 'finance' apply?",
            },
        )

    async def test_threshold(self):
        class Query(DataModel):
            query: str

        decision_model = DecisionModel(model="typesafe/jev-latest")
        mock_decision_model(
            decision_model,
            {
                "label_0": {"type": "noul", "noul": 0.9},
                "label_1": {"type": "noul", "noul": 0.2},
                "label_2": {"type": "noul", "noul": 0.6},
            },
        )
        multi_decision = MultiDecision(
            question="Which topics?",
            labels=["science", "finance", "sports"],
            decision_model=decision_model,
            threshold=0.7,
        )
        result = await multi_decision(Query(query="q"))
        self.assertEqual(result.get_json(), {"choices": ["science"]})
        restored = MultiDecision.from_config(multi_decision.get_config())
        self.assertEqual(restored.threshold, 0.7)
        with self.assertRaisesRegex(ValueError, "requires a `decision_model`"):
            MultiDecision(
                question="Which topics?",
                labels=["science", "finance"],
                language_model=LanguageModel(model="ollama/mistral"),
                threshold=0.7,
            )
