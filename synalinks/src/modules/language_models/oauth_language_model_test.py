# License Apache 2.0: (c) 2026 Synalinks contributors

import pytest

from synalinks.src import testing
from synalinks.src.language_models.oauth_language_model import OAuthLanguageModel
from synalinks.src.language_models.oauth_language_model import _extract_json
from synalinks.src.language_models.oauth_language_model import _make_strict_schema
from synalinks.src.language_models.oauth_language_model import ask_llm_via_cli


class OAuthLanguageModelTest(testing.TestCase):
    """Smoke tests that exercise the pure-Python helpers without spawning
    subprocesses, so CI does not need claude/codex/gemini binaries."""

    async def test_unknown_provider_returns_error_string(self):
        raw, parsed = await ask_llm_via_cli(provider="bogus", prompt="hi")
        self.assertTrue(raw.startswith("❌"))
        self.assertIsNone(parsed)

    def test_make_strict_schema_recursive(self):
        src = {
            "type": "object",
            "properties": {
                "foo": {"type": "string"},
                "bar": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "additionalProperties": True,
                },
            },
            "additionalProperties": True,
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"y": {"type": "boolean"}},
                }
            },
        }
        out = _make_strict_schema(src)

        self.assertEqual(out["additionalProperties"], False)
        self.assertEqual(sorted(out["required"]), ["bar", "foo"])

        inner = out["properties"]["bar"]
        self.assertEqual(inner["additionalProperties"], False)
        self.assertEqual(inner["required"], ["x"])

        defn = out["$defs"]["Inner"]
        self.assertEqual(defn["additionalProperties"], False)
        self.assertEqual(defn["required"], ["y"])

        # original input must not be mutated
        self.assertEqual(src["additionalProperties"], True)

    def test_extract_json_balanced_and_fenced(self):
        target = {"k": 1, "n": "value"}

        raw_text = '{"k": 1, "n": "value"}'
        fenced = "before\n```json\n" + raw_text + "\n```\nafter"
        prose = "Here is the answer:\n" + raw_text + "\nThanks."

        for variant in (raw_text, fenced, prose):
            self.assertEqual(_extract_json(variant), target)

        self.assertIsNone(_extract_json("no json here"))

    def test_constructor_signature_matches_parent(self):
        lm = OAuthLanguageModel(
            model="codex/gpt-5.2",
            timeout=60,
            retry=3,
        )
        self.assertEqual(lm.model, "codex/gpt-5.2")
        self.assertEqual(lm.provider, "codex")
        self.assertEqual(lm.cli_model, "gpt-5.2")
        self.assertEqual(lm.timeout, 60)
        self.assertEqual(lm.retry, 3)
        self.assertEqual(lm.caching, False)

    def test_serialization_roundtrip(self):
        lm = OAuthLanguageModel(
            model="claude/claude-sonnet-4-6",
            timeout=120,
            retry=4,
            caching=True,
        )
        config = lm.get_config()
        self.assertEqual(config["model"], "claude/claude-sonnet-4-6")
        self.assertEqual(config["timeout"], 120)
        self.assertEqual(config["retry"], 4)
        self.assertEqual(config["caching"], True)

        rebuilt = OAuthLanguageModel.from_config(config)
        self.assertEqual(rebuilt.model, "claude/claude-sonnet-4-6")
        self.assertEqual(rebuilt.provider, "claude")
        self.assertEqual(rebuilt.cli_model, "claude-sonnet-4-6")

    def test_invalid_model_string_raises(self):
        with pytest.raises(ValueError, match="expects model"):
            OAuthLanguageModel(model="just-a-name-no-slash")
        with pytest.raises(ValueError, match="provider must be one of"):
            OAuthLanguageModel(model="openai/gpt-4o")
        with pytest.raises(ValueError, match="model"):
            OAuthLanguageModel(model=None)
