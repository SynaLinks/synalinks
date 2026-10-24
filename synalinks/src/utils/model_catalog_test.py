# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import functools
import json
import os
from unittest.mock import patch

import httpx

from synalinks.src import testing
from synalinks.src.modules.decision_models.decision_model import DecisionModel
from synalinks.src.modules.embedding_models.embedding_model import EmbeddingModel
from synalinks.src.modules.language_models.language_model import LanguageModel
from synalinks.src.utils import model_catalog

OLLAMA_CAPABILITIES = {
    "llama3:latest": ["completion", "tools"],
    "llava:latest": ["completion", "vision"],
    "bge-m3:latest": ["embedding"],
    "laya:latest": ["decision"],
}


def mock_server(handler):
    """Route the catalog's HTTP requests to `handler`; collect the requests."""
    requests = []

    def record(request):
        requests.append(request)
        return handler(request)

    client = functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(record))
    return patch.object(model_catalog.httpx, "AsyncClient", client), requests


def ollama(request):
    if request.url.path == "/api/tags":
        names = list(OLLAMA_CAPABILITIES)
        return httpx.Response(200, json={"models": [{"name": n} for n in names]})
    model = json.loads(request.content)["model"]
    return httpx.Response(200, json={"capabilities": OLLAMA_CAPABILITIES[model]})


def openai_compatible(request):
    return httpx.Response(
        200, json={"data": [{"id": "Qwen/Qwen3-4B"}, {"id": "Qwen/Qwen3-8B"}]}
    )


class SupportedModelsTest(testing.TestCase):
    async def test_ollama_lists_chat_embedding_and_decision_models_apart(self):
        patcher, requests = mock_server(ollama)
        with patcher:
            chat = await LanguageModel.supported_models("ollama")
            embedding = await EmbeddingModel.supported_models("ollama")
            decision = await DecisionModel.supported_models("ollama")
        self.assertEqual(chat, ["ollama/llama3:latest", "ollama/llava:latest"])
        self.assertEqual(embedding, ["ollama/bge-m3:latest"])
        self.assertEqual(decision, ["ollama/laya:latest"])
        self.assertEqual(str(requests[0].url), "http://localhost:11434/api/tags")

    async def test_ollama_without_capabilities_keeps_every_model(self):
        def handler(request):
            if request.url.path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": "old:latest"}]})
            return httpx.Response(200, json={})

        patcher, _ = mock_server(handler)
        with patcher:
            models = await LanguageModel.supported_models(
                "ollama", api_base="http://gpu:11434/"
            )
        self.assertEqual(models, ["ollama/old:latest"])

    @patch.dict(
        os.environ,
        {"HOSTED_VLLM_API_BASE": "http://gpu:8000", "HOSTED_VLLM_API_KEY": "secret"},
    )
    async def test_vllm_asks_its_server(self):
        patcher, requests = mock_server(openai_compatible)
        with patcher:
            models = await LanguageModel.supported_models("vllm")
        self.assertEqual(models, ["vllm/Qwen/Qwen3-4B", "vllm/Qwen/Qwen3-8B"])
        # The base URL variable is used, with the `/v1` vLLM serves under.
        self.assertEqual(str(requests[0].url), "http://gpu:8000/v1/models")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer secret")

    async def test_sglang_asks_its_local_server(self):
        patcher, requests = mock_server(openai_compatible)
        with patcher:
            models = await LanguageModel.supported_models("sglang")
        self.assertEqual(models, ["sglang/Qwen/Qwen3-4B", "sglang/Qwen/Qwen3-8B"])
        self.assertEqual(str(requests[0].url), "http://localhost:30000/v1/models")

    async def test_explicit_api_base_wins(self):
        patcher, requests = mock_server(openai_compatible)
        with patcher:
            models = await DecisionModel.supported_models(
                "hosted_vllm", api_base="http://box:9000/v1"
            )
        self.assertEqual(models[0], "hosted_vllm/Qwen/Qwen3-4B")
        self.assertEqual(str(requests[0].url), "http://box:9000/v1/models")
        self.assertNotIn("Authorization", requests[0].headers)

    async def test_unreachable_server_raises(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        patcher, _ = mock_server(handler)
        with patcher:
            with self.assertRaisesRegex(ConnectionError, "vllm.*refused"):
                await LanguageModel.supported_models("vllm")

    async def test_server_error_raises(self):
        patcher, _ = mock_server(lambda request: httpx.Response(500))
        with patcher:
            with self.assertRaises(ConnectionError):
                await LanguageModel.supported_models("ollama")

    @patch.dict(os.environ, {}, clear=True)
    async def test_no_endpoint_raises(self):
        with self.assertRaisesRegex(ValueError, "HOSTED_VLLM_API_BASE"):
            await DecisionModel.supported_models("hosted_vllm")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"})
    async def test_openai_with_a_key_lists_the_account_models(self):
        def handler(request):
            ids = [
                "gpt-4o-mini",
                "ft:gpt-4o-mini:acme::abc123",
                "text-embedding-3-small",
                "tts-1",
                "gpt-6-luna",
            ]
            return httpx.Response(200, json={"data": [{"id": i} for i in ids]})

        patcher, requests = mock_server(handler)
        with patcher:
            chat = await LanguageModel.supported_models("openai")
            embedding = await EmbeddingModel.supported_models("openai")
            decision = await DecisionModel.supported_models("openai")
        self.assertEqual(
            chat,
            [
                "openai/ft:gpt-4o-mini:acme::abc123",
                "openai/gpt-4o-mini",
                "openai/gpt-6-luna",
            ],
        )
        self.assertEqual(embedding, ["openai/text-embedding-3-small"])
        self.assertEqual(decision, ["openai/gpt-6-luna"])
        self.assertEqual(str(requests[0].url), "https://api.openai.com/v1/models")
        self.assertEqual(requests[0].headers["Authorization"], "Bearer sk-test")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-bad"})
    async def test_openai_with_a_rejected_key_raises(self):
        patcher, _ = mock_server(lambda request: httpx.Response(401))
        with patcher:
            with self.assertRaisesRegex(ConnectionError, "401"):
                await LanguageModel.supported_models("openai")

    @patch.dict(os.environ, {}, clear=True)
    async def test_catalog_providers_follow_the_model_type(self):
        chat = await LanguageModel.supported_models("openai")
        embedding = await EmbeddingModel.supported_models("openai")
        decision = await DecisionModel.supported_models("typesafe")
        self.assertIn("openai/gpt-4o-mini", chat)
        self.assertNotIn("openai/text-embedding-3-small", chat)
        self.assertIn("openai/text-embedding-3-small", embedding)
        self.assertNotIn("openai/gpt-4o-mini", embedding)
        self.assertIn("typesafe/jev-latest", decision)
        self.assertEqual(chat, sorted(chat))

    @patch.dict(os.environ, {}, clear=True)
    async def test_openai_decision_models_are_the_ones_serving_decisions(self):
        models = await DecisionModel.supported_models("openai")
        self.assertIn("openai/gpt-6-luna", models)
        self.assertNotIn("openai/gpt-4o-mini", models)

    async def test_openrouter_decision_models_are_filtered(self):
        def handler(request):
            ids = ["openai/gpt-6-luna", "openai/gpt-4o-mini", "typesafe/jev-latest"]
            return httpx.Response(200, json={"data": [{"id": i} for i in ids]})

        patcher, requests = mock_server(handler)
        with patcher:
            models = await DecisionModel.supported_models("openrouter")
        self.assertEqual(
            models, ["openrouter/openai/gpt-6-luna", "openrouter/typesafe/jev-latest"]
        )
        self.assertEqual(str(requests[0].url), "https://openrouter.ai/api/v1/models")

    async def test_family_providers_are_listed_together(self):
        bedrock = await LanguageModel.supported_models("bedrock")
        self.assertTrue(all(model.startswith("bedrock/") for model in bedrock))
        self.assertFalse(any(model.startswith("bedrock/bedrock") for model in bedrock))

    async def test_unsupported_provider_raises(self):
        with self.assertRaisesRegex(ValueError, "Unsupported language model provider"):
            await LanguageModel.supported_models("typesafe")
        with self.assertRaisesRegex(ValueError, "Unsupported decision model provider"):
            await DecisionModel.supported_models("anthropic")


class ProviderInfoTest(testing.TestCase):
    def test_server_provider(self):
        self.assertEqual(
            LanguageModel.provider_info("vllm"),
            {
                "provider": "vllm",
                "env": [
                    {"name": "HOSTED_VLLM_API_BASE", "required": False, "secret": False},
                    {"name": "HOSTED_VLLM_API_KEY", "required": False, "secret": True},
                ],
                "default_api_base": "http://localhost:8000/v1",
                "lists_models_from": "server",
            },
        )

    def test_catalog_provider(self):
        info = EmbeddingModel.provider_info("voyage")
        self.assertEqual(
            info["env"], [{"name": "VOYAGE_API_KEY", "required": True, "secret": True}]
        )
        self.assertEqual(info["lists_models_from"], "catalog")

    def test_decision_provider_comes_from_litellm(self):
        info = DecisionModel.provider_info("typesafe")
        self.assertEqual(
            info["env"],
            [
                {"name": "TYPESAFE_API_KEY", "required": True, "secret": True},
                {"name": "TYPESAFE_API_BASE", "required": False, "secret": False},
            ],
        )
        self.assertEqual(info["default_api_base"], "https://api.typesafe.ai")
        # Without a default endpoint, the base URL variable is required.
        cloudflare = DecisionModel.provider_info("cloudflare")
        self.assertTrue(cloudflare["env"][-1]["required"])
        self.assertEqual(
            DecisionModel.provider_info("hosted_vllm")["lists_models_from"], "server"
        )
        # Ollama, called as `hosted_vllm` by litellm, keeps its own entry.
        self.assertEqual(
            DecisionModel.provider_info("ollama"), LanguageModel.provider_info("ollama")
        )

    def test_every_supported_provider_has_info(self):
        for cls in (LanguageModel, EmbeddingModel, DecisionModel):
            for provider in cls.supported_providers():
                info = cls.provider_info(provider)
                self.assertEqual(info["provider"], provider)

    def test_info_is_a_copy(self):
        LanguageModel.provider_info("openai")["env"][0]["name"] = "changed"
        self.assertEqual(
            LanguageModel.provider_info("openai")["env"][0]["name"], "OPENAI_API_KEY"
        )

    def test_unsupported_provider_raises(self):
        with self.assertRaises(ValueError):
            EmbeddingModel.provider_info("anthropic")


class ModelInfoTest(testing.TestCase):
    def test_known_model(self):
        info = LanguageModel.model_info("openai/gpt-4o-mini")
        self.assertEqual(info["max_input_tokens"], 128000)
        self.assertIn("vision", info["capabilities"])
        self.assertIn("function_calling", info["capabilities"])

    def test_decision_model(self):
        info = DecisionModel.model_info("typesafe/jev-1.14.0")
        self.assertEqual(info["output_cost_per_token"], 0.0)
        self.assertIn("box_questions", info["capabilities"])

    def test_unprefixed_table_entry(self):
        # Bedrock models are keyed without their provider in litellm's table.
        info = LanguageModel.model_info("bedrock/ai21.jamba-1-5-large-v1:0")
        self.assertIsNotNone(info)

    def test_unknown_model(self):
        self.assertIsNone(LanguageModel.model_info("vllm/Qwen/Qwen3-4B"))
        # Another provider's model of the same name is not this one.
        self.assertIsNone(LanguageModel.model_info("groq/claude-sonnet-4-5"))
