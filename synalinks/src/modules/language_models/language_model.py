# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import copy
import logging
import os
import warnings

import litellm
import orjson
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatRole
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib
from synalinks.src.utils.nlp_utils import shorten_text

litellm.drop_params = True
litellm.disable_aiohttp_transport = True
litellm.drop_params = True


@synalinks_export(
    [
        "synalinks.LanguageModel",
        "synalinks.language_models.LanguageModel",
    ]
)
class LanguageModel(Module):
    """A language model API wrapper.

    A language model is a type of AI model designed to generate, and interpret human
    language. It is trained on large amounts of text data to learn patterns and
    structures in language. Language models can perform various tasks such as text
    generation, translation, summarization, and answering questions.

    We support providers that implement *constrained structured output*
    like Azure, Ollama or Mistral. In addition we support providers that otherwise
    allow to constrain the use of a specific tool like Groq or Anthropic.

    For the complete list of models, please refer to the providers documentation.

    **Using OpenAI models**

    ```python
    import synalinks
    import os

    os.environ["OPENAI_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="openai/gpt-4o-mini",
    )
    ```

    **Using Groq models**

    ```python
    import synalinks
    import os

    os.environ["GROQ_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="groq/llama3-8b-8192",
    )
    ```

    **Using Anthropic models**

    ```python
    import synalinks
    import os

    os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="anthropic/claude-3-sonnet-20240229",
    )
    ```

    **Using Mistral models**

    ```python
    import synalinks
    import os

    os.environ["MISTRAL_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="mistral/codestral-latest",
    )
    ```

    **Using Ollama models**

    ```python
    import synalinks
    import os

    language_model = synalinks.LanguageModel(
        model="ollama/mistral",
    )
    ```

    **Using Azure models**

    ```python
    import synalinks
    import os

    os.environ["AZURE_API_KEY"] = "your-api-key"
    os.environ["AZURE_API_BASE"] = "your-api-base"
    os.environ["AZURE_API_VERSION"] = "your-api-version"

    language_model = synalinks.LanguageModel(
        model="azure/<your_deployment_name>",
    )
    ```

    **Using Google Gemini models**

    ```python
    import synalinks
    import os

    os.environ["GEMINI_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="gemini/gemini-2.5-pro",
    )
    ```

    **Using XAI models**

    ```python
    import synalinks
    import os

    os.environ["XAI_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="xai/grok-code-fast-1",
    )
    ```

    **Using Cohere models**

    ```python
    import synalinks
    import os

    os.environ["COHERE_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="cohere/command-r-plus",
    )
    ```

    **Using DeepSeek models**

    ```python
    import synalinks
    import os

    os.environ["DEEPSEEK_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="deepseek/deepseek-chat",
    )
    ```

    **Using Together AI models**

    ```python
    import synalinks
    import os

    os.environ["TOGETHER_AI_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    )
    ```

    **Using OpenRouter models**

    ```python
    import synalinks
    import os

    os.environ["OPENROUTER_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="openrouter/anthropic/claude-3-haiku",
    )
    ```

    **Using AWS Bedrock models**

    ```python
    import synalinks
    import os

    os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-key"
    os.environ["AWS_REGION_NAME"] = "us-east-1"

    language_model = synalinks.LanguageModel(
        model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    )
    ```

    **Using Doubleword models**

    Doubleword exposes an OpenAI-compatible API. The `doubleword/`
    prefix is rewritten to `openai/` internally and `api_base` is
    defaulted to `https://api.doubleword.ai/v1`, so structured outputs
    flow through the standard OpenAI path. Set `OPENAI_API_KEY` to your
    Doubleword API key (or pass `api_base` explicitly to override).

    ```python
    import synalinks
    import os

    os.environ["OPENAI_API_KEY"] = "your-doubleword-api-key"

    language_model = synalinks.LanguageModel(
        model="doubleword/qwen-qwen3-5-397b-a17b-fp8-dottxt",
    )
    ```

    To cascade models in case there is anything wrong with
    the model provider (hence making your pipelines more robust).
    Use the `fallback` argument like in this example:

    ```python
    import synalinks
    import os

    os.environ["GEMINI_API_KEY"] = "your-api-key"
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="anthropic/claude-3-sonnet-20240229",
        fallback=synalinks.LanguageModel(
            model="gemini/gemini-3-flash-preview",
        )
    )
    ```

    **Note**: Obviously, use an `.env` file and `.gitignore` to avoid
    putting your API keys in the code or a config file that can lead to
    leackage when pushing it into repositories.

    Args:
        model (str): The model to use.
        api_base (str): Optional. The endpoint to use.
        timeout (int): Optional. The timeout value in seconds (Default to 600).
        retry (int): Optional. The number of retry (default to 5).
        fallback (LanguageModel): Optional. The language model to fallback
            if anything is wrong.
        caching (bool): Optional. Enable caching of LM calls (Default to False).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        hooks (list): Optional. Hooks to attach to this module's calls.
        **default_kwargs: Optional. Default generation parameters (e.g.
            `temperature`, `top_p`, `top_k`, `max_tokens`, `reasoning_effort`)
            forwarded to every call. Per-call kwargs override these.
    """

    def __init__(
        self,
        model=None,
        api_base=None,
        timeout=600,
        retry=5,
        fallback=None,
        caching=False,
        name=None,
        description=None,
        hooks=None,
        **default_kwargs: object,
    ):
        super().__init__(
            trainable=False,
            name=name,
            description=description,
            hooks=hooks,
        )
        # `messages` may be passed as a Pydantic DataModel; the strict
        # JsonDataModel guard would otherwise reject it.
        self._allow_non_json_data_model_positional_args = True
        if model is None:
            raise ValueError("You need to set the `model` argument for any LanguageModel")
        model_provider = model.split("/")[0]
        if model_provider == "ollama":
            # Switch from `ollama` to `ollama_chat`
            # because it have better performance due to the chat prompts
            model = model.replace("ollama", "ollama_chat")
        if model_provider == "vllm":
            model = model.replace("vllm", "hosted_vllm")
        if model_provider == "doubleword":
            # Doubleword is OpenAI-compatible (strict JSON schema + same
            # request/response shape) — route via litellm's `openai`
            # provider with the Doubleword endpoint as `api_base`.
            model = model.replace("doubleword", "openai", 1)
            if not api_base:
                api_base = "https://api.doubleword.ai/v1"
        self.model = model
        if fallback is not None:
            # Lazy import: `get` lives in the package __init__ which imports
            # this file at load time.
            from synalinks.src.modules.language_models import get as _get_lm

            fallback = _get_lm(fallback)
        self.fallback = fallback
        if self.model.startswith("ollama") and not api_base:
            self.api_base = "http://localhost:11434"
        else:
            self.api_base = api_base
        if self.model.startswith("hosted_vllm") and not api_base:
            self.api_base = os.environ.get(
                "HOSTED_VLLM_API_BASE", "http://localhost:8000"
            )
        self.timeout = timeout
        self.retry = retry
        self.caching = caching
        self.default_kwargs = default_kwargs
        self.cumulated_cost = 0.0
        self.last_call_cost = 0.0
        # No state depends on the input shape, so mark built up-front and
        # skip Module's auto-build path (which would try to trace `call`).
        self.built = True

    async def call(self, messages, schema=None, streaming=False, **kwargs):
        """
        Call method to generate a response using the language model.

        Args:
            messages (dict): A formatted dict of chat messages.
            schema (dict): The target JSON schema for structed output (optional).
                If None, output a ChatMessage-like answer.
            streaming (bool): Enable streaming (optional). Default to False.
                Can be enabled only if schema is None.
            **kwargs (keyword arguments): The additional keywords arguments
                forwarded to the LM call.
        Returns:
            (dict): The generated structured response.
        """
        formatted_messages = messages.get_json().get("messages", [])
        input_kwargs = copy.deepcopy(kwargs)
        # Merge instance-level defaults; per-call kwargs win.
        kwargs = {**self.default_kwargs, **kwargs}
        schema = copy.deepcopy(schema)
        provider = self.model.split("/")[0]

        # Handle reasoning_effort parameter - just forward to litellm if supported
        reasoning_effort = kwargs.pop("reasoning_effort", "none")
        schema_had_thinking = bool(schema) and "thinking" in (
            schema.get("properties") or {}
        )
        if reasoning_effort not in ("none", "disable"):
            if litellm.supports_reasoning(model=self.model):
                kwargs["reasoning_effort"] = reasoning_effort
                if schema_had_thinking:
                    # The LM produces a native reasoning trace via
                    # `reasoning_content` — strip `thinking` from the LM
                    # schema to save tokens; we re-inject it after the call.
                    schema["properties"].pop("thinking", None)
                    required = schema.get("required")
                    if isinstance(required, list) and "thinking" in required:
                        required.remove("thinking")

        if schema:
            if (
                self.model.startswith("groq")
                or self.model.startswith("cohere")
                or self.model.startswith("openrouter")
                or self.model.startswith("bedrock")
            ):
                # Use a tool created on the fly. These providers either
                # don't support native JSON schema (cohere, most bedrock
                # models) or proxy heterogeneous backends with mixed
                # support (openrouter), so tool-call structured output
                # is the most reliable path.
                kwargs.update(
                    {
                        "tools": [
                            {
                                "function": {
                                    "name": "structured_output",
                                    "description": "Generate a valid JSON output",
                                    "parameters": schema.get("properties"),
                                },
                                "type": "function",
                            }
                        ],
                        "tool_choice": {
                            "type": "function",
                            "function": {"name": "structured_output"},
                        },
                    }
                )
            elif self.model.startswith("anthropic"):
                # Use response_format for Anthropic - LiteLLM handles this correctly:
                # - For newer models (sonnet-4.5, opus-4.1): uses native output_format
                # - For older models: uses tool call with proper tool_choice handling
                #   (auto when thinking is enabled, forced otherwise)
                kwargs.update(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "schema": schema,
                            },
                        },
                    }
                )
            elif self.model.startswith("ollama") or self.model.startswith("mistral"):
                # Use constrained structured output for ollama/mistral
                kwargs.update(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {"schema": schema},
                            "strict": True,
                        },
                    }
                )
            elif (
                self.model.startswith("openai")
                or self.model.startswith("azure")
                or self.model.startswith("deepseek")
                or self.model.startswith("together_ai")
            ):
                # Use constrained structured output for openai/azure
                # plus deepseek and together_ai which expose
                # OpenAI-compatible APIs that honor the same payload.
                # OpenAI/Azure require the field  "additionalProperties"
                # Also OpenAI/Azure disallow the field "description" in $ref
                if "properties" in schema:
                    for prop_key, prop_value in schema["properties"].items():
                        if "$ref" in prop_value and "description" in prop_value:
                            del prop_value["description"]
                kwargs.update(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_output",
                                "strict": True,
                                "schema": schema,
                            },
                        }
                    }
                )
            elif self.model.startswith("gemini"):
                kwargs.update(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "schema": schema,
                            },
                            "strict": True,
                        }
                    }
                )
            elif self.model.startswith("xai"):
                kwargs.update(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "schema": schema,
                            },
                            "strict": True,
                        }
                    }
                )
            elif self.model.startswith("hosted_vllm"):
                kwargs.update(
                    {
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "structured_output",
                                "schema": schema,
                            },
                            "strict": True,
                        }
                    }
                )
            else:
                provider = self.model.split("/")[0]
                raise ValueError(
                    f"LM provider '{provider}' not supported yet, please ensure that"
                    " they support constrained structured output and fill an issue."
                )

        if self.api_base:
            kwargs.update(
                {
                    "api_base": self.api_base,
                }
            )
        if streaming and schema:
            streaming = False
        if streaming:
            kwargs.update({"stream": True})
        # Enable prompt caching for the system instructions
        # (that only change during training not inference)
        if provider in ("gemini", "anthropic"):
            system_message_with_cache_control = {
                **formatted_messages[0],
                "cache_control": {"type": "ephemeral"},
            }
            formatted_messages[0] = system_message_with_cache_control
        try:
            return await self._call_with_retry(
                formatted_messages,
                schema,
                streaming,
                schema_had_thinking,
                **kwargs,
            )
        except Exception as e:
            warnings.warn(f"All retries failed for {self}: {e}")
            if self.fallback:
                return await self.fallback(
                    messages,
                    schema=schema,
                    streaming=streaming,
                    **input_kwargs,
                )
            else:
                return None

    async def _call_with_retry(
        self, formatted_messages, schema, streaming, schema_had_thinking, **kwargs
    ):
        """Perform the LM call with tenacity retry logic."""
        logger = logging.getLogger(__name__)

        @retry(
            stop=stop_after_attempt(self.retry),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _do_call():
            response_str = ""
            try:
                response = await litellm.acompletion(
                    model=self.model,
                    messages=formatted_messages,
                    timeout=self.timeout,
                    caching=self.caching,
                    **kwargs,
                )
                if hasattr(response, "_hidden_params"):
                    if "response_cost" in response._hidden_params:
                        response_cost = response._hidden_params["response_cost"]
                        if response_cost is not None:
                            self.last_call_cost = response_cost
                            self.cumulated_cost += response_cost
                if streaming:
                    return StreamingIterator(response)
                if not response.get("choices"):
                    raise ValueError(
                        "Empty response from the language model: no choices returned."
                    )
                response_message = response["choices"][0]["message"]
                if self.model.startswith("groq") and schema:
                    # Groq uses tool_calls for structured output
                    response_str = response_message["tool_calls"][0]["function"][
                        "arguments"
                    ]
                else:
                    # Anthropic and other providers use response_format,
                    # which returns content in message["content"]
                    response_str = response_message["content"]
                    if not response_str:
                        raise ValueError(
                            "Empty response from the language model: no content returned."
                        )
                    response_str = response_str.strip()
                reasoning_content = response_message.get("reasoning_content")
                if schema:
                    json_instance = orjson.loads(response_str)
                    if reasoning_content and schema_had_thinking:
                        json_instance["thinking"] = reasoning_content
                else:
                    json_instance = {
                        "role": ChatRole.ASSISTANT,
                        "thinking": reasoning_content,
                        "content": response_str,
                        "tool_call_id": None,
                        "tool_calls": [],
                        "created_at": None,
                    }
                return json_instance
            except Exception as e:
                warnings.warn(
                    f"Error occured while trying to call {self}: "
                    + str(e)
                    + f"\nReceived response={shorten_text(response_str)}"
                )
                raise

        return await _do_call()

    def _obj_type(self):
        return "LanguageModel"

    def get_config(self):
        config = {
            "model": self.model,
            "api_base": self.api_base,
            "timeout": self.timeout,
            "retry": self.retry,
            "caching": self.caching,
            "name": self.name,
            "description": self.description,
            **self.default_kwargs,
        }
        if self.fallback:
            fallback_config = {
                "fallback": serialization_lib.serialize_synalinks_object(
                    self.fallback,
                )
            }
            return {**fallback_config, **config}
        else:
            return config

    @classmethod
    def from_config(cls, config):
        if "fallback" in config:
            fallback = serialization_lib.deserialize_synalinks_object(
                config.pop("fallback")
            )
            return cls(fallback=fallback, **config)
        else:
            return cls(**config)

    def __repr__(self):
        api_base = f" api_base={self.api_base}" if self.api_base else ""
        return f"<LanguageModel model={self.model}{api_base}>"


class StreamingIterator:
    """Async iterator over LM stream chunks.

    Wraps litellm's `CustomStreamWrapper` (which is async-iterable via
    `__aiter__`/`__anext__`) and yields one normalized dict per
    non-empty chunk: `{"role": "assistant", "thinking": ..., "content": ...}`.
    Chunks containing only role/finish markers are skipped so reasoning-only
    deltas don't terminate the stream prematurely.

    Also accepts a plain sync iterator — useful for tests that mock
    `litellm.acompletion`.
    """

    def __init__(self, iterator):
        self._iterator = iterator
        self._is_async = hasattr(iterator, "__anext__") or hasattr(iterator, "__aiter__")

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            try:
                if self._is_async:
                    chunk = await self._iterator.__anext__()
                else:
                    chunk = next(self._iterator)
            except (StopAsyncIteration, StopIteration):
                raise StopAsyncIteration
            delta = chunk["choices"][0].get("delta") or {}
            content = delta.get("content")
            thinking = delta.get("reasoning_content")
            if content or thinking:
                out = {"role": ChatRole.ASSISTANT}
                if thinking:
                    out["thinking"] = thinking
                if content:
                    out["content"] = content
                return out
