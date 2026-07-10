# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import copy
import logging
import os
import time
import warnings

import litellm
import orjson
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import stop_after_attempt

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import ChatMessages
from synalinks.src.backend import ChatRole
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend.common.op_scope import current_op_scope
from synalinks.src.backend.common.op_scope import current_trajectory_start
from synalinks.src.backend.pydantic.chat_completions import to_chat_completion_message
from synalinks.src.backend.pydantic.media import resolve_content_media
from synalinks.src.modules.core.tool import Tool
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib
from synalinks.src.utils.file_cache import FileCache
from synalinks.src.utils.nlp_utils import shorten_text
from synalinks.src.utils.retry_utils import rate_limit_aware_wait

litellm.drop_params = True
litellm.disable_aiohttp_transport = True
litellm.drop_params = True


def _safe_get(obj, key, default=None):
    """Get `key` from obj that may be a dict or a pydantic-style object."""
    if obj is None:
        return default
    if hasattr(obj, "get"):
        v = obj.get(key)
    else:
        v = getattr(obj, key, None)
    return default if v is None else v


def _to_int(v):
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


# Long-tail Usage fields we sum into `<phase>_cumulated_details` for
# introspection (no dedicated metric — read via `lm.<phase>_cumulated_details`).
_PROMPT_DETAIL_KEYS = (
    "audio_tokens",
    "text_tokens",
    "image_tokens",
    "video_tokens",
    "web_search_requests",
    "character_count",
    "image_count",
    "video_length_seconds",
)
_COMPLETION_DETAIL_KEYS = (
    "audio_tokens",
    "text_tokens",
    "image_tokens",
    "video_tokens",
    "accepted_prediction_tokens",
    "rejected_prediction_tokens",
)
_SERVER_TOOL_KEYS = ("web_search_requests", "tool_search_requests")
_CACHE_CREATION_TTL_KEYS = (
    "ephemeral_5m_input_tokens",
    "ephemeral_1h_input_tokens",
)


def _extract_lm_extras(usage, response):
    """Pull tier-1 counters + long-tail details from a LiteLLM ModelResponse.

    Returns ``(cached_tokens, cache_creation_tokens, reasoning_tokens, extras)``
    where ``extras`` is a dict of long-tail key -> incremental count.
    """
    prompt_details = _safe_get(usage, "prompt_tokens_details")
    completion_details = _safe_get(usage, "completion_tokens_details")
    server_tool = _safe_get(usage, "server_tool_use")
    hidden = getattr(response, "_hidden_params", None) or {}

    cached = _to_int(_safe_get(prompt_details, "cached_tokens"))
    cache_creation = _to_int(_safe_get(prompt_details, "cache_creation_tokens"))
    reasoning = _to_int(_safe_get(completion_details, "reasoning_tokens"))

    extras = {}
    for key in _PROMPT_DETAIL_KEYS:
        v = _safe_get(prompt_details, key)
        if v:
            extras[f"prompt_{key}"] = v
    cache_create_ttl = _safe_get(prompt_details, "cache_creation_token_details")
    for key in _CACHE_CREATION_TTL_KEYS:
        v = _safe_get(cache_create_ttl, key)
        if v:
            extras[f"cache_creation_{key}"] = v
    for key in _COMPLETION_DETAIL_KEYS:
        v = _safe_get(completion_details, key)
        if v:
            extras[f"completion_{key}"] = v
    for key in _SERVER_TOOL_KEYS:
        v = _safe_get(server_tool, key)
        if v:
            extras[f"server_{key}"] = v
    overhead = (
        hidden.get("litellm_overhead_time_ms") if isinstance(hidden, dict) else None
    )
    if overhead is not None:
        extras["litellm_overhead_time_ms"] = overhead
    return cached, cache_creation, reasoning, extras


def _accumulate(obj, phase_prefix, increments, extras):
    """Bump ``obj.{phase_prefix}cumulated_<suffix>`` by each ``increments[suffix]``
    and merge ``extras`` into ``obj.{phase_prefix}cumulated_details``.
    """
    for suffix, delta in increments.items():
        attr = f"{phase_prefix}cumulated_{suffix}"
        setattr(obj, attr, getattr(obj, attr) + delta)
    if extras:
        details_attr = f"{phase_prefix}cumulated_details"
        details = getattr(obj, details_attr, None)
        if details is None:
            details = {}
            setattr(obj, details_attr, details)
        for k, v in extras.items():
            details[k] = details.get(k, 0) + v


def _message_to_wire(message):
    """Convert a synalinks ChatMessage to an OpenAI Chat Completions
    wire-format dict.

    Delegates to `to_chat_completion_message`, the single source of truth
    for the synalinks<->OpenAI mapping (tool-call envelope wrapping with
    `arguments` JSON-encoded, dict content JSON-encoded on tool results,
    and `thinking`/`thinking_blocks` mapped onto `reasoning_content`/
    `thinking_blocks` for DeepSeek- and Anthropic-style providers). Unset
    fields are dropped, but `content` is always emitted (as null when
    empty) since some providers expect the key to be present.
    """
    wire = to_chat_completion_message(message).model_dump(exclude_none=True)
    wire.setdefault("content", None)
    return wire


def _tool_to_wire(tool):
    """Convert a synalinks Tool to an OpenAI Chat Completions wire-format
    tool declaration dict."""
    if not isinstance(tool, Tool):
        raise TypeError(f"Expected synalinks.modules.Tool, got {type(tool).__name__}")
    function = {"name": tool.name, "parameters": tool.get_tool_schema()}
    if tool.description:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _cached_message_to_chunks(json_instance):
    """Turn a cached assistant ChatMessage back into wire-shaped stream
    chunks, so a `cache_dir` hit on a `streaming=True` call can be replayed
    through a `StreamingIterator` (as a single chunk carrying the full
    content and reasoning)."""
    delta = {}
    if json_instance.get("reasoning_content"):
        delta["reasoning_content"] = json_instance["reasoning_content"]
    if json_instance.get("content"):
        delta["content"] = json_instance["content"]
    return iter([{"choices": [{"delta": delta}]}])


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
        model="gemini/gemini-3.1-flash-lite-preview",
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
            model="gemini/gemini-3.1-flash-lite-preview",
        )
    )
    ```

    **Note**: Obviously, use an `.env` file and `.gitignore` to avoid
    putting your API keys in the code or a config file that can lead to
    leackage when pushing it into repositories.

    **Controlling reasoning effort**

    Reasoning ("thinking") models can be steered with the `reasoning_effort`
    parameter, passed either as a default in the constructor or per call. It
    accepts:

    - `"none"` (the default): send nothing, leaving the model at its provider
      default reasoning behavior.
    - `"disable"`: actively turn native reasoning *off*. This only has an
      effect for providers that reason by default, i.e. ollama's thinking
      models (`qwen3`, `deepseek-r1`, ...); it maps to ollama's `think=False`
      toggle and is safely ignored by non-thinking ollama models. Opt-in
      providers (OpenAI, Anthropic, Gemini, ...) reason only when explicitly
      enabled, so there is nothing to send and this value is a no-op for them.
    - any other value (e.g. `"low"`, `"medium"`, `"high"`): forwarded to the
      provider as the reasoning effort, but only when the model supports
      reasoning (otherwise it is silently dropped).

    ```python
    import synalinks

    language_model = synalinks.LanguageModel(
        model="openai/o3-mini",
        reasoning_effort="medium",
    )
    ```

    Args:
        model (str): The model to use.
        api_base (str): Optional. The endpoint to use.
        timeout (int): Optional. The timeout value in seconds (Default to 600).
        retry (int): Optional. The number of retry (default to 5).
        retry_max_wait (int): Optional. Max seconds to wait between retries when a
            rate-limit `Retry-After` header is honored (default to 60).
        fallback (LanguageModel): Optional. The language model to fallback
            if anything is wrong.
        caching (bool): Optional. Enable caching of LM calls (Default to False).
        cache_dir (str): Optional. Directory for a persistent on-disk response
            cache. When set, every successful response is saved as a JSON
            file keyed by the full request (model, messages, schema and
            generation parameters), and identical requests are answered from
            disk without calling the provider — including across runs and
            processes. Streamed responses are cached once the stream is fully
            consumed (an abandoned stream is not cached) and a cache hit on a
            streaming call is replayed as a stream. (Default to None,
            disabled).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        hooks (list): Optional. Hooks to attach to this module's calls.
        **default_kwargs: Optional. Default generation parameters (e.g.
            `temperature`, `top_p`, `top_k`, `max_tokens`, `reasoning_effort`)
            forwarded to every call. Per-call kwargs override these. See
            "Controlling reasoning effort" above for the `reasoning_effort`
            values and their semantics.
    """

    def __init__(
        self,
        model=None,
        api_base=None,
        timeout=600,
        retry=5,
        retry_max_wait=60,
        fallback=None,
        caching=False,
        cache_dir=None,
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
            raise ValueError(
                "You need to set the `model` argument for any LanguageModel"
            )
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
        self.retry_max_wait = retry_max_wait
        self.caching = caching
        self.cache_dir = cache_dir
        self._file_cache = FileCache(cache_dir) if cache_dir else None
        self.default_kwargs = default_kwargs
        self.cumulated_cost = 0.0
        self.last_call_cost = 0.0
        # All-time counters across every LM call (training + inference).
        # Useful for raw debugging; operational metrics use the
        # inference-scoped counters below instead.
        self.cumulated_calls = 0
        self.cumulated_prompt_tokens = 0
        self.cumulated_completion_tokens = 0
        self.cumulated_tokens = 0
        self.cumulated_elapsed_s = 0.0
        self.cumulated_cached_tokens = 0
        self.cumulated_cache_creation_tokens = 0
        self.cumulated_reasoning_tokens = 0
        # Streaming latency counters: number of streamed calls and the summed
        # time-to-first-token / time-to-last-token (seconds, measured from the
        # provider request start). Averaged by the streaming metrics. Only
        # populated for `streaming=True` calls.
        self.cumulated_streaming_calls = 0
        self.cumulated_streaming_ttft_s = 0.0
        self.cumulated_streaming_ttlt_s = 0.0
        # Whole-trajectory time-to-first-token: for a streamed call made inside
        # an agent trajectory, the wall-clock from the (outermost) agent's start
        # to the first token of the final answer -- includes every tool-calling
        # round, unlike `streaming_ttft_s` which times the final LM call only.
        # Only populated when a `trajectory_scope` is active (see op_scope).
        self.cumulated_trajectory_calls = 0
        self.cumulated_trajectory_ttft_s = 0.0
        # Failure counters: a call that exhausts all retries (`failed_calls`)
        # and each time the `fallback` chain is invoked because of it
        # (`fallback_activations`). Successful calls bump `cumulated_calls`;
        # these are tracked separately so error rate is observable.
        self.cumulated_failed_calls = 0
        self.cumulated_fallback_activations = 0
        # Number of calls answered from the on-disk `cache_dir` cache
        # (no provider request, no token counters bumped).
        self.cumulated_cache_hits = 0
        self.cumulated_details = {}
        self.last_call_prompt_tokens = 0
        self.last_call_completion_tokens = 0
        self.last_call_tokens = 0
        self.last_call_elapsed_s = 0.0
        self.last_call_cached_tokens = 0
        self.last_call_cache_creation_tokens = 0
        self.last_call_reasoning_tokens = 0
        # Phase-scoped counters — populated based on `synalinks_op_scope` set
        # by the trainer: "inference" inside `predict_on_batch`, "reward"
        # inside `compute_reward`, "optimizer" inside `optimizer.optimize`.
        # Calls made outside any scope (e.g. standalone debugging) are
        # tracked only in the all-time `cumulated_*` set above.
        #
        # Tier 1 extras (first-class, drive dedicated KPI metrics):
        #   cached_tokens, cache_creation_tokens, reasoning_tokens.
        # Tier 2 long tail (multimodal split, tool use, LiteLLM overhead)
        # lives in `<phase>_cumulated_details` — a dict accumulated per call.
        for _phase in ("inference", "reward", "optimizer"):
            setattr(self, f"{_phase}_cumulated_calls", 0)
            setattr(self, f"{_phase}_cumulated_prompt_tokens", 0)
            setattr(self, f"{_phase}_cumulated_completion_tokens", 0)
            setattr(self, f"{_phase}_cumulated_tokens", 0)
            setattr(self, f"{_phase}_cumulated_elapsed_s", 0.0)
            setattr(self, f"{_phase}_cumulated_cost", 0.0)
            setattr(self, f"{_phase}_cumulated_cached_tokens", 0)
            setattr(self, f"{_phase}_cumulated_cache_creation_tokens", 0)
            setattr(self, f"{_phase}_cumulated_reasoning_tokens", 0)
            setattr(self, f"{_phase}_cumulated_failed_calls", 0)
            setattr(self, f"{_phase}_cumulated_fallback_activations", 0)
            setattr(self, f"{_phase}_cumulated_cache_hits", 0)
            setattr(self, f"{_phase}_cumulated_streaming_calls", 0)
            setattr(self, f"{_phase}_cumulated_streaming_ttft_s", 0.0)
            setattr(self, f"{_phase}_cumulated_streaming_ttlt_s", 0.0)
            setattr(self, f"{_phase}_cumulated_trajectory_calls", 0)
            setattr(self, f"{_phase}_cumulated_trajectory_ttft_s", 0.0)
            setattr(self, f"{_phase}_cumulated_details", {})
        # No state depends on the input shape, so mark built up-front and
        # skip Module's auto-build path (which would try to trace `call`).
        self.built = True

    def _record_event(self, suffix):
        """Bump an all-time + phase-scoped operational counter by 1 (used for
        `failed_calls` / `fallback_activations`). Phase follows the active
        `op_scope`, matching how successful-call counters are attributed.
        """
        op = current_op_scope()
        _accumulate(self, "", {suffix: 1}, None)
        if op is not None:
            _accumulate(self, f"{op}_", {suffix: 1}, None)

    async def call(
        self,
        messages,
        schema=None,
        tools=None,
        tool_schemas=None,
        streaming=False,
        **kwargs,
    ):
        """
        Call method to generate a response using the language model.

        Args:
            messages (dict): A formatted dict of chat messages.
            schema (dict): The target JSON schema for structed output (optional).
                If None, output a ChatMessage-like answer.
            tools (list | dict): Optional iterable or `{name: Tool}` mapping of
                `synalinks.modules.Tool` the LM may call. Mutually exclusive
                with `schema` — schema forces structured output, tools let the
                LM choose; they cannot both apply to the same call. In the
                function-calling agent pattern, the tool-call generator uses
                `tools` and the final generator uses `schema`.
            tool_schemas (list): Optional iterable of already-wire-formatted
                tool declaration dicts (OpenAI `{"type": "function", ...}`
                shape) passed through verbatim, appended after any converted
                `tools`. Use this to expose tools you already have a schema for
                without wrapping them in a `synalinks.modules.Tool`. Subject to
                the same mutual exclusivity with `schema` as `tools`.
            streaming (bool): Enable streaming (optional). Default to False.
                Can be enabled only if schema is None.
            **kwargs (keyword arguments): The additional keywords arguments
                forwarded to the LM call.
        Returns:
            (dict): The generated structured response.
        """
        if not messages:
            return None
        if schema and (tools or tool_schemas):
            raise ValueError(
                "`schema` and `tools` cannot be passed to the same LM call: "
                "schema forces structured output, while tools let the LM choose "
                "which to call. Split into two calls — typically the tool-call "
                "generator uses `tools` and the final generator uses `schema`."
            )
        input_kwargs = copy.deepcopy(kwargs)
        # Merge instance-level defaults; per-call kwargs win.
        kwargs = {**self.default_kwargs, **kwargs}
        schema = copy.deepcopy(schema)
        provider = self.model.split("/")[0]

        # Single pass: messages → OpenAI wire shape (nested tool_call
        # envelopes, JSON-string arguments) and synalinks Tools → wire tool
        # declarations. The schema branches below may override `tools` /
        # `tool_choice` for providers that need structured-output-as-tool;
        # cache_control is applied to the system message after this.
        # `messages` here is typically a JsonDataModel from `ops.predict`;
        # wrap it in `ChatMessages` so its `before`-validator converts the
        # dict list into typed `ChatMessage` instances the converter needs.
        _typed_messages = (
            messages
            if hasattr(messages, "messages")
            else ChatMessages(messages=messages.get("messages", []))
        )
        formatted_messages = [_message_to_wire(m) for m in _typed_messages.messages]
        # Inline any deferred multimodal references (a dataset's image_url/
        # input_audio parts carrying a url/path) into base64 payloads, for this
        # batch only. Content already inlined at construction is left untouched.
        formatted_messages = await resolve_content_media(formatted_messages)
        if tools or tool_schemas:
            wire_tools = []
            if tools:
                if isinstance(tools, dict):
                    tools = tools.values()
                wire_tools.extend(_tool_to_wire(t) for t in tools)
            if tool_schemas:
                # Already in OpenAI wire shape — pass through verbatim.
                wire_tools.extend(tool_schemas)
            kwargs["tools"] = wire_tools

        # Handle reasoning_effort:
        #   "none"    -> send nothing; leave the model at its provider default.
        #   "disable" -> actively turn native reasoning OFF. This only needs an
        #                explicit flag where reasoning is ON by default, i.e.
        #                ollama's thinking models (qwen3, deepseek-r1, ...), which
        #                reason unless told not to. `think=False` is the ollama
        #                toggle and is safely ignored by non-thinking ollama
        #                models. Opt-in providers (OpenAI, Anthropic, Gemini, ...)
        #                reason only when enabled, so there is nothing to send.
        #   otherwise -> forward the effort to litellm when the model supports it.
        reasoning_effort = kwargs.pop("reasoning_effort", "none")
        schema_had_thinking = bool(schema) and "thinking" in (
            schema.get("properties") or {}
        )
        if reasoning_effort == "disable":
            if self.model.startswith("ollama"):
                kwargs["think"] = False
        elif reasoning_effort != "none":
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
        # Persistent on-disk cache: keyed by the fully-formatted request
        # (model, wire messages, schema and final kwargs), so any change to
        # prompts, tools or generation parameters yields a different entry.
        # The `stream` flag is excluded from the key: a streamed call stores
        # the same assistant ChatMessage a non-streamed schema-less call does
        # (streaming forces schema=None), so both share one entry — a hit is
        # replayed as a single-chunk stream when `streaming=True`.
        cache_key = None
        if self._file_cache is not None:
            cache_key = self._file_cache.make_key(
                {
                    "model": self.model,
                    "messages": formatted_messages,
                    "schema": schema,
                    "kwargs": {k: v for k, v in kwargs.items() if k != "stream"},
                }
            )
            if cache_key is not None:
                cached_json = self._file_cache.get(cache_key)
                if cached_json is not None:
                    self._record_event("cache_hits")
                    if streaming:
                        return StreamingIterator(_cached_message_to_chunks(cached_json))
                    return JsonDataModel(
                        json=cached_json,
                        schema=schema if schema else ChatMessage.get_schema(),
                        name=f"{self.name}_response",
                    )
        try:
            response = await self._call_with_retry(
                formatted_messages,
                schema,
                streaming,
                schema_had_thinking,
                **kwargs,
            )
            if cache_key is not None and response is not None:
                if streaming:
                    # Cache once the caller drains the stream; an abandoned
                    # stream is incomplete and is not cached.
                    file_cache = self._file_cache
                    response._on_complete = lambda json_instance: file_cache.set(
                        cache_key, json_instance
                    )
                else:
                    self._file_cache.set(cache_key, response.get_json())
            return response
        except Exception as e:
            warnings.warn(f"All retries failed for {self}: {e}")
            self._record_event("failed_calls")
            if self.fallback:
                self._record_event("fallback_activations")
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
            # Honor a rate-limit `Retry-After` header (e.g. Azure OpenAI TPM
            # throttling); fall back to exponential backoff for other errors.
            wait=rate_limit_aware_wait(max_wait=self.retry_max_wait),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _do_call():
            response_str = ""
            try:
                t0 = time.perf_counter()
                response = await litellm.acompletion(
                    model=self.model,
                    messages=formatted_messages,
                    timeout=self.timeout,
                    caching=self.caching,
                    **kwargs,
                )
                elapsed_s = time.perf_counter() - t0
                op_scope = current_op_scope()
                response_cost = None
                if hasattr(response, "_hidden_params"):
                    if "response_cost" in response._hidden_params:
                        response_cost = response._hidden_params["response_cost"]
                        if response_cost is not None:
                            self.last_call_cost = response_cost
                # Streaming usage isn't known until the stream completes,
                # so skip counter updates in that case.
                if not streaming:
                    usage = response.get("usage") or {}
                    prompt_tokens = int(usage.get("prompt_tokens") or 0)
                    completion_tokens = int(usage.get("completion_tokens") or 0)
                    total_tokens = int(
                        usage.get("total_tokens") or (prompt_tokens + completion_tokens)
                    )
                    cached, cache_creation, reasoning, extras = _extract_lm_extras(
                        usage, response
                    )
                    self.last_call_prompt_tokens = prompt_tokens
                    self.last_call_completion_tokens = completion_tokens
                    self.last_call_tokens = total_tokens
                    self.last_call_elapsed_s = elapsed_s
                    self.last_call_cached_tokens = cached
                    self.last_call_cache_creation_tokens = cache_creation
                    self.last_call_reasoning_tokens = reasoning
                    flat_increments = {
                        "calls": 1,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "tokens": total_tokens,
                        "elapsed_s": elapsed_s,
                        "cached_tokens": cached,
                        "cache_creation_tokens": cache_creation,
                        "reasoning_tokens": reasoning,
                    }
                    if response_cost is not None:
                        flat_increments["cost"] = response_cost
                    _accumulate(self, "", flat_increments, extras)
                    if op_scope is not None:
                        _accumulate(self, f"{op_scope}_", flat_increments, extras)
                if streaming:
                    return StreamingIterator(
                        response,
                        language_model=self,
                        op_scope=op_scope,
                        request_start=t0,
                        trajectory_start=current_trajectory_start(),
                    )
                if not response.get("choices"):
                    raise ValueError(
                        "Empty response from the language model: no choices returned."
                    )
                response_message = response["choices"][0]["message"]
                wire_tool_calls = _safe_get(response_message, "tool_calls", None)
                if self.model.startswith("groq") and schema:
                    # Groq uses tool_calls for structured output
                    response_str = response_message["tool_calls"][0]["function"][
                        "arguments"
                    ]
                else:
                    # Anthropic and other providers use response_format,
                    # which returns content in message["content"]
                    response_str = response_message["content"]
                    if not response_str and not wire_tool_calls:
                        raise ValueError(
                            "Empty response from the language model: no content "
                            "or tool_calls returned."
                        )
                    response_str = response_str.strip() if response_str else ""
                reasoning_content = response_message.get("reasoning_content")
                thinking_blocks = response_message.get("thinking_blocks")
                if schema:
                    json_instance = orjson.loads(response_str)
                    if reasoning_content and schema_had_thinking:
                        json_instance["thinking"] = reasoning_content
                else:
                    # Parse OpenAI's nested tool-call envelope into the
                    # synalinks `ToolCall` shape (`{id, type, function:
                    # {name, arguments}}`), decoding the JSON `arguments`
                    # string back into a dict.
                    parsed_tool_calls = None
                    if wire_tool_calls:
                        parsed_tool_calls = []
                        for tc in wire_tool_calls:
                            fn = _safe_get(tc, "function")
                            args = _safe_get(fn, "arguments", "")
                            parsed_tool_calls.append(
                                {
                                    "id": _safe_get(tc, "id"),
                                    "type": "function",
                                    "function": {
                                        "name": _safe_get(fn, "name"),
                                        "arguments": orjson.loads(args) if args else {},
                                    },
                                }
                            )
                    json_instance = {
                        "role": ChatRole.ASSISTANT,
                        "content": response_str,
                    }
                    if reasoning_content:
                        json_instance["reasoning_content"] = reasoning_content
                    if thinking_blocks:
                        json_instance["thinking_blocks"] = thinking_blocks
                    if parsed_tool_calls:
                        json_instance["tool_calls"] = parsed_tool_calls
                return JsonDataModel(
                    json=json_instance,
                    schema=schema if schema else ChatMessage.get_schema(),
                    name=f"{self.name}_response",
                )
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
            "retry_max_wait": self.retry_max_wait,
            "caching": self.caching,
            "cache_dir": self.cache_dir,
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
    non-empty chunk: `{"role": "assistant", "reasoning_content": ..., "content": ...}`.
    Chunks containing only role/finish markers are skipped so reasoning-only
    deltas don't terminate the stream prematurely.

    Also accepts a plain sync iterator — useful for tests that mock
    `litellm.acompletion`.

    When `language_model` and `request_start` are provided, the iterator times
    each consumed stream and, once exhausted, records two latencies onto the
    LM's operational counters (all-time + the phase captured at request time):
    time-to-first-token (request start -> first non-empty chunk) and
    time-to-last-token (request start -> final non-empty chunk). These drive
    the `AvgTimeToFirstToken` / `AvgTimeToLastToken` metrics. A stream that the
    caller abandons before exhaustion is not recorded (no terminal timing).

    When `trajectory_start` is also provided (an agent set a `trajectory_scope`),
    a third latency is recorded: whole-trajectory time-to-first-token, measured
    from the outermost agent's start (so it includes every tool-calling round)
    to the first token of the final answer. This drives
    `AvgTrajectoryTimeToFirstToken`.
    """

    def __init__(
        self,
        iterator,
        *,
        language_model=None,
        op_scope=None,
        request_start=None,
        trajectory_start=None,
    ):
        self._iterator = iterator
        self._is_async = hasattr(iterator, "__anext__") or hasattr(
            iterator, "__aiter__"
        )
        self._language_model = language_model
        self._op_scope = op_scope
        self._request_start = request_start
        self._trajectory_start = trajectory_start
        self._first_token_time = None
        self._last_token_time = None
        self._recorded = False
        # Accumulates every yielded token so the full assistant message can
        # be handed to `_on_complete` (the on-disk cache writer) once the
        # stream is drained. An abandoned stream never fires the callback.
        self._content_parts = []
        self._reasoning_parts = []
        self._on_complete = None

    def __aiter__(self):
        return self

    def _fire_on_complete(self):
        """Hand the fully-accumulated assistant message to `_on_complete`
        (at most once, and only when the stream actually yielded tokens)."""
        callback = self._on_complete
        if callback is None or not (self._content_parts or self._reasoning_parts):
            return
        self._on_complete = None
        json_instance = {
            "role": ChatRole.ASSISTANT,
            "content": "".join(self._content_parts),
        }
        if self._reasoning_parts:
            json_instance["reasoning_content"] = "".join(self._reasoning_parts)
        callback(json_instance)

    def _record_latencies(self):
        """Accumulate time-to-first / time-to-last token onto the LM counters.

        Idempotent and a no-op when no token was ever yielded or no timing
        context was supplied (e.g. the test-only sync-iterator path).
        """
        if self._recorded:
            return
        self._recorded = True
        lm = self._language_model
        if lm is None or self._request_start is None or self._first_token_time is None:
            return
        increments = {
            "streaming_calls": 1,
            "streaming_ttft_s": self._first_token_time - self._request_start,
            "streaming_ttlt_s": self._last_token_time - self._request_start,
        }
        # Whole-trajectory TTFT, only when this streamed call ran inside an agent
        # trajectory: time from the (outermost) agent's start to the first token.
        if self._trajectory_start is not None:
            increments["trajectory_calls"] = 1
            increments["trajectory_ttft_s"] = (
                self._first_token_time - self._trajectory_start
            )
        _accumulate(lm, "", increments, None)
        if self._op_scope is not None:
            _accumulate(lm, f"{self._op_scope}_", increments, None)

    async def __anext__(self):
        while True:
            try:
                if self._is_async:
                    chunk = await self._iterator.__anext__()
                else:
                    chunk = next(self._iterator)
            except (StopAsyncIteration, StopIteration):
                self._record_latencies()
                self._fire_on_complete()
                raise StopAsyncIteration
            delta = chunk["choices"][0].get("delta") or {}
            content = delta.get("content")
            thinking = delta.get("reasoning_content")
            if content or thinking:
                now = time.perf_counter()
                if self._first_token_time is None:
                    self._first_token_time = now
                self._last_token_time = now
                out = {"role": ChatRole.ASSISTANT}
                if thinking:
                    out["reasoning_content"] = thinking
                    self._reasoning_parts.append(thinking)
                if content:
                    out["content"] = content
                    self._content_parts.append(content)
                return out

    async def aconsume(self, name=None):
        """Drain the stream to exhaustion and return a concrete assistant
        `ChatMessage` `JsonDataModel`.

        The result is the same shape a non-streamed schema-less LM call
        returns (role `assistant`, joined `content`, joined `reasoning_content`
        when present), so a drained stream is indistinguishable downstream from
        an ordinary prediction -- which is what lets the batched predict /
        evaluate loops stream and still score the output. Exhausting the stream
        here also records the time-to-first / time-to-last token latencies.
        """
        content_parts = []
        reasoning_parts = []
        async for chunk in self:
            content = chunk.get("content")
            reasoning = chunk.get("reasoning_content")
            if content:
                content_parts.append(content)
            if reasoning:
                reasoning_parts.append(reasoning)
        json_instance = {
            "role": ChatRole.ASSISTANT,
            "content": "".join(content_parts),
        }
        if reasoning_parts:
            json_instance["reasoning_content"] = "".join(reasoning_parts)
        return JsonDataModel(
            json=json_instance,
            schema=ChatMessage.get_schema(),
            name=name,
        )
