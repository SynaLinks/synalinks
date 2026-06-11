# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import copy
import logging
import time
import warnings

import litellm
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import stop_after_attempt

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import Embeddings
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend.common.op_scope import current_op_scope
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib
from synalinks.src.utils.retry_utils import rate_limit_aware_wait

litellm.disable_aiohttp_transport = True


def _safe_get(obj, key, default=None):
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


# Embedding-specific long-tail keys (smaller than the LM set — embeddings
# have no completion phase, no reasoning, no tool use).
_EM_PROMPT_DETAIL_KEYS = (
    "audio_tokens",
    "text_tokens",
    "image_tokens",
    "video_tokens",
    "character_count",
    "image_count",
)


def _extract_em_extras(usage, response):
    """Pull tier-1 counters + long-tail details from a LiteLLM EmbeddingResponse.

    Returns ``(cached_tokens, extras)``.
    """
    prompt_details = _safe_get(usage, "prompt_tokens_details")
    hidden = getattr(response, "_hidden_params", None) or {}
    cached = _to_int(_safe_get(prompt_details, "cached_tokens"))
    extras = {}
    for key in _EM_PROMPT_DETAIL_KEYS:
        v = _safe_get(prompt_details, key)
        if v:
            extras[f"prompt_{key}"] = v
    overhead = (
        hidden.get("litellm_overhead_time_ms") if isinstance(hidden, dict) else None
    )
    if overhead is not None:
        extras["litellm_overhead_time_ms"] = overhead
    return cached, extras


def _accumulate(obj, phase_prefix, increments, extras):
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


@synalinks_export(
    [
        "synalinks.EmbeddingModel",
        "synalinks.embedding_models.EmbeddingModel",
    ]
)
class EmbeddingModel(Module):
    """An embedding model API wrapper.

    Embedding models are a type of machine learning model used to convert
    high-dimensional data, such as text into lower-dimensional vector
    representations while preserving the semantic meaning and relationships
    within the data. These vector representations, known as embeddings,
    allow for more efficient and effective processing in various tasks.

    Many providers are available like Gemini, Azure, Vertex AI or Ollama.

    For the complete list of models, please refer to the providers documentation.

    **Using Gemini models**

    ```python
    import synalinks
    import os

    os.environ["GEMINI_API_KEY"] = "your-api-key"

    embedding_model = synalinks.EmbeddingModel(
        model="gemini/text-embedding-004",
    )
    ```

    **Using Azure models**

    ```python
    import synalinks
    import os

    os.environ["AZURE_API_KEY"] = "your-api-key"
    os.environ["AZURE_API_BASE"] = "your-api-base"
    os.environ["AZURE_API_VERSION"] = "your-api-version"

    embedding_model = synalinks.EmbeddingModel(
        model="azure/<your_deployment_name>",
    )
    ```

    **Using VertexAI models**

    ```python
    import synalinks
    import os

    embedding_model = synalinks.EmbeddingModel(
        model="vertex_ai/text-embedding-004",
        vertex_project = "hardy-device-38811", # Your Project ID
        vertex_location = "us-central1",  # Project location
    )
    ```

    **Using Ollama models**

    ```python
    import synalinks

    embedding_model = synalinks.EmbeddingModel(
        model="ollama/mxbai-embed-large",
    )
    ```

    **Note**: Obviously, use an `.env` file and `.gitignore` to avoid
    putting your API keys in the code or a config file that can lead to
    leackage when pushing it into repositories.

    Args:
        model (str): The model to use.
        api_base (str): Optional. The endpoint to use.
        retry (int): Optional. The number of retry.
        retry_max_wait (int): Optional. Max seconds to wait between retries when a
            rate-limit `Retry-After` header is honored (default to 60).
        fallback (EmbeddingModel): Optional. The embedding model to fallback
            if anything is wrong.
        caching (bool): Enables caching (Default to True).
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        hooks (list): Optional. Hooks to attach to this module's calls.
        **default_kwargs: Optional. Default parameters (e.g. `dimensions`,
            `encoding_format`) forwarded to every call. Per-call kwargs
            override these.
    """

    def __init__(
        self,
        *,
        model=None,
        api_base=None,
        retry=5,
        retry_max_wait=60,
        fallback=None,
        caching=True,
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
        if model is None:
            raise ValueError(
                "You need to set the `model` argument for any EmbeddingModel"
            )
        self.model = model
        if self.model.startswith("ollama") and not api_base:
            self.api_base = "http://localhost:11434"
        else:
            self.api_base = api_base
        self.retry = retry
        self.retry_max_wait = retry_max_wait
        if fallback is not None:
            # Lazy import: `get` lives in the package __init__ which imports
            # this file at load time.
            from synalinks.src.modules.embedding_models import get as _get_em

            fallback = _get_em(fallback)
        self.fallback = fallback
        self.caching = caching
        self.default_kwargs = default_kwargs
        # All-time counters across every embedding call (training + inference).
        # Operational metrics use the inference-scoped counters below instead.
        self.cumulated_calls = 0
        self.cumulated_prompt_tokens = 0
        self.cumulated_tokens = 0
        self.cumulated_vectors = 0
        self.cumulated_elapsed_s = 0.0
        self.cumulated_cost = 0.0
        self.cumulated_cached_tokens = 0
        # Failure counters: a call that exhausts all retries (`failed_calls`)
        # and each `fallback` invocation it triggers (`fallback_activations`).
        self.cumulated_failed_calls = 0
        self.cumulated_fallback_activations = 0
        self.cumulated_details = {}
        self.last_call_prompt_tokens = 0
        self.last_call_tokens = 0
        self.last_call_vectors = 0
        self.last_call_elapsed_s = 0.0
        self.last_call_cost = 0.0
        # Phase-scoped counters — populated based on `synalinks_op_scope` set
        # by the trainer: "inference" inside `predict_on_batch`, "reward"
        # inside `compute_reward`, "optimizer" inside `optimizer.optimize`.
        # `cached_tokens` is the only tier-1 extra that makes sense for
        # embeddings (no completion phase, no reasoning, no tool use).
        for _phase in ("inference", "reward", "optimizer"):
            setattr(self, f"{_phase}_cumulated_calls", 0)
            setattr(self, f"{_phase}_cumulated_prompt_tokens", 0)
            setattr(self, f"{_phase}_cumulated_tokens", 0)
            setattr(self, f"{_phase}_cumulated_vectors", 0)
            setattr(self, f"{_phase}_cumulated_elapsed_s", 0.0)
            setattr(self, f"{_phase}_cumulated_cost", 0.0)
            setattr(self, f"{_phase}_cumulated_cached_tokens", 0)
            setattr(self, f"{_phase}_cumulated_failed_calls", 0)
            setattr(self, f"{_phase}_cumulated_fallback_activations", 0)
            setattr(self, f"{_phase}_cumulated_details", {})
        # No state depends on the input shape, so mark built up-front and
        # skip Module's auto-build path (which would try to trace `call`).
        self.built = True

    def _record_event(self, suffix):
        """Bump an all-time + phase-scoped operational counter by 1 (used for
        `failed_calls` / `fallback_activations`), attributed to the active
        `op_scope` like the successful-call counters.
        """
        op = current_op_scope()
        _accumulate(self, "", {suffix: 1}, None)
        if op is not None:
            _accumulate(self, f"{op}_", {suffix: 1}, None)

    async def call(self, inputs, **kwargs):
        """
        Call method to get dense embeddings vectors.

        Args:
            inputs (EmbeddingRequest | JsonDataModel): An `EmbeddingRequest`
                wrapping the text(s) to embed (single `str` or `list[str]`).

        Returns:
            (Embeddings): The corresponding embedding vectors, wrapped as
                an `Embeddings` JsonDataModel.
        """
        if not inputs:
            return None
        texts = inputs.get("texts")
        input_kwargs = copy.deepcopy(kwargs)
        # Merge instance-level defaults; per-call kwargs win.
        kwargs = {**self.default_kwargs, **kwargs}
        try:
            return await self._call_with_retry(texts, **kwargs)
        except Exception as e:
            warnings.warn(f"All retries failed for {self}: {e}")
            self._record_event("failed_calls")
            if self.fallback:
                self._record_event("fallback_activations")
                return await self.fallback(
                    inputs,
                    **input_kwargs,
                )
            else:
                return None

    async def _call_with_retry(self, texts, **kwargs):
        """Perform the embedding call with tenacity retry logic."""
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
            try:
                t0 = time.perf_counter()
                if self.api_base:
                    response = await litellm.aembedding(
                        model=self.model,
                        input=texts,
                        api_base=self.api_base,
                        caching=self.caching,
                        **kwargs,
                    )
                else:
                    response = await litellm.aembedding(
                        model=self.model,
                        input=texts,
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
                vectors = []
                for data in response["data"]:
                    vectors.append(data["embedding"])
                n_vectors = len(vectors)
                usage = response.get("usage") or {}
                prompt_tokens = int(usage.get("prompt_tokens") or 0)
                total_tokens = int(usage.get("total_tokens") or prompt_tokens)
                cached, extras = _extract_em_extras(usage, response)
                self.last_call_prompt_tokens = prompt_tokens
                self.last_call_tokens = total_tokens
                self.last_call_vectors = n_vectors
                self.last_call_elapsed_s = elapsed_s
                flat_increments = {
                    "calls": 1,
                    "prompt_tokens": prompt_tokens,
                    "tokens": total_tokens,
                    "vectors": n_vectors,
                    "elapsed_s": elapsed_s,
                    "cached_tokens": cached,
                }
                if response_cost is not None:
                    flat_increments["cost"] = response_cost
                _accumulate(self, "", flat_increments, extras)
                if op_scope is not None:
                    _accumulate(self, f"{op_scope}_", flat_increments, extras)
                return JsonDataModel(
                    json={"embeddings": vectors},
                    schema=Embeddings.get_schema(),
                    name=f"{self.name}_response",
                )
            except Exception as e:
                warnings.warn(f"Error occured while trying to call {self}: {e}")
                raise

        return await _do_call()

    def _obj_type(self):
        return "EmbeddingModel"

    def get_config(self):
        config = {
            "model": self.model,
            "api_base": self.api_base,
            "retry": self.retry,
            "retry_max_wait": self.retry_max_wait,
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
        return f"<EmbeddingModel model={self.model}{api_base}>"
