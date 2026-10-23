# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""What a frontend needs to offer a model picker: the providers, the
environment variables each one needs, the models it serves and their info.

Hosted providers list their models from litellm's model table. Self-hosted
ones (Ollama, vLLM...) and open catalogs (OpenRouter, Doubleword) are asked
for the models they actually serve.
"""

import asyncio
import os

import httpx
import litellm

OLLAMA = "ollama"
OPENAI_COMPATIBLE = "openai_compatible"

LIST_TIMEOUT = 10.0

# The endpoint litellm's model table lists for the chat models that also
# answer decisions (e.g. `gpt-6-luna`).
DECISIONS_ENDPOINT = "/v1/decisions"


def env_var(name, required=True, secret=True):
    """Describe an environment variable a provider reads.

    Args:
        name (str): The variable name.
        required (bool): Whether calls fail without it.
        secret (bool): Whether it holds a credential (to mask in a UI).

    Returns:
        (dict): `{"name", "required", "secret"}`.
    """
    return {"name": name, "required": required, "secret": secret}


# The providers of language and embedding models: the environment variables
# they read, their default endpoint, and the server API to list models from
# (`None` for the providers listed from litellm's model table).
PROVIDERS = {
    "anthropic": {"env": [env_var("ANTHROPIC_API_KEY")]},
    "azure": {
        "env": [
            env_var("AZURE_API_KEY"),
            env_var("AZURE_API_BASE", secret=False),
            env_var("AZURE_API_VERSION", secret=False),
        ]
    },
    "bedrock": {
        "env": [
            env_var("AWS_ACCESS_KEY_ID"),
            env_var("AWS_SECRET_ACCESS_KEY"),
            env_var("AWS_REGION_NAME", secret=False),
        ]
    },
    "cohere": {"env": [env_var("COHERE_API_KEY")]},
    "deepseek": {"env": [env_var("DEEPSEEK_API_KEY")]},
    "doubleword": {
        "env": [env_var("OPENAI_API_KEY")],
        "default_api_base": "https://api.doubleword.ai/v1",
        "server": OPENAI_COMPATIBLE,
    },
    "gemini": {"env": [env_var("GEMINI_API_KEY")]},
    "groq": {"env": [env_var("GROQ_API_KEY")]},
    "huggingface": {"env": [env_var("HUGGINGFACE_API_KEY")]},
    "mirai": {
        "env": [],
        "default_api_base": "http://localhost:8000/v1",
        "server": OPENAI_COMPATIBLE,
    },
    "mistral": {"env": [env_var("MISTRAL_API_KEY")]},
    "ollama": {
        "env": [],
        "default_api_base": "http://localhost:11434",
        "server": OLLAMA,
    },
    "openai": {
        "env": [
            env_var("OPENAI_API_KEY"),
            env_var("OPENAI_API_BASE", required=False, secret=False),
        ]
    },
    "openrouter": {
        "env": [env_var("OPENROUTER_API_KEY")],
        "default_api_base": "https://openrouter.ai/api/v1",
        "server": OPENAI_COMPATIBLE,
    },
    "together_ai": {"env": [env_var("TOGETHERAI_API_KEY")]},
    "vertex_ai": {
        "env": [
            env_var("VERTEXAI_PROJECT", secret=False),
            env_var("VERTEXAI_LOCATION", secret=False),
            env_var("GOOGLE_APPLICATION_CREDENTIALS", required=False, secret=False),
        ]
    },
    "vllm": {
        "env": [
            env_var("HOSTED_VLLM_API_BASE", required=False, secret=False),
            env_var("HOSTED_VLLM_API_KEY", required=False),
        ],
        "default_api_base": "http://localhost:8000/v1",
        "server": OPENAI_COMPATIBLE,
    },
    "voyage": {"env": [env_var("VOYAGE_API_KEY")]},
    "xai": {"env": [env_var("XAI_API_KEY")]},
}


def check_provider(provider, supported, kind):
    """Raise a `ValueError` naming the supported providers if `provider` is not one.

    Args:
        provider (str): The provider prefix to check.
        supported (list): The supported provider prefixes.
        kind (str): The model type, for the message (e.g. `"language model"`).
    """
    if provider not in supported:
        raise ValueError(
            f"Unsupported {kind} provider {provider!r}. Use one of {sorted(supported)}."
        )


def decision_provider(provider):
    """Return the `PROVIDERS`-like entry of a litellm decisions provider.

    Its environment variables and default endpoint come from litellm's
    decisions config of the provider.

    Args:
        provider (str): A litellm decisions provider, e.g. `"typesafe"`.

    Returns:
        (dict): The `env`, `default_api_base` and `server` of the provider.
    """
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_decisions_config(
        model="", provider=LlmProviders(provider)
    )
    default_api_base = config.get_default_api_base()
    if config.api_key_env:
        env = [env_var(config.api_key_env[0], required=config.api_key_required)]
    else:
        env = [dict(var) for var in PROVIDERS.get(provider, {}).get("env", [])]
    if config.api_base_env:
        env.append(
            env_var(
                config.api_base_env[0], required=default_api_base is None, secret=False
            )
        )
    return {
        "env": env,
        "default_api_base": default_api_base,
        "server": (
            OPENAI_COMPATIBLE if provider in ("hosted_vllm", "openrouter") else None
        ),
    }


def provider_info(provider, info):
    """Return what a frontend needs to set a provider up.

    Args:
        provider (str): The provider prefix, e.g. `"ollama"`.
        info (dict): The provider's entry (see `PROVIDERS`).

    Returns:
        (dict): `provider`, `env` (the variables to set, see `env_var`),
            `default_api_base` (or `None`) and `lists_models_from`
            (`"server"` when the models are asked to the provider's server,
            else `"catalog"`).
    """
    return {
        "provider": provider,
        "env": [dict(var) for var in info.get("env", [])],
        "default_api_base": info.get("default_api_base"),
        "lists_models_from": "server" if info.get("server") else "catalog",
    }


def _in_family(provider, litellm_provider):
    """Whether a litellm model table provider belongs to a synalinks provider."""
    if provider is None or litellm_provider == provider:
        return True
    if provider == "bedrock":
        return litellm_provider == "bedrock_converse"
    if provider == "vertex_ai":
        return litellm_provider.startswith("vertex_ai-")
    return False


def catalog_models(provider, modes, endpoint=None):
    """List a provider's models from litellm's model table.

    Args:
        provider (str): The provider prefix, or `None` for every provider
            (each model is then prefixed by its litellm provider).
        modes (tuple): The litellm modes to keep, e.g. `("chat",)`.
        endpoint (str): Optional. Also keep the models of other modes that
            list this endpoint among their `supported_endpoints`.

    Returns:
        (list): The sorted `provider/model` IDs.
    """
    models = set()
    for key, entry in litellm.model_cost.items():
        litellm_provider = entry.get("litellm_provider") or ""
        kept = entry.get("mode") in modes or (
            endpoint is not None and endpoint in (entry.get("supported_endpoints") or [])
        )
        if not kept or not _in_family(provider, litellm_provider):
            continue
        for prefix in (f"{litellm_provider}/", f"{provider}/"):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        models.add(f"{provider or litellm_provider}/{key}")
    return sorted(models)


async def server_models(provider, info, api_base=None, ollama_capability=None, keep=None):
    """Ask a provider's server for the models it serves.

    Args:
        provider (str): The provider prefix, used to prefix the model IDs.
        info (dict): The provider's entry (see `PROVIDERS`).
        api_base (str): Optional. The server endpoint. Defaults to the
            provider's base URL variable when set, then to its default.
        ollama_capability (str): Optional. For Ollama, keep only the models
            reporting this capability (`"completion"` or `"embedding"`).
        keep (set): Optional. Keep only the models the server names so.

    Returns:
        (list): The sorted `provider/model` IDs.

    Raises:
        ValueError: If no endpoint is known.
        ConnectionError: If the server cannot be reached or answers an error.
    """
    env = info.get("env", [])
    if api_base is None:
        api_base = next(
            (
                os.environ[var["name"]]
                for var in env
                if var["name"].endswith("_API_BASE") and os.environ.get(var["name"])
            ),
            info.get("default_api_base"),
        )
    if not api_base:
        names = [var["name"] for var in env if var["name"].endswith("_API_BASE")]
        raise ValueError(
            f"No endpoint to list the {provider} models from: pass `api_base`"
            + (f" or set `{names[0]}`." if names else ".")
        )
    api_base = api_base.rstrip("/")
    api_key = next(
        (
            os.environ[var["name"]]
            for var in env
            if var["secret"] and os.environ.get(var["name"])
        ),
        None,
    )
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=LIST_TIMEOUT, headers=headers) as client:
            if info.get("server") == OLLAMA:
                response = await client.get(f"{api_base}/api/tags")
                response.raise_for_status()
                names = [model["name"] for model in response.json().get("models", [])]
                if ollama_capability is not None:

                    async def capabilities(name):
                        shown = await client.post(
                            f"{api_base}/api/show", json={"model": name}
                        )
                        shown.raise_for_status()
                        # Servers older than capabilities reporting keep every model.
                        return shown.json().get("capabilities") or [ollama_capability]

                    found = await asyncio.gather(*(capabilities(name) for name in names))
                    names = [
                        name
                        for name, caps in zip(names, found)
                        if ollama_capability in caps
                    ]
            else:
                # vLLM serves its OpenAI-compatible API under `/v1`.
                if not api_base.endswith("/v1"):
                    api_base = f"{api_base}/v1"
                response = await client.get(f"{api_base}/models")
                response.raise_for_status()
                names = [model["id"] for model in response.json().get("data", [])]
            if keep is not None:
                names = [name for name in names if name in keep]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
        raise ConnectionError(
            f"Could not list the {provider} models from {api_base}: "
            f"{e or type(e).__name__}"
        ) from e
    return sorted(f"{provider}/{name}" for name in set(names))


async def account_models(provider, catalog, api_base=None):
    """Narrow a provider's catalog to the models the account can use.

    For OpenAI with `OPENAI_API_KEY` set, the live `/v1/models` list is asked
    and the catalog models it holds are kept, along with the fine-tuned models
    (`ft:<base>:...`) of a kept base model. Without a key, and for the other
    providers, the catalog is returned as is.

    Args:
        provider (str): The provider prefix.
        catalog (list): The provider's models from `catalog_models()`.
        api_base (str): Optional. The OpenAI-compatible endpoint to ask
            (defaults to `OPENAI_API_BASE`, then to OpenAI's).

    Returns:
        (list): The sorted `provider/model` IDs.

    Raises:
        ConnectionError: If the live list cannot be fetched (e.g. a
            rejected key).
    """
    if provider != "openai" or not os.environ.get("OPENAI_API_KEY"):
        return catalog
    info = {
        "env": PROVIDERS["openai"]["env"],
        "default_api_base": "https://api.openai.com/v1",
        "server": OPENAI_COMPATIBLE,
    }
    known = set(catalog)
    return [
        model
        for model in await server_models(provider, info, api_base)
        if model in known
        or (model.startswith("openai/ft:") and f"openai/{model.split(':')[1]}" in known)
    ]


def model_info(model, litellm_model=None):
    """Return a model's info from litellm's model table.

    Args:
        model (str): The model, e.g. `"openai/gpt-4o-mini"`.
        litellm_model (str): Optional. The name litellm knows the model by,
            when the provider is routed under another name.

    Returns:
        (dict): `max_input_tokens`, `max_output_tokens`,
            `input_cost_per_token`, `output_cost_per_token` (each `None` when
            unknown) and `capabilities` (e.g. `["function_calling",
            "vision"]`), or `None` when litellm does not know the model.
    """
    litellm_model = litellm_model or model
    provider, _, name = litellm_model.partition("/")
    entry = litellm.model_cost.get(litellm_model)
    if entry is None:
        entry = litellm.model_cost.get(name)
        if entry is None or not _in_family(provider, entry.get("litellm_provider", "")):
            return None
    return {
        "max_input_tokens": entry.get("max_input_tokens"),
        "max_output_tokens": entry.get("max_output_tokens"),
        "input_cost_per_token": entry.get("input_cost_per_token"),
        "output_cost_per_token": entry.get("output_cost_per_token"),
        "capabilities": sorted(
            key[len("supports_") :]
            for key, value in entry.items()
            if key.startswith("supports_") and value is True
        ),
    }
