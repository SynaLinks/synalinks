# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import copy
import json
import warnings

import litellm

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatRole
from synalinks.src.saving import serialization_lib
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.nlp_utils import shorten_text

litellm.drop_params = True


@synalinks_export(
    [
        "synalinks.LanguageModel",
        "synalinks.language_models.LanguageModel",
    ]
)
class LanguageModel(SynalinksSaveable):
    """A language model API wrapper.

    A language model is a type of AI model designed to generate, and interpret human
    language. It is trained on large amounts of text data to learn patterns and
    structures in language. Language models can perform various tasks such as text
    generation, translation, summarization, and answering questions.

    We support providers that implement *constrained structured output*
    like OpenAI, Azure, Ollama or Mistral. In addition we support providers that otherwise
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

    **Using Azure OpenAI models**

    ```python
    import synalinks
    import os

    os.environ["AZURE_API_KEY"] = "your-api-key"
    os.environ["AZURE_API_BASE"] = "your-api-key"
    os.environ["AZURE_API_VERSION"] = "your-api-key"

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
        model="gemini/gemini-1.5-pro",
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

    To cascade models in case there is anything wrong with
    the model provider (hence making your pipelines more robust).
    Use the `fallback` argument like in this example:

    ```python
    import synalinks
    import os

    os.environ["OPENAI_API_KEY"] = "your-api-key"
    os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

    language_model = synalinks.LanguageModel(
        model="anthropic/claude-3-sonnet-20240229",
        fallback=synalinks.LanguageModel(
            model="openai/gpt-4o-mini",
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
        caching (bool): Optional. Enable caching of LM calls (Default to True).
    """

    def __init__(
        self,
        model=None,
        api_base=None,
        timeout=600,
        retry=5,
        fallback=None,
        caching=True,
    ):
        if model is None:
            raise ValueError("You need to set the `model` argument for any LanguageModel")
        model_provider = model.split("/")[0]
        if model_provider == "ollama":
            # Switch from `ollama` to `ollama_chat`
            # because it have better performance due to the chat prompts
            model = model.replace("ollama", "ollama_chat")
        self.model = model
        self.fallback = fallback
        if self.model.startswith("ollama") and not api_base:
            self.api_base = "http://localhost:11434"
        else:
            self.api_base = api_base
        self.timeout = timeout
        self.retry = retry
        self.caching = caching

    async def __call__(self, messages, schema=None, streaming=False, **kwargs):
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
        json_instance = {}
        input_kwargs = copy.deepcopy(kwargs)
        schema = copy.deepcopy(schema)
        if schema:
            if self.model.startswith("groq"):
                # Use a tool created on the fly for groq
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
                # Use a tool created on the fly for anthropic
                kwargs.update(
                    {
                        "tools": [
                            {
                                "name": "structured_output",
                                "description": "Generate a valid JSON output",
                                "input_schema": {
                                    "type": "object",
                                    "properties": schema.get("properties"),
                                    "required": schema.get("required"),
                                },
                            }
                        ],
                        "tool_choice": {
                            "type": "tool",
                            "name": "structured_output",
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
            elif self.model.startswith("openai") or self.model.startswith("azure"):
                # Use constrained structured output for openai
                # OpenAI require the field  "additionalProperties"
                # Also OpenAI disallow the field "description" in $ref
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
        for i in range(self.retry):
            try:
                response_str = ""
                response = await litellm.acompletion(
                    model=self.model,
                    messages=formatted_messages,
                    timeout=self.timeout,
                    caching=self.caching,
                    **kwargs,
                )
                if streaming:
                    return StreamingIterator(response)
                if (
                    self.model.startswith("groq") or self.model.startswith("anthropic")
                ) and schema:
                    response_str = response["choices"][0]["message"]["tool_calls"][0][
                        "function"
                    ]["arguments"]
                else:
                    response_str = response["choices"][0]["message"]["content"].strip()
                if schema:
                    json_instance = json.loads(response_str)
                else:
                    json_instance = {
                        "role": ChatRole.ASSISTANT,
                        "content": response_str,
                        "tool_call_id": None,
                        "tool_calls": [],
                    }
                return json_instance
            except Exception as e:
                warnings.warn(
                    f"Error occured while trying to call {self}: "
                    + str(e)
                    + f"\nReceived response={shorten_text(response_str)}"
                )
            await asyncio.sleep(1)
        if self.fallback:
            return self.fallback(
                messages,
                schema=schema,
                streaming=streaming,
                **input_kwargs,
            )
        else:
            return None

    def _obj_type(self):
        return "LanguageModel"

    def get_config(self):
        config = {
            "model": self.model,
            "api_base": self.api_base,
            "timeout": self.timeout,
            "retry": self.retry,
            "caching": self.caching,
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
    def __init__(self, iterator):
        self._iterator = iterator

    def __iter__(self):
        return self

    def __next__(self):
        content = self._iterator.__next__()["choices"][0]["delta"]["content"]
        if content:
            return {"role": ChatRole.ASSISTANT, "content": content}
        else:
            raise StopIteration
