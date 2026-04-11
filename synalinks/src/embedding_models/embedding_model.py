# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import logging
import warnings

import litellm
from tenacity import before_sleep_log
from tenacity import retry
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from synalinks.src.api_export import synalinks_export
from synalinks.src.saving import serialization_lib
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable

litellm.disable_aiohttp_transport = True


@synalinks_export(
    [
        "synalinks.EmbeddingModel",
        "synalinks.embedding_models.EmbeddingModel",
    ]
)
class EmbeddingModel(SynalinksSaveable):
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
        fallback (EmbeddingModel): Optional. The embedding model to fallback
            if anything is wrong.
        caching (bool): Enables caching (Default to True).
    """

    def __init__(
        self,
        model=None,
        api_base=None,
        retry=5,
        fallback=None,
        caching=True,
    ):
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
        self.fallback = fallback
        self.caching = caching

    async def __call__(self, texts, **kwargs):
        """
        Call method to get dense embeddings vectors

        Args:
            texts (list): A list of texts to embed.

        Returns:
            (list): The list of corresponding vectors.
        """

        try:
            return await self._call_with_retry(texts, **kwargs)
        except Exception as e:
            warnings.warn(
                f"All retries failed for {self}: {e}"
            )
            if self.fallback:
                return await self.fallback(
                    texts,
                    **kwargs,
                )
            else:
                return None

    async def _call_with_retry(self, texts, **kwargs):
        """Perform the embedding call with tenacity retry logic."""
        logger = logging.getLogger(__name__)

        @retry(
            stop=stop_after_attempt(self.retry),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _do_call():
            try:
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
                vectors = []
                for data in response["data"]:
                    vectors.append(data["embedding"])
                return {"embeddings": vectors}
            except Exception as e:
                warnings.warn(
                    f"Error occured while trying to call"
                    f" {self}: {e}"
                )
                raise

        return await _do_call()

    def _obj_type(self):
        return "EmbeddingModel"

    def get_config(self):
        config = {
            "model": self.model,
            "api_base": self.api_base,
            "retry": self.retry,
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
