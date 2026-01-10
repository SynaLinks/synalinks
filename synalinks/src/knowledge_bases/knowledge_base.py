# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.knowledge_bases import database_adapters
from synalinks.src.saving import serialization_lib
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.naming import auto_name


@synalinks_export("synalinks.KnowledgeBase")
class KnowledgeBase(SynalinksSaveable):
    """A knowledge base for storing and retrieving structured data.

    The KnowledgeBase provides a unified interface for storing structured data
    with support for full-text search and optional vector similarity search.
    It uses DuckDB as the underlying storage engine.

    ### Basic Usage

    ```python
    import synalinks

    class Document(synalinks.DataModel):
        id: str
        title: str
        content: str

    # Create a knowledge base without embeddings (full-text search only)
    knowledge_base = synalinks.KnowledgeBase(
        uri="duckdb://my_database.db",
        data_models=[Document],
    )

    # Store a document
    doc = Document(id="1", title="Hello", content="Hello World!")
    await knowledge_base.update(doc.to_json_data_model())

    # Retrieve by ID
    result = await knowledge_base.get("1", [Document.to_symbolic_data_model()])

    # Full-text search
    results = await knowledge_base.fulltext_search("Hello", k=10)
    ```

    ### With Vector Similarity Search

    ```python
    embedding_model = synalinks.EmbeddingModel(
        model="ollama/mxbai-embed-large"
    )

    knowledge_base = synalinks.KnowledgeBase(
        uri="duckdb://./my_database.db",
        data_models=[Document],
        embedding_model=embedding_model,
        metric="cosine",
    )

    # Hybrid search (combines full-text and vector similarity)
    results = await knowledge_base.hybrid_search("semantic query", k=10)
    ```

    ### Retrieving Table Definitions

    ```python
    # Get all symbolic data models (table definitions) from the database
    symbolic_models = knowledge_base.get_symbolic_data_models()

    for model in symbolic_models:
        print(model.get_schema())
        # {'title': 'Document', 'type': 'object', 'properties': {...}, ...}
    ```

    Args:
        uri (str): The database connection URI. Use "duckdb://path/to/db.db"
            for DuckDB. If not provided, uses an in-memory database.
        data_models (list): Optional list of DataModel or SymbolicDataModel
            classes to create tables for.
        embedding_model (EmbeddingModel): Optional embedding model for
            vector similarity search.
        metric (str): The distance metric for vector search.
            Options: "cosine", "l2seq", "ip" (default: "cosine").
        wipe_on_start (bool): Whether to clear the database on initialization
            (default: False).
        name (str): Optional name for the knowledge base (used for serialization).
    """

    def __init__(
        self,
        uri=None,
        data_models=None,
        embedding_model=None,
        metric="cosine",
        wipe_on_start=False,
        name=None,
    ):
        self.adapter = database_adapters.get(uri)(
            uri=uri,
            data_models=data_models,
            embedding_model=embedding_model,
            metric=metric,
            wipe_on_start=wipe_on_start,
            name=name,
        )
        self.uri = uri
        self.data_models = data_models or []
        self.embedding_model = embedding_model
        self.metric = metric
        self.wipe_on_start = wipe_on_start
        if not name:
            self.name = auto_name("knowledge_base")
        else:
            self.name = name

    async def update(
        self,
        data_model_or_data_models: Union[Any, List[Any]],
    ) -> Union[Any, List[Any]]:
        """Insert or update records in the knowledge base.

        Args:
            data_model_or_data_models (JsonDataModel | List[JsonDataModel]):
                A single JsonDataModel or a list of JsonDataModels to insert
                or update. Uses the first field as the primary key for upserts.

        Returns:
            The primary key value(s) of the inserted/updated records.
        """
        return await self.adapter.update(data_model_or_data_models)

    async def get(
        self,
        id_or_ids: Any,
        data_models: Optional[List[Any]] = None,
    ) -> Optional[Any]:
        """Retrieve a record by its primary key.

        Args:
            id_or_ids: The primary key value to look up.
            data_models: Optional list of SymbolicDataModels to search in.
                If not provided, searches all tables.

        Returns:
            JsonDataModel if found, None otherwise.
        """
        return await self.adapter.get(id_or_ids, data_models=data_models)

    async def getall(
        self,
        data_model: Any,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Any]:
        """Retrieve all records from a table with pagination.

        Args:
            data_model: The SymbolicDataModel representing the table to query.
            limit: Maximum number of records to return (default: 50).
            offset: Number of records to skip (default: 0).

        Returns:
            List of JsonDataModels.
        """
        return await self.adapter.getall(data_model, limit=limit, offset=offset)

    async def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Execute a raw SQL query against the knowledge base.

        Args:
            query (str): The SQL query to execute.
            params (dict): Optional list of parameters for parameterized queries.
            **kwargs (Any): Additional options (e.g., read_only=True/False).

        Returns:
            List of result dictionaries.
        """
        return await self.adapter.query(query, params=params, **kwargs)

    async def similarity_search(
        self,
        text_or_texts: Union[str, List[str]],
        data_models: Optional[List[Any]] = None,
        k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search using embeddings.

        Requires an embedding_model to be configured.

        Args:
            text_or_texts: Query text or list of query texts.
            data_models: Optional list of SymbolicDataModels to search in.
            k: Maximum number of results to return (default: 10).
            threshold: Optional maximum distance threshold for filtering.

        Returns:
            List of matching records with similarity scores.
        """
        return await self.adapter.similarity_search(
            text_or_texts,
            data_models=data_models,
            k=k,
            threshold=threshold,
        )

    async def fulltext_search(
        self,
        text_or_texts: Union[str, List[str]],
        data_models: Optional[List[Any]] = None,
        k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform full-text search using BM25 ranking.

        Searches text fields (description, text, content, message, name,
        query, question) for matching documents.

        Args:
            text_or_texts: Query text or list of query texts.
            data_models: Optional list of SymbolicDataModels to search in.
            k: Maximum number of results to return (default: 10).
            threshold: Optional minimum BM25 score threshold.

        Returns:
            List of matching records with relevance scores.
        """
        return await self.adapter.fulltext_search(
            text_or_texts,
            data_models=data_models,
            k=k,
            threshold=threshold,
        )

    async def hybrid_search(
        self,
        text_or_texts: Union[str, List[str]],
        data_models: Optional[List[Any]] = None,
        k: int = 10,
        k_rank: int = 60,
        similarity_threshold: Optional[float] = None,
        fulltext_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity and full-text.

        Uses Reciprocal Rank Fusion (RRF) to combine results from both
        similarity search and full-text search. Falls back to full-text
        search only if no embedding model is configured.

        Args:
            text_or_texts: Query text or list of query texts.
            data_models: Optional list of SymbolicDataModels to search in.
            k: Maximum number of results to return (default: 10).
            k_rank: RRF smoothing constant. Lower values emphasize top ranks
                more strongly (default: 60).
            similarity_threshold: Optional threshold for vector similarity.
            fulltext_threshold: Optional threshold for full-text relevance.

        Returns:
            List of matching records with combined scores.
        """
        return await self.adapter.hybrid_search(
            text_or_texts,
            data_models=data_models,
            k=k,
            k_rank=k_rank,
            similarity_threshold=similarity_threshold,
            fulltext_threshold=fulltext_threshold,
        )

    def get_symbolic_data_models(self) -> List[Any]:
        """Retrieve all symbolic data models (table definitions) from the database.

        Returns a list of SymbolicDataModel objects representing each table
        in the database. This is useful for introspecting the database schema
        or for passing to search methods to limit the search scope.

        Returns:
            list: List of symbolic data models representing the database tables.

        Example:
            ```python
            symbolic_models = knowledge_base.get_symbolic_data_models()
            for model in symbolic_models:
                schema = model.get_schema()
                print(f"Table: {schema['title']}")
                print(f"Fields: {list(schema['properties'].keys())}")
            ```
        """
        return self.adapter.get_symbolic_data_models()

    def get_config(self):
        config = {
            "uri": self.uri,
            "name": self.name,
            "metric": self.metric,
            "wipe_on_start": self.wipe_on_start,
        }
        data_models_config = {
            "data_models": [
                (
                    serialization_lib.serialize_synalinks_object(
                        data_model.to_symbolic_data_model(
                            name="data_model" + (f"_{i}_" if i > 0 else "_") + self.name
                        )
                    )
                    if not is_symbolic_data_model(data_model)
                    else serialization_lib.serialize_synalinks_object(data_model)
                )
                for i, data_model in enumerate(self.data_models)
            ]
        }
        embedding_model_config = {}
        if self.embedding_model:
            embedding_model_config = {
                "embedding_model": serialization_lib.serialize_synalinks_object(
                    self.embedding_model,
                )
            }
        return {
            **data_models_config,
            **embedding_model_config,
            **config,
        }

    @classmethod
    def from_config(cls, config):
        data_models_config = config.pop("data_models", [])
        data_models = [
            serialization_lib.deserialize_synalinks_object(data_model)
            for data_model in data_models_config
        ]
        embedding_model = None
        if "embedding_model" in config:
            embedding_model = serialization_lib.deserialize_synalinks_object(
                config.pop("embedding_model"),
            )
        return cls(
            data_models=data_models,
            embedding_model=embedding_model,
            **config,
        )
