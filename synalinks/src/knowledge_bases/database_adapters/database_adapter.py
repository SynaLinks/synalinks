# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union


class DatabaseAdapter:
    """Base class for database adapters.

    DatabaseAdapter provides a unified interface for storing and retrieving
    structured data with optional embedding-based similarity search capabilities.

    Subclasses must implement the abstract methods to provide concrete
    database functionality.
    """

    def __init__(
        self,
        uri=None,
        embedding_model=None,
        data_models=None,
        metric="cosine",
        wipe_on_start=False,
        name=None,
        **kwargs,
    ):
        """Initialize the database adapter.

        Args:
            uri (str): The database connection URI or path.
            embedding_model (EmbeddingModel): Optional embedding model for
                vector similarity search.
            data_models (list): Optional list of SymbolicDataModel or DataModel
                classes to create tables for.
            metric (str): Distance metric for vector search. Options depend on
                the specific adapter implementation.
            wipe_on_start (bool): Whether to clear the database on initialization.
            name (str): Optional name for the adapter instance.
        """
        self.uri = uri
        self.embedding_model = embedding_model
        self.data_models = data_models or []
        self.metric = metric
        self.name = name

        if wipe_on_start:
            self.wipe_database()

    def wipe_database(self):
        """Clear all data from the database.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `wipe_database()` method"
        )

    def get_symbolic_data_models(self):
        """Retrieve all data models from the database schema.

        Returns:
            List[SymbolicDataModel]: List of symbolic data models representing
                the database schema.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the "
            "`get_symbolic_data_models()` method"
        )

    async def update(
        self,
        data_model_or_data_models: Union[Any, List[Any]],
    ) -> Union[Any, List[Any]]:
        """Insert or update records in the database.

        Args:
            data_model_or_data_models: A single JsonDataModel or a list of
                JsonDataModels to insert or update.

        Returns:
            The primary key value(s) of the inserted/updated records.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `update()` method"
        )

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

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `get()` method"
        )

    async def getall(
        self,
        data_model: Any,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Any]:
        """Retrieve all records from a table with pagination.

        Args:
            data_model: The SymbolicDataModel representing the table to query.
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            List of JsonDataModels.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `getall()` method"
        )

    async def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Execute a raw query against the database.

        Args:
            query: The query string (SQL or other query language).
            params: Optional parameters for parameterized queries.
            **kwargs: Additional adapter-specific options.

        Returns:
            List of result dictionaries.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `query()` method"
        )

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
            text_or_texts: Query text or list of query texts to search for.
            data_models: Optional list of SymbolicDataModels to search in.
            k: Maximum number of results to return.
            threshold: Optional similarity threshold for filtering results.

        Returns:
            List of matching records with similarity scores.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `similarity_search()` method"
        )

    async def fulltext_search(
        self,
        text_or_texts: Union[str, List[str]],
        data_models: Optional[List[Any]] = None,
        k: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform full-text search on text fields.

        Args:
            text_or_texts: Query text or list of query texts to search for.
            data_models: Optional list of SymbolicDataModels to search in.
            k: Maximum number of results to return.
            threshold: Optional relevance threshold for filtering results.

        Returns:
            List of matching records with relevance scores.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `fulltext_search()` method"
        )

    async def hybrid_search(
        self,
        text_or_texts: Union[str, List[str]],
        data_models: Optional[List[Any]] = None,
        k: int = 10,
        similarity_threshold: Optional[float] = None,
        fulltext_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector similarity and full-text search.

        Uses Reciprocal Rank Fusion (RRF) to combine results from both
        similarity search and full-text search.

        Args:
            text_or_texts: Query text or list of query texts to search for.
            data_models: Optional list of SymbolicDataModels to search in.
            k: Maximum number of results to return.
            similarity_threshold: Optional threshold for vector similarity.
            fulltext_threshold: Optional threshold for full-text relevance.

        Returns:
            List of matching records with combined scores.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} should implement the `hybrid_search()` method"
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} uri={self.uri}>"
