from synalinks.src.knowledge_bases.graph_database_adapters.graph_database_adapter import (
    GraphDatabaseAdapter,
)
from synalinks.src.knowledge_bases.graph_database_adapters.ladybug_adapter import (
    LadybugAdapter,
)


def get(uri):
    """Resolve a graph-database URI to its adapter class.

    Mirrors :func:`database_adapters.get`: ``None`` returns the
    default adapter (LadybugDB) so a no-args ``KnowledgeBase()`` can
    auto-pair both stores under ``synalinks_home()``.
    """
    if not uri:
        return LadybugAdapter
    if uri.startswith("ladybug"):
        return LadybugAdapter
    raise ValueError(f"No graph database adapter registered for uri: {uri!r}")
