from synalinks.src.api_export import synalinks_export
from synalinks.src.knowledge_bases.knowledge_base import KnowledgeBase
from synalinks.src.saving import serialization_lib

ALL_OBJECTS = {
    KnowledgeBase,
}

ALL_OBJECTS_DICT = {cls.__name__.lower(): cls for cls in ALL_OBJECTS}

# URI scheme prefixes that route to a specific adapter. Mirrors the
# dispatch tables in `database_adapters.get` and
# `graph_database_adapters.get`. A schemeless identifier (e.g.
# `":memory:"`) doesn't match either set and is treated as dual-store:
# we prepend each scheme so DuckDB and LadybugDB both get the location.
_SQL_URI_SCHEMES = ("duckdb",)
_GRAPH_URI_SCHEMES = ("ladybug",)


def _is_sql_uri(uri: str) -> bool:
    return any(uri.startswith(scheme) for scheme in _SQL_URI_SCHEMES)


def _is_graph_uri(uri: str) -> bool:
    return any(uri.startswith(scheme) for scheme in _GRAPH_URI_SCHEMES)


@synalinks_export("synalinks.knowledge_bases.serialize")
def serialize(knowledge_base):
    """Returns the knowledge base configuration as a Python dict.

    Args:
        knowledge_base (KnowledgeBase): A `KnowledgeBase` instance to serialize.

    Returns:
        Python dict which contains the configuration of the knowledge base.
    """
    return serialization_lib.serialize_synalinks_object(knowledge_base)


@synalinks_export("synalinks.knowledge_bases.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Synalinks knowledge base object via its configuration.

    Args:
        config (dict): KnowledgeBase configuration dictionary.
        custom_objects (dict): Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Synalinks KnowledgeBase instance.
    """
    if config["class_name"].lower() in ALL_OBJECTS_DICT:
        config["class_name"] = config["class_name"].lower()

    return serialization_lib.deserialize_synalinks_object(
        config,
        module_objects=ALL_OBJECTS_DICT,
        custom_objects=custom_objects,
    )


@synalinks_export("synalinks.knowledge_bases.get")
def get(identifier):
    """Retrieves a Synalinks KnowledgeBase instance.

    Args:
        identifier (str | dict | KnowledgeBase | None): KnowledgeBase
            identifier, one of:
            - String: a URI. SQL schemes (e.g.
              `"duckdb://./my_database.db"`) construct a
              `KnowledgeBase(uri=identifier)`; graph schemes (e.g.
              `"ladybug://./graph.lb"`) construct a
              `KnowledgeBase(graph_uri=identifier)`. A schemeless
              location (e.g. `":memory:"`) constructs a dual-store
              `KnowledgeBase(uri="duckdb://<loc>", graph_uri="ladybug://<loc>")`.
            - Dictionary: configuration dictionary.
            - Synalinks KnowledgeBase instance (returned unchanged).
            - ``None``: resolved against
              ``synalinks.set_default_knowledge_base(...)``; if no
              default is configured, returns a dual-store
              `KnowledgeBase()` (DuckDB + LadybugDB auto-paired under
              ``synalinks_home()``).

    Returns:
        (KnowledgeBase): A KnowledgeBase instance. Never ``None``.
    """
    if identifier is None:
        from synalinks.src.backend.config import default_knowledge_base

        identifier = default_knowledge_base()
        if identifier is None:
            # No explicit default — fall back to the auto-paired
            # DuckDB + LadybugDB instance under `synalinks_home()`.
            return KnowledgeBase()

    if isinstance(identifier, dict):
        obj = deserialize(identifier)
    elif isinstance(identifier, str):
        if _is_graph_uri(identifier):
            obj = KnowledgeBase(graph_uri=identifier)
        elif _is_sql_uri(identifier):
            obj = KnowledgeBase(uri=identifier)
        else:
            # Schemeless location (e.g. ":memory:") — mirror it to
            # both default adapters so the dual store is reachable.
            obj = KnowledgeBase(
                uri=f"duckdb://{identifier}",
                graph_uri=f"ladybug://{identifier}",
            )
    else:
        obj = identifier

    if isinstance(obj, KnowledgeBase):
        return obj
    raise ValueError(f"Could not interpret knowledge base identifier: {identifier}")
