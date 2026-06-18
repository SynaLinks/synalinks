# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Internal helpers shared by the `Path*Search` modules.

Path modules take TWO entity endpoints (subj + obj). Each endpoint
has the same three-way `schema` / `entity_model` / `label` contract
as the single-endpoint `Entity*Search` modules, and serializes the
same way. This module centralizes that logic so each path module
stays a thin wrapper around its underlying KB method.
"""

from synalinks.src.backend import is_symbolic_data_model
from synalinks.src.saving import serialization_lib


def resolve_endpoint(schema, entity_model, label, side):
    """Resolve ``(schema, label)`` for one path endpoint.

    Mirrors the single-endpoint entity-search contract: derive schema from
    ``entity_model`` if needed and the label from the schema's ``title`` if
    needed. The label is **optional** — when neither a label nor a schema to
    derive it from is given it stays ``None``, and the caller lets the language
    model infer the endpoint label per call. ``side`` (``"subj"`` / ``"obj"``)
    is kept for symmetry with callers.
    """
    if schema is None and entity_model is not None:
        schema = entity_model.get_schema()
    if label is None and schema is not None:
        label = schema.get("title") or None
    return schema, label


def serialize_entity_model(em, name):
    """Serialize one path endpoint's entity_model for `get_config`."""
    if em is None:
        return None
    if not is_symbolic_data_model(em):
        em = em.to_symbolic_data_model(name=name)
    return serialization_lib.serialize_synalinks_object(em)


def deserialize_entity_model(serialized):
    """Deserialize one path endpoint's entity_model from `from_config`."""
    if serialized is None:
        return None
    return serialization_lib.deserialize_synalinks_object(serialized)
