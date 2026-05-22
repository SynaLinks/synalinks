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
    """Resolve (schema, label) for one path endpoint.

    Mirrors the three-way contract used in the single-endpoint
    entity-search modules: derive schema from entity_model if needed,
    derive label from schema's title if needed, raise if neither is
    available. ``side`` is ``"subj"`` or ``"obj"`` and is used only
    in the error message.
    """
    if schema is None and entity_model is not None:
        schema = entity_model.get_schema()
    if schema is None and label is None:
        raise ValueError(
            f"One of `{side}_schema`, `{side}_entity_model`, or "
            f"`{side}_label` is required"
        )
    if label is None:
        label = schema.get("title")
        if not label:
            raise ValueError(
                f"Could not infer `{side}_label` from `{side}_schema` "
                f"(no `title`); pass `{side}_label` explicitly."
            )
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
