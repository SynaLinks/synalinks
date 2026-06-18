# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""Shared helpers for retriever modules that let the LM **infer** the target.

When a retriever is constructed without a fixed target (no ``table_name`` /
``label`` / endpoint label), it must let the language model choose one per call.
These helpers build an extra JSON-schema field — constrained to a JSON-schema
``enum`` of the knowledge base's actual tables / entity labels / relation labels,
so the LM cannot pick a nonexistent target — and concatenate it onto the query
generator's input schema (`concatenate_schema`). The retriever then reads the
inferred value back out of the generated query.
"""

from synalinks.src.backend import concatenate_schema


def enum_field_schema(field, description, values):
    """One-property object schema for an LM-inferred identifier.

    A required string ``field``; constrained to ``values`` (the KB's actual
    tables / labels) as a JSON-schema ``enum`` when any are known, so the LM must
    pick a real target. With no known values it stays a free string (the
    description should then name the options).
    """
    prop = {"type": "string", "description": description}
    values = [v for v in (values or []) if v]
    if values:
        prop["enum"] = sorted(dict.fromkeys(values))
    return {
        "type": "object",
        "additionalProperties": False,
        "title": field.replace("_", " ").title(),
        "properties": {field: prop},
        "required": [field],
    }


def concat_infer_fields(base_schema, specs):
    """Concatenate inferred enum fields onto ``base_schema``.

    ``specs`` is a list of ``(field, description, values)``. Returns the combined
    JSON schema (the query generator's target) so the LM emits the search queries
    AND each inferred identifier in one structured output.
    """
    schema = base_schema
    for field, description, values in specs:
        schema = concatenate_schema(schema, enum_field_schema(field, description, values))
    return schema


def _titles(models):
    out = []
    for model in models or []:
        try:
            title = model.get_schema().get("title")
        except Exception:  # noqa: BLE001 - introspection is best-effort
            title = None
        if title:
            out.append(title)
    return out


def kb_table_names(knowledge_base):
    """Available SQL table names in the knowledge base (best-effort, may be [])."""
    try:
        return _titles(knowledge_base.get_symbolic_data_models())
    except Exception:  # noqa: BLE001 - no adapter / not connected
        return []


def kb_entity_labels(knowledge_base):
    """Available graph entity (node) labels (best-effort, may be [])."""
    try:
        return _titles(knowledge_base.get_symbolic_entities())
    except Exception:  # noqa: BLE001 - no graph adapter
        return []


def kb_relation_labels(knowledge_base):
    """Available graph relation labels (best-effort, may be [])."""
    try:
        return _titles(knowledge_base.get_symbolic_relations())
    except Exception:  # noqa: BLE001 - no graph adapter
        return []
