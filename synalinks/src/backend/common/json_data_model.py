# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import copy
import inspect

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.data_model_operators import DataModelOperators
from synalinks.src.backend.common.data_model_operators import DataModelTransforms
from synalinks.src.backend.common.json_schema_utils import standardize_schema
from synalinks.src.backend.common.symbolic_data_model import SymbolicDataModel
from synalinks.src.utils.async_utils import run_maybe_nested
from synalinks.src.utils.naming import auto_name


@synalinks_export("synalinks.JsonDataModel")
class JsonDataModel(DataModelOperators, DataModelTransforms):
    """A backend-independent dynamic data model.

    This structure is the one flowing in the pipelines as
    the backend data models are only used for the variable/data model declaration.

    Args:
        schema (dict): The JSON object's schema. If not provided,
            uses the data model to infer it.
        json (dict): The JSON object's json. If not provided,
            uses the data model to infer it.
        data_model (DataModel | JsonDataModel): The data model to use to
            infer the schema and json.
        name (str): Optional. The name of the data model, automatically
            inferred if not provided.

    Examples:

    **Creating a `JsonDataModel` with a DataModel's schema and json:**

    ```python
    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    json = {"query": "What is the capital of France?"}

    data_model = JsonDataModel(
        schema=Query.get_schema(),
        json=json,
    )
    ```

    **Creating a `JsonDataModel` with a data_model:**

    ```python
    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    query_instance = Query(
        query="What is the capital of France?"
    )
    data_model = JsonDataModel(
        data_model=query_instance,
    )
    ```

    **Creating a `JsonDataModel` with `to_json_data_model()`:**

    ```python
    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    data_model = Query(
        query="What is the capital of France?",
    ).to_json_data_model()
    ```
    """

    def __init__(
        self,
        schema=None,
        json=None,
        data_model=None,
        name=None,
    ):
        name = name or auto_name(self.__class__.__name__)
        self.name = name
        self._schema = None
        self._json = None

        if not data_model and not schema and not json:
            raise ValueError("Initializing without arguments is not permited.")
        if not schema and not data_model:
            raise ValueError(
                "You should specify at least one argument between "
                "`data_model` or `schema`."
            )
        if not schema and not json and not data_model:
            raise ValueError(
                "You should specify at least one argument between `data_model` or `json`."
            )
        if data_model:
            if not schema:
                schema = data_model.get_schema()
            if not json:
                if inspect.isclass(data_model):
                    raise ValueError(
                        "Couldn't get the JSON data from the `data_model` argument, "
                        "the `data_model` needs to be instanciated. "
                        f"Received data_model={data_model}."
                    )
                json = data_model.get_json()

        self._schema = standardize_schema(schema)
        self._json = json

    def to_symbolic_data_model(self):
        """Converts the JsonDataModel to a SymbolicDataModel.

        Returns:
            (SymbolicDataModel): The symbolic data model.
        """
        return SymbolicDataModel(schema=self._schema)

    def get_json(self):
        """Gets the current json of the JSON object.

        Returns:
            (dict): The current json of the JSON object.
        """
        return self._json

    def get_schema(self):
        """Gets the schema of the JSON object.

        Returns:
            (dict): The JSON schema.
        """
        return self._schema

    def prettify_schema(self):
        """Get a pretty version of the JSON schema for display.

        Returns:
            (dict): The indented JSON schema.
        """
        import orjson

        return orjson.dumps(self._schema, option=orjson.OPT_INDENT_2).decode()

    def prettify_json(self):
        """Get a pretty version of the JSON object for display.

        Returns:
            (str): The indented JSON object.
        """
        import orjson

        return orjson.dumps(self._json, option=orjson.OPT_INDENT_2).decode()

    def _apply_op(self, op, *args):
        return run_maybe_nested(op.call(*args))

    def get(self, key, default=None):
        """Get wrapper to make it easier to access JSON fields.

        Args:
            key (str): The key to access.
            default (any): The default value if key not found.
        """
        return self._json.get(key, default)

    def __getitem__(self, key):
        """Get item wrapper to make it easier to access JSON fields.

        Args:
            key (str): The key to access.
        """
        return self._json[key]

    def keys(self):
        """Keys wrapper to make it easier to access JSON fields."""
        return self._json.keys()

    def values(self):
        """Values wrapper to make it easier to access JSON fields."""
        return self._json.values()

    def items(self):
        """Items wrapper to make it easier to access JSON fields."""
        return self._json.items()

    def update(self, kv_dict):
        """Update wrapper to make it easier to modify JSON fields.

        Args:
            kv_dict (dict): The key/json dict to update.
        """
        self._json.update(kv_dict)

    def clone(self, name=None):
        """Clone a data model and give it a different name."""

        clone = copy.deepcopy(self)
        if name:
            clone.name = name
        else:
            clone.name = auto_name("clone_" + self.name)
        return clone

    @staticmethod
    def _resolve_ref_schema(spec, defs):
        """Resolve a property/items spec to its target schema in ``defs``.

        Handles a direct ``$ref`` and an ``anyOf`` / ``oneOf`` union with a
        single ``$ref`` variant (the common "Optional[X]" / single-type
        list shapes). Returns ``None`` when the spec doesn't point at a
        single resolvable def; callers fall back to skipping the item.
        """
        if not isinstance(spec, dict):
            return None
        ref = spec.get("$ref")
        if not ref:
            for union_key in ("anyOf", "oneOf"):
                variants = [v for v in spec.get(union_key, []) if v.get("$ref")]
                if len(variants) == 1:
                    ref = variants[0]["$ref"]
                    break
        if not ref:
            return None
        return copy.deepcopy(defs.get(ref.rsplit("/", 1)[-1]))

    def _attach_nested_defs(self, schema, defs):
        """Attach the subset of ``defs`` that ``schema`` references."""
        nested_defs = {}
        for obj_key, obj_schema in copy.deepcopy(defs).items():
            if str(schema).find(f"#/$defs/{obj_key}") > 0:
                nested_defs[obj_key] = obj_schema
        if nested_defs:
            schema.update({"$defs": nested_defs})
        return schema

    def get_nested_entity(self, key):
        """Retrieve a nested Entity and convert it to a JsonDataModel.

        The entity's `label` field is used as the discriminator to look up
        its schema in the parent's `$defs`. When the label value isn't a
        `$defs` key (the free-form case, where `label` is open data rather
        than a `Literal` matching the class name), the field's *declared*
        schema (its `$ref`) is used as a fallback. Returns ``None`` when the
        value at ``key`` has no ``label`` or when no schema resolves.

        Args:
            key (str): The field name holding the nested entity.

        Returns:
            (JsonDataModel | None): A typed JsonDataModel wrapping the
                nested entity, or ``None`` if the value isn't an entity.
        """
        json = copy.deepcopy(self.get(key))
        if not json or "label" not in json:
            return None
        parent_schema = self.get_schema()
        defs = parent_schema.get("$defs") or {}
        schema = copy.deepcopy(defs.get(json.get("label")))
        if not schema:
            # Free-form: fall back to the field's declared type.
            field_spec = (parent_schema.get("properties") or {}).get(key)
            schema = self._resolve_ref_schema(field_spec, defs)
        if not schema:
            return None

        schema = self._attach_nested_defs(schema, defs)
        return JsonDataModel(json=json, schema=schema, name=key + "_" + self.name)

    def get_nested_entity_list(self, key):
        """Retrieve a nested Entity list and convert it to typed JsonDataModels.

        Each item is resolved against the parent's `$defs` using its
        ``label`` field as discriminator. When the label value isn't a
        `$defs` key (the free-form case, where `label` is open data rather
        than a `Literal` matching the class name), the list's *declared*
        item schema (`items.$ref`) is used as a fallback. Items without a
        ``label`` field or with no resolvable schema are skipped.

        Args:
            key (str): The field name holding the list of nested entities.

        Returns:
            (list[JsonDataModel]): One JsonDataModel per resolved entity.
        """
        items = self.get(key) or []
        outputs = []
        parent_schema = self.get_schema()
        defs = parent_schema.get("$defs") or {}
        # Declared item schema for this list field, resolved once and reused
        # as the fallback for free-form labels that aren't $defs keys.
        field_spec = (parent_schema.get("properties") or {}).get(key) or {}
        field_item_schema = self._resolve_ref_schema(field_spec.get("items"), defs)
        for i, item_json in enumerate(items):
            if "label" not in item_json:
                continue
            schema = copy.deepcopy(defs.get(item_json.get("label")))
            if not schema:
                schema = copy.deepcopy(field_item_schema)
            if not schema:
                continue
            schema = self._attach_nested_defs(schema, defs)
            outputs.append(
                JsonDataModel(
                    json=item_json,
                    schema=schema,
                    name=key + f"_{i}_" + self.name,
                )
            )
        return outputs

    def __repr__(self):
        return f"<JsonDataModel schema={self._schema}, json={self._json}>"


@synalinks_export(
    [
        "synalinks.utils.is_json_data_model",
        "synalinks.backend.is_json_data_model",
    ]
)
def is_json_data_model(x):
    """Returns whether `x` is a backend-independent data model.

    Args:
        x (any): The object to check.

    Returns:
        (bool): True if `x` is a backend-independent data model, False otherwise.
    """
    return isinstance(x, JsonDataModel)
