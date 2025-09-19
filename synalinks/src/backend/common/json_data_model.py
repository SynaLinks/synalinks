# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import copy
import inspect

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.json_schema_utils import standardize_schema
from synalinks.src.backend.common.symbolic_data_model import SymbolicDataModel
from synalinks.src.utils.async_utils import run_maybe_nested
from synalinks.src.utils.naming import auto_name


@synalinks_export("synalinks.JsonDataModel")
class JsonDataModel:
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
        import json

        return json.dumps(self._schema, indent=2)

    def prettify_json(self):
        """Get a pretty version of the JSON object for display.

        Returns:
            (str): The indented JSON object.
        """
        import json

        return json.dumps(self._json, indent=2)

    def __add__(self, other):
        """Concatenates this data model with another.

        Args:
            other (JsonDataModel | DataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel): The concatenated data model.
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.Concat().call(self, other),
        )

    def __radd__(self, other):
        """Concatenates another data model with this one.

        Args:
            other (JsonDataModel | DataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel): The concatenated data model.
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.Concat().call(other, self),
        )

    def __and__(self, other):
        """Perform a `logical_and` with another data model.

        If one of them is None, output None. If both are provided,
        then concatenates this data model with the other.

        Args:
            other (JsonDataModel | DataModel): The other data model to concatenate with.

        Returns:
            (JsonDataModel | None): The concatenated data model or None
                based on the `logical_and` table.
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.And().call(self, other),
        )

    def __rand__(self, other):
        """Perform a `logical_and` (reverse) with another data model.

        If one of them is None, output None. If both are provided,
        then concatenates the other data model with this one.

        Args:
            other (JsonDataModel | DataModel): The other data model to concatenate with.

        Returns:
            (JsonDataModel | None): The concatenated data model or None
                based on the `logical_and` table.
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.And().call(other, self),
        )

    def __or__(self, other):
        """Perform a `logical_or` with another data model

        If one of them is None, output the other one. If both are provided,
        then concatenates this data model with the other.

        Args:
            other (JsonDataModel | DataModel): The other data model to concatenate with.

        Returns:
            (JsonDataModel | None): The concatenation of data model if both are provided,
                or the non-None data model or None if none are provided.
                (See `logical_or` table).
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.Or().call(self, other),
        )

    def __ror__(self, other):
        """Perform a `logical_or` (reverse) with another data model

        If one of them is None, output the other one. If both are provided,
        then concatenates the other data model with this one.

        Args:
            other (JsonDataModel | DataModel): The other data model to concatenate with.

        Returns:
            (JsonDataModel | None): The concatenation of data model if both are provided,
                or the non-None data model or None if none are provided.
                (See `logical_or` table).
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.Or().call(other, self),
        )

    def __xor__(self, other):
        """Perform a `logical_xor` with another data model.

        If one of them is `None`, output the other one. If both are provided,
        then the output is `None`.

        Args:
            other (SymbolicDataModel): The other data model to concatenate with.

        Returns:
            (SymbolicDataModel | None): `None` if both are
                provided, or the non-None data model if one is provided
                or `None` if none are provided. (See `logical_xor` table).
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.Xor().call(self, other),
        )

    def __rxor__(self, other):
        """Perform a `logical_xor` (reverse) with another data model.

        If one of them is None, output the other one. If both are provided,
        then concatenates the other data model with this one.

        Args:
            other (SymbolicDataModel | DataModel): The other data model to concatenate
                with.

        Returns:
            (SymbolicDataModel | None): `None` if both are
                provided, or the non-None data model if one is provided
                or `None` if none are provided. (See `logical_xor` table).
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.Xor().call(other, self),
        )

    def __contains__(self, other):
        """Check if the schema of `other` is contained in this one.

        Args:
            other (SymbolicDataModel | DataModel): The other data model to compare with.

        Returns:
            (bool): True if all properties of `other` are present in this one.
        """
        from synalinks.src.backend.common.json_schema_utils import contains_schema

        return contains_schema(self.get_schema(), other.get_schema())

    def factorize(self):
        """Factorizes the data model.

        Returns:
            (JsonDataModel): The factorized data model.
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.Factorize().call(self),
        )

    def in_mask(self, mask=None, recursive=True):
        """Applies a mask to **keep only** specified keys of the data model.

        Args:
            mask (list): The mask to be applied.
            recursive (bool): Optional. Whether to apply the mask recursively.
                Defaults to True.

        Returns:
            (JsonDataModel): The data model with the mask applied.
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.InMask(mask=mask, recursive=recursive).call(self),
        )

    def out_mask(self, mask=None, recursive=True):
        """Applies a mask to **remove** specified keys of the data model.

        Args:
            mask (list): The mask to be applied.
            recursive (bool): Optional. Whether to apply the mask recursively.
                Defaults to True.

        Returns:
            (JsonDataModel): The data model with the mask applied.
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.OutMask(mask=mask, recursive=recursive).call(self),
        )

    def prefix(self, prefix=None):
        """Add a prefix to **all** the data model fields (non-recursive).

        Args:
            prefix (str): the prefix to add.

        Returns:
            (JsonDataModel): The data model with the prefix added.
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.Prefix(prefix=prefix).call(self),
        )

    def suffix(self, suffix=None):
        """Add a suffix to **all** the data model fields (non-recursive).

        Args:
            suffix (str): the suffix to add.

        Returns:
            (JsonDataModel): The data model with the suffix added.
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.Suffix(suffix=suffix).call(self),
        )

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
            clone.name = auto_name(self.name + "_clone")
        return clone

    def get_nested_entity(self, key):
        """Retrieve a nested Entity and convert it to a JsonDataModel"""
        json = copy.deepcopy(self.get(key))
        if "label" in json:
            schema_key = json.get("label")
        else:
            return None
        schema = copy.deepcopy(self.get_schema().get("$defs").get(schema_key))

        defs = {}
        for obj_key, obj_schema in copy.deepcopy(self.get_schema().get("$defs")).items():
            if str(schema).find(f"#/$defs/{obj_key}") > 0:
                defs[obj_key] = obj_schema

        if defs:
            schema.update({"$defs": defs})

        if schema:
            return JsonDataModel(json=json, schema=schema, name=self.name + "_" + key)
        else:
            return None

    def get_nested_entity_list(self, key):
        """Retrieve a nested Entity list and convert it to a list of JsonDataModel"""
        json = self.get(key)
        outputs = []
        for i, data_model_json in enumerate(json):
            if "label" in data_model_json:
                schema_key = data_model_json.get("label")
                schema = self.get_schema().get("$defs").get(schema_key)
                defs = {}
                for obj_key, obj_schema in self.get_schema().get("$defs").items():
                    if str(schema).find(f"#/$defs/{obj_key}") > 0:
                        defs[obj_key] = obj_schema
                if defs:
                    schema.update({"$defs": defs})

                outputs.append(
                    JsonDataModel(
                        json=data_model_json,
                        schema=schema,
                        name=self.name + "_" + key + f"_{i}",
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
