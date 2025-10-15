# Modified from: keras/src/backend/common/keras_tensor.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.json_schema_utils import is_schema_equal
from synalinks.src.backend.common.json_schema_utils import standardize_schema
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.async_utils import run_maybe_nested
from synalinks.src.utils.naming import auto_name


@synalinks_export("synalinks.SymbolicDataModel")
class SymbolicDataModel(SynalinksSaveable):
    """A symbolic backend-independent data model.

    A `SymbolicDataModel` is a container for a JSON schema and can be used to represent
        data structures in a backend-agnostic way. It can record history and is used in
        symbolic operations (in the Functional API and to compute output specs).

    A "symbolic data model" can be understood as a placeholder for data specification,
        it does not contain any actual data, only a schema. It can be used for building
        Functional models, but it cannot be used in actual computations.

    Args:
        data_model (DataModel): Optional. The data_model used to extract the schema.
        schema (dict): Optional. The JSON schema to be used. If the schema is not
            provided, the data_model argument should be used to infer it.
        record_history (bool): Optional. Boolean indicating if the history
            should be recorded. Defaults to `True`.
        name (str): Optional. A unique name for the data model. Automatically generated
            if not set.

    Examples:

    **Creating a `SymbolicDataModel` with a backend data model metaclass:**

    ```python
    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    data_model = SymbolicDataModel(data_model=Query)
    ```

    **Creating a `SymbolicDataModel` with a backend data model metaclass's schema:**

    ```python
    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    data_model = SymbolicDataModel(schema=Query.get_schema())
    ```

    **Creating a `SymbolicDataModel` with `to_symbolic_data_model()`:**

    using a backend data model metaclass

    ```python
    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    data_model = Query.to_symbolic_data_model()
    ```
    """

    def __init__(
        self,
        data_model=None,
        schema=None,
        record_history=True,
        name=None,
    ):
        self.name = name or auto_name(self.__class__.__name__)
        self._record_history = record_history
        self._schema = None
        if not schema and not data_model:
            raise ValueError(
                "You should specify at least one argument between "
                "`data_model` or `schema`"
            )
        if schema and data_model:
            if not is_schema_equal(schema, data_model.get_schema()):
                raise ValueError(
                    "Attempting to create a SymbolicDataModel "
                    "with both `schema` and `data_model` argument "
                    "but their schemas are incompatible "
                    f"received schema={schema} and "
                    f"data_model.get_schema()={data_model.get_schema()}."
                )
            self._schema = standardize_schema(schema)
        else:
            if schema:
                self._schema = standardize_schema(schema)
            if data_model:
                self._schema = standardize_schema(data_model.get_schema())

    @property
    def record_history(self):
        """Whether the history is being recorded."""
        return self._record_history

    @record_history.setter
    def record_history(self, value):
        self._record_history = value

    def get_schema(self):
        """Gets the JSON schema of the data model.

        Returns:
            (dict): The JSON schema.
        """
        return self._schema

    def get_json(self):
        """Gets the current value of the JSON object (impossible in `SymbolicDataModel`).

        Implemented to help the user to identifying issues.

        Raises:
            ValueError: The help message.
        """
        raise ValueError(
            "Attempting to retrieve the JSON value from a symbolic data model "
            "this operation is not possible, make sure that your `call()` "
            "is correctly implemented, if so then you likely need to implement "
            " `compute_output_spec()` in your subclassed module."
        )

    def prettify_schema(self):
        """Get a pretty version of the JSON schema for display.

        Returns:
            (dict): The indented JSON schema.
        """
        return json.dumps(self._schema, indent=2)

    def __repr__(self):
        return f"<SymbolicDataModel schema={self._schema}, name={self.name}>"

    def __add__(self, other):
        """Concatenates this data model with another.

        Args:
            other (SymbolicDataModel | DataModel):
                The other data model to concatenate with.

        Returns:
            (SymbolicDataModel): The concatenated data model.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.Concat().symbolic_call(self, other))

    def __radd__(self, other):
        """Concatenates (reverse) another data model with this one.

        Args:
            other (SymbolicDataModel | DataModel):
                The other data model to concatenate with.

        Returns:
            (SymbolicDataModel): The concatenated data model.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.Concat().symbolic_call(other, self))

    def __and__(self, other):
        """Perform a `logical_and` with another data model.

        If one of them is None, output None. If both are provided,
        then concatenates this data model with the other.

        Args:
            other (SymbolicDataModel | DataModel): The other data model to concatenate
                with.

        Returns:
            (SymbolicDataModel | None): The concatenated data model or None
                based on the `logical_and` table.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.And().symbolic_call(self, other))

    def __rand__(self, other):
        """Perform a `logical_and` (reverse) with another data model.

        If one of them is None, output None. If both are provided,
        then concatenates the other data model with this one.

        Args:
            other (SymbolicDataModel | DataModel): The other data model to concatenate
                with.

        Returns:
            (SymbolicDataModel | None): The concatenated data model or None
                based on the `logical_and` table.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.And().symbolic_call(other, self))

    def __or__(self, other):
        """Perform a `logical_or` with another data model

        If one of them is None, output the other one. If both are provided,
        then concatenates this data model with the other.

        Args:
            other (SymbolicDataModel): The other data model to concatenate with.

        Returns:
            (SymbolicDataModel | None): The concatenation of data model if both are
                provided, or the non-None data model or None if none are provided.
                (See `logical_or` table).
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.Or().symbolic_call(self, other))

    def __ror__(self, other):
        """Perform a `logical_or` (reverse) with another data model

        If one of them is None, output the other one. If both are provided,
        then concatenates the other data model with this one.

        Args:
            other (SymbolicDataModel | DataModel): The other data model to concatenate
                with.

        Returns:
            (SymbolicDataModel | None): The concatenation of data model if both are
                provided, or the non-None data model or None if none are provided.
                (See `logical_or` table).
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.Or().symbolic_call(other, self))

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

        return run_maybe_nested(ops.Xor().symbolic_call(self, other))

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

        return run_maybe_nested(ops.Xor().symbolic_call(other, self))

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
            (SymbolicDataModel): The factorized data model.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.Factorize().symbolic_call(self))

    def in_mask(self, mask=None, recursive=True):
        """Applies a mask to **keep only** specified keys of the data model.

        Args:
            mask (list): The mask to be applied (list of keys).
            recursive (bool): Optional. Whether to apply the mask recursively.
                Defaults to `True`.

        Returns:
            (SymbolicDataModel): The data model with the mask applied.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.InMask(mask=mask, recursive=True).symbolic_call(self))

    def out_mask(self, mask=None, recursive=True):
        """Applies an mask to **remove** specified keys of the data model.

        Args:
            mask (list): The mask to be applied (list of keys).
            recursive (bool): Optional. Whether to apply the mask recursively.
                Defaults to `True`.

        Returns:
            (SymbolicDataModel): The data model with the mask applied.
        """
        from synalinks.src import ops

        return run_maybe_nested(
            ops.OutMask(mask=mask, recursive=True).symbolic_call(self)
        )

    def prefix(self, prefix=None):
        """Add a prefix to **all** the data model fields (non-recursive).

        Args:
            prefix (str): the prefix to add

        Returns:
            (SymbolicDataModel): The data model with the prefix added.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.Prefix(prefix=prefix).symbolic_call(self))

    def suffix(self, suffix=None):
        """Add a suffix to **all** the data model fields (non-recursive).

        Args:
            suffix (str): the suffix to add

        Returns:
            (SymbolicDataModel): The data model with the suffix added.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.Suffix(suffix=suffix).symbolic_call(self))

    def get(self, key, default=None):
        """Get wrapper to make easier to access fields.

        Implemented to help the user to identifying issues.

        Args:
            key (str): The key to access.
            default (any): The default value if key not found.

        Raises:
            ValueError: The help message.
        """
        raise ValueError(
            f"Attempting to get '{key}' from a symbolic data model "
            "this operation is not possible, make sure that your `call()` "
            "is correctly implemented, if so then you likely need to implement "
            " `compute_output_spec()` in your subclassed module."
        )

    def __getitem__(self, key):
        """Get item wrapper to make it easier to access JSON fields.

        Implemented to help the user to identifying issues.

        Args:
            key (str): The key to access.
        """
        raise ValueError(
            f"Attempting to get '{key}' from a symbolic data model "
            "this operation is not possible, make sure that your `call()` "
            "is correctly implemented, if so then you likely need to implement "
            " `compute_output_spec()` in your subclassed module."
        )

    def keys(self):
        """Keys wrapper to make it easier to access JSON fields."""
        if "properties" in self.get_schema():
            return self.get_schema()["properties"].keys()
        else:
            return []

    def values(self):
        """Values wrapper to make it easier to access JSON fields.

        Implemented to help the user to identifying issues.
        """
        raise ValueError(
            "Attempting to get '.values()' from a symbolic data model "
            "this operation is not possible, make sure that your `call()` "
            "is correctly implemented, if so then you likely need to implement "
            " `compute_output_spec()` in your subclassed module."
        )

    def items(self):
        """Items wrapper to make it easier to access JSON fields.

        Implemented to help the user to identifying issues.
        """
        raise ValueError(
            "Attempting to get '.items()' from a symbolic data model "
            "this operation is not possible, make sure that your `call()` "
            "is correctly implemented, if so then you likely need to implement "
            " `compute_output_spec()` in your subclassed module."
        )

    def update(self, kv_dict):
        """Update wrapper to make easier to modify fields.

        Implemented to help the user to identifying issues.

        Args:
            kv_dict (dict): The key/value dict to update.

        Raises:
            ValueError: The help message.
        """
        raise ValueError(
            f"Attempting to update keys {list(kv_dict.keys())} from a symbolic "
            "data model this operation is not possible, make sure that your `call()` "
            "is correctly implemented, if so then you likely need to implement "
            " `compute_output_spec()` in your subclassed module."
        )

    def clone(self, name=None):
        """Clone a symbolic data model and give it a different name."""
        import copy

        clone = copy.deepcopy(self)
        if name:
            clone.name = name
        else:
            clone.name = "clone_" + self.name
        return clone

    def get_config(self):
        config = {
            "name": self.name,
            "schema": self.get_schema(),
            "record_history": self.record_history,
        }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def any_symbolic_data_models(args=None, kwargs=None):
    """Checks if any of the arguments are symbolic data models.

    Args:
        args (tuple): Optional. The positional arguments to check.
        kwargs (dict): Optional. The keyword arguments to check.

    Returns:
        (bool): True if any of the arguments are symbolic data models, False otherwise.
    """
    args = args or ()
    kwargs = kwargs or {}
    for x in tree.flatten((args, kwargs)):
        if is_symbolic_data_model(x):
            return True
    return False


@synalinks_export(
    [
        "synalinks.utils.is_symbolic_data_model",
        "synalinks.backend.is_symbolic_data_model",
    ]
)
def is_symbolic_data_model(x):
    """Returns whether `x` is a synalinks data model.

    A "synalinks data model" is a *symbolic data model*, such as a data model
    that was created via `Input()`. A "symbolic data model"
    can be understood as a placeholder for data specification -- it does not
    contain any actual data, only a schema.
    It can be used for building Functional models, but it
    cannot be used in actual computations.

    Args:
        x (any): The object to check.

    Returns:
        (bool): True if `x` is a symbolic data model, False otherwise.
    """
    return isinstance(x, SymbolicDataModel)
