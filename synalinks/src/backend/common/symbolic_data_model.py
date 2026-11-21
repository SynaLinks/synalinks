# Modified from: keras/src/backend/common/keras_tensor.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import orjson

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.data_model_operators import DataModelOperators
from synalinks.src.backend.common.data_model_operators import DataModelTransforms
from synalinks.src.backend.common.data_model_operators import apply_symbolic_op
from synalinks.src.backend.common.json_schema_utils import is_schema_equal
from synalinks.src.backend.common.json_schema_utils import standardize_schema
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.naming import auto_name


@synalinks_export("synalinks.SymbolicDataModel")
class SymbolicDataModel(DataModelOperators, DataModelTransforms, SynalinksSaveable):
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
        return orjson.dumps(self._schema, option=orjson.OPT_INDENT_2).decode()

    def __repr__(self):
        return f"<SymbolicDataModel schema={self._schema}, name={self.name}>"

    _apply_op = apply_symbolic_op

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
