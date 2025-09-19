# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import inspect

import pydantic
from typing_extensions import ClassVar

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.json_data_model import JsonDataModel
from synalinks.src.backend.common.symbolic_data_model import SymbolicDataModel
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.async_utils import run_maybe_nested

IS_THREAD_SAFE = True


class MetaDataModel(type(pydantic.BaseModel)):
    """The metaclass data model.

    This class defines operations at the metaclass level.
    Allowing to use Synalinks Python operators with `DataModel` types.
    """

    def __add__(cls, other):
        """Concatenates this data model with another.

        Args:
            other (SymbolicDataModel | DataModel):
                The other data model to concatenate with.

        Returns:
            (SymbolicDataModel): The concatenated data model.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.Concat().symbolic_call(cls, other))

    def __radd__(cls, other):
        """Concatenates (reverse) another data model with this one.

        Args:
            other (SymbolicDataModel | DataModel):
                The other data model to concatenate with.

        Returns:
            (SymbolicDataModel): The concatenated data model.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.Concat().symbolic_call(other, cls))

    def __and__(cls, other):
        """Perform a `logical_and` with another data model.

        If one of them is `None`, output `None`. If both are provided,
        then concatenates this data model with the other.

        Args:
            other (SymbolicDataModel | DataModel): The other data model to concatenate
                with.

        Returns:
            (SymbolicDataModel | None): The concatenated data model or `None`
                based on the `logical_and` table.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.And().symbolic_call(cls, other))

    def __rand__(cls, other):
        """Perform a `logical_and` (reverse) with another data model.

        If one of them is `None`, output `None`. If both are provided,
        then concatenates the other data model with this one.

        Args:
            other (SymbolicDataModel | DataModel): The other data model to concatenate
                with.

        Returns:
            (SymbolicDataModel | None): The concatenated data model or `None`
                based on the `logical_and` table.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.And().symbolic_call(other, cls))

    def __or__(cls, other):
        """Perform a `logical_or` with another data model.

        If one of them is `None`, output the other one. If both are provided,
        then concatenates this data model with the other.

        Args:
            other (SymbolicDataModel): The other data model to concatenate with.

        Returns:
            (SymbolicDataModel | None): The concatenation of data model if both are
                provided, or the non-None data model or None if none are provided.
                (See `logical_or` table).
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.Or().symbolic_call(cls, other))

    def __ror__(cls, other):
        """Perform a `logical_or` (reverse) with another data model.

        If one of them is `None`, output the other one. If both are provided,
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

        return run_maybe_nested(ops.Or().symbolic_call(other, cls))

    def __xor__(cls, other):
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

        return run_maybe_nested(ops.Xor().symbolic_call(cls, other))

    def __rxor__(cls, other):
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

        return run_maybe_nested(ops.Xor().symbolic_call(other, cls))

    def __contains__(cls, other):
        """Check if the schema of `other` is contained in this one.

        Args:
            other (SymbolicDataModel | DataModel): The other data model to compare with.

        Returns:
            (bool): True if all properties of `other` are present in this one.
        """
        from synalinks.src.backend.common.json_schema_utils import contains_schema

        return contains_schema(cls.get_schema(), other.get_schema())

    def factorize(cls):
        """Factorizes the data model.

        Returns:
            (SymbolicDataModel): The factorized data model.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.Factorize().symbolic_call(cls))

    def in_mask(cls, mask=None, recursive=True):
        """Applies a mask to **keep only** specified keys of the data model.

        Args:
            mask (list): The mask to be applied (list of keys).
            recursive (bool): Optional. Whether to apply the mask recursively.
                Defaults to `True`.

        Returns:
            (SymbolicDataModel): The data model with the mask applied.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.InMask(mask=mask, recursive=True).symbolic_call(cls))

    def out_mask(cls, mask=None, recursive=True):
        """Applies an mask to **remove** specified keys of the data model.

        Args:
            mask (list): The mask to be applied (list of keys).
            recursive (bool): Optional. Whether to apply the mask recursively.
                Defaults to `True`.

        Returns:
            (SymbolicDataModel): The data model with the mask applied.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.OutMask(mask=mask, recursive=True).symbolic_call(cls))

    def prefix(cls, prefix=None):
        """Add a prefix to **all** the data model fields (non-recursive).

        Args:
            prefix (str): the prefix to add

        Returns:
            (SymbolicDataModel): The data model with the prefix added.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.Prefix(prefix=prefix).symbolic_call(cls))

    def suffix(cls, suffix=None):
        """Add a suffix to **all** the data model fields (non-recursive).

        Args:
            suffix (str): the suffix to add

        Returns:
            (SymbolicDataModel): The data model with the suffix added.
        """
        from synalinks.src import ops

        return run_maybe_nested(ops.Suffix(suffix=suffix).symbolic_call(cls))


class DataModel(pydantic.BaseModel, SynalinksSaveable, metaclass=MetaDataModel):
    """The backend-dependent data model.

    This data model uses Pydantic to provide, JSON schema inference
    and JSON serialization.

    Examples:

    **Creating a DataModel for structured output**

    ```python
    class AnswerWithReflection(synalinks.DataModel):
        thinking: str = synalinks.Field(
            description="Your step by step thinking",
        )
        reflection: str = synalinks.Field(
            description="The reflection about your thinking",
        )
        answer: str = synalinks.Field(
            description="The correct answer",
        )

    language_model = synalinks.LanguageModel("ollama/mistral")

    generator = synalinks.Generator(
        data_model=AnswerWithReflection,
        language_model=language_model,
    )
    ```
    """

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(extra="forbid")

    @classmethod
    def get_schema(cls):
        """Gets the JSON schema of the data model.

        Returns:
            (dict): The JSON schema.
        """
        return cls.model_json_schema()

    @classmethod
    def keys(cls):
        """Gets the JSON properties keys of the data model.

        Returns:
            (dict): The JSON schema.
        """
        schema = cls.get_schema()
        if "properties" in schema:
            return schema["properties"].keys()
        else:
            return []

    @classmethod
    def prettify_schema(cls):
        """Get a pretty version of the JSON schema for display.

        Returns:
            (str): The indented JSON schema.
        """
        import json

        return json.dumps(cls.get_schema(), indent=2)

    @classmethod
    def to_symbolic_data_model(cls, name=None):
        """Converts the data model to a symbolic data model.

        Args:
            name (str): Optional. The name of the symbolic data model.
                If None, a name will be given automatically.

        Returns:
            (SymbolicDataModel): The symbolic data model.
        """
        return SymbolicDataModel(schema=cls.get_schema(), name=name)

    def get_json(self):
        """Gets the JSON value of the data model.

        Returns:
            (dict): The JSON value.
        """
        return self.model_dump(mode="json")

    def prettify_json(self):
        """Get a pretty version of the JSON object for display.

        Returns:
            (str): The indented JSON object.
        """
        import json

        return json.dumps(self.get_json(), indent=2)

    def __repr__(self):
        return f"<DataModel json={self.get_json()}, schema={self.get_schema()}>"

    def get(self, key, default=None):
        """Get wrapper to make it easier to access JSON fields.

        Args:
            key (str): The key to access.
            default (any): The default value if key not found.
        """
        try:
            self.__getattribute__(key)
        except Exception:
            return default

    def __getitem__(self, key):
        """Get item wrapper to make it easier to access JSON fields.

        Args:
            key (str): The key to access.
        """
        return self.__getattribute__(key)

    def to_json_data_model(self, name=None):
        """Converts the data model to a backend-independent data model.

        Args:
            name (str): Optional. The name of the json data model.
                If None, a name will be given automatically.

        Returns:
            (JsonDataModel): The backend-independent data model.
        """
        return JsonDataModel(
            schema=self.get_schema(),
            json=self.get_json(),
            name=name,
        )

    def __add__(self, other):
        """Concatenates this data model with another.

        Args:
            other (JsonDataModel | DataModel | SymbolicDataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel | SymbolicDataModel): The concatenated data model.
                If one of them is a metaclass or symbolic data model,
                then output a `SymbolicDataModel`.
        """
        from synalinks.src import ops

        if any_meta_class(self, other):
            return run_maybe_nested(ops.Concat().symbolic_call(self, other))
        else:
            return run_maybe_nested(ops.Concat()(self, other))

    def __radd__(self, other):
        """Concatenates another data model with this one.

        Args:
            other (JsonDataModel | DataModel | SymbolicDataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel | SymbolicDataModel): The concatenated data model.
                If one of them is a metaclass or symbolic data model,
                then output a `SymbolicDataModel`.
        """
        from synalinks.src import ops

        if any_meta_class(self, other):
            return run_maybe_nested(
                ops.Concat().symbolic_call(other, self),
            )
        else:
            return run_maybe_nested(
                ops.Concat()(other, self),
            )

    def __and__(self, other):
        """Perform a `logical_and` with another data model.

        If one of them is None, output None. If both are provided,
        then concatenates the other data model with this one.

        If the other is a metaclass or symbolic data model, output a symbolic data model.

        Args:
            other (JsonDataModel | SymbolicDataModel | DataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel | SymbolicDataModel | None): The concatenated data model or
                `None` based on the `logical_and` table.
        """
        from synalinks.src import ops

        if any_meta_class(self, other):
            return run_maybe_nested(
                ops.Add().symbolic_call(self, other),
            )
        else:
            return run_maybe_nested(
                ops.Add()(self, other),
            )

    def __rand__(self, other):
        """Perform a `logical_and` (reverse) with another data model.

        If one of them is None, output None. If both are provided,
        then concatenates the other data model with this one.

        If the other is a metaclass or symbolic data model, output a symbolic data model.

        Args:
            other (JsonDataModel | SymbolicDataModel | DataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel | SymbolicDataModel | None): The concatenated data model or
                `None` based on the `logical_and` table.
        """
        from synalinks.src import ops

        if any_meta_class(other, self):
            return run_maybe_nested(
                ops.Add().symbolic_call(other, self),
            )
        else:
            return run_maybe_nested(
                ops.Add()(other, self),
            )

    def __or__(self, other):
        """Perform a `logical_or` with another data model

        If one of them is None, output the other one. If both are provided,
        then concatenates this data model with the other.

        If the other is a metaclass or symbolic data model, output a symbolic data model.

        Args:
            other (JsonDataModel | SymbolicDataModel | DataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel | SymbolicDataModel | None): The concatenation of data model
                if both are provided, or the non-None data model or None if none are
                provided. (See `logical_or` table).
        """
        from synalinks.src import ops

        if any_meta_class(self, other):
            return run_maybe_nested(
                ops.Or().symbolic_call(self, other),
            )
        else:
            return run_maybe_nested(
                ops.Or()(self, other),
            )

    def __ror__(self, other):
        """Perform a `logical_or` (reverse) with another data model

        If one of them is None, output the other one. If both are provided,
        then concatenates the other data model with this one.

        If the other is a metaclass or symbolic data model, output a symbolic data model.

        Args:
            other (JsonDataModel | SymbolicDataModel | DataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel | SymbolicDataModel | None): The concatenation of data model
                if both are provided, or the non-None data model or None if none are
                provided. (See `logical_or` table).
        """
        from synalinks.src import ops

        if any_meta_class(other, self):
            return run_maybe_nested(
                ops.Or().symbolic_call(other, self),
            )
        else:
            return run_maybe_nested(
                ops.Or()(other, self),
            )

    def __xor__(self, other):
        """Perform a `logical_xor` with another data model.

        If one of them is None, output the other one. If both are provided,
        then output None.

        Args:
            other (SymbolicDataModel): The other data model to concatenate with.

        Returns:
            (SymbolicDataModel | None): `None` if both are
                provided, or the non-None data model if one is provided
                or `None` if none are provided. (See `logical_xor` table).
        """
        from synalinks.src import ops

        if any_meta_class(self, other):
            return run_maybe_nested(
                ops.Xor().symbolic_call(self, other),
            )
        else:
            return run_maybe_nested(
                ops.Xor()(self, other),
            )

    def __rxor__(self, other):
        """Perform a `logical_xor` (reverse) with another data model.

        If one of them is None, output the other one. If both are provided,
        then output None.

        Args:
            other (SymbolicDataModel | DataModel): The other data model to concatenate
                with.

        Returns:
            (SymbolicDataModel | None): `None` if both are
                provided, or the non-None data model if one is provided
                or `None` if none are provided. (See `logical_xor` table).
        """
        from synalinks.src import ops

        if any_meta_class(other, self):
            return run_maybe_nested(
                ops.Xor().symbolic_call(other, self),
            )
        else:
            return run_maybe_nested(
                ops.Xor()(other, self),
            )

    def __contains__(cls, other):
        """Check if the schema of `other` is contained in this one.

        Args:
            other (SymbolicDataModel | DataModel): The other data model to compare with.

        Returns:
            (bool): True if all properties of `other` are present in this one.
        """
        from synalinks.src.backend.common.json_schema_utils import contains_schema

        return contains_schema(cls.get_schema(), other.get_schema())

    def get_config(self):
        return self.get_json()

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def is_data_model(x):
    """Returns whether `x` is a DataModel.

    Args:
        x (any): The object to check.

    Returns:
        (bool): True if `x` is a DataModel, False otherwise.
    """
    return isinstance(x, DataModel)


def any_data_model(args=None, kwargs=None):
    """Check if any of the arguments are backend-dependent data models.

    Args:
        args (tuple): Optional. The positional arguments to check.
        kwargs (dict): Optional. The keyword arguments to check.

    Returns:
        (bool): True if any of the arguments are meta classes, False otherwise.
    """
    args = args or ()
    kwargs = kwargs or {}
    for x in tree.flatten((args, kwargs)):
        if is_meta_class(x):
            return True
    return False


def any_meta_class(args=None, kwargs=None):
    """Check if any of the arguments are meta classes.

    This happen when using a `DataModel` without instanciating it.
    In Synalinks this is used when declaring data models for schema inference.

    Args:
        args (tuple): Optional. The positional arguments to check.
        kwargs (dict): Optional. The keyword arguments to check.

    Returns:
        (bool): True if any of the arguments are meta classes, False otherwise.
    """
    args = args or ()
    kwargs = kwargs or {}
    for x in tree.flatten((args, kwargs)):
        if is_meta_class(x):
            return True
    return False


@synalinks_export(
    [
        "synalinks.utils.is_meta_class",
        "synalinks.backend.is_meta_class",
    ]
)
def is_meta_class(x):
    """Returns whether `x` is a meta class.

    A meta class is a python type. This method checks if the data model provided
    if a meta class, allowing to detect if the `DataModel` have been instanciated.
    Meta classes are using in Synalinks when declaring data models for schema inference.

    Args:
        x (any): The object to check.

    Returns:
        (bool): True if `x` is a meta class, False otherwise.
    """
    return inspect.isclass(x)
