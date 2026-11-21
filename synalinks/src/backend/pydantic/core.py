# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import inspect

import pydantic
from typing_extensions import ClassVar

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.data_model_operators import DataModelOperators
from synalinks.src.backend.common.data_model_operators import DataModelTransforms
from synalinks.src.backend.common.data_model_operators import apply_symbolic_op
from synalinks.src.backend.common.json_data_model import JsonDataModel
from synalinks.src.backend.common.symbolic_data_model import SymbolicDataModel
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils.async_utils import run_maybe_nested

IS_THREAD_SAFE = True


class MetaDataModel(DataModelOperators, DataModelTransforms, type(pydantic.BaseModel)):
    """The metaclass data model.

    This class defines operations at the metaclass level.
    Allowing to use Synalinks Python operators with `DataModel` types.
    """

    _apply_op = apply_symbolic_op


class DataModel(
    pydantic.BaseModel, SynalinksSaveable, DataModelOperators, metaclass=MetaDataModel
):
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
        import orjson

        return orjson.dumps(cls.get_schema(), option=orjson.OPT_INDENT_2).decode()

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

        Optional fields whose value is `None` are omitted from the output;
        the schema still declares them, so consumers should treat a missing
        key as `None`.

        Serialization is duck-typed (``serialize_as_any``): a field declared
        with a parent type (e.g. ``List[Entity]``) dumps each item with the
        *instance's* fields, so `Entity`/`Relation` subclasses keep their
        properties inside a generic `KnowledgeGraph`.

        Returns:
            (dict): The JSON value.
        """
        return self.model_dump(mode="json", exclude_none=True, serialize_as_any=True)

    def prettify_json(self):
        """Get a pretty version of the JSON object for display.

        Returns:
            (str): The indented JSON object.
        """
        import orjson

        return orjson.dumps(self.get_json(), option=orjson.OPT_INDENT_2).decode()

    def __repr__(self):
        return f"<DataModel json={self.get_json()}, schema={self.get_schema()}>"

    def get(self, key, default=None):
        """Get wrapper to make it easier to access JSON fields.

        Args:
            key (str): The key to access.
            default (any): The default value if key not found.
        """
        try:
            return self.__getattribute__(key)
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

    def get_nested_entity(self, key):
        """Retrieve a nested Entity and convert it to a JsonDataModel.

        The entity's `label` field is used as the discriminator to look up
        its schema in this model's `$defs`. Returns ``None`` when the value
        at ``key`` has no ``label`` or when the schema cannot be resolved.

        Args:
            key (str): The field name holding the nested entity.

        Returns:
            (JsonDataModel | None): A typed JsonDataModel wrapping the
                nested entity, or ``None`` if the value isn't an entity.
        """
        return self.to_json_data_model().get_nested_entity(key)

    def get_nested_entity_list(self, key):
        """Retrieve a nested Entity list and convert it to typed JsonDataModels.

        Each item is resolved against this model's `$defs` using its
        ``label`` field as discriminator. Items without a ``label`` field
        or an unresolved schema are skipped.

        Args:
            key (str): The field name holding the list of nested entities.

        Returns:
            (list[JsonDataModel]): One JsonDataModel per resolved entity.
        """
        return self.to_json_data_model().get_nested_entity_list(key)

    def _apply_op(self, op, *args):
        if any_meta_class(args):
            return run_maybe_nested(op.symbolic_call(*args))
        return run_maybe_nested(op(*args))

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
