# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import inspect
import logging
import typing

import docstring_parser
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib

JsonSchema = typing.Union[
    typing.Dict[str, typing.Any],
    typing.List[typing.Any],
    str,
    int,
    float,
    bool,
    None,
]


def json_schema_type(py_type: typing.Any) -> JsonSchema:
    """Convert a Python type to a JSON schema type."""
    mapping = {
        int: "integer",
        float: "number",
        bool: "boolean",
        str: "string",
        type(None): "null",
    }

    # Check if type is a basic type
    if py_type in mapping:
        return mapping[py_type]

    # For unparameterized list and dict types
    if py_type is list or py_type is typing.List:
        return {"type": "array", "items": {}}
    if py_type is dict or py_type is typing.Dict:
        return {"type": "object", "additionalProperties": {}}

    origin = typing.get_origin(py_type)
    args = typing.get_args(py_type)

    if origin is typing.Union:
        # Handle Optional[type] which is Union[type, None]
        if len(args) == 2 and type(None) in args:
            return json_schema_type(args[0])
        else:
            return [json_schema_type(arg) for arg in args]

    if origin is list or origin is typing.List:
        schema_type = json_schema_type(args[0])
        if isinstance(schema_type, dict):
            return {"type": "array", "items": schema_type}
        return {
            "type": "array",
            "items": {"type": json_schema_type(args[0])},
        }

    if origin is dict or origin is typing.Dict:
        schema_type = json_schema_type(args[1])
        if isinstance(schema_type, dict):
            return {"type": "object", "additionalProperties": schema_type}
        return {"type": "object", "additionalProperties": {"type": schema_type}}

    raise ValueError(f"Cannot convert {py_type} to a JSON schema type")


def get_param_schema(
    param_name: str,
    param: inspect.Parameter,
    type_hints: typing.Dict[str, typing.Any],
    doc_parsed: docstring_parser.Docstring,
) -> JsonSchema:
    """Create a schema for a single parameter."""
    if param_name not in type_hints:
        raise ValueError(f"Missing type hint for parameter '{param_name}'")
    param_type = type_hints[param_name]
    param_type_str = json_schema_type(param_type)
    descriptions = (p.description for p in doc_parsed.params if p.arg_name == param_name)
    param_doc = next(descriptions, None)
    if param_doc is None:
        raise ValueError(f"Missing description for parameter '{param_name}' in docstring")

    param_schema = {}
    param_schema["description"] = param_doc.replace("\n", " ")
    param_schema["title"] = param_name.title().replace("_", " ")
    if isinstance(param_type_str, dict):
        param_schema.update(**param_type_str)
    else:
        param_schema["type"] = param_type_str

    if param.default is not param.empty:
        param_schema["default"] = param.default

    return param_schema


@synalinks_export(["synalinks.modules.Tool", "synalinks.Tool"])
class Tool(Module):
    """A module that wraps an async function as a callable tool.

    The `Tool` module allows you to wrap any async function and use it as a
    module within Synalinks programs. It automatically extracts the function's
    schema from its type hints and docstring.

    Example:

    ```python
    import synalinks

    @synalinks.saving.register_synalinks_serializable()
    async def calculate(expression: str):
        \"\"\"Calculate the result of a mathematical expression.

        Args:
            expression (str): The mathematical expression to calculate.
        \"\"\"
        result = eval(expression)
        return {"result": result}

    tool = synalinks.Tool(calculate)
    result = await tool(expression="2 + 2")
    ```

    Important:
        **No Optional Parameters**: All function parameters must be required.
        Optional parameters with default values are not supported because
        OpenAI and other providers require all parameters to be required
        in their structured output JSON schemas.

        **Complete Docstring Required**: The wrapped function must have a
        complete docstring with an `Args:` section that documents every
        parameter. The Tool extracts parameter descriptions from the docstring
        to build the JSON schema sent to the language model. Missing
        descriptions will raise a ValueError.

        Example of a properly documented tool function:

        ```python
        async def search(query: str, limit: int):
            \"\"\"Search the database for matching records.

            Args:
                query (str): The search query string.
                limit (int): Maximum number of results to return.
            \"\"\"
            # Implementation here
            return {"results": [...]}
        ```

    Args:
        func (Callable): The async function to wrap as a tool.
        name (str): Optional. The name of the module. Defaults to the function name.
        description (str): Optional. The description of the module.
            Defaults to the function's docstring short description.
        trainable (bool): Whether the module's variables should be trainable.
            Defaults to False since tools typically don't have trainable state.
    """

    def __init__(
        self,
        func: typing.Callable,
        name=None,
        description=None,
        trainable=False,
    ):
        self._func = func
        if not inspect.iscoroutinefunction(self._func):
            raise TypeError(f"{self._func.__name__} is not an asynchronous function")

        doc = inspect.getdoc(func)
        if not doc:
            raise ValueError(f"The tool ({self._func.__name__}) must have a docstring")

        self._docstring = docstring_parser.parse(doc)
        self._signature = inspect.signature(func)
        self._type_hints = typing.get_type_hints(func)
        self._params_schema = {}
        self._required_params = []

        self._parse_arguments()

        # Use function name if no name provided
        if not name:
            name = self._func.__name__

        # Use docstring short description if no description provided
        if not description:
            description = self._docstring.short_description or ""

        if not description:
            logging.warning(
                f"The tool ({name}) has no description. "
                "This is unsafe behavior and may lead to issues."
            )

        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )

    def _parse_arguments(self):
        """Parse the function arguments to build the input parameter schema."""
        for param_name, param in self._signature.parameters.items():
            param_schema = get_param_schema(
                param_name,
                param,
                self._type_hints,
                self._docstring,
            )
            self._params_schema[param_name] = param_schema
            if param.default is param.empty:
                self._required_params.append(param_name)

    def _build_output_schema(self):
        """Build the output schema from the function's return type hint.

        Since tools must always return a dict, this method ensures
        the output schema is always of type "object".
        """
        return_type = self._type_hints.get("return", None)
        base_schema = {
            "type": "object",
            "title": f"{self.name}_output",
        }

        if return_type is None:
            # No return type hint, use generic object schema
            base_schema["additionalProperties"] = True
            return base_schema

        origin = typing.get_origin(return_type)
        args = typing.get_args(return_type)

        # Handle Dict[K, V] - extract value type for additionalProperties
        if origin is dict or origin is typing.Dict:
            if len(args) >= 2:
                value_type = args[1]
                try:
                    value_schema = json_schema_type(value_type)
                    if isinstance(value_schema, dict):
                        base_schema["additionalProperties"] = value_schema
                    else:
                        base_schema["additionalProperties"] = {"type": value_schema}
                except ValueError:
                    base_schema["additionalProperties"] = True
            else:
                base_schema["additionalProperties"] = True
            return base_schema

        # Handle TypedDict - extract properties from annotations
        if hasattr(return_type, "__annotations__"):
            properties = {}
            required = []
            for field_name, field_type in return_type.__annotations__.items():
                try:
                    field_schema = json_schema_type(field_type)
                    if isinstance(field_schema, dict):
                        properties[field_name] = field_schema
                    else:
                        properties[field_name] = {"type": field_schema}
                    # Check if field is required (not Optional)
                    if typing.get_origin(field_type) is not typing.Union:
                        required.append(field_name)
                    elif type(None) not in typing.get_args(field_type):
                        required.append(field_name)
                except ValueError:
                    properties[field_name] = {}
            if properties:
                base_schema["properties"] = properties
                if required:
                    base_schema["required"] = required
                base_schema["additionalProperties"] = False
                return base_schema

        # Fallback to generic object schema
        base_schema["additionalProperties"] = True
        return base_schema

    def get_input_schema(self) -> dict:
        """Get the JSON schema for this tool's input parameters.

        Returns:
            dict: The JSON schema describing the tool's input parameters.
        """
        return {
            "additionalProperties": False,
            "description": self._docstring.short_description,
            "properties": self._params_schema,
            "required": self._required_params,
            "title": self.name.title().replace("_", " "),
            "type": "object",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def call(
        self, training: bool = False, **kwargs: typing.Any
    ) -> typing.Optional[JsonDataModel]:
        """Execute the wrapped function with the provided arguments.

        Args:
            training (bool): Whether in training mode. Not used by Tool but
                included for consistency with other modules.
            **kwargs (Any): The arguments to pass to the wrapped function.

        Returns:
            JsonDataModel: The result wrapped in a JsonDataModel with the output schema.
        """
        result = await self._func(**kwargs)
        if result is None:
            return None
        if isinstance(result, dict):
            return JsonDataModel(
                json=result,
                schema=self._build_output_schema(),
                name=f"{self.name}_output",
            )
        # Wrap non-dict results in a dict
        return JsonDataModel(
            json={"result": result},
            schema=self._build_output_schema(),
            name=f"{self.name}_output",
        )

    async def compute_output_spec(
        self, training: bool = False, **kwargs: typing.Any
    ) -> SymbolicDataModel:
        """Compute the output specification for the tool.

        Uses the function's schema to define the output structure.

        Args:
            training (bool): Whether in training mode.
            **kwargs (Any): The input arguments.

        Returns:
            SymbolicDataModel: A SymbolicDataModel with the tool's output schema.
        """
        return SymbolicDataModel(
            schema=self._build_output_schema(),
            name=self.name,
        )

    def get_tool_schema(self) -> dict:
        """Get the JSON schema for this tool's parameters.

        Returns:
            dict: The JSON schema describing the tool's input parameters.
        """
        schema = {
            "additionalProperties": False,
            "description": self._docstring.short_description,
            "properties": self._params_schema,
            "required": self._required_params,
            "title": self.name.title().replace("_", " "),
            "type": "object",
        }
        return schema

    def get_config(self):
        config = {
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        func_config = {"func": serialization_lib.serialize_synalinks_object(self._func)}
        return {**config, **func_config}

    @classmethod
    def from_config(cls, config):
        func = serialization_lib.deserialize_synalinks_object(config.pop("func"))
        return cls(func=func, **config)
