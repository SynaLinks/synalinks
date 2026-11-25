# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import inspect
import logging
import typing

import docstring_parser
import pydantic
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend.pydantic.media import Audio
from synalinks.src.backend.pydantic.media import Image
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib
from synalinks.src.utils.tool_utils import docstring_description
from synalinks.src.utils.tool_utils import get_params_schema
from synalinks.src.utils.tool_utils import inline_refs


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
        if not all(char in "0123456789+-*/(). " for char in expression):
            return {"result": None, "log": "Error: invalid characters"}
        result = eval(expression, {"__builtins__": None}, {})
        return {"result": result}

    tool = synalinks.Tool(calculate)
    result = await tool(expression="2 + 2")
    ```

    Important:
        **Optional Parameters**: Parameters with default values are
        supported. A defaulted parameter is left out of the schema's
        `required` list and its default is emitted in the schema, so the
        language model may omit it and the function's default applies.
        (`Optional[T]` is treated as `T`; the `None` member is dropped.)
        The "every property must be required" rule that some providers
        enforce only applies to *strict structured output*, which is a
        separate code path from tool calling.

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

        **Images and audio**: A tool can show images to the language model,
        or play it audio, by returning `synalinks.Image` / `synalinks.Audio`
        values in its dict (alone, or in a list). The agents attach them to
        the tool result so a model that takes them gets them, and they stay
        in the returned trajectory for the user:

        ```python
        async def render(path: str):
            \"\"\"Render a chart.

            Args:
                path (str): The chart file.
            \"\"\"
            return {"path": path, "image": synalinks.Image(path=path)}
        ```

    Args:
        func (Callable): The async function to wrap as a tool.
        name (str): Optional. The name of the module. Defaults to the function name.
        description (str): Optional. The description of the module.
            Defaults to the function's docstring description (its summary
            and the paragraphs before the Args section).
        trainable (bool): Whether the module's variables should be trainable.
            Defaults to False since tools typically don't have trainable state.
    """

    def __init__(
        self,
        func: typing.Callable,
        *,
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
        self._type_hints = typing.get_type_hints(func)
        self._params_schema, self._required_params = get_params_schema(
            func, self._docstring
        )

        # Use function name if no name provided
        if not name:
            name = self._func.__name__

        # Use the docstring's description if no description provided
        if not description:
            description = docstring_description(self._docstring)

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
        # A tool has no state to build. An agent calls its tools only for
        # real, never symbolically, so left unbuilt they made every agent
        # program look unbuilt to `fit` and `evaluate`.
        self.built = True

    def _build_output_schema(self):
        """Build the output schema from the function's return type hint.

        Since tools must always return a dict, this method ensures
        the output schema is always of type "object".
        """
        return_type = self._type_hints.get("return")
        schema = {}
        if return_type is not None:
            try:
                schema = inline_refs(pydantic.TypeAdapter(return_type).json_schema())
            except (pydantic.PydanticUserError, ValueError):
                pass
        if schema.get("type") != "object":
            schema = {"type": "object", "additionalProperties": True}
        # A TypedDict or BaseModel result has exactly its declared keys
        schema.setdefault("additionalProperties", False)
        schema["title"] = f"{self.name}_output"
        return schema

    def get_input_schema(self) -> dict:
        """Get the JSON schema for this tool's input parameters.

        Returns:
            dict: The JSON schema describing the tool's input parameters.
        """
        return {
            "additionalProperties": False,
            "description": docstring_description(self._docstring),
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
            # Returned media become their content parts, to stay JSON.
            result = dict(result)
            for key, value in result.items():
                if isinstance(value, (Image, Audio)):
                    result[key] = value.to_content_part()
                elif isinstance(value, list):
                    result[key] = [
                        v.to_content_part() if isinstance(v, (Image, Audio)) else v
                        for v in value
                    ]
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
        return self.get_input_schema()

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
