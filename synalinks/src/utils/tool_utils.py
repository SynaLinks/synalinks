# MIT License
# Copyright (c) 2023 Lucas Lofaro

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Modified from: llmfuncs
# Original authors: Lucas Lofaro
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import inspect
import logging
import typing

import docstring_parser


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
        # this is a special case to handle Optional[type] which is just syntactic sugar
        # around Union[type, None]
        if len(args) == 2 and type(None) in args:
            # Assuming the None is always last
            return json_schema_type(args[0])
        else:
            return [json_schema_type(arg) for arg in args]

    if origin is list or origin is typing.List:
        # For simplicity, we're assuming all elements in the list are of the same type
        schema_type = json_schema_type(args[0])
        if isinstance(schema_type, dict):
            return {"type": "array", "items": schema_type}
        return {
            "type": "array",
            "items": {"type": json_schema_type(args[0])},
        }

    if origin is dict or origin is typing.Dict:
        # For simplicity, we're assuming all keys are strings
        # and all values are of the same type
        schema_type = json_schema_type(args[1])
        if isinstance(schema_type, dict):
            return {"type": "object", "additionalProperties": schema_type}
        return {"type": "object", "additionalProperties": {"type": schema_type}}

    # The type is not supported
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

    # Check if the param_type_str is already a dictionary.
    if isinstance(param_type_str, dict):
        param_schema = param_type_str
    else:
        param_schema = {"type": param_type_str}

    param_schema["description"] = param_doc

    if param.default is not param.empty:
        param_schema["default"] = param.default

    return param_schema


class Tool:
    def __init__(self, func: typing.Callable, include_return=False):
        self._func = func
        self._include_return = include_return

        doc = inspect.getdoc(func)
        if not doc:
            raise ValueError(f"The tool ({self.name()}) must have a docstring")

        self._docstring = docstring_parser.parse(doc)
        self._signature = inspect.signature(func)
        self._type_hints = typing.get_type_hints(func)
        self._params_schema = {}
        self._required_params = []

        self._parse_arguments()

        if not self.description():
            logging.warning(
                "The tool (%s) has no description. " % self.name() +
                "This is unsafe behavior and may lead to issues."
            )

    def __call__(self, *args, **kwargs):
        return self._func(*args, **kwargs)
    
    async def async__call__(self, *args, **kwargs):
        if not inspect.iscoroutinefunction(self._func):
            raise TypeError(f"{self.name()} is not an asynchronous function")

        return await self._func(*args, **kwargs)

    def _parse_arguments(self):
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

    def _has_return(self):
        return self._signature.return_annotation is not self._signature.empty

    def description(self) -> str:
        return self._docstring.short_description or ""

    def name(self) -> str:
        return self._func.__name__

    def func_schema(self):
        func_schema = {
            "name": self.name(),
            "description": self._docstring.short_description,
            "parameters": {
                "type": "object",
                "properties": self._params_schema,
            },
        }

        if self._required_params:
            func_schema["parameters"]["required"] = self._required_params

        if self._include_return and self._has_return():
            schema_type = json_schema_type(self._signature.return_annotation)
            func_schema["return"] = schema_type

        return func_schema

    def obj_schema(self):
        obj_schema = {
            "title": self.name().title().replace("_", " "),
            "description": self._docstring.short_description,
            "type": "object",
            "properties": self._params_schema,
            "additionalProperties": False,
        }

        if self._required_params:
            obj_schema["required"] = self._required_params

        return obj_schema


def toolkit_to_static_prompt(toolkit: typing.List[Tool] = None) -> str:
    """Convert a toolkit into a static text prompt.

    Args:
        toolkit: The set of available tools

    Returns:
        A string describing each available tool by name and description.
    """
    toolkit = toolkit or []

    if not toolkit:
        return "The toolkit is empty. No tools available."

    prompt = "The toolkit contains %d tool%s:\n\n" % (len(toolkit), "s" if len(toolkit) > 1 else "")

    description = [f"- ({_.name()}) {_.description()}\n" for _ in toolkit]

    prompt = prompt + "".join(description)

    return prompt
