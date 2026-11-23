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
from synalinks.src.saving import serialization_lib
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable


def inline_refs(schema: dict) -> dict:
    """Replace each `$ref` of a pydantic schema by its `$defs` entry.

    A tool schema gets nested inside bigger schemas (the tool calls one),
    where a `#/$defs/...` ref would resolve against the wrong root.
    Recursive types cannot be inlined and raise a `ValueError`.
    """
    defs = schema.get("$defs", {})

    def inline(node, seen):
        if isinstance(node, list):
            return [inline(v, seen) for v in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            if name in seen:
                raise ValueError(f"Recursive type '{name}' is not supported in tools")
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            return {**inline(defs[name], seen | {name}), **inline(siblings, seen)}
        return {k: inline(v, seen) for k, v in node.items() if k != "$defs"}

    return inline(schema, frozenset())


def get_params_schema(func: typing.Callable, docstring: docstring_parser.Docstring):
    """The JSON schema of `func`'s parameters, built by pydantic from the type
    hints, each with its description taken from the docstring's Args.

    Returns:
        (tuple): The `properties` dict and the `required` list.
    """
    type_hints = typing.get_type_hints(func)
    descriptions = {p.arg_name: p.description for p in docstring.params}
    for name in inspect.signature(func).parameters:
        if name not in type_hints:
            raise ValueError(f"Missing type hint for parameter '{name}'")
        if descriptions.get(name) is None:
            raise ValueError(f"Missing description for parameter '{name}' in docstring")
    schema = inline_refs(pydantic.TypeAdapter(func).json_schema())
    for name, prop in schema["properties"].items():
        prop["description"] = descriptions[name].replace("\n", " ")
    return schema["properties"], schema.get("required", [])


def docstring_description(docstring: docstring_parser.Docstring) -> str:
    """A docstring's whole description: its summary and the paragraphs after
    it, without the Args/Returns sections.

    `docstring_parser` cuts the summary at its first line break: a summary
    wrapped over two lines keeps only its first line in `short_description`,
    and the rest of the sentence starts `long_description`.
    """
    short = docstring.short_description or ""
    long = docstring.long_description or ""
    if not long:
        return short
    return short + ("\n\n" if docstring.blank_after_short_description else "\n") + long


@synalinks_export("synalinks.utils.Tool")
class Tool(SynalinksSaveable):
    def __init__(self, func: typing.Callable):
        self._func = func
        if not inspect.iscoroutinefunction(self._func):
            raise TypeError(f"{self.name} is not an asynchronous function")

        doc = inspect.getdoc(func)
        if not doc:
            raise ValueError(f"The tool ({self.name}) must have a docstring")

        self._docstring = docstring_parser.parse(doc)
        self._params_schema, self._required_params = get_params_schema(
            func, self._docstring
        )

        if not self.description:
            logging.warning(
                "The tool (%s) has no description. " % self.name
                + "This is unsafe behavior and may lead to issues."
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def __call__(self, *args, **kwargs):
        return await self._func(*args, **kwargs)

    @property
    def description(self) -> str:
        return docstring_description(self._docstring)

    @property
    def name(self) -> str:
        return self._func.__name__

    def get_tool_schema(self):
        schema = {
            "additionalProperties": False,
            "description": self.description,
            "properties": self._params_schema,
            "required": self._required_params,
            "title": self.name.title().replace("_", " "),
            "type": "object",
        }
        return schema

    def get_config(self):
        func_config = {"func": serialization_lib.serialize_synalinks_object(self._func)}
        return {**func_config}

    @classmethod
    def from_config(cls, config):
        func = serialization_lib.deserialize_synalinks_object(config.pop("func"))
        return cls(func)
