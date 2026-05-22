# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import inspect

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import DataModel
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.modules.module import Module
from synalinks.src.saving import serialization_lib


@synalinks_export(["synalinks.modules.Lambda", "synalinks.Lambda"])
class Lambda(Module):
    """Wraps an arbitrary callable as a `Module`.

    The `Lambda` module allows you to plug a plain function (sync or async)
    into a Synalinks program without having to subclass `Module` yourself. It is
    useful for inserting custom, stateless data transformations into the
    program graph.

    Because the output schema cannot in general be inferred from an arbitrary
    callable, you must provide either a `data_model` (a `DataModel` subclass)
    or a `schema` (a JSON schema dict) describing the shape of the function's
    return value.

    The wrapped function is invoked with a `JsonDataModel` as its single
    positional argument and is expected to return either:

    * a `dict` matching the output schema,
    * a `DataModel` or `JsonDataModel` instance, or
    * `None`.

    Any callable works — Python `lambda`s, named sync functions, or named
    async functions are all supported. Named, decorated functions are required
    when you need the program to be serializable; `lambda`s cannot be saved.

    **Inline `lambda` for short transforms:**

    ```python
    import synalinks

    class Query(synalinks.DataModel):
        query: str

    class UppercaseQuery(synalinks.DataModel):
        query: str

    x0 = synalinks.Input(data_model=Query)
    x1 = await synalinks.Lambda(
        function=lambda x: {"query": x.get_json()["query"].upper()},
        data_model=UppercaseQuery,
    )(x0)
    ```

    **Named sync function returning a `dict`:**

    ```python
    @synalinks.saving.register_synalinks_serializable()
    def uppercase(inputs):
        data = inputs.get_json()
        return {"query": data["query"].upper()}

    x1 = await synalinks.Lambda(
        function=uppercase,
        data_model=UppercaseQuery,
    )(x0)
    ```

    **Named async function (useful when the transform performs I/O):**

    ```python
    @synalinks.saving.register_synalinks_serializable()
    async def uppercase(inputs):
        data = inputs.get_json()
        return {"query": data["query"].upper()}

    x1 = await synalinks.Lambda(
        function=uppercase,
        data_model=UppercaseQuery,
    )(x0)
    ```

    **Returning a `DataModel` instance instead of a `dict`:**

    ```python
    @synalinks.saving.register_synalinks_serializable()
    async def uppercase(inputs):
        return UppercaseQuery(query=inputs.get_json()["query"].upper())

    x1 = await synalinks.Lambda(
        function=uppercase,
        data_model=UppercaseQuery,
    )(x0)
    ```

    **Using a raw JSON `schema` instead of a `DataModel`:**

    ```python
    x1 = await synalinks.Lambda(
        function=lambda x: {"query": x.get_json()["query"].upper()},
        schema=UppercaseQuery.get_schema(),
    )(x0)
    ```

    **Returning `None` to short-circuit downstream branches:**

    ```python
    # Filter: forward the input only when the query is non-empty,
    # otherwise emit None and let downstream `|` / `Branch` skip it.
    @synalinks.saving.register_synalinks_serializable()
    async def non_empty(inputs):
        data = inputs.get_json()
        return data if data.get("query") else None

    x1 = await synalinks.Lambda(
        function=non_empty,
        data_model=Query,
    )(x0)
    ```

    **Full program example:**

    ```python
    import synalinks
    import asyncio

    async def main():

        class Query(synalinks.DataModel):
            query: str

        class UppercaseQuery(synalinks.DataModel):
            query: str

        @synalinks.saving.register_synalinks_serializable()
        async def uppercase(inputs):
            data = inputs.get_json()
            return {"query": data["query"].upper()}

        x0 = synalinks.Input(data_model=Query)
        x1 = await synalinks.Lambda(
            function=uppercase,
            data_model=UppercaseQuery,
        )(x0)

        program = synalinks.Program(
            inputs=x0,
            outputs=x1,
            name="shouter",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    Args:
        function (Callable): The function (sync or async) to wrap. It receives
            the module's input as a `JsonDataModel` and should return a `dict`,
            a `DataModel` / `JsonDataModel`, or `None`.
        schema (dict): Optional. The target JSON schema. If not provided, use
            the `data_model` to infer it.
        data_model (DataModel): Optional. The `DataModel` class describing the
            output schema. Either this or `schema` must be provided.
        name (str): Optional. The name of the module.
        description (str): Optional. The description of the module.
        trainable (bool): Whether the module's variables should be trainable.
            Defaults to `False` since `Lambda` has no inherent trainable state.
    """

    def __init__(
        self,
        function,
        *,
        schema=None,
        data_model=None,
        name=None,
        description=None,
        trainable=False,
    ):
        super().__init__(
            name=name,
            description=description,
            trainable=trainable,
        )
        if not callable(function):
            raise TypeError(
                f"`function` must be callable. Received: {function!r} "
                f"(of type {type(function)})."
            )
        if not schema and not data_model:
            raise ValueError(
                "Lambda requires either `data_model` or `schema` to describe "
                "the function's return value."
            )
        if not schema:
            schema = data_model.get_schema()
        self.function = function
        self.schema = schema
        self._is_coroutine = inspect.iscoroutinefunction(function)
        self.built = True

    async def call(self, inputs):
        if inputs is None:
            return None
        if self._is_coroutine:
            result = await self.function(inputs)
        else:
            result = self.function(inputs)
        if result is None:
            return None
        if isinstance(result, JsonDataModel):
            return result
        if isinstance(result, DataModel):
            return result.to_json_data_model(name=self.name)
        if isinstance(result, dict):
            return JsonDataModel(
                json=result,
                schema=self.schema,
                name=self.name,
            )
        raise TypeError(
            f"`function` for Lambda '{self.name}' must return a dict, a "
            f"DataModel/JsonDataModel, or None. Received: {result!r} "
            f"(of type {type(result)})."
        )

    async def compute_output_spec(self, inputs):
        return SymbolicDataModel(
            schema=self.schema,
            name=self.name,
        )

    def get_config(self):
        config = {
            "schema": self.schema,
            "name": self.name,
            "description": self.description,
            "trainable": self.trainable,
        }
        function_config = {
            "function": serialization_lib.serialize_synalinks_object(self.function),
        }
        return {**config, **function_config}

    @classmethod
    def from_config(cls, config):
        function = serialization_lib.deserialize_synalinks_object(config.pop("function"))
        return cls(function=function, **config)
