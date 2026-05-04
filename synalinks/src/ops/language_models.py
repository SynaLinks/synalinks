# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import ChatMessage
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import any_symbolic_data_models
from synalinks.src.modules.language_models.language_model import StreamingIterator
from synalinks.src.ops.operation import Operation
from synalinks.src.saving import serialization_lib


class Predict(Operation):
    """Perform a prediction using a `LanguageModel`."""

    def __init__(
        self,
        schema=None,
        data_model=None,
        language_model=None,
        streaming=False,
        name=None,
        description=None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
        )
        if not schema and data_model:
            schema = data_model.get_schema()
        self.schema = schema
        self.data_model = data_model
        self.language_model = language_model
        if schema and streaming:
            streaming = False
        self.streaming = streaming
        self.lm_kwargs = kwargs

    async def call(self, x):
        value = await self.language_model(
            x,
            schema=self.schema,
            streaming=self.streaming,
            **self.lm_kwargs,
        )
        if isinstance(value, StreamingIterator):
            return value
        if not value:
            return None
        if self.schema:
            return JsonDataModel(json=value, schema=self.schema, name=self.name)
        else:
            return JsonDataModel(
                json=value, schema=ChatMessage.get_schema(), name=self.name
            )

    async def compute_output_spec(self, x):
        if self.schema:
            return SymbolicDataModel(schema=self.schema, name=self.name)
        else:
            return SymbolicDataModel(schema=ChatMessage.get_schema(), name=self.name)

    def get_config(self):
        config = {
            "schema": self.schema,
            "streaming": self.streaming,
            "name": self.name,
            "description": self.description,
        }
        language_model_config = serialization_lib.serialize_synalinks_object(
            self.language_model
        )
        config.update({"lm_kwargs": self.lm_kwargs})
        return {"language_model": language_model_config, **config}

    @classmethod
    def from_config(cls, config):
        language_model = serialization_lib.deserialize_synalinks_object(
            config.pop("language_model")
        )
        lm_kwargs = config.pop("lm_kwargs")
        return cls(language_model=language_model, **config, **lm_kwargs)


@synalinks_export(["synalinks.ops.predict", "synalinks.ops.json.predict"])
async def predict(
    x,
    schema=None,
    data_model=None,
    language_model=None,
    streaming=False,
    name=None,
    description=None,
    **kwargs,
):
    """Perform a prediction using a `LanguageModel`.

    Predict consist in predicting a target data_model from an input data_model.
    This function uses a backend DataModel to get the target schema.

    Args:
        x (JsonDataModel | SymbolicDataModel): the input data model.
        data_model (DataModel): The target data model.
        language_model (LanguageModel): The language model to use
        streaming (bool): Enable streaming if True (Default to False)
        name (str): Optional. The name of the operation.
        description (str): Optional. The description of the operation.
        **kwargs (keyword arguments): Additional keywords forwarded to the
            LanguageModel call.

    Returns:
        (JsonDataModel | SymbolicDataModel): The resulting data model.
    """
    if any_symbolic_data_models(x):
        return await Predict(
            schema=schema,
            data_model=data_model,
            language_model=language_model,
            streaming=False,
            name=name,
            description=description,
        ).symbolic_call(x)
    return await Predict(
        schema=schema,
        data_model=data_model,
        language_model=language_model,
        streaming=streaming,
        name=name,
        description=description,
        **kwargs,
    )(x)
