# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import is_meta_class
from synalinks.src.initializers.initializer import Initializer


@synalinks_export(
    [
        "synalinks.initializers.Empty",
        "synalinks.initializers.empty",
    ]
)
class Empty(Initializer):
    """
    Initialize a variable with the data_model default value.

    Args:
        schema (dict): The JSON object schema. If not provided,
            uses the `data_model` to infer it.
        value (dict): The initial JSON object. If the value is not provided the
            `data_model` should be provided. When a value is provided, a schema
            should be provided.
        data_model (DataModel): The backend data_model to use.

    """

    def get_config(self):
        return {
            "schema": self._schema,
            "json": self._json,
        }

    def __call__(self, data_model=None):
        """Returns a JSON object initialized as specified by the initializer."""
        if data_model:
            if not is_meta_class(data_model):
                self._json = data_model.get_json()
            else:
                self._json = data_model().get_json()
            self._schema = data_model.get_schema()
        return self._json
