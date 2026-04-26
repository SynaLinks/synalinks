# Modified from: keras/src/initializers/initializer.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import is_meta_class
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable


@synalinks_export(["synalinks.Initializer", "synalinks.initializers.Initializer"])
class Initializer(SynalinksSaveable):
    """Initializer base class: all synalinks initializers inherit from this class.

    Initializers should implement a `__call__()` method with the following
    signature:

    ```python
    def __call__(self, **kwargs):
        # returns a JSON object dict containing values.
    ```

    Optionally, you can also implement the method `get_config()` and the class
    method `from_config` in order to support serialization, just like with
    any synalinks object.

    Note that we don't have to implement `from_config()`
    if the constructor arguments of the class and the keys in the config returned
    by `get_config()` are the same. In this case, the default `from_config()`
    works fine.
    """

    def __init__(
        self,
        schema=None,
        json=None,
        data_model=None,
    ):
        if not schema and data_model:
            schema = data_model.get_schema()
        if not json and data_model:
            if not is_meta_class(data_model):
                json = data_model.get_json()
            else:
                json = data_model().get_json()
        self._schema = schema
        self._json = json

    def get_json(self):
        return self._json

    def get_schema(self):
        return self._schema

    def __call__(self, data_model=None, **kwargs):
        """Returns a JSON object initialized as specified by the initializer.

        Args:
            data_model (DataModel): The data_model used to create the JSON data.
                If not provided, should use the initializer config
        """
        raise NotImplementedError(
            "Initializer subclasses must implement the `__call__()` method."
        )

    def get_config(self):
        """Returns the initializer's configuration as a JSON-serializable dict.

        Returns:
            A JSON-serializable Python dict.
        """
        return {}

    def _obj_type(self):
        return "Initializer"

    @classmethod
    def from_config(cls, config):
        """Instantiates an initializer from a configuration dictionary.

        Example:

        ```python
        # Note that the data_models used in the initializer **must**
        # have a default value for each field

        class Instructions(synalinks.DataModel)
            instructions: List[str] = []

        initializer = synalinks.initializers.Empty(data_model=Instructions)
        config = initializer.get_config()
        initializer = Empty.from_config(config)
        ```

        Args:
            config: A Python dictionary, the output of `get_config()`.

        Returns:
            An `Initializer` instance.
        """
        return cls(**config)

    def clone(self):
        return self.__class__.from_config(self.get_config())
