# Modified from: keras/src/backend/common/variables.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src import backend
from synalinks.src.backend.common import global_state
from synalinks.src.backend.common.json_data_model import JsonDataModel
from synalinks.src.backend.common.json_schema_utils import standardize_schema
from synalinks.src.backend.common.name_scope import current_path
from synalinks.src.backend.common.stateless_scope import get_stateless_scope
from synalinks.src.backend.common.stateless_scope import in_stateless_scope
from synalinks.src.backend.common.symbolic_data_model import SymbolicDataModel
from synalinks.src.utils.naming import auto_name


class Variable:
    """A backend-agnostic variable in synalinks.

    A `Variable` acts as a container for state.
    It holds a JSON object value with the corresponding schema and can
    be updated by the optimizers.

    A Variable is different from a JsonDataModel as it can be modified by the optimizers

    Note that the DataModel used for the variable declaration
    **must have a default value** for each of its field.

    Examples:

    **Initializing a `Variable` with a dict:**

    ```python
    from typing import List
    import synalinks

    class Instructions(synalinks.DataModel):
        instructions: List[str] = []

    initial_data = {
        "instructions": [
            "For any problem involving division, always round the quotient to "
            "the nearest even number, regardless of the remainder."
        ],
    }
    variable_from_dict = synalinks.Variable(
        initializer=initial_data,
        data_model=Instructions,
    )
    ```

    **Using a synalinks initializer to create a `Variable`:**

    ```python
    from typing import List
    import synalinks

    class Instructions(synalinks.DataModel):
        instructions: List[str] = []

    from synalinks.initializers import Empty

    variable_from_initializer = synalinks.Variable(
        initializer=Empty(data_model=Instructions)
    )
    ```

    **Updating the value of a `Variable`:**

    ```python
    new_json = {
        "instructions": [
            "When performing division, always check if the division results "
            "in a whole number. If not, express the result as a fraction or "
            "a decimal, depending on the context of the problem."
        ],
    }
    variable_from_dict.assign(new_json)
    ```

    **Marking a `Variable` as non-trainable:**

    ```python
    from typing import List
    import synalinks

    class Instructions(synalinks.DataModel):
        instructions: List[str] = []

    from synalinks.initializers import Empty

    non_trainable_variable = synalinks.Variable(
        initializer=Empty(data_model=Instructions), trainable=False
    )
    ```

    Args:
        initializer (str | dict | Initializer): Initial value (dict) or callable
            (Initializer) for initialization. If a callable is used, it should
            take the arguments `data_model`.
        data_model (DataModel): The backend-dependent data model used as spec.
        trainable (bool): Optional. Boolean indicating if variable is trainable.
            Defaults to `True`.
        name (str): Optional. A unique name for the variable. Automatically generated
            if not set.
    """

    def __init__(
        self,
        initializer=None,
        data_model=None,
        trainable=True,
        name=None,
    ):
        name = name or auto_name(self.__class__.__name__)
        if not isinstance(name, str) or "/" in name:
            raise ValueError(
                "Argument `name` must be a string and "
                "cannot contain character `/`. "
                f"Received: name={name}"
            )
        self._name = name
        parent_path = current_path()
        if parent_path:
            self._path = current_path() + "/" + name
        else:
            self._path = name
        self._initializer = None
        self._data_model = data_model
        self._trainable = bool(trainable)

        if in_stateless_scope():
            if callable(initializer):
                self._json = None
                self._initializer = initializer
                self._schema = standardize_schema(data_model.get_schema())
                register_uninitialized_variable(self)
            else:
                raise ValueError(
                    "You are attempting to create a variable "
                    "while in a stateless scope. This is disallowed. "
                    "Make sure that all variables are created "
                    "before you start using your module/program objects.\n\n"
                    "In some cases, you might be seeing this error "
                    "because you need to "
                    "implement a `def build(self, input_schema)` method "
                    "on your module/program, which will "
                    "create its variables.\n\n"
                    "In some other cases, you might be seeing this error "
                    "because you are instantiating a `Variable` and "
                    "assigning it to a module without going through "
                    "self.add_variable(). Always prefer "
                    "using these methods "
                    "(with a `data_model` and `initializer` argument)."
                )
        else:
            if callable(initializer):
                self._initialize_with_initializer(initializer)
            else:
                if data_model is None:
                    raise ValueError(
                        "When creating a Variable from an a dict,"
                        "the `data_model` argument should be specified. "
                        f"Received: initializer={initializer} "
                        f"and data_model={data_model}"
                    )
                value = initializer
                self._initialize(value)
                self._schema = standardize_schema(data_model.get_schema())

    def _deferred_initialize(self):
        """Deferred initialization of the variable.

        Raises:
            ValueError: If the variable is already initialized or
                if attempting to initialize while in a stateless scope.
        """
        if self._json is not None:
            raise ValueError(f"Variable '{self._path}' is already initialized.")

        if in_stateless_scope():
            raise ValueError(
                "You are attempting to initialize a variable "
                "while in a stateless scope. This is disallowed. "
                "Make sure that all variables are initialized "
                "before you start using your layer/model objects."
            )
        self._initialize_with_initializer(self._initializer)
        self._initializer = None

    def get_json(self):
        """The current value of the variable.

        Returns:
            (dict): The current value of the variable.
        """
        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                return value
        if self._json is None:
            # Uninitialized variable. Return a placeholder.
            # This is fine because it's only ever used
            # in during schema inference / graph tracing
            # (anything else would be a bug, to be fixed.)
            return self._initializer(data_model=self._data_model)
        return self._json

    def prettify_schema(self):
        """Get a pretty version of the JSON schema for display.

        Returns:
            (dict): The indented JSON schema.
        """
        import json

        return json.dumps(self.get_schema(), indent=2)

    def prettify_json(self):
        """Get a pretty version of the JSON object for display.

        Returns:
            (dict): The indented JSON object.
        """
        import json

        return json.dumps(self.get_json(), indent=2)

    def assign(self, value):
        """Assigns a new value to the variable.

        Args:
            value (dict | DataModel | JsonDataModel): The new value to be assigned.
                The value can be an instanciated data model or JSON dict.

        Returns:
            (dict): The assigned value.

        Raises:
            ValueError: If the schema of the target variable and
                the value are incompatible.
        """
        if backend.is_data_model(value):
            value = value.get_json()
        if in_stateless_scope():
            scope = get_stateless_scope()
            scope.add_update((self, value))
        else:
            self._direct_assign(value)
        return value

    def _direct_assign(self, json):
        """Directly assigns a new value to the variable.

        Args:
            json (dict): The new json value to be assigned.
        """
        self._json = json

    def get_schema(self):
        """The schema of the variable.

        Returns:
            (dict): The JSON schema of the variable.
        """
        return self._schema

    def __contains__(self, other):
        """Check if the schema of `other` is contained within the schema of this variable.

        Args:
            other (SymbolicDataModel | DataModel): The data model to compare against this variable's schema.

        Returns:
            bool: True if `other`'s schema is a subset of this variable's schema.
        """
        from synalinks.src.backend.common.json_schema_utils import contains_schema

        return contains_schema(self.get_schema(), other.get_schema())

    @property
    def trainable(self):
        """Whether the variable is trainable."""
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = bool(value)

    @property
    def name(self):
        """The name of the variable."""
        return self._name

    @property
    def path(self):
        """The path of the variable within the program or module."""
        return self._path

    def __repr__(self):
        json = None
        if self._json is not None:
            json = self._json
        json_str = f", json={json}" if json is not None else ""
        return f"<Variable path={self.path}, schema={self._schema}{json_str}>"

    def _initialize_with_initializer(self, initializer):
        """Initializes the variable using an initializer object.

        Args:
            initializer (Initializer): The initializer to be used.
        """
        value = initializer()
        self._schema = standardize_schema(initializer.get_schema())
        self._initialize(value)

    def _initialize(self, json):
        """Initializes the variable with a given json dict.

        Args:
            json (dict): The initial value (JSON object dict).
        """
        self._json = json

    def to_json_data_model(self):
        """Convert the variable into a `JsonDataModel`.

        Returns:
            (JsonDataModel): The equivalent backend-independent data model
        """
        return JsonDataModel(schema=self.get_schema(), json=self.get_json())

    def to_symbolic_data_model(self):
        """Convert the variable into a `SymbolicDataModel`.

        Returns:
            (SymbolicDataModel): The equivalent symbolic data model
        """
        return SymbolicDataModel(schema=self._schema)

    def get(self, key, default_value=None):
        """Get wrapper to make easier to access fields.

        Args:
            key (str): The key to access.
        """
        return self._json.get(key, default_value)

    def update(self, kv_dict):
        """Update wrapper to make easier to modify fields.

        Args:
            kv_dict (dict): The key/value dict to update.
        """
        self._json.update(kv_dict)


def register_uninitialized_variable(variable):
    uninitialized_variables = global_state.get_global_attribute(
        "uninitialized_variables", [], set_to_default=True
    )
    uninitialized_variables.append(variable)


def initialize_all_variables():
    collection = global_state.get_global_attribute("uninitialized_variables")
    if collection:
        for v in collection:
            v._deferred_initialize()
    global_state.set_global_attribute("uninitialized_variables", [])
