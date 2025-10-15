# Modified from: keras/src/models/model.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import inspect
import json
import typing
import warnings

from synalinks.src import utils
from synalinks.src.api_export import synalinks_export
from synalinks.src.modules import Module
from synalinks.src.trainers.trainer import Trainer
from synalinks.src.utils import file_utils
from synalinks.src.utils import io_utils
from synalinks.src.utils import summary_utils
from synalinks.src.utils.nlp_utils import remove_numerical_suffix


@synalinks_export(["synalinks.Program", "synalinks.programs.Program"])
class Program(Trainer, Module):
    """A program grouping modules into an object with training/inference features.

    There is four ways to instantiate a `Program`:

    ## With the "Functional API"

    You start from `Input`, you chain modules calls to specify the program's structure,
    and finally, you create your program from inputs and outputs:

    ```python
    import synalinks
    import asyncio
    
    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    class AnswerWithThinking(synalinks.DataModel):
        thinking: str = synalinks.Field(
            description="Your step by step thinking process",
        )
        answer: float = synalinks.Field(
            description="The correct numerical answer",
        )
    
    async def main():

        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )

        x0 = synalinks.Input(data_model=Query)
        x1 = await synalinks.Generator(
            data_model=AnswerWithThinking,
            language_model=language_model,
        )(x0)

        program = synalinks.Program(
            inputs=x0,
            outputs=x1,
            name="chain_of_thought",
            description="Useful to answer in a step by step manner.",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    Note: Only dicts, lists, and tuples of input data models are supported. Nested
    inputs are not supported (e.g. lists of list or dicts of dict).

    ## By subclassing the `Program` class

    In that case, you should define your
    modules in `__init__()` and you should implement the program's structure
    in `call()` .

    ```python
    import synalinks
    import asyncio
    
    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    class AnswerWithThinking(synalinks.DataModel):
        thinking: str = synalinks.Field(
            description="Your step by step thinking process",
        )
        answer: float = synalinks.Field(
            description="The correct numerical answer",
        )
        
    class ChainOfThought(synalinks.Program):
        \"\"\"Useful to answer in a step by step manner.
        
        The first line of the docstring is provided as description
        for the program if not provided in the `super().__init__()`.
        In a similar way the name is automatically infered based on
        the class name if not provided.
        \"\"\"

        def __init__(
            self,
            language_model=None,
            name=None,
            description=None,
            trainable=True,
        ):
            super().__init__(
                name=name,
                description=description,
                trainable=trainable,
            )
            self.answer = synalinks.Generator(
                data_model=AnswerWithThinking,
                language_model=language_model,
                name="generator_"+self.name,
            )

        async def call(self, inputs, training=False):
            if not inputs:
                return None
            x = await self.answer(inputs, training=training)
            return x

        def get_config(self):
            config = {
                "name": self.name,
                "description": self.description,
                "trainable": self.trainable,
            }
            language_model_config = \
            {
                "language_model": synalinks.saving.serialize_synalinks_object(
                    self.language_model
                )
            }
            return {**config, **language_model_config}

        @classmethod
        def from_config(cls, config):
            language_model = synalinks.saving.deserialize_synalinks_object(
                config.pop("language_model")
            )
            return cls(language_model=language_model, **config)
    
    async def main():

        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )

        program = ChainOfThought(
            language_model=language_model,
        )
    ```

    If you subclass `Program`, you can optionally have
    a `training` argument (boolean) in `call()`, which you can use to specify
    a different behavior in training and inference.

    Once the program is created, you can config the program with rewards and metrics
    with `program.compile()`, train the program with `program.fit()`, or use the program
    to do prediction with `program.predict()` or `program()`.

    To understand the difference between `program.predict()` or `program()`, read the
    [FAQ](https://synalinks.github.io/synalinks/FAQ/#whats-the-difference-between-program-methods-predict-and-__call__).

    ## Mixing the subclassing and the `Functional` API

    This way of programming is recommended to encapsulate your application while 
    providing an easy to use setup. It is the recommended way for most users as 
    it avoid making your program/agents from scratch.
    In that case, you should implement only the `__init__()` and `build()` methods.

    ```python
    import synalinks
    import asyncio

    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    class AnswerWithThinking(synalinks.DataModel):
        thinking: str = synalinks.Field(
            description="Your step by step thinking process",
        )
        answer: float = synalinks.Field(
            description="The correct numerical answer",
        )

    async def main():

        class ChainOfThought(synalinks.Program):
            \"\"\"Useful to answer in a step by step manner.\"\"\"

            def __init__(
                self,
                language_model=None,
                name=None,
                description=None,
                trainable=True,
            ):
                super().__init__(
                    name=name,
                    description=description,
                    trainable=trainable,
                )

                self.language_model = language_model
            
            async def build(self, inputs):
                outputs = await synalinks.Generator(
                    data_model=AnswerWithThinking,
                    language_model=self.language_model,
                )(inputs)

                # Create your program using the functional API
                super().__init__(
                    inputs=inputs,
                    outputs=outputs,
                    name=self.name,
                    description=self.description,
                    trainable=self.trainable,
                )

        language_model = synalinks.LanguageModel(
            model="ollama/mistral",
        )

        program = ChainOfThought(
            language_model=language_model,
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```

    This allows you to not have to implement the `call()` and serialization methods
    (`get_config()` and `from_config()`). The program will be built for any inputs 
    the first time called.

    ## With the `Sequential` class

    In addition, `synalinks.Sequential` is a special case of program where
    the program is purely a stack of single-input, single-output modules.

    ```python
    import synalinks
    import asyncio
    
    class Query(synalinks.DataModel):
        query: str = synalinks.Field(
            description="The user query",
        )

    class AnswerWithThinking(synalinks.DataModel):
        thinking: str = synalinks.Field(
            description="Your step by step thinking process",
        )
        answer: float = synalinks.Field(
            description="The correct numerical answer",
        )

    async def main():

        language_model = synalinks.LanguageModel(model="ollama/mistral")

        program = synalinks.Sequential(
            [
                synalinks.Input(
                    data_model=Query,
                ),
                synalinks.Generator(
                    data_model=AnswerWithThinking,
                    language_model=language_model,
                ),
            ],
            name="chain_of_thought",
            description="Useful to answer in a step by step manner.",
        )

    if __name__ == "__main__":
        asyncio.run(main())
    ```
    """

    def __new__(cls, *args, **kwargs):
        # Signature detection for usage of `Program` as a `Functional`
        if functional_init_arguments(args, kwargs) and cls == Program:
            from synalinks.src.programs.functional import Functional

            return Functional.__new__(Functional, *args, **kwargs)
        return typing.cast(cls, super().__new__(cls))

    def __init__(self, *args, **kwargs):
        Trainer.__init__(self)
        from synalinks.src.programs import functional

        # Signature detection for usage of a `Program` subclass
        # as a `Functional` subclass
        if functional_init_arguments(args, kwargs):
            inject_functional_program_class(self.__class__)
            functional.Functional.__init__(self, *args, **kwargs)
        else:
            Module.__init__(self, *args, **kwargs)

    async def call(self, *args, **kwargs):
        raise NotImplementedError(
            f"Program {self.__class__.__name__} does not have a `call()` "
            "method implemented."
        )

    @property
    def modules(self):
        return list(self._flatten_modules(include_self=False, recursive=False))

    @modules.setter
    def modules(self, _):
        raise AttributeError(
            "`Program.modules` attribute is reserved and should not be used. "
            "Please use another name."
        )

    def get_module(self, name=None, index=None):
        """Retrieves a module based on either its name (unique) or index.

        If `name` and `index` are both provided, `index` will take precedence.
        Indices are based on order of horizontal graph traversal (bottom-up).

        Args:
            name (str): String, name of module.
            index (int): Integer, index of module.

        Returns:
            (Module): A module instance.
        """
        if index is not None and name is not None:
            raise ValueError(
                "Provide only a module name or a module index. Received: "
                f"index={index}, name={name}."
            )
        if index is not None:
            if len(self.modules) <= index:
                raise ValueError(
                    f"Was asked to retrieve module at index {index}"
                    f" but program only has {len(self.modules)}"
                    " modules."
                )
            else:
                return self.modules[index]

        if name is not None:
            for module in self.modules:
                if module.name == name:
                    return module
            raise ValueError(
                f"No such module: {name}. Existing modules are: "
                f"{list(module.name for module in self.modules)}."
            )
        raise ValueError("Provide either a module name or module index at `get_module`.")

    def summary(
        self,
        line_length=None,
        positions=None,
        print_fn=None,
        expand_nested=False,
        show_trainable=False,
        module_range=None,
    ):
        """Prints a string summary of the program.

        Args:
            line_length (int): Total length of printed lines
                (e.g. set this to adapt the display to different
                terminal window sizes).
            positions (list): Relative or absolute positions of log elements
                in each line. If not provided, becomes
                `[0.3, 0.6, 0.70, 1.]`. Defaults to `None`.
            print_fn (Callable): Print function to use. By default, prints to `stdout`.
                If `stdout` doesn't work in your environment, change to `print`.
                It will be called on each line of the summary.
                You can set it to a custom function
                in order to capture the string summary.
            expand_nested (bool): Whether to expand the nested models.
                Defaults to `False`.
            show_trainable (bool): Whether to show if a module is trainable.
                Defaults to `False`.
            module_range (list | tuple): a list or tuple of 2 strings,
                which is the starting module name and ending module name
                (both inclusive) indicating the range of modules to be printed
                in summary. It also accepts regex patterns instead of exact
                names. In this case, the start predicate will be
                the first element that matches `module_range[0]`
                and the end predicate will be the last element
                that matches `module_range[1]`.
                By default `None` considers all modules of the model.

        Raises:
            ValueError: if `summary()` is called before the model is built.
        """
        summary_utils.print_summary(
            self,
            line_length=line_length,
            positions=positions,
            print_fn=print_fn,
            expand_nested=expand_nested,
            show_trainable=show_trainable,
            module_range=module_range,
        )

    def save(self, filepath, overwrite=True, **kwargs):
        """Saves a program as a `.json` file.

        Example:

        ```python
        import synalinks

        class Query(synalinks.DataModel):
            query: str

        class AnswerWithRationale(synalinks.DataModel):
            rationale: str
            answer: str

        language_model = LanguageModel("ollama/mistral")

        program = synalinks.Sequential(
            [
                synalinks.Input(data_model=Query),
                synalinks.Generator(
                    data_model=AnswerWithRationale,
                    language_model=language_model,
                ),
            ],
        )

        program.save("program.json")
        loaded_program = synalinks.programs.program_from_json("program.json")
        ```

        The saved `.json` file contains:

        - The program's configuration (architecture)
        - The program's variables
        - The program's optimizer's state (if any)
        - The program's reward's state (if any)

        Thus programs can be reinstantiated in the exact same state.

        Args:
            filepath (str | os.PathLike): `str` or `os.PathLike` object.
                The path where to save the model. Must end in `.json`.
            overwrite (bool): Whether we should overwrite any existing program at
                the target location, or instead ask the user via
                an interactive prompt. Default to `True`.
        """
        from synalinks.src.saving import serialization_lib

        filepath = file_utils.path_to_string(filepath)
        if not filepath.endswith(".json"):
            raise ValueError(
                f"The filepath should ends with '.json', received filepath={filepath}"
            )
        program_config = serialization_lib.serialize_synalinks_object(self)
        variables_config = self.get_state_tree()
        program_config.update({"variables": variables_config})
        program_config_string = json.dumps(program_config, indent=2)
        if file_utils.exists(filepath) and not overwrite:
            io_utils.ask_to_proceed_with_overwrite(filepath)
        with open(filepath, "w") as f:
            f.write(program_config_string)

    async def build_from_config(self, config):
        if not config:
            return
        status = False
        if "input_schema" in config:
            # Case: all inputs are in the first arg (possibly nested).
            if utils.is_default(self.build):
                status = self._build_by_run_for_single_pos_arg(config["input_schema"])
            else:
                try:
                    await self.build(config["input_schema"])
                    status = True
                except:
                    pass
            self._build_schemas_dict = config

        elif "schemas_dict" in config:
            # Case: inputs were recorded as multiple keyword arguments.
            if utils.is_default(self.build):
                status = self._build_by_run_for_kwargs(config["schemas_dict"])
            else:
                try:
                    await self.build(**config["schemas_dict"])
                    status = True
                except:
                    pass
            self._build_schemas_dict = config["schemas_dict"]

        if not status:
            warnings.warn(
                f"Program '{self.name}' had a build config, but the program "
                "cannot be built automatically in "
                "`build_from_config(config)`. "
                "You should implement "
                "`def build_from_config(self, config)`, "
                "and you might also want to implement the method "
                " that generates the config at saving time, "
                "`def get_build_config(self)`. "
                "The method `build_from_config()` is meant to "
                "create the state of the model (i.e. its variables) "
                "upon deserialization.",
                stacklevel=2,
            )

    def to_json(self, **kwargs):
        """Returns a JSON string containing the network configuration.

        ```python
        json_string = program.to_json()
        ```

        To load a network from a JSON save file, use
        `synalinks.programs.program_from_json(json_string, custom_objects={...})`.

        Args:
            **kwargs (keyword arguments): Additional keyword arguments to be passed to
                `json.dumps()`.

        Returns:
            (str): A JSON string.
        """
        from synalinks.src.saving import serialization_lib

        program_config = serialization_lib.serialize_synalinks_object(self)
        return json.dumps(program_config, **kwargs)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from synalinks.src.programs.functional import Functional

        functional_config_keys = [
            "name",
            "modules",
            "input_modules",
            "output_modules",
        ]
        is_functional_config = all(key in config for key in functional_config_keys)
        argspec = inspect.getfullargspec(cls.__init__)
        functional_init_args = inspect.getfullargspec(Functional.__init__).args[1:]
        revivable_as_functional = (
            cls in {Functional, Program}
            or argspec.args[1:] == functional_init_args
            or (argspec.varargs == "args" and argspec.varkw == "kwargs")
        )
        if is_functional_config and revivable_as_functional:
            # Revive Functional model
            # (but not Functional subclasses with a custom __init__)
            from synalinks.src.programs.functional import functional_from_config

            return functional_from_config(cls, config, custom_objects=custom_objects)

        # Either the model has a custom __init__, or the config
        # does not contain all the information necessary to
        # revive a Functional model. This happens when the user creates
        # subclassed models where `get_config()` is returning
        # insufficient information to be considered a Functional model.
        # In this case, we fall back to provide all config into the
        # constructor of the class.
        try:
            return cls(**config)
        except TypeError as e:
            raise TypeError(
                "Unable to revive program from config. When overriding "
                "the `get_config()` method, make sure that the "
                "returned config contains all items used as arguments "
                f"in the  constructor to {cls}, "
                "which is the default behavior. "
                "You can override this default behavior by defining a "
                "`from_config(cls, config)` class method to specify "
                "how to create an "
                f"instance of {cls.__name__} from its config.\n\n"
                f"Received config={config}\n\n"
                f"Error encountered during deserialization: {e}"
            )

    def get_state_tree(self):
        """Retrieves tree-like structure of program variables.

        This method allows retrieval of different program variables (trainable,
        non-trainable, optimizer, and metrics). The variables are returned in a
        nested dictionary format, where the keys correspond to the variable
        names and the values are the nested representations of the variables.

        Example:

        ```python
        program.compile(
            optimizer=synalinks.optimizers.RandomFewShot(),
            reward=synalinks.rewards.ExactMatch(),
        )
        program.fit(x=x_train, y=y_train)
        state_tree = program.get_state_tree()
        ```

        Returns:
            (dict): A dictionary containing the nested representations of the
                requested variables. The keys are the variable names, and the
                values are the corresponding nested dictionaries.
        """
        variables = {}
        variables["trainable_variables"] = self._create_nested_dict(
            self.trainable_variables
        )
        variables["non_trainable_variables"] = self._create_nested_dict(
            self.non_trainable_variables
        )
        if self.optimizer:
            variables["optimizer_trainable_variables"] = self._create_nested_dict(
                self.optimizer.trainable_variables
            )
            variables["optimizer_non_trainable_variables"] = self._create_nested_dict(
                self.optimizer.non_trainable_variables
            )
        variables["metrics_variables"] = self._create_nested_dict(self.metrics_variables)
        return variables

    def _create_nested_dict(self, variables):
        flat_dict = {}
        for v in variables:
            if v.path in flat_dict:
                raise ValueError(
                    "The following variable path is found twice in the program: "
                    f"'{v.path}'. `get_state_tree()` can only be called when "
                    "all variable paths are unique. Make sure to give unique "
                    "names to your modules (and other objects)."
                )
            flat_dict[v.path] = v.get_json()

        nested_dict = {}
        for path, value in flat_dict.items():
            parts = path.split("/")
            current_dict = nested_dict
            for part in parts[:-1]:
                if part not in current_dict:
                    current_dict[part] = {}
                current_dict = current_dict[part]
            current_dict[parts[-1]] = value

        return nested_dict

    def set_state_tree(self, state_tree):
        """Assigns values to variables of the program.

        This method takes a dictionary of nested variable values, which
        represents the state tree of the program, and assigns them to the
        corresponding variables of the program. The dictionary keys represent the
        variable names (e.g., `'trainable_variables'`, `'optimizer_variables'`),
        and the values are nested dictionaries containing the variable
        paths and their corresponding values.

        Args:
            state_tree (dict): A dictionary representing the state tree of the program.
                The keys are the variable names, and the values are nested
                dictionaries representing the variable paths and their values.
        """
        for k, v in state_tree.items():
            path_value_dict = self._flatten_nested_dict(v)
            if k == "trainable_variables":
                self._assign_variable_values(self.trainable_variables, path_value_dict)
            elif k == "non_trainable_variables":
                self._assign_variable_values(
                    self.non_trainable_variables, path_value_dict
                )
            elif k == "optimizer_trainable_variables":
                if self.optimizer:
                    self._assign_variable_values(
                        self.optimizer.trainable_variables, path_value_dict
                    )
            elif k == "optimizer_non_trainable_variables":
                if self.optimizer:
                    self._assign_variable_values(
                        self.optimizer.non_trainable_variables, path_value_dict
                    )
            elif k == "metrics_variables":
                self._assign_variable_values(self.metrics_variables, path_value_dict)
            else:
                warnings.warn(f"Unknown variable name: {k}")

    def _assign_variable_values(self, variables, path_value_dict):
        for full_path, value in path_value_dict.items():
            path = "/".join(full_path.split("/")[:-1])
            field_name = full_path.split("/")[-1]
            for variable in variables:
                if remove_numerical_suffix(variable.path) == path:
                    variable.get_json()[field_name] = value

    def _flatten_nested_dict(self, nested_dict):
        flat_dict = {}

        def _flatten(current_dict, prefix=""):
            for key, value in current_dict.items():
                if isinstance(value, dict):
                    _flatten(value, prefix + key + "/")
                else:
                    flat_dict[prefix + key] = value

        _flatten(nested_dict)
        return flat_dict

    def save_variables(self, filepath, overwrite=True):
        """Saves all module variables to a `.variables.json` file.

        Args:
            filepath (str | pathlib.Path): `str` or `pathlib.Path` object.
                Path where to save the program. Must end in `.variables.json`.
            overwrite (bool): Whether we should overwrite any existing program
                at the target location, or instead ask the user
                via an interactive prompt.
        """
        filepath = file_utils.path_to_string(filepath)
        if not filepath.endswith(".variables.json"):
            raise ValueError(
                "The filepath should ends with '.variables.json', "
                f"received filepath={filepath}"
            )
        config = self.get_state_tree()
        config_string = json.dumps(config, indent=2)
        if file_utils.exists(filepath) and not overwrite:
            io_utils.ask_to_proceed_with_overwrite(filepath)
        with open(filepath, "w") as f:
            f.write(config_string)

    def load_variables(self, filepath):
        """Load all module variables from a `.variable.json` file.

        Args:
            filepath (str | pathlib.Path): `str` or `pathlib.Path` object.
                Path to load the program's variables from.
                Must end in `.variables.json`.
        """
        filepath = file_utils.path_to_string(filepath)
        if not filepath.endswith(".variables.json"):
            raise ValueError(
                "The filepath should ends with '.variables.json', "
                f"received filepath={filepath}"
            )
        with open(filepath, "r") as f:
            state_tree_config = json.loads(f.read())
        self.set_state_tree(state_tree_config)

    @classmethod
    def load(cls, filepath, custom_objects=None):
        """Load a program from a JSON file.

        Example:

        ```python
        import synalinks

        loaded_program = synalinks.Program.load("program.json")
        ```

        Args:
            filepath (str | pathlib.Path): `str` or `pathlib.Path` object.
                Path to load the program's variables from.
                Must end in `.variables.json`.
            custom_objects (dict): Optional dictionary mapping names
                (strings) to custom classes or functions to be
                considered during deserialization.

        Returns:
            (Program): A Synalinks program instance (uncompiled).
        """
        filepath = file_utils.path_to_string(filepath)
        if not filepath.endswith(".json"):
            raise ValueError(
                f"The filepath should ends with '.json', received filepath={filepath}"
            )
        with open(filepath, "r") as f:
            json_config = f.read()
        return program_from_json(json_config, custom_objects=custom_objects)


@synalinks_export("synalinks.programs.program_from_json")
def program_from_json(json_string, custom_objects=None):
    """Parses a JSON program configuration string and returns a program instance.

    Example:

    ```python
    import synalinks

    class Query(synalinks.DataModel):
        query: str

    class AnswerWithRationale(synalinks.DataModel):
        rationale: str
        answer: str

    language_model = LanguageModel("ollama/mistral")

    program = synalinks.Sequential(
        [
            synalinks.Input(data_model=Query),
            synalinks.Generator(
                data_model=AnswerWithRationale,
                language_model=language_model,
            ),
        ],
    )

    config = program.to_json()
    loaded_program = synalinks.programs.program_from_json(config)
    ```

    Args:
        json_string (str): JSON string encoding a program configuration.
        custom_objects (dict): Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        (Program): A Synalinks program instance (uncompiled).
    """
    from synalinks.src.saving import serialization_lib

    program_config = json.loads(json_string)
    variables_config = program_config.get("variables")
    program = serialization_lib.deserialize_synalinks_object(
        program_config, custom_objects=custom_objects
    )
    program.set_state_tree(variables_config)
    return program


def functional_init_arguments(args, kwargs):
    return (
        (len(args) == 2)
        or (len(args) == 1 and "outputs" in kwargs)
        or ("inputs" in kwargs and "outputs" in kwargs)
    )


def inject_functional_program_class(cls):
    """Inject `Functional` into the hierarchy of this class if needed."""
    from synalinks.src.programs import functional

    if cls is Program:
        return functional.Functional
    # In case there is any multiple inheritance, we stop injecting the
    # class if synalinks model is not in its class hierarchy.
    if cls is object:
        return object

    cls.__bases__ = tuple(inject_functional_program_class(base) for base in cls.__bases__)
    # Trigger any `__new__` class swapping that needed to happen on `Functional`
    # but did not because functional was not in the class hierarchy.
    cls.__new__(cls)

    return cls
