# The Sequential class

Bases: `Program`

`Sequential` groups a linear stack of modules into a `Program`.

Examples:

```
program = synalinks.Sequential(
    name="chain_of_thought",
    description="Useful to answer in a step by step manner."
)
program.add(
    synalinks.Input(
            data_program=Query,
        )
)
program.add(
    synalinks.Generator(
        data_program=AnswerWithRationale,
        language_program=language_program,
    )
)

# Note that you can also omit the initial `Input`.
# In that case the program doesn't have any variables until the first call
# to a training/evaluation method (since it isn't yet built):

program = synalinks.Sequential(
    name="chain_of_thought",
    description="Useful to answer in a step by step manner."
)
program.add(
    synalinks.Generator(
        data_program=AnswerWithRationale,
        language_program=language_program,
    )
)
# program.variables not created yet

# Whereas if you specify an `Input`, the program gets built
# continuously as you are adding modules:

program = synalinks.Sequential(
    name="chain_of_thought",
    description="Useful to answer in a step by step manner."
)
program.add(
    synalinks.Input(
        data_program=Query,
    )
)
program.add(
    synalinks.Generator(
        data_program=AnswerWithRationale,
        language_program=language_program,
    )
)

# Note that when using the delayed-build pattern (no input specified),
# the program gets built the first time you call `fit`, `eval`, or `predict`,
# or the first time you call the program on some input data.
```

Source code in `synalinks/src/programs/sequential.py`

````
@synalinks_export(["synalinks.Sequential", "synalinks.programs.Sequential"])
class Sequential(Program):
    """`Sequential` groups a linear stack of modules into a `Program`.

    Examples:

    ```python
    program = synalinks.Sequential(
        name="chain_of_thought",
        description="Useful to answer in a step by step manner."
    )
    program.add(
        synalinks.Input(
                data_program=Query,
            )
    )
    program.add(
        synalinks.Generator(
            data_program=AnswerWithRationale,
            language_program=language_program,
        )
    )

    # Note that you can also omit the initial `Input`.
    # In that case the program doesn't have any variables until the first call
    # to a training/evaluation method (since it isn't yet built):

    program = synalinks.Sequential(
        name="chain_of_thought",
        description="Useful to answer in a step by step manner."
    )
    program.add(
        synalinks.Generator(
            data_program=AnswerWithRationale,
            language_program=language_program,
        )
    )
    # program.variables not created yet

    # Whereas if you specify an `Input`, the program gets built
    # continuously as you are adding modules:

    program = synalinks.Sequential(
        name="chain_of_thought",
        description="Useful to answer in a step by step manner."
    )
    program.add(
        synalinks.Input(
            data_program=Query,
        )
    )
    program.add(
        synalinks.Generator(
            data_program=AnswerWithRationale,
            language_program=language_program,
        )
    )

    # Note that when using the delayed-build pattern (no input specified),
    # the program gets built the first time you call `fit`, `eval`, or `predict`,
    # or the first time you call the program on some input data.

    ```
    """

    def __new__(cls, *args, **kwargs):
        return typing.cast(cls, super().__new__(cls))

    def __init__(self, modules=None, trainable=True, name=None, description=None):
        if description is None:
            raise ValueError(
                "All Sequential programs must have a `description`, "
                "please add it to the constructor arguments"
            )
        super().__init__(trainable=trainable, name=name, description=description)
        self._functional = None
        self._modules = []
        if modules:
            for module in modules:
                self.add(module, rebuild=False)
            run_maybe_nested(self._maybe_rebuild())

    def add(self, module, rebuild=True):
        """Adds a module instance on top of the module stack.

        Args:
            module (Module): Module instance.
            rebuild (bool): If `True` rebuild the program.
        """
        # If we are passed a SymbolicDataModel created by synalinks.Input(), we
        # extract the input module from its synalinks history and use that.
        if hasattr(module, "_synalinks_history"):
            origin_module = module._synalinks_history[0]
            if isinstance(origin_module, InputModule):
                module = origin_module
        if not isinstance(module, Module):
            raise ValueError(
                "Only instances of `synalinks.Module` can be "
                f"added to a Sequential program. Received: {module} "
                f"(of type {type(module)})"
            )
        if not self._is_module_name_unique(module):
            raise ValueError(
                "All modules added to a Sequential program "
                f"should have unique names. Name '{module.name}' is already "
                "the name of a module in this program. Update the `name` argument "
                "to pass a unique name."
            )
        if (
            isinstance(module, InputModule)
            and self._modules
            and isinstance(self._modules[0], InputModule)
        ):
            raise ValueError(
                f"Sequential program '{self.name}' has already been configured "
                f"to use input schema {self._modules[0].input_schema}. You cannot "
                f"add a different Input module to it."
            )

        self._modules.append(module)
        if rebuild:
            run_maybe_nested(self._maybe_rebuild())
        else:
            self.built = False
            self._functional = None

    def pop(self, rebuild=True):
        """Removes the last module in the program.

        Args:
            rebuild (bool): If `True` rebuild the program.
        """
        module = self._modules.pop()
        self.built = False
        self._functional = None
        if rebuild:
            run_maybe_nested(self._maybe_rebuild())
        return module

    async def _maybe_rebuild(self):
        self.built = False
        self._functional = None
        if isinstance(self._modules[0], InputModule) and len(self._modules) > 1:
            input_schema = self._modules[0].get_schema()
            await self.build(Input(schema=input_schema))
        elif hasattr(self._modules[0], "input_schema") and len(self._modules) > 1:
            # We can build the Sequential program if the first module has the
            # `input_schema` property. This is most commonly found in Functional
            # program.
            input_schema = self._modules[0].input_schema
            await self.build(Input(schema=input_schema))

    def _lock_state(self):
        # Unlike other modules, Sequential is mutable after build.
        pass

    def _obj_type(self):
        return "Sequential"

    async def build(self, inputs):
        try:
            input_schema = standardize_schema(inputs.get_schema())
        except Exception:
            # Do not attempt to build if the program does not have a single
            # input.
            return
        if not self._modules:
            raise ValueError(
                f"Sequential program {self.name} cannot be built because it has "
                "no modules. Call `program.add(module)`."
            )
        if isinstance(self._modules[0], InputModule):
            if self._modules[0].get_schema() != input_schema:
                raise ValueError(
                    f"Sequential program '{self.name}' has already been "
                    "configured to use input schema "
                    f"{self._modules[0].get_schema()}. You cannot build it "
                    f"with input_schema {input_schema}"
                )
        else:
            self._modules = [InputModule(schema=input_schema)] + self._modules

        # Build functional program
        inputs = self._modules[0].output
        x = inputs
        for module in self._modules[1:]:
            try:
                x = await module(x)
            except NotImplementedError:
                # Can happen if spec inference is not implemented.
                # TODO: consider reverting inbound nodes on modules processed.
                return
            except TypeError as e:
                signature = inspect.signature(module.call)
                positional_args = [
                    param
                    for param in signature.parameters.values()
                    if param.default == inspect.Parameter.empty
                ]
                if len(positional_args) != 1:
                    raise ValueError(
                        "Modules added to a Sequential program "
                        "can only have a single positional argument, "
                        f"the input data model. Module {module.__class__.__name__} "
                        f"has multiple positional arguments: {positional_args}"
                    )
                raise e
        outputs = x
        self._functional = Functional(inputs=inputs, outputs=outputs)
        self.built = True

    async def call(self, inputs, training=None):
        if self._functional:
            return await self._functional.call(inputs, training=training)
        # Fallback: Just apply the module sequence.
        # This typically happens if `inputs` is a nested struct.
        for module in self.modules:
            # During each iteration, `inputs` are the inputs to `module`, and
            # `outputs` are the outputs of `module` applied to `inputs`. At the
            # end of each iteration `inputs` is set to `outputs` to prepare for
            # the next module.
            kwargs = {}
            if module._call_has_training_arg and training is not None:
                kwargs["training"] = training
            outputs = await module(inputs, **kwargs)
            inputs = outputs
        return outputs

    @property
    def modules(self):
        """Unlike Keras, also output the potentially auto-generated `InputModule`"""
        return self._modules

    @modules.setter
    def modules(self, _):
        raise AttributeError(
            "`Sequential.modules` attribute is reserved and should not be used. "
            "Use `add()` and `pop()` to change the modules in this program."
        )

    async def compute_output_spec(self, inputs, training=None):
        if self._functional:
            return await self._functional.compute_output_spec(
                inputs,
                training=training,
            )
        # Direct application
        for module in self.modules:
            outputs = await module.compute_output_spec(inputs, training=training)
            inputs = outputs
        return outputs

    @property
    def input_schema(self):
        if self._functional:
            return self._functional.input_schema
        raise AttributeError(
            f"Sequential program '{self.name}' has no defined input schema yet."
        )

    @property
    def output_schema(self):
        if self._functional:
            return self._functional.output_schema
        raise AttributeError(
            f"Sequential program '{self.name}' has no defined output schema yet."
        )

    @property
    def inputs(self):
        if self._functional:
            return self._functional.inputs
        raise AttributeError(
            f"Sequential program '{self.name}' has no defined inputs yet."
        )

    @property
    def outputs(self):
        if self._functional:
            return self._functional.outputs
        raise AttributeError(
            f"Sequential program '{self.name}' has no defined outputs yet."
        )

    def _is_module_name_unique(self, module):
        for ref_module in self._modules:
            if module.name == ref_module.name and ref_module is not module:
                return False
        return True

    def get_config(self):
        serialize_fn = serialization_lib.serialize_synalinks_object
        module_configs = []
        for module in self.modules:
            module_configs.append(serialize_fn(module))
        config = Program.get_config(self)
        config["name"] = self.name
        config["description"] = self.description
        config["modules"] = copy.deepcopy(module_configs)
        if self._functional is not None:
            config["build_input_schema"] = self._modules[0].input_schema
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if "name" in config:
            name = config["name"]
            build_input_schema = config.get("build_input_schema")
            module_configs = config["modules"]
        else:
            name = None
            module_configs = config
        if "description" in config:
            description = config["description"]
        else:
            description = None
        program = cls(name=name, description=description)
        for module_config in module_configs:
            module = serialization_lib.deserialize_synalinks_object(
                module_config,
                custom_objects=custom_objects,
            )
            program.add(module)
        if (
            not program._functional
            and "build_input_schema" in locals()
            and build_input_schema
            and isinstance(build_input_schema, (tuple, list))
        ):
            program.build(build_input_schema)
        return program
````

## `modules`

Unlike Keras, also output the potentially auto-generated `InputModule`

## `add(module, rebuild=True)`

Adds a module instance on top of the module stack.

Parameters:

| Name      | Type     | Description                  | Default    |
| --------- | -------- | ---------------------------- | ---------- |
| `module`  | `Module` | Module instance.             | *required* |
| `rebuild` | `bool`   | If True rebuild the program. | `True`     |

Source code in `synalinks/src/programs/sequential.py`

```
def add(self, module, rebuild=True):
    """Adds a module instance on top of the module stack.

    Args:
        module (Module): Module instance.
        rebuild (bool): If `True` rebuild the program.
    """
    # If we are passed a SymbolicDataModel created by synalinks.Input(), we
    # extract the input module from its synalinks history and use that.
    if hasattr(module, "_synalinks_history"):
        origin_module = module._synalinks_history[0]
        if isinstance(origin_module, InputModule):
            module = origin_module
    if not isinstance(module, Module):
        raise ValueError(
            "Only instances of `synalinks.Module` can be "
            f"added to a Sequential program. Received: {module} "
            f"(of type {type(module)})"
        )
    if not self._is_module_name_unique(module):
        raise ValueError(
            "All modules added to a Sequential program "
            f"should have unique names. Name '{module.name}' is already "
            "the name of a module in this program. Update the `name` argument "
            "to pass a unique name."
        )
    if (
        isinstance(module, InputModule)
        and self._modules
        and isinstance(self._modules[0], InputModule)
    ):
        raise ValueError(
            f"Sequential program '{self.name}' has already been configured "
            f"to use input schema {self._modules[0].input_schema}. You cannot "
            f"add a different Input module to it."
        )

    self._modules.append(module)
    if rebuild:
        run_maybe_nested(self._maybe_rebuild())
    else:
        self.built = False
        self._functional = None
```

## `pop(rebuild=True)`

Removes the last module in the program.

Parameters:

| Name      | Type   | Description                  | Default |
| --------- | ------ | ---------------------------- | ------- |
| `rebuild` | `bool` | If True rebuild the program. | `True`  |

Source code in `synalinks/src/programs/sequential.py`

```
def pop(self, rebuild=True):
    """Removes the last module in the program.

    Args:
        rebuild (bool): If `True` rebuild the program.
    """
    module = self._modules.pop()
    self.built = False
    self._functional = None
    if rebuild:
        run_maybe_nested(self._maybe_rebuild())
    return module
```
