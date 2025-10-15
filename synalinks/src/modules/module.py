# Modified from: keras/src/layers/layer.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import collections
import inspect
import uuid
import warnings
from functools import wraps

from synalinks.src import backend
from synalinks.src import initializers
from synalinks.src import tree
from synalinks.src import utils
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import is_trainable
from synalinks.src.backend.common import global_state
from synalinks.src.backend.common.name_scope import current_path
from synalinks.src.hooks.hook_list import HookList
from synalinks.src.metrics import Metric
from synalinks.src.ops.operation import Operation
from synalinks.src.saving.synalinks_saveable import SynalinksSaveable
from synalinks.src.utils import python_utils
from synalinks.src.utils import tracking
from synalinks.src.utils.async_utils import run_maybe_nested
from synalinks.src.utils.naming import auto_name

if backend.backend() == "pydantic":
    from synalinks.src.backend.pydantic.module import PydanticModule as BackendModule
else:
    raise RuntimeError(
        f"Backend '{backend.backend()}' must implement a module mixin class."
    )


@synalinks_export(["synalinks.Module", "synalinks.modules.Module"])
class Module(BackendModule, Operation, SynalinksSaveable):
    """This is the class from which all modules inherit.

    A module is a callable object that takes as input one or more data models and
    that outputs one or more data models. It involves *computation*, defined
    in the `call()` method, and a *state* (the variables). State can be
    created:

    * in `__init__()`, for instance via `self.add_variable()`;
    * in the optional `build()` method, which is invoked by the first
      `__call__()` to the module, and supplies the schema(s) of the input(s),
      which may not have been known at initialization time.

    Modules are recursively composable: If you assign a Module instance as an
    attribute of another Module, the outer Module will start tracking the variables
    created by the inner module. Nested modules should be instantiated in the
    `__init__()` method or `build()` method.

    Users will just instantiate a module and then treat it as a callable.

    Args:
        trainable (bool): Boolean, whether the module's variables should be trainable.
        name (str): String name of the module.

    We recommend that descendants of `Module` implement the following methods:

    * `__init__()`: Defines custom modules attributes, and creates module variables
        that do not depend on input schemas, using `add_variable()`,
        or other state.
    * `build(self, input_schema)`: This method can be used to create variables that
        depend on the schemas(s) of the input(s), using `add_variable()`, or other
        state. `__call__()` will automatically build the module
        (if it has not been built yet) by calling `build()`.
    * `call(self, *args, **kwargs)`: Called in `__call__` after making
        sure `build()` has been called. `call()` performs the logic of applying
        the module to the input arguments.
        Two reserved keyword arguments you can optionally use in `call()` are:
            1. `training` (boolean, whether the call is in inference mode or
                training mode).
        A typical signature for this method is `call(self, inputs)`, and user
        could optionally add `training` if the module need it.
    * `get_config(self)`: Returns a dictionary containing the configuration
        used to initialize this module. If the keys differ from the arguments
        in `__init__()`, then override `from_config(self)` as well.
        This method is used when saving
        the module or a program that contains this module.
    """

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)  # , *args, **kwargs)

        # Wrap the user-provided `build` method in the `build_wrapper`
        # to add name scope support and serialization support.
        original_build_method = obj.build

        @wraps(original_build_method)
        async def build_wrapper(*args, **kwargs):
            with obj._open_name_scope():
                obj._path = current_path()
                await original_build_method(*args, **kwargs)
            # Record build config.
            signature = inspect.signature(original_build_method)
            obj._build_schemas_dict = signature.bind(*args, **kwargs).arguments
            # Set built, post build actions, and lock state.
            obj.built = True
            obj._post_build()
            obj._lock_state()

        obj.build = build_wrapper
        return obj

    def __init__(
        self,
        *,
        trainable=True,
        name=None,
        description=None,
        hooks=None,
        **kwargs,
    ):
        BackendModule.__init__(self)
        self._lock = False
        Operation.__init__(self, name=name, description=description)
        if kwargs:
            raise ValueError(
                "Unrecognized keyword arguments "
                f"passed to {self.__class__.__name__}: {kwargs}"
            )
        self._path = None  # Will be determined in `build_wrapper`
        self.built = False
        self._input_spec = None
        self._called = False

        self._trainable = trainable
        self._rewards = []
        self._reward_ids = set()
        self._rewards_override = []

        self._call_signature = inspect.signature(self.call)
        call_signature_parameters = [
            p.name for p in self._call_signature.parameters.values()
        ]
        self._call_has_training_arg = "training" in call_signature_parameters
        # Whether to automatically convert inputs to `call()`.
        self._convert_input_args = True
        # Whether to allow non-json object as positional arguments in `call()`.
        self._allow_non_json_data_model_positional_args = False
        # Dict of schemas that were used to call `build()`.
        self._build_schemas_dict = None
        # Parent path
        self._parent_path = None
        self._hooks = HookList(
            hooks=hooks,
            module=self,
        )
        self._initialize_tracker()

    @tracking.no_automatic_dependency_tracking
    def _initialize_tracker(self):
        if hasattr(self, "_tracker"):
            return

        trainable_variables = []
        non_trainable_variables = []
        modules = []
        metrics = []
        self._tracker = tracking.Tracker(
            {
                "trainable_variables": (
                    lambda x: isinstance(x, backend.Variable) and x.trainable,
                    trainable_variables,
                ),
                "non_trainable_variables": (
                    lambda x: isinstance(x, backend.Variable) and not x.trainable,
                    non_trainable_variables,
                ),
                "metrics": (lambda x: isinstance(x, Metric), metrics),
                "modules": (
                    lambda x: isinstance(x, Module) and not isinstance(x, Metric),
                    modules,
                ),
            },
            exclusions={"non_trainable_variables": ["trainable_variables"]},
        )

        self._trainable_variables = trainable_variables
        self._non_trainable_variables = non_trainable_variables
        self._modules = modules
        self._metrics = metrics

    @property
    def path(self):
        """The path of the module.

        If the module has not been built yet, it will be `None`.
        """
        return self._path

    @property
    def input_spec(self):
        return self._input_spec

    @input_spec.setter
    def input_spec(self, value):
        self._input_spec = value

    @classmethod
    def get_arity(cls):
        # Inspect the call method to get the number of parameters
        sig = inspect.signature(cls.call)
        # Exclude 'self' and 'training' from the parameter count
        return len(
            [
                param
                for param in sig.parameters.values()
                if param.name not in ["self", "training"]
            ]
        )

    @python_utils.default
    async def build(self, *args, **kwargs):
        self._check_super_called()
        if utils.is_default(self.build) and might_have_unbuilt_state(self):
            try:
                if isinstance(args, tuple):
                    args = list(args)
                if len(args) == 1 or backend.is_data_model(args[0]):
                    args[0] = args[0].to_symbolic_data_model()
                else:
                    args = tree.map_structure(
                        lambda x: (
                            x.to_symbolic_data_model() if backend.is_data_model(x) else x
                        ),
                        args,
                    )
                await self.__call__(*args, **kwargs)
            except Exception as e:
                warnings.warn(
                    f"`build()` was called on module '{self.name}', however "
                    "the module does not have a `build()` method implemented "
                    "and it looks like it has unbuilt state. This will cause "
                    "the module to be marked as built, despite not being "
                    "actually built, which may cause failures down the line. "
                    "Make sure to implement a proper `build()` method."
                    f"Exception encountered: ''{e}''"
                )
        self.built = True

    def _lock_state(self):
        """Prevent further state updates, called automatically in `build()`."""
        if not self._tracker.locked:
            self._tracker.lock(
                msg=(
                    "You cannot add new elements of state "
                    "(variables or sub-modules) "
                    "to a module that is already built. All state "
                    "must be created in the `__init__()` method or "
                    "in the `build()` method."
                )
            )

    def get_build_config(self):
        """Returns a dictionary with the modules's input schema.

        This method returns a config dict that can be used by
        `build_from_config(config)` to create all states (e.g. Variables and
        Lookup tables) needed by the module.

        By default, the config only contains the input schema that the module
        was built with. If you're writing a custom module that creates state in
        an unusual way, you should override this method to make sure this state
        is already created when Synalinks attempts to load its value upon model
        loading.

        Returns:
            (dict): A dict containing the input schema associated with the module.
        """
        if self._build_schemas_dict is not None:
            if len(self._build_schemas_dict) == 1:
                return {
                    "input_schema": tuple(self._build_schemas_dict.values())[0],
                }
            else:
                return {"schemas_dict": self._build_schemas_dict}

    def build_from_config(self, config):
        """Builds the module's states with the supplied config dict.

        By default, this method calls the `build()` method,
        which creates variables based on the module's input schema in the supplied
        config. If your config contains other information needed to load the
        module's state, you should override this method.

        Args:
            config (dict): Dict containing the input schema associated with this module.
        """
        if config:
            if "input_schema" in config:
                run_maybe_nested(
                    self.build(backend.SymbolicDataModel(schema=config["input_schema"]))
                )
            elif "schemas_dict" in config:
                symbolic_inputs = {}
                for key, schema in config["schemas_dict"].items():
                    symbolic_inputs[key] = backend.SymbolicDataModel(schema=schema)
                run_maybe_nested(self.build(**symbolic_inputs))
            self.built = True

    def _obj_type(self):
        return "Module"

    @property
    def metrics(self):
        """List of all metrics."""
        metrics = list(self._metrics)
        for module in self._modules:
            metrics.extend(module.metrics)
        return metrics

    @property
    def metrics_variables(self):
        """List of all metric variables."""
        vars = []
        for metric in self.metrics:
            vars.extend(metric.variables)
        return vars

    def _get_own_rewards(self):
        if backend.in_stateless_scope():
            rewards = []
            scope = backend.get_stateless_scope()
            for reward in scope.rewards:
                if id(reward) in self._reward_ids:
                    rewards.append(reward)
            return rewards
        else:
            return self._rewards[:]

    @property
    def rewards(self):
        """List of scalar rewards from `add_reward` and submodules."""
        if self._rewards_override:
            return self._rewards_override
        rewards = self._get_own_rewards()
        for module in self._flatten_modules(include_self=False):
            rewards.extend(module._get_own_rewards())
        return rewards

    def _clear_rewards(self):
        if backend.in_stateless_scope():
            scope = backend.get_stateless_scope()
            if scope.collect_rewards:
                for x in scope.rewards:
                    if id(x) in self._reward_ids:
                        scope.rewards.remove(x)
        self._rewards.clear()
        self._reward_ids.clear()
        for module in self._modules:
            module._clear_rewards()

    def add_variable(
        self,
        initializer=None,
        data_model=None,
        trainable=True,
        name=None,
        description=None,
    ):
        """Add a variable to the module

        Args:
            initializer (dict | Initializer): Initializer object to use to
                populate the initial variable value. Can be a JSON dict containing the
                initial value. If unspecified, defaults to `initializers.Empty`.
            data_model (DataModel): The DataModel used to infer the schema
                and default value.
            trainable (bool): Boolean, whether the variable should be trainable via
                optimization or whether its updates are managed manually. Defaults
                to `True`.
            name (string): String name of the variable. Useful for debugging purposes.
            description (string): String description of the variable. Used by the
                optimizers to infer the role of the variable. Required if the data
                model do not have a docstring.

        Returns:
            (Variable): The created variable
        """
        self._check_super_called()
        if initializer is None:
            initializer = initializers.Empty(data_model=data_model)
        trainable = trainable and is_trainable(data_model)
        with backend.name_scope(self.name, caller=self):
            variable = backend.Variable(
                initializer=initializer,
                data_model=data_model,
                trainable=trainable,
                name=name,
                description=description,
            )
        self._track_variable(variable)
        return variable

    @property
    def trainable(self):
        """Settable boolean, whether this module should be trainable or not."""
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        """Sets trainable attribute for the module and its submodules.

        When this value is changed during training (e.g. with a
        `Callback`) you need to call the parent
        `Program.make_train_function` with `force=True` in order to
        recompile the training graph.

        Args:
            value (bool): Boolean with the desired state for the module's trainable
                attribute.
        """
        value = bool(value)
        self._trainable = value
        for v in self._trainable_variables:
            v.trainable = value
        for module in self._modules:
            module.trainable = value

    def add_hook(self, hook):
        self._hooks.add_hook(hook)

    @property
    def variables(self):
        """List of all module state.

        Note that metrics variables are not included here, use
        `metrics_variables` to visit all the metric variables.

        Returns:
            (list): The list of the variables.
        """
        # Return all `Variables` associate with the module including metrics
        # and random seeds. Also deduplicate them.
        variables = []
        seen_ids = set()
        for v in self._trainable_variables + self._non_trainable_variables:
            if id(v) not in seen_ids:
                variables.append(v)
                seen_ids.add(id(v))
        for module in self._modules:
            for v in module.variables:
                if id(v) not in seen_ids:
                    variables.append(v)
                    seen_ids.add(id(v))
        return variables

    @property
    def trainable_variables(self):
        """List of all trainable module state.

        Returns:
            (list): The list of trainable variables.
        """
        if not self.trainable:
            return []
        return [v for v in self.variables if v.trainable]

    @property
    def non_trainable_variables(self):
        """List of all non-trainable module state.

        Returns:
            (list): The list of non-trainable variables.
        """
        if not self.trainable:
            return self.variables
        return [v for v in self.variables if not v.trainable]

    def get_variable(self, name=None, index=None):
        """Retrieves a variable based on either its name (unique) or index.

        If `name` and `index` are both provided, `index` will take precedence.
        Indices are based on order of instantiation.

        Args:
            name (string): The name of the variable.
            index (int): The index of the variable.

        Returns:
            (Variable): The returned variable.
        """
        if index is not None and name is not None:
            raise ValueError(
                "Provide only a variable name or a variable index. Received: "
                f"index={index}, name={name}."
            )
        if index is not None:
            if len(self.variables) <= index:
                raise ValueError(
                    f"Was asked to retrieve variable at index {index}"
                    f" but module only has {len(self.modules)}"
                    " variables."
                )
            else:
                return self.variables[index]

        if name is not None:
            for variable in self.variables:
                if variable.name == name:
                    return variable
            raise ValueError(
                f"No such variable: {name}. Existing variables are: "
                f"{list(variable.name for variable in self.variables)}."
            )
        raise ValueError(
            "Provide either a variable name or variable index at `get_variable`."
        )

    async def __call__(self, *args, **kwargs):
        call_id = str(uuid.uuid4())
        if self._hooks:
            self._hooks.on_call_begin(
                call_id=call_id,
                inputs=args,
            )
        self._check_super_called()
        self._called = True

        #####################################
        # 0. Convert tuple inputs to list for convenience
        if isinstance(args, tuple):
            args = list(args)

        #####################################
        # 1. Convert any DataModel positional arguments to JsonDataModel
        # This operation is performed to make the computation backend independent
        # and the making the data models dynamically modifiable

        # Used to avoid expensive `tree` operations in the most common case.
        if len(args) == 1 and backend.is_data_model(args[0]):
            args[0] = args[0].to_json_data_model(name="inputs_" + self.name)
        else:
            args = self._maybe_convert_inputs(args)

        ##########################################################
        # 2. Enforce that only JsonDataModels or SymbolicDataModel
        # can be passed positionally.
        if not self._allow_non_json_data_model_positional_args:
            for arg in tree.flatten(args):
                if not is_json_data_model_or_symbolic_data_model(arg) and arg is not None:
                    raise ValueError(
                        "Only input JsonDataModel, DataModel or SymbolicDataModel"
                        " may be passed as positional arguments. The following argument "
                        f"value should be passed as a keyword argument: {arg} "
                        f"(of type {type(arg)})"
                    )

        # Caches info about `call()` signature, args, kwargs.
        call_spec = CallSpec(self._call_signature, args, kwargs)

        ############################################
        # 3. Check input spec for 1st positional arg.
        # TODO: consider extending this to all args and kwargs.
        self._assert_input_compatibility(call_spec.first_arg)

        ################
        # 4. Call build
        with self._open_name_scope():
            await self._maybe_build(call_spec)

        ##########################
        # 5. Infer training value
        # Training phase for `Module.call` is set via (in order of priority):
        # (1) The `training` argument passed to this `Module.call`, if not None
        # (2) The training argument of an outer `Module.call`.
        # (4) Any non-None default value for `training` in the call signature
        # (5) False (treating the module as if it's in inference)

        # Maintains info about the `Module.call` stack
        # across nested calls.
        call_context = self._get_call_context()

        # This is the value explicitly passed by the user
        training = call_spec.user_arguments_dict.get("training", None)
        if training is None:
            # Wasn't passed explicitly: use context value
            training = call_context.training
            if training is None:
                # Get signature default value
                training = call_spec.arguments_dict.get("training", None)
        call_context.training = training
        if self._call_has_training_arg and training is not None:
            # Only populate arg if it has a concrete value
            kwargs["training"] = training

        ####################
        # 6. Call the module.
        try:
            with self._open_name_scope():
                outputs = await super().__call__(*args, **kwargs)

            if not self.built:
                self.built = True
        except Exception as e:
            if self._hooks:
                self._hooks.on_call_end(
                    call_id=call_id,
                    exception=str(e),
                )
            raise e
        finally:
            # Destroy call context if we created it
            self._maybe_reset_call_context()
        if self._hooks:
            self._hooks.on_call_end(
                call_id=call_id,
                outputs=outputs,
            )
        return outputs

    async def call(self, *args, **kwargs):
        raise self._not_implemented_error(self.call)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"name={self.name}, description='{self.description}', built={self.built}>"
        )

    def __str__(self):
        return self.__repr__()

    def __setattr__(self, name, value):
        # Track Variables, Modules, Metrics.
        name, value = self._setattr_hook(name, value)
        if name != "_tracker":
            if not hasattr(self, "_tracker"):
                self._initialize_tracker()
            value = self._tracker.track(value)
        return super().__setattr__(name, value)

    def __delattr__(self, name):
        obj = getattr(self, name)
        if isinstance(obj, backend.Variable):
            import gc

            # It will take a short amount of time for the corresponding buffer
            # to be actually removed from the device.
            # https://stackoverflow.com/a/74631949
            self._untrack_variable(obj)
            super().__delattr__(name)
            gc.collect()
        else:
            super().__delattr__(name)

    def _check_super_called(self):
        if getattr(self, "_lock", True):
            raise RuntimeError(
                f"In module '{self.__class__.__name__}', you forgot to call "
                "`super().__init__()` as the first statement "
                "in the `__init__()` method. Go add it!"
            )

    def _assert_input_compatibility(self, first_arg):
        # TODO perform check using schemas
        pass

    def _maybe_convert_inputs(self, inputs):
        counter = {"i": 0}

        def convert_fn(x):
            if backend.is_data_model(x):
                if counter["i"] > 0:
                    result = x.to_json_data_model(
                        name=f"{self.name}_inputs_{counter['i']}"
                    )
                else:
                    result = x.to_json_data_model(name=f"{self.name}_inputs")
                counter["i"] += 1
                return result
            return x

        return tree.map_structure(
            convert_fn,
            inputs,
        )
        return inputs

    def _get_call_context(self):
        """Returns currently active `CallContext`."""
        module_call_ctx = global_state.get_global_attribute("current_call_ctx")
        if module_call_ctx is None:
            # Enter new call context.
            module_call_ctx = CallContext(entry_module=self)
            global_state.set_global_attribute("current_call_ctx", module_call_ctx)
            self._clear_rewards()
        return module_call_ctx

    def _maybe_reset_call_context(self):
        module_call_ctx = global_state.get_global_attribute("current_call_ctx")
        if module_call_ctx is None or module_call_ctx.entry_module == self:
            global_state.set_global_attribute("current_call_ctx", None)

    def _flatten_modules(self, include_self=True, recursive=True):
        modules = []
        if include_self:
            modules.append(self)
        seen_object_ids = set()
        deque = collections.deque(self._modules)
        while deque:
            module = deque.popleft()
            if id(module) in seen_object_ids:
                continue
            seen_object_ids.add(id(module))
            modules.append(module)
            # Introspect recursively through submodules.
            if recursive:
                deque.extendleft(module._modules)
        return modules

    def _not_implemented_error(self, attr, msg=None):
        if callable(attr):
            attr_name = attr.__name__
            attr_type = "method"
        else:
            attr_name = str(attr)
            attr_type = "attribute"
        msg = " " + msg if msg is not None else ""
        return NotImplementedError(
            f"Module {self.__class__.__name__} does not have a `{attr_name}` "
            f"{attr_type} implemented.{msg}"
        )

    def _track_variable(self, variable):
        if variable.trainable:
            self._tracker.add_to_store("trainable_variables", variable)
        else:
            self._tracker.add_to_store("non_trainable_variables", variable)
        if not self.trainable:
            variable.trainable = False
        self._post_track_variable(variable)

    def _untrack_variable(self, variable):
        previous_lock_state = self._tracker.locked
        self._tracker.unlock()
        self._tracker.untrack(variable)
        if previous_lock_state is True:
            self._tracker.lock()
        self._post_untrack_variable(variable)

    @python_utils.default
    def get_config(self):
        self._check_super_called()
        base_config = super().get_config()
        config = {
            "trainable": self.trainable,
        }
        return {**base_config, **config}

    def _open_name_scope(self):
        if self._parent_path is None:
            self._parent_path = current_path()
        return backend.name_scope(self.name, caller=self)

    async def _maybe_build(self, call_spec):
        if self.built:
            return

        # If the module has a build method, call it with our input schemas.
        if not utils.is_default(self.build):
            if len(call_spec.first_arg) == 1:
                await self.build(call_spec.first_arg[0])
            else:
                await self.build(call_spec.first_arg)
            # Check input spec again (after build, since self.input_spec
            # may have been updated
            self._assert_input_compatibility(call_spec.first_arg)
            return

        # Otherwise, attempt to build the module by calling it on symbolic input.
        if might_have_unbuilt_state(self):
            try:
                if not utils.is_default(self.compute_output_spec):
                    await self.compute_output_spec(**call_spec.arguments_dict)
                else:
                    await backend.compute_output_spec(
                        self.call, **call_spec.arguments_dict
                    )
            except Exception as e:
                if call_spec.eager:
                    # Will let the actual eager call do state-building
                    return
                warnings.warn(
                    f"Module '{self.name}' looks like it has unbuilt state, but "
                    "Synalinks is not able to trace the module `call()` in order to "
                    "build it automatically. Possible causes:\n"
                    "1. The `call()` method of your module may be crashing. Try "
                    "to `__call__()` the module eagerly on some test input "
                    "first to see if it works. "
                    "2. If the `call()` method is correct, then you may need "
                    "to implement the `def build(self, inputs)` or "
                    "`def compute_output_spec(inputs, training=False)` method on "
                    "your module."
                    f"Exception encountered: ''{e}''"
                )
        self.built = True

    def clone(self, name=None):
        clone = self.from_config(self.get_config())
        if name:
            clone.name = name
        else:
            clone.name = auto_name(self.name)
        return clone


def is_json_data_model_or_symbolic_data_model(x, allow_none=False):
    if allow_none and x is None:
        return True
    return backend.is_json_data_model(x) or backend.is_symbolic_data_model(x)


class CallSpec:
    def __init__(self, signature, args, kwargs):
        # `training` is a special kwargs that is always available in
        # a module, if user specifies them in their call without adding to spec,
        # we remove them to be able to bind variables. If User is not using
        # `training` anyway so we can ignore.
        if "training" in kwargs and "training" not in signature.parameters:
            kwargs.pop("training")
            bound_args = signature.bind(*args, **kwargs)
        else:
            bound_args = signature.bind(*args, **kwargs)
        self.user_arguments_dict = {k: v for k, v in bound_args.arguments.items()}
        bound_args.apply_defaults()
        arg_dict = {}
        arg_names = []
        data_arg_dict = {}
        data_args = []
        data_arg_names = []
        nested_data_arg_names = []
        for name, value in bound_args.arguments.items():
            arg_dict[name] = value
            arg_names.append(name)
            if is_json_data_model_or_symbolic_data_model(value):
                data_args.append(value)
                data_arg_names.append(name)
                data_arg_dict[name] = value
            elif tree.is_nested(value) and len(value) > 0:
                flat_values = tree.flatten(value)
                if all(
                    is_json_data_model_or_symbolic_data_model(x, allow_none=True)
                    for x in flat_values
                ):
                    data_args.append(value)
                    data_arg_names.append(name)
                    data_arg_dict[name] = value
                    nested_data_arg_names.append(name)
                elif any(
                    is_json_data_model_or_symbolic_data_model(x) for x in flat_values
                ):
                    raise ValueError(
                        "In a nested call() argument, "
                        "you cannot mix JsonDataModels and non-JsonDataModels. "
                        "Received invalid mixed argument: "
                        f"{name}={value}"
                    )
        self.arguments_dict = arg_dict
        self.argument_names = arg_names
        self.data_arguments_dict = data_arg_dict
        self.data_arguments_names = data_arg_names
        self.nested_data_argument_names = nested_data_arg_names
        self.first_arg = arg_dict[arg_names[0]]
        if all(backend.is_json_data_model(x) for x in self.data_arguments_dict.values()):
            self.eager = True
        else:
            self.eager = False


class CallContext:
    def __init__(self, entry_module):
        self.entry_module = entry_module
        self.training = None


def might_have_unbuilt_state(module):
    return any(not module.built for module in module._modules)
