# Modified from: keras/src/ops/operation.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import inspect
import textwrap

import docstring_parser

from synalinks.src import backend
from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend.common.symbolic_data_model import any_symbolic_data_models
from synalinks.src.ops.node import Node
from synalinks.src.utils import python_utils
from synalinks.src.utils.naming import auto_name


@synalinks_export("synalinks.Operation")
class Operation:
    def __init__(self, name=None, description=None):
        if name is None:
            name = auto_name(self.__class__.__name__)
        if description is None:
            if self.__class__.__doc__:
                description = docstring_parser.parse(
                    self.__class__.__doc__
                ).short_description
            else:
                description = ""
        if not isinstance(name, str) or "/" in name:
            raise ValueError(
                "Argument `name` must be a string and "
                "cannot contain character `/`. "
                f"Received: name={name} (of type {type(name)})"
            )
        self.name = name
        self.description = description
        self._inbound_nodes = []
        self._outbound_nodes = []

    async def __call__(self, *args, **kwargs):
        if any_symbolic_data_models(args, kwargs):
            return await self.symbolic_call(*args, **kwargs)
        else:
            return await self.call(*args, **kwargs)

    async def symbolic_call(self, *args, **kwargs):
        # Perform schema inference.
        outputs = await self.compute_output_spec(*args, **kwargs)
        # Record a new node in the operations graph.
        # The Node wires itself to inbound and outbound ops.  The
        # Node constructor updates this op's self._inbound_nodes,
        # sets _synalinks_history on the outputs, and adds itself to the
        # `_outbound_nodes` of the ops that produced the inputs to this
        # call.
        Node(operation=self, call_args=args, call_kwargs=kwargs, outputs=outputs)
        return outputs

    async def call(self, *args, **kwargs):
        raise NotImplementedError

    @python_utils.default
    async def compute_output_spec(self, *args, **kwargs):
        try:
            return await backend.compute_output_spec(self.call, *args, **kwargs)
        except Exception as e:
            new_e = e.__class__(
                "Could not automatically infer the output spec of "
                f"'{self.name}' (of type {self.__class__.__name__}). "
                f"Either the `{self.__class__.__name__}.call()` method "
                f"is incorrect, or you need to implement the "
                f"`{self.__class__.__name__}.compute_output_spec()` "
                "method. "
                f"Error encountered:\n\n{e}"
            )
            raise new_e.with_traceback(e.__traceback__) from None

    def __new__(cls, *args, **kwargs):
        """We override __new__ to saving serializable constructor arguments.

        These arguments are used to auto-generate an object serialization
        config, which enables user-created subclasses to be serializable
        out of the box in most cases without forcing the user
        to manually implement `get_config()`.
        """
        instance = super(Operation, cls).__new__(cls)

        # Generate a config to be returned by default by `get_config()`.
        arg_names = inspect.getfullargspec(cls.__init__).args
        kwargs.update(dict(zip(arg_names[1 : len(args) + 1], args)))

        # For safety, we only rely on auto-configs for a small set of
        # serializable types.
        supported_types = (str, int, float, bool, type(None))
        try:
            flat_arg_values = tree.flatten(kwargs)
            auto_config = True
            for value in flat_arg_values:
                if not isinstance(value, supported_types):
                    auto_config = False
                    break
        except TypeError:
            auto_config = False
        try:
            instance._lock = False
            if auto_config:
                from synalinks.src.saving import serialization_lib

                instance._auto_config = serialization_lib.SerializableDict(**kwargs)
            else:
                instance._auto_config = None
            instance._lock = True
        except RecursionError:
            # Setting an instance attribute in __new__ has the potential
            # to trigger an infinite recursion if a subclass overrides
            # setattr in an unsafe way.
            pass
        return instance

    @python_utils.default
    def get_config(self):
        """Returns the config of the object.

        An object config is a Python dictionary (serializable)
        containing the information needed to re-instantiate it.
        """
        config = {
            "name": self.name,
            "description": self.description,
        }

        if not python_utils.is_default(self.get_config):
            # In this case the subclass implements get_config()
            return config

        # In this case the subclass doesn't implement get_config():
        # Let's see if we can autogenerate it.
        if getattr(self, "_auto_config", None) is not None:
            xtra_args = set(config.keys())
            config.update(self._auto_config.config)
            # Remove args non explicitly supported
            argspec = inspect.getfullargspec(self.__init__)
            if argspec.varkw != "kwargs":
                for key in xtra_args - xtra_args.intersection(argspec.args[1:]):
                    config.pop(key, None)
            return config
        else:
            raise NotImplementedError(
                textwrap.dedent(f"""
        Object {self.__class__.__name__} was created by passing
        non-serializable argument values in `__init__()`,
        and therefore the object must override `get_config()` in
        order to be serializable. Please implement `get_config()`.""")
            )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __repr__(self):
        return f"<Operation name={self.name} description={self.description}>"

    @property
    def input(self):
        """Retrieves the input data model(s) of a symbolic operation.

        Only returns the data model(s) corresponding to the *first time*
        the operation was called.

        Returns:
            (list | SymbolicDataModel): Input data model or list of input data models.
        """
        return self._get_node_attribute_at_index(0, "input_data_models", "input")

    @property
    def output(self):
        """Retrieves the output data model(s) of a layer.

        Only returns the data model(s) corresponding to the *first time*
        the operation was called.

        Returns:
            (list | SymbolicDataModel): Output data model or list of output data models.
        """
        return self._get_node_attribute_at_index(0, "output_data_models", "output")

    def _get_node_attribute_at_index(self, node_index, attr, attr_name):
        """Private utility to retrieves an attribute (e.g. inputs) from a node.

        This is used to implement the properties:
        - output
        - input

        Args:
            node_index: Integer index of the node from which
                to retrieve the attribute.
            attr: Exact node attribute name.
            attr_name: Human-readable attribute name, for error messages.

        Returns:
            The operation's attribute `attr` at the node of index `node_index`.
        """
        if not self._inbound_nodes:
            raise AttributeError(
                f"The module {self.name} has never been called "
                f"and thus has no defined {attr_name}."
            )
        if not len(self._inbound_nodes) > node_index:
            raise ValueError(
                f"Asked to get {attr_name} at node "
                f"{node_index}, but the operation has only "
                f"{len(self._inbound_nodes)} inbound nodes."
            )
        values = getattr(self._inbound_nodes[node_index], attr)
        if isinstance(values, list) and len(values) == 1:
            return values[0]
        else:
            return values

    # Hooks for backend layer classes
    def _post_build(self):
        """Can be overridden for per backend post build actions."""
        pass

    def _setattr_hook(self, name, value):
        """Can be overridden for per backend post build actions."""
        return name, value

    def _post_track_variable(self, variable):
        """Can be overridden for per backend post track actions."""
        pass

    def _post_untrack_variable(self, variable):
        """Can be overridden for per backend post untrack actions."""
        pass
