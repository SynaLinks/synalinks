# Modified from: keras/src/models/functional.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import copy
import inspect
import typing
import warnings

from synalinks.src import backend
from synalinks.src import tree
from synalinks.src.modules import Input
from synalinks.src.modules import InputModule
from synalinks.src.modules import Module
from synalinks.src.ops.function import Function
from synalinks.src.ops.function import _build_map
from synalinks.src.ops.function import make_node_key
from synalinks.src.ops.node import Node
from synalinks.src.ops.node import SynalinksHistory
from synalinks.src.programs.program import Program
from synalinks.src.saving.serialization_lib import deserialize_synalinks_object
from synalinks.src.saving.serialization_lib import serialize_synalinks_object
from synalinks.src.utils import tracking
from synalinks.src.utils.async_utils import run_maybe_nested


class Functional(Function, Program):
    """A `Functional` program is a `Program` defined as a directed graph of modules.

    Three types of `Program` exist: subclassed `Program`, `Functional` program,
    and `Sequential` (a special case of `Functional`).

    A `Functional` program can be instantiated by passing two arguments to
    `__init__()`. The first argument is the `synalinks.Input` objects
    that represent the inputs to the model.
    The second argument specifies the output data_models that represent
    the outputs of this program. Both arguments can be a nested structure
    of data_models.
    """

    def __new__(cls, *args, **kwargs):
        return typing.cast(cls, super().__new__(cls))

    @tracking.no_automatic_dependency_tracking
    def __init__(self, inputs, outputs, name=None, description=None, **kwargs):
        if isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, backend.SymbolicDataModel) and k != v.name:
                    warnings.warn(
                        "When providing `inputs` as a dict, all keys in the "
                        "dict must match the names of the corresponding "
                        f"data_models. Received key '{k}' mapping to value {v} "
                        f"which has name '{v.name}'. Change the data_model name to "
                        f"'{k}' (via `Input(..., name='{k}')`)"
                    )

        trainable = kwargs.pop("trainable", None)
        flat_inputs = tree.flatten(inputs)
        flat_outputs = tree.flatten(outputs)
        for x in flat_inputs:
            if not isinstance(x, backend.SymbolicDataModel):
                raise ValueError(
                    "All `inputs` values must be SymbolicDataModels. Received: "
                    f"inputs={inputs} including invalid value {x} of "
                    f"type {type(x)}"
                )
        for x in flat_outputs:
            if not isinstance(x, backend.SymbolicDataModel):
                raise ValueError(
                    "All `outputs` values must be SymbolicDataModels. Received: "
                    f"outputs={outputs} including invalid value {x} of "
                    f"type {type(x)}"
                )

        if not all(is_input_symbolic_data_model(t) for t in flat_inputs):
            inputs, outputs = clone_graph_nodes(inputs, outputs)

        Function.__init__(self, inputs, outputs, name=name, description=description)

        if trainable is not None:
            self.trainable = trainable

        self._modules = self.modules
        self.built = True
        self._convert_input_args = False
        self._allow_non_data_model_positional_args = True
        output_modules = [x._synalinks_history[0] for x in self.outputs]
        self.output_names = [x.name for x in output_modules]

    def _lock_state(self):
        # Unlike other modules, we allow Functional state to be mutable after
        # build. E.g. to attach a module to a model that is not part of the
        # functional DAG.
        pass

    def _obj_type(self):
        return "Functional"

    @property
    def modules(self):
        modules = []
        for operation in self._operations:
            if isinstance(operation, Module):
                modules.append(operation)
        return modules

    @modules.setter
    def modules(self, _):
        raise AttributeError(
            "`Program.modules` attribute is reserved and should not be used. "
            "Please use another name."
        )

    async def call(self, inputs, training=None):
        # Add support for training
        inputs = self._standardize_inputs(inputs)
        outputs = await self._run_through_graph(
            inputs, operation_fn=lambda op: operation_fn(op, training=training)
        )
        return unpack_singleton(outputs)

    async def compute_output_spec(self, inputs, training=None):
        # From Function
        return await super().compute_output_spec(inputs)

    async def build(self, inputs):
        self.built = True

    @property
    def input_schema(self):
        input_schemas = tree.map_structure(lambda x: x.get_schema(), self.inputs)
        if isinstance(input_schemas, (list, tuple)) and len(input_schemas) == 1:
            return input_schemas[0]
        return input_schemas

    @property
    def output_schema(self):
        output_schemas = tree.map_structure(lambda x: x.get_schema(), self.outputs)
        if isinstance(output_schemas, (list, tuple)) and len(output_schemas) == 1:
            return output_schemas[0]
        return output_schemas

    def _assert_input_compatibility(self, *args):
        return super(Program, self)._assert_input_compatibility(*args)

    def _maybe_warn_inputs_struct_mismatch(self, inputs, raise_exception=False):
        try:
            # We first normalize to tuples before performing the check to
            # suppress warnings when encountering mismatched tuples and lists.
            tree.assert_same_structure(
                tree.lists_to_tuples(inputs),
                tree.lists_to_tuples(self._inputs_struct),
            )
        except Exception:
            model_inputs_struct = tree.map_structure(
                lambda x: x.name, self._inputs_struct
            )
            inputs_struct = tree.map_structure(lambda x: f"{x}", inputs)
            msg = (
                "The structure of `inputs` doesn't match the expected "
                f"structure.\nExpected: {model_inputs_struct}\n"
                f"Received: inputs={inputs_struct}"
            )
            if raise_exception:
                raise ValueError(msg)
            warnings.warn(msg)

    def _convert_inputs_to_json_data_models(self, flat_inputs):
        converted = []
        for x, symb_input in zip(flat_inputs, self._inputs):
            if x is None:  # TODO: check if optional
                converted.append(x)
            else:
                if backend.is_data_model(x):
                    converted.append(
                        backend.JsonDataModel(
                            value=x.json(), schema=symb_input.get_schema()
                        )
                    )
                else:
                    converted.append(x)
        return converted

    def _standardize_inputs(self, inputs):
        raise_exception = False
        if isinstance(inputs, dict) and not isinstance(self._inputs_struct, dict):
            # This is to avoid warning
            # when we have reconciable dict/list structs
            if hasattr(self._inputs_struct, "__len__") and all(
                isinstance(i, backend.SymbolicDataModel) for i in self._inputs_struct
            ):
                expected_keys = set(i.name for i in self._inputs_struct)
                keys = set(inputs.keys())
                if expected_keys.issubset(keys):
                    inputs = [inputs[i.name] for i in self._inputs_struct]
                else:
                    raise_exception = True
            elif isinstance(self._inputs_struct, backend.SymbolicDataModel):
                if self._inputs_struct.name in inputs:
                    inputs = [inputs[self._inputs_struct.name]]
                else:
                    raise_exception = True
            else:
                raise_exception = True

        self._maybe_warn_inputs_struct_mismatch(inputs, raise_exception=raise_exception)

        flat_inputs = tree.flatten(inputs)
        flat_inputs = self._convert_inputs_to_json_data_models(flat_inputs)
        return flat_inputs

    @property
    def input(self):
        return self._inputs_struct

    @property
    def output(self):
        return self._outputs_struct

    def get_config(self):
        if not functional_like_constructor(self.__class__):
            # Subclassed networks are not serializable
            # (unless serialization is implemented by
            # the author of the subclassed network).
            return Program.get_config(self)

        config = {
            "name": self.name,
            "trainable": self.trainable,
        }
        # Build a map from a module unique name (make_node_key)
        # to the index of the nodes that are saved in the config.
        # Only nodes in network_nodes are saved.
        node_reindexing_map = {}
        for operation in self.operations:
            if issubclass(operation.__class__, Functional):
                # Functional models start with a pre-existing node
                # linking their input to output.
                kept_nodes = 1
            else:
                kept_nodes = 0
            for original_node_index, node in enumerate(operation._inbound_nodes):
                node_key = make_node_key(operation, original_node_index)
                if node_key in self._nodes:
                    # i.e. we mark it to be saved
                    node_reindexing_map[node_key] = kept_nodes
                    kept_nodes += 1

        # serialize and save the modules in module_configs
        module_configs = []
        for operation in self.operations:  # From the earliest modules on.
            filtered_inbound_nodes = []
            for original_node_index, node in enumerate(operation._inbound_nodes):
                node_key = make_node_key(operation, original_node_index)
                if node_key in self._nodes:
                    # The node is relevant to the model:
                    # add to filtered_inbound_nodes.
                    node_data = serialize_node(node, own_nodes=self._nodes)
                    if node_data is not None:
                        filtered_inbound_nodes.append(node_data)

            serialize_obj_fn = serialize_synalinks_object
            module_config = serialize_obj_fn(operation)
            module_config["name"] = operation.name
            module_config["inbound_nodes"] = filtered_inbound_nodes
            module_configs.append(module_config)
        config["modules"] = module_configs

        # Gather info about inputs and outputs.
        def get_data_model_config(data_model):
            operation = data_model._synalinks_history[0]
            node_index = data_model._synalinks_history[1]
            data_model_index = data_model._synalinks_history[2]
            node_key = make_node_key(operation, node_index)
            assert node_key in self._nodes
            new_node_index = node_reindexing_map[node_key]
            return [operation.name, new_node_index, data_model_index]

        def map_data_models(data_models):
            if isinstance(data_models, backend.SymbolicDataModel):
                return [get_data_model_config(data_models)]
            return tree.map_structure(get_data_model_config, data_models)

        config["input_modules"] = map_data_models(self._inputs_struct)
        config["output_modules"] = map_data_models(self._outputs_struct)
        return copy.deepcopy(config)


def functional_from_config(cls, config, custom_objects=None):
    """Instantiates a Functional program from its config (from `get_config()`).

    Args:
        cls: Class of the program, e.g. a custom subclass of `Program`.
        config: Output of `get_config()` for the original model instance.
        custom_objects: Optional dict of custom objects.

    Returns:
        An instance of `cls`.
    """
    # Module instances created during
    # the graph reconstruction process
    created_modules = {}

    # Dictionary mapping module instances to
    # node data that specifies a module call.
    # It acts as a queue that maintains any unprocessed
    # module call until it becomes possible to process it
    # (i.e. until the input data models to the call all exist).
    unprocessed_nodes = {}

    def add_unprocessed_node(module, node_data):
        """Add node to module list

        Arg:
            module: module object
            node_data: Node data specifying module call
        """
        if module not in unprocessed_nodes:
            unprocessed_nodes[module] = [node_data]
        else:
            unprocessed_nodes[module].append(node_data)

    def process_node(module, node_data):
        """Reconstruct node by linking to inbound modules

        Args:
            module: Module to process
            node_data: List of module configs
        """
        args, kwargs = deserialize_node(node_data, created_modules)
        # Call module on its inputs, thus creating the node
        # and building the module if needed.
        run_maybe_nested(module(*args, **kwargs))

    def process_module(module_data):
        """Deserializes a module and index its inbound nodes.

        Args:
            module_data: module config dict.
        """
        module_name = module_data["name"]

        # Instantiate module.
        module = deserialize_synalinks_object(module_data, custom_objects=custom_objects)
        created_modules[module_name] = module

        # Gather module inputs.
        inbound_nodes_data = module_data["inbound_nodes"]
        for node_data in inbound_nodes_data:
            # We don't process nodes (i.e. make module calls)
            # on the fly because the inbound node may not yet exist,
            # in case of module shared at different topological depths
            # (e.g. a model such as A(B(A(B(x)))))
            add_unprocessed_node(module, node_data)

    # Extract config used to instantiate Functional model from the config. The
    # remaining config will be passed as keyword arguments to the Model
    # constructor.
    functional_config = {}
    for key in ["modules", "input_modules", "output_modules"]:
        functional_config[key] = config.pop(key)
    for key in ["name", "trainable"]:
        if key in config:
            functional_config[key] = config.pop(key)
        else:
            functional_config[key] = None

    # First, we create all modules and enqueue nodes to be processed
    for module_data in functional_config["modules"]:
        process_module(module_data)

    # Then we process nodes in order of module depth.
    # Nodes that cannot yet be processed (if the inbound node
    # does not yet exist) are re-enqueued, and the process
    # is repeated until all nodes are processed.
    while unprocessed_nodes:
        for module_data in functional_config["modules"]:
            module = created_modules[module_data["name"]]

            # Process all nodes in module, if not yet processed
            if module in unprocessed_nodes:
                node_data_list = unprocessed_nodes[module]

                # Process nodes in order
                node_index = 0
                while node_index < len(node_data_list):
                    node_data = node_data_list[node_index]
                    try:
                        process_node(module, node_data)

                    # If the node does not have all inbound modules
                    # available, stop processing and continue later
                    except IndexError:
                        break

                    node_index += 1

                # If not all nodes processed then store unprocessed nodes
                if node_index < len(node_data_list):
                    unprocessed_nodes[module] = node_data_list[node_index:]
                # If all nodes processed remove the module
                else:
                    del unprocessed_nodes[module]

    # Create list of input and output data models and return new class
    name = functional_config["name"]
    trainable = functional_config["trainable"]

    def get_data_model(module_name, node_index, data_model_index):
        assert module_name in created_modules
        module = created_modules[module_name]
        if isinstance(module, Functional):
            # Functional models start out with a built-in node.
            node_index -= 1
        module_output_data_models = module._inbound_nodes[node_index].output_data_models
        return module_output_data_models[data_model_index]

    def map_data_models(data_models):
        if (
            isinstance(data_models, list)
            and len(data_models) == 3
            and isinstance(data_models[0], str)
        ):
            # Leaf
            return get_data_model(*data_models)
        if isinstance(data_models, dict):
            return {k: map_data_models(v) for k, v in data_models.items()}
        if isinstance(data_models, tuple):
            return tuple([map_data_models(v) for v in data_models])
        return [map_data_models(v) for v in data_models]

    input_data_models = map_data_models(functional_config["input_modules"])
    output_data_models = map_data_models(functional_config["output_modules"])
    if isinstance(input_data_models, list) and len(input_data_models) == 1:
        input_data_models = input_data_models[0]
    if isinstance(output_data_models, list) and len(output_data_models) == 1:
        output_data_models = output_data_models[0]

    return cls(
        inputs=input_data_models,
        outputs=output_data_models,
        name=name,
        trainable=trainable,
        **config,
    )


def operation_fn(operation, training):
    def call(*args, **kwargs):
        if (
            hasattr(operation, "_call_has_training_arg")
            and operation._call_has_training_arg
            and training is not None
        ):
            kwargs["training"] = training
        return operation(*args, **kwargs)

    return call


def functional_like_constructor(cls):
    init_args = inspect.getfullargspec(cls.__init__).args[1:]
    functional_init_args = inspect.getfullargspec(Functional.__init__).args[1:]
    if init_args == functional_init_args:
        return True
    return False


def unpack_singleton(x):
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    return x


def serialize_node(node, own_nodes=()):
    if not node.input_data_models:
        # Does not need to be serialized.
        return

    def serialize_symbolic_datamodel(x):
        # Serialize SymbolicDataModel while converting
        # node indices to only include nodes relevant to `own_nodes`.
        if isinstance(x, backend.SymbolicDataModel):
            operation, node_index, data_model_index = x._synalinks_history
            irrelevant_node_count = 0
            for i, node in enumerate(operation._inbound_nodes[:node_index]):
                node_key = make_node_key(operation, i)
                if node_key not in own_nodes:
                    irrelevant_node_count += 1
            x._synalinks_history = SynalinksHistory(
                operation, node_index - irrelevant_node_count, data_model_index
            )
            serialized = serialize_synalinks_object(x)
            x._synalinks_history = SynalinksHistory(
                operation, node_index, data_model_index
            )
            return serialized
        return x

    args = node.arguments.args
    kwargs = node.arguments.kwargs

    args = tree.map_structure(serialize_symbolic_datamodel, args)
    kwargs = tree.map_structure(serialize_symbolic_datamodel, kwargs)
    return {
        "args": serialize_synalinks_object(args),
        "kwargs": serialize_synalinks_object(kwargs),
    }


def deserialize_node(node_data, created_modules):
    """Return (args, kwargs) for calling the node module."""
    if not node_data:
        return [], {}

    if isinstance(node_data, list):
        # Legacy case.
        input_data_models = []
        for input_data in node_data:
            inbound_module_name = input_data[0]
            inbound_node_index = input_data[1]
            inbound_data_model_index = input_data[2]
            if len(input_data) == 3:
                kwargs = {}
            elif len(input_data) == 4:
                kwargs = input_data[3]
            else:
                raise ValueError("Cannot deserialize the program (invalid config data?)")
            inbound_module = created_modules[inbound_module_name]

            # Raise an error if the corresponding module node
            # has not yet been created
            if len(inbound_module._inbound_nodes) <= inbound_node_index:
                raise IndexError(
                    "Module node index out of bounds.\n"
                    f"inbound_module = {inbound_module}\n"
                    "inbound_module._inbound_nodes = "
                    f"{inbound_module._inbound_nodes}\n"
                    f"inbound_node_index = {inbound_node_index}"
                )
            inbound_node = inbound_module._inbound_nodes[inbound_node_index]
            input_data_models.append(
                inbound_node.output_data_models[inbound_data_model_index]
            )
        return [unpack_singleton(input_data_models)], kwargs

    args = deserialize_synalinks_object(node_data["args"])
    kwargs = deserialize_synalinks_object(node_data["kwargs"])

    def convert_revived_data_model(x):
        if isinstance(x, backend.SymbolicDataModel):
            history = x._pre_serialization_synalinks_history
            if history is None:
                return x
            module = created_modules.get(history[0], None)
            if module is None:
                raise ValueError(f"Unknown module: {history[0]}")
            inbound_node_index = history[1]
            inbound_data_model_index = history[2]
            if len(module._inbound_nodes) <= inbound_node_index:
                raise IndexError(
                    "Module node index out of bounds.\n"
                    f"inbound_module = {module}\n"
                    f"inbound_module._inbound_nodes = {module._inbound_nodes}\n"
                    f"inbound_node_index = {inbound_node_index}"
                )
            inbound_node = module._inbound_nodes[inbound_node_index]
            return inbound_node.output_data_models[inbound_data_model_index]
        return x

    args = tree.map_structure(convert_revived_data_model, args)
    kwargs = tree.map_structure(convert_revived_data_model, kwargs)
    return args, kwargs


def is_input_symbolic_data_model(x):
    (
        operation,
        node_index,
        _,
    ) = x._synalinks_history
    node = operation._inbound_nodes[node_index]
    return node.is_input


def clone_single_symbolic_data_model(x):
    return backend.SymbolicDataModel(schema=x.get_schema(), name=x.name + "_clone")


def clone_symbolic_data_models(data_models, sd_id_mapping):
    def swap(x):
        if not isinstance(x, backend.SymbolicDataModel):
            return x
        if id(x) in sd_id_mapping:
            return sd_id_mapping[id(x)]
        new_x = clone_single_symbolic_data_model(x)
        sd_id_mapping[id(x)] = new_x
        return new_x

    return tree.map_structure(swap, data_models)


def find_nodes_by_inputs_and_outputs(inputs, outputs):
    nodes, _ = _build_map(inputs, outputs)
    return nodes


def clone_graph_nodes(inputs, outputs):
    """Clone the `Node` between the inputs and output data_models.

    This function is used to create a new functional program from any intermediate
    symbolic data models. The clone of the nodes mimic the behavior of reconstructing
    the functional graph network by re-executing all the `__call__()` methods.
    The cloned nodes will be appended to the modules.

    Note that a new `synalinks.Input` will be created for any items in the
    `inputs`

    Args:
    inputs: A nested structure of `SymbolicDataModel` instances.
    outputs: A nested structure of `SymbolicDataModel` instances.

    Returns:
        A pair of inputs and outputs, with cloned `SymbolicDataModel` instances.
        They can be used to create a new functional program.
    """
    nodes_to_clone = find_nodes_by_inputs_and_outputs(inputs, outputs)
    cloned_inputs = []
    cloned_outputs = []
    # We not only need to create copies of Nodes (mimic the calls), also need to
    # clone symbolic data models to avoid the override of _synalinks_history attached on
    # the symbolic data model. The following dict is used to track any synalinks
    # data_model we cloned. The key is the string ID of the original synalinks
    # data_model, and value is the cloned symbolic data model instance.
    sd_id_mapping = {}
    op_id_mapping = {}

    for sd_input in tree.flatten(inputs):
        if is_input_symbolic_data_model(sd_input):
            # For any existing symbolic data model from synalinks.Input, leave them as is.
            cloned_inputs.append(sd_input)
            sd_id_mapping[id(sd_input)] = sd_input
        else:
            # We need to create a new symbolic data model for any intermediate data_model
            cloned_input = Input(
                schema=sd_input.get_schema(),
                name=sd_input.name + "CLONE",
            )
            cloned_inputs.append(cloned_input)
            sd_id_mapping[id(sd_input)] = cloned_input
            op_id_mapping[id(sd_input._synalinks_history[0])] = (
                cloned_input._synalinks_history[0]
            )
    cloned_inputs = tree.pack_sequence_as(inputs, cloned_inputs)

    for sd_output in tree.flatten(outputs):
        cpy = clone_single_symbolic_data_model(sd_output)
        # We reuse the _synalinks_history here, which contains the old information.
        cpy._synalinks_history = sd_output._synalinks_history
        cloned_outputs.append(cpy)
        sd_id_mapping[id(sd_output)] = cpy
    cloned_outputs = tree.pack_sequence_as(outputs, cloned_outputs)

    for node in nodes_to_clone:
        if id(node.operation) in op_id_mapping:
            operation = op_id_mapping[id(node.operation)]
        else:
            operation = node.operation
        # Clone any symbolic data model to avoid override of _synalinks_history
        # Or reuse an existing symbolic data model if it has already been cloned.
        output_copy = clone_symbolic_data_models(node.output_data_models, sd_id_mapping)
        if not isinstance(operation, InputModule):
            call_args_copy = clone_symbolic_data_models(
                node.arguments.args, sd_id_mapping
            )
            call_kwargs_copy = clone_symbolic_data_models(
                node.arguments.kwargs, sd_id_mapping
            )
        else:
            call_args_copy = ()
            call_kwargs_copy = {}
        # Creating new nodes based on the existing node information.  Node wires
        # itself to inbound and outbound modules.  The Node constructor actually
        # updates this module's self._inbound_nodes, sets _synalinks_history on the
        # outputs, and adds itself to the `_outbound_nodes` of the modules that
        # produced the inputs to this module call.
        Node(
            operation,
            call_args=call_args_copy,
            call_kwargs=call_kwargs_copy,
            outputs=output_copy,
        )
    return cloned_inputs, cloned_outputs
