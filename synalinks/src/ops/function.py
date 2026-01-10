# Modified from: keras/src/ops/function.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import asyncio
import collections

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import is_schema_equal
from synalinks.src.ops.operation import Operation


@synalinks_export("synalinks.Function")
class Function(Operation):
    """Class that encapsulates a computation graph of operations.

    You can use a `Function` to capture the computation graph linking
    some input data_models to some output data_models, and reapply the same
    computation on new inputs.

    A `Function` is similar to a Functional Program, with the difference
    that it is stateless (it does not track state variables)
    and does not implement the `Module` API.

    Example:

    ```python
    import synalinks

    class Query(synalinks.DataModel):
        query: str

    class Answer(synalinks.DataModel):
        answer: str

    input_1 = synalinks.SymbolicDataModel(data_model=Query)
    input_2 = synalinks.SymbolicDataModel(data_model=Answer)
    output = synalinks.ops.concatenate([input_1, input_2])
    fn = synalinks.Function(inputs=[input_1, input_2], outputs=output)

    input_1_val = synalinks.JsonDataModel(
        data_model=Query(query="What is the capital of France?")
    )
    input_2_val = synalinks.JsonDataModel(data_model=Answer(answer="Paris"))
    output_val = fn([input_1_val, input_2_val])
    ```

    Args:
        inputs (SymbolicDataModel | tuple | list | dict): `SymbolicDataModel`
            instance or nested structured of `SymbolicDataModel` instances.
        outputs (SymbolicDataModel | tuple | list | dict): `SymbolicDataModel`
            instance or nested structured of `SymbolicDataModel` instances.
            They should be computable given only the values of `inputs`.
        name (str): Optional. The name of the function operation.
        description (str): Optional. The description of the function operation.
    """

    def __init__(self, inputs, outputs, name=None, description=None):
        super().__init__(name=name, description=description)

        self._inputs_struct = tree.map_structure(lambda x: x, inputs)
        self._outputs_struct = tree.map_structure(lambda x: x, outputs)
        self._inputs = tree.flatten(inputs)
        self._outputs = tree.flatten(outputs)
        if not self._inputs:
            raise ValueError(
                "`inputs` argument cannot be empty. Received:\n"
                f"inputs={inputs}\n"
                f"outputs={outputs}"
            )
        if not self._outputs:
            raise ValueError(
                "`outputs` argument cannot be empty. Received:\n"
                f"inputs={inputs}\n"
                f"outputs={outputs}"
            )

        nodes, nodes_by_depth, operations, operations_by_depth = map_graph(
            self._inputs, self._outputs
        )
        self._nodes = nodes
        self._nodes_by_depth = nodes_by_depth
        self._operations = operations
        self._operations_by_depth = operations_by_depth

    @property
    def operations(self):
        return self._operations[:]

    @property
    def inputs(self):
        """Flat list of the symbolic inputs of the Function."""
        return self._inputs

    @property
    def outputs(self):
        """Flat list of the symbolic outputs of the Function."""
        return self._outputs

    async def compute_output_spec(self, inputs):
        self._assert_input_compatibility(inputs)
        # Check if input schemas are identical to ref input schemas,
        # if so take a shortcut.
        shortcut = True
        for x, x_ref in zip(tree.flatten(inputs), self._inputs):
            if not is_schema_equal(x.get_schema(), x_ref.get_schema()):
                shortcut = False
                break
        if shortcut:
            return tree.map_structure(
                lambda x: SymbolicDataModel(schema=x.get_schema()),
                self._outputs_struct,
            )
        # No luck; take the long road through the graph.
        # Original Keras used a cache to avoid recomputing all this
        # when known input shapes where seen again. Perhaps a good
        # idea to bring that back.
        return await self._run_through_graph(
            inputs, operation_fn=lambda op: op.compute_output_spec
        )

    def compute_output_schema(self, input_schema):
        # Wrap `input_schema` into the structure of SymbolicDataModel to utilize
        # `compute_output_spec`.
        input_shape_struct = tree.map_shape_structure(
            lambda x: SymbolicDataModel(schema=x), input_schema
        )
        output_spec = self.compute_output_spec(input_shape_struct)
        return tree.map_structure(lambda x: x.get_schema(), output_spec)

    async def call(self, inputs):
        """Computes output data_models for new inputs."""
        self._assert_input_compatibility(inputs)
        return await self._run_through_graph(inputs, operation_fn=lambda op: op)

    async def _run_through_graph(self, inputs, operation_fn, call_fn=None):
        """Execute the graph.

        At each node we compute outputs via
        `operation_fn(node.operation)(*args, **kwargs)`.
        """
        inputs = tree.flatten(inputs)

        # Dictionary mapping reference data_models to computed data_models.
        data_model_dict = {}
        for x, y in zip(self.inputs, inputs):
            data_model_dict[id(x)] = y

        nodes_by_depth = self._nodes_by_depth
        depth_keys = list(nodes_by_depth.keys())
        depth_keys.sort(reverse=True)

        async def compute_node(node, operation_fn, call_fn):
            args, kwargs = node.arguments.fill_in(data_model_dict)
            op = operation_fn(node.operation)
            if call_fn is not None:
                outputs = await call_fn(op, *args, **kwargs)
            else:
                outputs = await op(*args, **kwargs)
            return outputs

        for depth in depth_keys:
            nodes = nodes_by_depth[depth]
            tasks = []

            for node in nodes:
                if not node.operation or node.is_input:
                    continue  # Input data_models already exist.

                if any(id(x) not in data_model_dict for x in node.input_data_models):
                    continue  # Node is not computable, try skipping.

                tasks.append(compute_node(node, operation_fn, call_fn))

            results = await asyncio.gather(*tasks)

            for i, node in enumerate(nodes):
                if not node.operation or node.is_input:
                    continue  # Input data_models already exist.

                if any(id(x) not in data_model_dict for x in node.input_data_models):
                    continue  # Node is not computable, try skipping.

                # Update data_model_dict.
                for x, y in zip(node.outputs, tree.flatten(results[i])):
                    data_model_dict[id(x)] = y

        output_data_models = []
        for x in self.outputs:
            try:
                output_data_models.append(data_model_dict[id(x)])
            except KeyError:
                raise KeyError(
                    f"Name conflict detected for x={x}: "
                    "Ensure that each data model have a "
                    "unique name. If it is the case, ensure that your inputs"
                    " match the program's structure"
                )

        return tree.pack_sequence_as(self._outputs_struct, output_data_models)

    def _assert_input_compatibility(self, inputs):
        try:
            tree.assert_same_structure(inputs, self._inputs_struct)
        except ValueError:
            raise ValueError(
                "Function was called with an invalid input structure. "
                f"Expected input structure: {self._inputs_struct}\n"
                f"Received input structure: {inputs}"
            )
        for x, x_ref in zip(tree.flatten(inputs), self._inputs):
            if not is_schema_equal(x.get_schema(), x_ref.get_schema()):
                raise ValueError(
                    f"{self.__class__.__name__} was passed "
                    f"incompatible inputs. For input '{x_ref.name}', "
                    f"expected schema {x_ref.get_schema()}, but received "
                    f"instead the a JsonDataModel with schema: {x.get_schema()}."
                )


def make_node_key(op, node_index):
    return str(id(op)) + "_ib-" + str(node_index)


def map_graph(inputs, outputs):
    """Validates a graph's topology and gather its operations and nodes.

    Args:
        inputs: List of input tensors.
        outputs: List of outputs tensors.

    Returns:
        A tuple `(nodes, nodes_by_depth, operations, operations_by_depth)`.
        - nodes: set of Node instances
        - nodes_by_depth: dict mapping ints (depth) to lists of node instances.
        - operations: list of Operation instances.
        - operations_by_depth: dict mapping ints (depth) to lists of Operation
            instances.
    """
    # "depth" is number of operations between output Node and the Node.
    # Nodes are ordered from inputs -> outputs.
    nodes_in_decreasing_depth, operation_indices = _build_map(inputs, outputs)
    network_nodes = {
        make_node_key(node.operation, node.operation._inbound_nodes.index(node))
        for node in nodes_in_decreasing_depth
    }

    nodes_depths = {}  # dict {node: depth value}
    operations_depths = {}  # dict {operation: depth value}

    for node in reversed(nodes_in_decreasing_depth):
        # If the depth is not set, the node has no outbound nodes (depth 0).
        depth = nodes_depths.setdefault(node, 0)

        # Update the depth of the corresponding operation
        previous_depth = operations_depths.get(node.operation, 0)
        # If we've seen this operation before at a higher depth,
        # we should use that depth instead of the node depth.
        # This is necessary for shared operations that have inputs at different
        # depth levels in the graph.
        depth = max(depth, previous_depth)
        operations_depths[node.operation] = depth
        nodes_depths[node] = depth

        # Update the depth of inbound nodes.
        # The "depth" of a node is the max of the depths
        # of all nodes it is connected to + 1.
        for node_dep in node.parent_nodes:
            previous_depth = nodes_depths.get(node_dep, 0)
            nodes_depths[node_dep] = max(depth + 1, previous_depth)

    # Handle inputs that are not connected to outputs.
    # We do not error out here because the inputs may be used to compute losses
    # and metrics.
    for input_t in inputs:
        input_operation = input_t._synalinks_history[0]
        if input_operation and input_operation not in operations_depths:
            operations_depths[input_operation] = 0
            operation_indices[input_operation] = -1
            nodes_depths[input_operation._inbound_nodes[0]] = 0
            network_nodes.add(make_node_key(input_operation, 0))

    # Build a dict {depth: list of nodes with this depth}
    nodes_by_depth = collections.defaultdict(list)
    for node, depth in nodes_depths.items():
        nodes_by_depth[depth].append(node)

    # Build a dict {depth: list of operations with this depth}
    operations_by_depth = collections.defaultdict(list)
    for operation, depth in operations_depths.items():
        operations_by_depth[depth].append(operation)

    # Get sorted list of operation depths.
    depth_keys = list(operations_by_depth.keys())
    depth_keys.sort(reverse=True)

    # Set self.operations ordered by depth.
    operations = []
    for depth in depth_keys:
        operations_for_depth = operations_by_depth[depth]
        # Network.operations needs to have a deterministic order:
        # here we order them by traversal order.
        operations_for_depth.sort(key=lambda x: operation_indices[x])
        operations.extend(operations_for_depth)

    # Get sorted list of node depths.
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    # Check that all data_models required are computable.
    # computable_data_models: all data_models in the graph
    # that can be computed from the inputs provided.
    computable_data_models = set()
    for x in inputs:
        computable_data_models.add(x)

    operations_with_complete_input = []  # To provide a better error msg.
    for depth in depth_keys:
        for node in nodes_by_depth[depth]:
            for x in tree.flatten(node.input_data_models):
                if x not in computable_data_models:
                    operation = node.operation
                    raise ValueError(
                        "Graph disconnected: cannot find parent for "
                        f"data_model {x} at operation '{operation}'. "
                        "The following previous operations were accessed "
                        f"without issue: {operations_with_complete_input}"
                    )
                operations_with_complete_input.append(node.operation.name)

            for x in tree.flatten(node.outputs):
                computable_data_models.add(x)

    # Ensure name unicity, which will be crucial for serialization
    # (since serialized nodes refer to operations by their name).
    all_names = [operation.name for operation in operations]
    for name in all_names:
        if all_names.count(name) != 1:
            raise ValueError(
                f'The name "{name}" is used {all_names.count(name)} '
                "times in the model. All operation names should be unique."
            )
    return network_nodes, nodes_by_depth, operations, operations_by_depth


def _build_map(inputs, outputs):
    """Topologically sort nodes in order from inputs to outputs.

    It uses a depth-first search to topologically sort nodes that appear in the
    _synalinks_history connectivity metadata of `outputs`.

    Args:
        outputs: the output data_models whose _synalinks_history metadata should be
                walked. This may be an arbitrary nested structure.

    Returns:
        A tuple like (ordered_nodes, operation_to_first_traversal_index)
        ordered_nodes: list of nodes appearing in the synalinks history,
            topologically sorted from original inputs to the `outputs`.
            (If outputs have different sets of ancestors, the inputs to one
            output may appear after a different output).
        operation_to_first_traversal_index:
            A dict mapping operation to the traversal index in the DFS where it
            is seen. Note: if a operation is shared by several nodes, the dict
            will onlystore the index corresponding to the *first* time the
            operation seen.
    """
    finished_nodes = set()
    nodes_in_progress = set()
    nodes_in_decreasing_depth = []  # nodes from inputs -> outputs.
    operation_indices = {}  # operation -> in traversal order.
    for output in tree.flatten(outputs):
        _build_map_helper(
            inputs,
            output,
            finished_nodes,
            nodes_in_progress,
            nodes_in_decreasing_depth,
            operation_indices,
        )
    return nodes_in_decreasing_depth, operation_indices


def _build_map_helper(
    inputs,
    data_model,
    finished_nodes,
    nodes_in_progress,
    nodes_in_decreasing_depth,
    operation_indices,
):
    """Recursive helper for `_build_map`."""
    (
        operation,
        node_index,
        _,
    ) = data_model._synalinks_history
    if not operation:
        return

    node = operation._inbound_nodes[node_index]

    # Don't repeat work for shared subgraphs
    if node in finished_nodes:
        return

    # Prevent cycles.
    if node in nodes_in_progress:
        raise ValueError(
            f"{data_model} from operation '{operation.name}' is part of a cycle."
        )

    # Store the traversal order for operation sorting.
    if operation not in operation_indices:
        operation_indices[operation] = len(operation_indices)

    # Propagate to all previous data_models connected to this node.
    nodes_in_progress.add(node)
    if not node.is_input and data_model not in tree.flatten(inputs):
        for data_model in node.input_data_models:
            _build_map_helper(
                inputs,
                data_model,
                finished_nodes,
                nodes_in_progress,
                nodes_in_decreasing_depth,
                operation_indices,
            )

    finished_nodes.add(node)
    nodes_in_progress.remove(node)
    nodes_in_decreasing_depth.append(node)
