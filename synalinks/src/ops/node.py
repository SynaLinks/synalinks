# Modified from: keras/src/ops/node.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import collections

from synalinks.src import tree
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.ops.symbolic_arguments import SymbolicArguments


class Node:
    """A `Node` describes an operation `__call__()` event.

    A Synalinks Function is a DAG with `Node` instances as nodes, and
    `SymbolicDataModel` instances as edges. Nodes aren't `Operation` instances,
    because a single operation could be called multiple times, which would
    result in graph cycles.

    A `__call__()` event involves input data_models (and other input arguments),
    the operation that was called, and the resulting output data_models.
    A `Node` will include all this information.

    Since a single `Operation` could be called multiple times,
    the `Node` instances are stored on operations as a list.
    Each time an operation is called, a node is added to `op._inbound_nodes`.
    Each time the output of an operation is used by another operation,
    a node is added to `op._outbound_nodes`.

    Every `SymbolicDataModel` instance has a `synalinksHistory` object attached,
    which tracks the `Node` that records the `__call__()` event that created
    the data_model. By recursively walking through `Node` instances
    via the `KerasHistory` metadata of `SymbolicDataModel` instances, once can
    retrieve the entire DAG of a synalinks Function.

    Args:
        operation (Operation): The Operation that was called in the `op.__call__()`
            event that this node represents.
        call_arg (positional arguments): The positional arguments the operation was
            called with.
        call_kwargs (keyword arguments): The keyword arguments the operation was
            called with.
        outputs (list): The output data models of the `op.__call__()` call.
    """

    def __init__(self, operation, call_args=None, call_kwargs=None, outputs=None):
        self.operation = operation
        self.arguments = SymbolicArguments(*call_args, **call_kwargs)
        self.outputs = [] if outputs is None else tree.flatten(outputs)
        for x in self.outputs:
            if not isinstance(x, SymbolicDataModel):
                raise ValueError(
                    "All operation outputs must be data_models. "
                    f"Operation {operation} returned a non-data_model. "
                    f"Non-data_model received: {x}"
                )

        zero_history = any(
            not x.record_history for x in self.arguments.symbolic_data_models
        )

        # If inputs don't have metadata yet, add it.
        if not zero_history:
            for data_model in self.arguments.symbolic_data_models:
                if not hasattr(data_model, "_synalinks_history"):
                    data_model._synalinks_history = SynalinksHistory(
                        operation=None, node_index=0, data_model_index=0
                    )

        # Wire up Node to Operations.
        self.operation._inbound_nodes.append(self)
        for dt in self.arguments.symbolic_data_models:
            inbound_op = dt._synalinks_history.operation
            if inbound_op is not None:  # It's a graph entry point.
                inbound_op._outbound_nodes.append(self)

        # Set metadata on outputs.
        if not zero_history:
            node_index = len(self.operation._inbound_nodes) - 1
            for i, data_model in enumerate(self.outputs):
                data_model._synalinks_history = SynalinksHistory(
                    operation=operation,
                    node_index=node_index,
                    data_model_index=i,
                )

        # Whether this is a root node.
        self.is_input = not self.arguments.symbolic_data_models

    def __repr__(self):
        if self.operation.description:
            operation_str = f"({self.operation.name}, '{self.operation.description}')"
        else:
            operation_str = f"{self.operation.name}"
        return f"<Node operation={operation_str}, id={id(self)}>"

    @property
    def input_data_models(self):
        return self.arguments.symbolic_data_models

    @property
    def output_data_models(self):
        return self.outputs

    @property
    def parent_nodes(self):
        """The parent `Node`s.

        Returns:
            all the `Node`s whose output this node immediately depends on.
        """
        node_deps = []
        for dt in self.arguments.symbolic_data_models:
            op = dt._synalinks_history.operation
            node_index = dt._synalinks_history.node_index
            if op is not None:  # `None` for `Input` data_models.
                node_deps.append(op._inbound_nodes[node_index])
        return node_deps


class SynalinksHistory(
    collections.namedtuple(
        "SynalinksHistory", ["operation", "node_index", "data_model_index"]
    )
):
    """Tracks the Operation call that created a DataModel.

    During construction of synalinks Functions, this metadata is added to
    each DataModel produced as the output of an Operation.
    This allows synalinks to track how each DataModel was produced, and
    this information is later retraced by the `Function` class to
    reconstruct the Operations graph.

    Attributes:
      operation: The Operation instance that produced the DataModel.
      node_index: The specific call to the Operation that produced this DataModel.
        Operations can be called multiple times in order to share weights. A new
        node is created every time an Operation is called. The corresponding
        node that represents the call event that produced the DataModel can be
        found at `op._inbound_nodes[node_index]`.
      data_model_index: The output index for this DataModel.
        Always zero if the Operation that produced this DataModel
        only has one output.
    """

    # Added to maintain memory and performance characteristics of `namedtuple`
    # while subclassing.
    __slots__ = ()


def is_symbolic_data_model(obj):
    return hasattr(obj, "_synalinks_history")
