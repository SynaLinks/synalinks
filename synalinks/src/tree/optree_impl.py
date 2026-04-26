# Modified from: keras/src/tree/optree_impl.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import optree
import optree.utils


def register_tree_node_class(cls):
    return optree.register_pytree_node_class(cls, namespace="synalinks")


def is_nested(structure):
    return not optree.tree_is_leaf(structure, none_is_leaf=True, namespace="synalinks")


def traverse(func, structure, top_down=True):
    # From https://github.com/google/jax/pull/19695
    def traverse_children():
        children, treedef = optree.tree_flatten(
            structure,
            is_leaf=lambda x: x is not structure,
            none_is_leaf=True,
            namespace="synalinks",
        )
        if treedef.num_nodes == 1 and treedef.num_leaves == 1:
            return structure
        else:
            return optree.tree_unflatten(
                treedef,
                [traverse(func, c, top_down=top_down) for c in children],
            )

    if top_down:
        ret = func(structure)
        if ret is None:
            return traverse_children()
    else:
        traversed_structure = traverse_children()
        ret = func(traversed_structure)
        if ret is None:
            return traversed_structure
    # Detect MAP_TO_NONE without tree_api import to avoid circular import.
    if isinstance(ret, type) and ret.__name__ == "MAP_TO_NONE":
        return None
    return ret


def flatten(structure):
    # optree.tree_flatten returns a pair (leaves, treespec) where the first
    # element is a list of leaf values and the second element is a treespec
    # representing the structure of the pytree.
    leaves, _ = optree.tree_flatten(structure, none_is_leaf=True, namespace="synalinks")
    return leaves


def map_structure(func, *structures):
    if not structures:
        raise ValueError("Must provide at least one structure")

    # Add check for same structures, otherwise optree just maps to shallowest.
    def func_with_check(*args):
        if not all(
            optree.tree_is_leaf(s, none_is_leaf=True, namespace="synalinks") for s in args
        ):
            raise ValueError("Structures don't have the same nested structure.")
        return func(*args)

    map_func = func_with_check if len(structures) > 1 else func

    return optree.tree_map(
        map_func, *structures, none_is_leaf=True, namespace="synalinks"
    )


def assert_same_structure(a, b):
    def check(a_leaf, b_leaf):
        if not optree.tree_is_leaf(
            a_leaf, none_is_leaf=True, namespace="synalinks"
        ) or not optree.tree_is_leaf(b_leaf, none_is_leaf=True, namespace="synalinks"):
            raise ValueError("Structures don't have the same nested structure.")
        return None

    optree.tree_map(check, a, b, none_is_leaf=True, namespace="synalinks")


def assert_same_paths(a, b):
    a_paths = set(optree.tree_paths(a, none_is_leaf=True, namespace="synalinks"))
    b_paths = set(optree.tree_paths(b, none_is_leaf=True, namespace="synalinks"))

    if a_paths != b_paths:
        msg = "`a` and `b` don't have the same paths."
        a_diff = a_paths.difference(b_paths)
        if a_diff:
            msg += f"\nPaths in `a` missing in `b`:\n{a_diff}"
        b_diff = b_paths.difference(a_paths)
        if b_diff:
            msg += f"\nPaths in `b` missing in `a`:\n{b_diff}"
        raise ValueError(msg)


def pack_sequence_as(structure, flat_sequence):
    _, treespec = optree.tree_flatten(structure, none_is_leaf=True, namespace="synalinks")
    return optree.tree_unflatten(treespec, flat_sequence)


def lists_to_tuples(structure):
    def list_to_tuple(instance):
        return tuple(instance) if isinstance(instance, list) else None

    return traverse(list_to_tuple, structure, top_down=False)
