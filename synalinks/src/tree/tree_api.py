# Modified from: keras/src/tree/tree_api.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import warnings

from synalinks.src.api_export import synalinks_export
from synalinks.src.utils.module_utils import optree

if optree.available:
    from synalinks.src.tree import optree_impl as tree_impl
else:
    from synalinks.src.tree import python_impl as tree_impl

    raise ImportError(
        "To use synalinks, you need to have `optree` installed. "
        "Install it via `pip install optree`"
    )


def register_tree_node_class(cls):
    return tree_impl.register_tree_node_class(cls)


@synalinks_export("synalinks.tree.MAP_TO_NONE")
class MAP_TO_NONE:
    """Special value for use with `traverse()`."""

    pass


@synalinks_export("synalinks.tree.is_nested")
def is_nested(structure):
    """Checks if a given structure is nested.

    Examples:

    >>> synalinks.tree.is_nested(42)
    False
    >>> synalinks.tree.is_nested({"foo": 42})
    True

    Args:
        structure: A structure to check.

    Returns:
        `True` if a given structure is nested, i.e. is a sequence, a mapping,
        or a namedtuple, and `False` otherwise.
    """
    return tree_impl.is_nested(structure)


@synalinks_export("synalinks.tree.traverse")
def traverse(func, structure, top_down=True):
    """Traverses the given nested structure, applying the given function.

    The traversal is depth-first. If `top_down` is True (default), parents
    are returned before their children (giving the option to avoid traversing
    into a sub-tree).

    Examples:

    >>> v = []
    >>> synalinks.tree.traverse(v.append, [(1, 2), [3], {"a": 4}], top_down=True)
    [(1, 2), [3], {'a': 4}]
    >>> v
    [[(1, 2), [3], {'a': 4}], (1, 2), 1, 2, [3], 3, {'a': 4}, 4]

    >>> v = []
    >>> synalinks.tree.traverse(v.append, [(1, 2), [3], {"a": 4}], top_down=False)
    [(1, 2), [3], {'a': 4}]
    >>> v
    [1, 2, (1, 2), 3, [3], 4, {'a': 4}, [(1, 2), [3], {'a': 4}]]

    Args:
        func: The function to be applied to each sub-nest of the structure.

        When traversing top-down:
            If `func(subtree) is None` the traversal continues into the
            sub-tree.
            If `func(subtree) is not None` the traversal does not continue
            into the sub-tree. The sub-tree will be replaced by `func(subtree)`
            in the returned structure (to replace the sub-tree with `None`, use
            the special value `MAP_TO_NONE`).

        When traversing bottom-up:
            If `func(subtree) is None` the traversed sub-tree is returned
            unaltered.
            If `func(subtree) is not None` the sub-tree will be replaced by
            `func(subtree)` in the returned structure (to replace the sub-tree
            with None, use the special value `MAP_TO_NONE`).

        structure: The structure to traverse.
        top_down: If True, parent structures will be visited before their
            children.

    Returns:
        The structured output from the traversal.

    Raises:
        TypeError: If `func` is not callable.
    """
    return tree_impl.traverse(func, structure, top_down=top_down)


@synalinks_export("synalinks.tree.flatten")
def flatten(structure):
    """Flattens a possibly nested structure into a list.

    In the case of dict instances, the sequence consists of the values,
    sorted by key to ensure deterministic behavior. However, instances of
    `collections.OrderedDict` are handled differently: their sequence order is
    used instead of the sorted keys. The same convention is followed in
    `pack_sequence_as`. This correctly unflattens dicts and `OrderedDict` after
    they have been flattened, or vice-versa.

    Dictionaries with non-sortable keys are not supported.

    Examples:

    >>> synalinks.tree.flatten([[1, 2, 3], [4, [5], [[6]]]])
    [1, 2, 3, 4, 5, 6]
    >>> synalinks.tree.flatten(None)
    [None]
    >>> synalinks.tree.flatten(1)
    [1]
    >>> synalinks.tree.flatten({100: 'world!', 6: 'Hello'})
    ['Hello', 'world!']

    Args:
        structure: An arbitrarily nested structure.

    Returns:
        A list, the flattened version of the input `structure`.
    """
    return tree_impl.flatten(structure)


@synalinks_export("synalinks.tree.map_structure")
def map_structure(func, *structures):
    """Maps `func` through given structures.

    Examples:

    >>> structure = [[1], [2], [3]]
    >>> synalinks.tree.map_structure(lambda v: v**2, structure)
    [[1], [4], [9]]
    >>> synalinks.tree.map_structure(lambda x, y: x * y, structure, structure)
    [[1], [4], [9]]

    >>> Foo = collections.namedtuple('Foo', ['a', 'b'])
    >>> structure = Foo(a=1, b=2)
    >>> synalinks.tree.map_structure(lambda v: v * 2, structure)
    Foo(a=2, b=4)

    Args:
        func: A callable that accepts as many arguments as there are structures.
        *structures: Arbitrarily nested structures of the same layout.

    Returns:
        A new structure with the same layout as the given ones.

    Raises:
        TypeError: If `structures` is empty or `func` is not callable.
        ValueError: If there is more than one items in `structures` and some of
            the nested structures don't match according to the rules of
            `assert_same_structure`.
    """
    return tree_impl.map_structure(func, *structures)


@synalinks_export("synalinks.tree.assert_same_structure")
def assert_same_structure(a, b, check_types=None):
    """Asserts that two structures are nested in the same way.

    This function verifies that the nested structures match. The leafs can be of
    any type. At each level, the structures must be of the same type and have
    the same number of elements. Instances of `dict`, `OrderedDict` and
    `defaultdict` are all considered the same as long as they have the same set
    of keys. However, `list`, `tuple`, `namedtuple` and `deque` are not the same
    structures. Two namedtuples with identical fields and even identical names
    are not the same structures.

    Examples:

    >>> synalinks.tree.assert_same_structure([(0, 1)], [(2, 3)])

    >>> Foo = collections.namedtuple('Foo', ['a', 'b'])
    >>> AlsoFoo = collections.namedtuple('Foo', ['a', 'b'])
    >>> synalinks.tree.assert_same_structure(Foo(0, 1), Foo(2, 3))
    >>> synalinks.tree.assert_same_structure(Foo(0, 1), AlsoFoo(2, 3))
    Traceback (most recent call last):
        ...
    ValueError: The two structures don't have the same nested structure.
    ...

    Args:
        a: an arbitrarily nested structure.
        b: an arbitrarily nested structure.
        check_types: Deprecated. The behavior of this flag was inconsistent, it
            no longer has any effect. For a looser check, use
            `assert_same_paths` instead, which considers `list`, `tuple`,
            `namedtuple` and `deque` as matching structures.

    Raises:
        ValueError: If the two structures `a` and `b` don't match.
    """
    if check_types is not None:
        if check_types:
            warnings.warn(
                "The `check_types` argument is deprecated and no longer has "
                "any effect, please remove.",
                DeprecationWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "The `check_types` argument is deprecated and no longer has "
                "any effect. For a looser check, use "
                "`synalinks.tree.assert_same_paths()`, which considers `list`, "
                "`tuple`, `namedtuple` and `deque` as matching",
                DeprecationWarning,
                stacklevel=2,
            )
    return tree_impl.assert_same_structure(a, b)


@synalinks_export("synalinks.tree.assert_same_paths")
def assert_same_paths(a, b):
    """Asserts that two structures have identical paths in their tree structure.

    This function verifies that two nested structures have the same paths.
    Unlike `assert_same_structure`, this function only checks the paths
    and ignores the collection types.
    For Sequences, to path is the index: 0, 1, 2, etc. For Mappings, the path is
    the key, for instance "a", "b", "c". Note that namedtuples also use indices
    and not field names for the path.

    Examples:
    >>> synalinks.tree.assert_same_paths([0, 1], (2, 3))
    >>> Point1 = collections.namedtuple('Point1', ['x', 'y'])
    >>> Point2 = collections.namedtuple('Point2', ['x', 'y'])
    >>> synalinks.tree.assert_same_paths(Point1(0, 1), Point2(2, 3))

    Args:
        a: an arbitrarily nested structure.
        b: an arbitrarily nested structure.

    Raises:
        ValueError: If the paths in structure `a` don't match the paths in
            structure `b`. The error message will include the specific paths
            that differ.
    """
    return tree_impl.assert_same_paths(a, b)


@synalinks_export("synalinks.tree.pack_sequence_as")
def pack_sequence_as(structure, flat_sequence):
    """Returns a given flattened sequence packed into a given structure.

    If `structure` is an atom, `flat_sequence` must be a single-item list; in
    this case the return value is `flat_sequence[0]`.

    If `structure` is or contains a dict instance, the keys will be sorted to
    pack the flat sequence in deterministic order. However, instances of
    `collections.OrderedDict` are handled differently: their sequence order is
    used instead of the sorted keys. The same convention is followed in
    `flatten`. This correctly repacks dicts and `OrderedDicts` after they have
    been flattened, or vice-versa.

    Dictionaries with non-sortable keys are not supported.

    Examples:

    >>> structure = {"key3": "", "key1": "", "key2": ""}
    >>> flat_sequence = ["value1", "value2", "value3"]
    >>> synalinks.tree.pack_sequence_as(structure, flat_sequence)
    {"key3": "value3", "key1": "value1", "key2": "value2"}

    >>> structure = (("a", "b"), ("c", "d", "e"), "f")
    >>> flat_sequence = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> synalinks.tree.pack_sequence_as(structure, flat_sequence)
    ((1.0, 2.0), (3.0, 4.0, 5.0), 6.0)

    >>> structure = {"key3": {"c": ("alpha", "beta"), "a": ("gamma")},
    ... "key1": {"e": "val1", "d": "val2"}}
    >>> flat_sequence = ["val2", "val1", 3.0, 1.0, 2.0]
    >>> synalinks.tree.pack_sequence_as(structure, flat_sequence)
    {'key3': {'c': (1.0, 2.0), 'a': 3.0}, 'key1': {'e': 'val1', 'd': 'val2'}}

    >>> structure = ["a"]
    >>> flat_sequence = [np.array([[1, 2], [3, 4]])]
    >>> synalinks.tree.pack_sequence_as(structure, flat_sequence)
    [array([[1, 2],
       [3, 4]])]

    >>> structure = ["a"]
    >>> flat_sequence = [synalinks.ops.ones([2, 2])]
    >>> synalinks.tree.pack_sequence_as(structure, flat_sequence)
    [array([[1., 1.],
       [1., 1.]]]

    Args:
        structure: Arbitrarily nested structure.
        flat_sequence: Flat sequence to pack.

    Returns:
        `flat_sequence` converted to have the same recursive structure as
        `structure`.

    Raises:
        TypeError: If `flat_sequence` is not iterable.
        ValueError: If `flat_sequence` cannot be repacked as `structure`; for
            instance, if `flat_sequence` has too few or too many elements.
    """
    return tree_impl.pack_sequence_as(structure, flat_sequence)


@synalinks_export("synalinks.tree.lists_to_tuples")
def lists_to_tuples(structure):
    """Returns the structure with list instances changed to tuples.

    Args:
        structure: Arbitrarily nested structure.

    Returns:
        The same structure but with tuples instead of lists.
    """
    return tree_impl.lists_to_tuples(structure)
