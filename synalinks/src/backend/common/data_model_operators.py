# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

"""The Python operators of the data models, defined once.

`SymbolicDataModel`, `JsonDataModel`, `DataModel` and its metaclass
`MetaDataModel` share these operators; each only says how to run an op
(`_apply_op`). See the `synalinks.ops` tables for the exact semantics.
"""

from synalinks.src.utils.async_utils import run_maybe_nested


class DataModelOperators:
    """Python operators (`+`, `&`, `|`, `^`, `~`, `in`) on data models."""

    def _apply_op(self, op, *args):
        """Run the `op` operation on `args` and return its output.

        Looked up on `type(self)`, so a `DataModel` class uses the one of its
        metaclass and an instance the one of its class.

        Args:
            op (Operation): The operation to run.
            *args: The operation inputs.
        """
        raise NotImplementedError

    def __add__(self, other):
        """Concatenates this data model with another.

        Args:
            other (JsonDataModel | DataModel | SymbolicDataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel | SymbolicDataModel): The concatenated data model,
                symbolic if one of them is a metaclass or symbolic data model.
        """
        from synalinks.src import ops

        return type(self)._apply_op(self, ops.Concat(), self, other)

    def __radd__(self, other):
        """Concatenates (reverse) another data model with this one.

        Args:
            other (JsonDataModel | DataModel | SymbolicDataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel | SymbolicDataModel): The concatenated data model,
                symbolic if one of them is a metaclass or symbolic data model.
        """
        from synalinks.src import ops

        return type(self)._apply_op(self, ops.Concat(), other, self)

    def __and__(self, other):
        """Perform a `logical_and` with another data model.

        If one of them is `None`, output `None`. If both are provided,
        then concatenates this data model with the other.

        Args:
            other (JsonDataModel | DataModel | SymbolicDataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel | SymbolicDataModel | None): The concatenated data
                model or `None` (see the `logical_and` table).
        """
        from synalinks.src import ops

        return type(self)._apply_op(self, ops.And(), self, other)

    def __rand__(self, other):
        """Perform a `logical_and` (reverse) with another data model.

        If one of them is `None`, output `None`. If both are provided,
        then concatenates the other data model with this one.

        Args:
            other (JsonDataModel | DataModel | SymbolicDataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel | SymbolicDataModel | None): The concatenated data
                model or `None` (see the `logical_and` table).
        """
        from synalinks.src import ops

        return type(self)._apply_op(self, ops.And(), other, self)

    def __or__(self, other):
        """Perform a `logical_or` with another data model.

        If one of them is `None`, output the other one. If both are provided,
        then concatenates this data model with the other.

        Args:
            other (JsonDataModel | DataModel | SymbolicDataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel | SymbolicDataModel | None): The concatenation if
                both are provided, the non-`None` data model if only one is,
                `None` otherwise (see the `logical_or` table).
        """
        from synalinks.src import ops

        return type(self)._apply_op(self, ops.Or(), self, other)

    def __ror__(self, other):
        """Perform a `logical_or` (reverse) with another data model.

        If one of them is `None`, output the other one. If both are provided,
        then concatenates the other data model with this one.

        Args:
            other (JsonDataModel | DataModel | SymbolicDataModel):
                The other data model to concatenate with.

        Returns:
            (JsonDataModel | SymbolicDataModel | None): The concatenation if
                both are provided, the non-`None` data model if only one is,
                `None` otherwise (see the `logical_or` table).
        """
        from synalinks.src import ops

        return type(self)._apply_op(self, ops.Or(), other, self)

    def __xor__(self, other):
        """Perform a `logical_xor` with another data model.

        If one of them is `None`, output the other one. If both are provided,
        then the output is `None`.

        Args:
            other (JsonDataModel | DataModel | SymbolicDataModel):
                The other data model.

        Returns:
            (JsonDataModel | SymbolicDataModel | None): `None` if both are
                provided, the non-`None` data model if only one is, `None`
                otherwise (see the `logical_xor` table).
        """
        from synalinks.src import ops

        return type(self)._apply_op(self, ops.Xor(), self, other)

    def __rxor__(self, other):
        """Perform a `logical_xor` (reverse) with another data model.

        If one of them is `None`, output the other one. If both are provided,
        then the output is `None`.

        Args:
            other (JsonDataModel | DataModel | SymbolicDataModel):
                The other data model.

        Returns:
            (JsonDataModel | SymbolicDataModel | None): `None` if both are
                provided, the non-`None` data model if only one is, `None`
                otherwise (see the `logical_xor` table).
        """
        from synalinks.src import ops

        return type(self)._apply_op(self, ops.Xor(), other, self)

    def __invert__(self):
        """Perform an inversion/negation (`logical_not`).

        Returns:
            (SymbolicDataModel | None): `None` for a concrete data model, a
                symbolic data model with the same schema for a metaclass or
                symbolic data model.
        """
        from synalinks.src import ops

        return type(self)._apply_op(self, ops.Not(), self)

    def __contains__(self, other):
        """Check if the schema of `other` is contained in this one,
        or if a string key exists.

        Args:
            other (SymbolicDataModel | DataModel | str): The other data model to compare
                with, or a string key to check for in the schema properties.

        Returns:
            (bool): True if all properties of `other` are present in this one,
                or if the string key exists in the schema properties.
        """
        if isinstance(other, str):
            return other in self.get_schema().get("properties", {})
        from synalinks.src.backend.common.json_schema_utils import contains_schema

        return contains_schema(self.get_schema(), other.get_schema())


class DataModelTransforms:
    """Schema transforms (`factorize`, masks, prefix, suffix) on data models.

    Kept apart from `DataModelOperators` so a `DataModel` instance does not
    get these names, which would shadow fields of the same name.
    """

    def factorize(self):
        """Factorizes the data model.

        Returns:
            (JsonDataModel | SymbolicDataModel): The factorized data model.
        """
        from synalinks.src import ops

        return type(self)._apply_op(self, ops.Factorize(), self)

    def in_mask(self, mask=None, pattern=None, recursive=True):
        """Applies a mask to **keep only** specified keys of the data model.

        Args:
            mask (list): The mask to be applied (list of keys).
            pattern (str): Optional. A regex pattern to match keys to keep.
            recursive (bool): Optional. Whether to apply the mask recursively.
                Defaults to `True`.

        Returns:
            (JsonDataModel | SymbolicDataModel): The data model with the mask applied.
        """
        from synalinks.src import ops

        return type(self)._apply_op(
            self, ops.InMask(mask=mask, pattern=pattern, recursive=recursive), self
        )

    def out_mask(self, mask=None, pattern=None, recursive=True):
        """Applies a mask to **remove** specified keys of the data model.

        Args:
            mask (list): The mask to be applied (list of keys).
            pattern (str): Optional. A regex pattern to match keys to remove.
            recursive (bool): Optional. Whether to apply the mask recursively.
                Defaults to `True`.

        Returns:
            (JsonDataModel | SymbolicDataModel): The data model with the mask applied.
        """
        from synalinks.src import ops

        return type(self)._apply_op(
            self, ops.OutMask(mask=mask, pattern=pattern, recursive=recursive), self
        )

    def prefix(self, prefix=None):
        """Add a prefix to **all** the data model fields (non-recursive).

        Args:
            prefix (str): the prefix to add

        Returns:
            (JsonDataModel | SymbolicDataModel): The data model with the prefix added.
        """
        from synalinks.src import ops

        return type(self)._apply_op(self, ops.Prefix(prefix=prefix), self)

    def suffix(self, suffix=None):
        """Add a suffix to **all** the data model fields (non-recursive).

        Args:
            suffix (str): the suffix to add

        Returns:
            (JsonDataModel | SymbolicDataModel): The data model with the suffix added.
        """
        from synalinks.src import ops

        return type(self)._apply_op(self, ops.Suffix(suffix=suffix), self)


def apply_symbolic_op(self, op, *args):
    """Run `op` symbolically on `args` (the `_apply_op` of symbolic models)."""
    return run_maybe_nested(op.symbolic_call(*args))
