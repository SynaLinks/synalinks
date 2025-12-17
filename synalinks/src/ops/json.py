# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import JsonDataModel
from synalinks.src.backend import SymbolicDataModel
from synalinks.src.backend import any_symbolic_data_models
from synalinks.src.backend import concatenate_json
from synalinks.src.backend import concatenate_schema
from synalinks.src.backend import factorize_json
from synalinks.src.backend import factorize_schema
from synalinks.src.backend import in_mask_json
from synalinks.src.backend import in_mask_schema
from synalinks.src.backend import out_mask_json
from synalinks.src.backend import out_mask_schema
from synalinks.src.backend import prefix_json
from synalinks.src.backend import prefix_schema
from synalinks.src.backend import suffix_json
from synalinks.src.backend import suffix_schema
from synalinks.src.ops.operation import Operation


class Concat(Operation):
    """Concatenate two data models together."""

    def __init__(
        self,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
        )

    async def call(self, x1, x2):
        if not x1:
            raise ValueError(f"Received x1={x1} and x2={x2}")
        if not x2:
            raise ValueError(f"Received x1={x1} and x2={x2}")
        json = concatenate_json(x1.get_json(), x2.get_json())
        schema = concatenate_schema(x1.get_schema(), x2.get_schema())
        return JsonDataModel(json=json, schema=schema, name=self.name)

    async def compute_output_spec(self, x1, x2):
        schema = concatenate_schema(x1.get_schema(), x2.get_schema())
        return SymbolicDataModel(schema=schema, name=self.name)


@synalinks_export(
    [
        "synalinks.ops.concat",
        "synalinks.ops.concatenate",
        "synalinks.ops.json.concat",
        "synalinks.ops.json.concatenate",
    ]
)
async def concat(x1, x2, name=None, description=None):
    """Concatenate two data models together.

    Concatenation consist in creating a new data model containing
    all the elements of the two inputs into a new one.
    Each field name is made unique if needed by adding a numerical suffix `_n`,
    with `n` being an incremental integer.

    This operation is implemented in the `+` Python operator.

    If any of the data models used is a metaclass or symbolic data model
    the output is a symbolic data model.

    If any of the inputs is None, then an exception is raised.

    If the keys are used more than once, a numerical suffix is added.

    Table:

    | `x1`   | `x2`   | Concat (`+`)      |
    | ------ | ------ | ----------------- |
    | `x1`   | `x2`   | `x1 + x2`         |
    | `x1`   | `None` | `Exception`       |
    | `None` | `x2`   | `Exception`       |
    | `None` | `None` | `Exception`       |

    Args:
        x1 (JsonDataModel | SymbolicDataModel): the first input data model.
        x2 (JsonDataModel | SymbolicDataModel): the second input data model.
        name (str): Optional. The name of the operation.
        description (str): Optional. The description of the operation.

    Returns:
        (JsonDataModel | SymbolicDataModel): The resulting data model
    """
    if any_symbolic_data_models(x1, x2):
        return await Concat(
            name=name,
            description=description,
        ).symbolic_call(x1, x2)
    return await Concat(
        name=name,
        description=description,
    )(x1, x2)


class And(Operation):
    """Perform a logical `And` operation between data models."""

    def __init__(
        self,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
        )

    async def call(self, x1, x2):
        if x1 and x2:
            json = concatenate_json(x1.get_json(), x2.get_json())
            schema = concatenate_schema(x1.get_schema(), x2.get_schema())
            return JsonDataModel(json=json, schema=schema, name=self.name)
        elif x1 and not x2:
            return None
        elif not x1 and x2:
            return None
        else:
            return None

    async def compute_output_spec(self, x1, x2):
        schema = concatenate_schema(x1.get_schema(), x2.get_schema())
        return SymbolicDataModel(schema=schema, name=self.name)


@synalinks_export(["synalinks.ops.logical_and", "synalinks.ops.json.logical_and"])
async def logical_and(x1, x2, name=None, description=None):
    """Perform a logical `And` operation between two data models.

    If one of the inputs is `None`, then this operation output `None`.
    If both inputs are provided, the output is a concatenation
    of the two given data models.

    This operation is implemented in the Python `&` operator.

    If any of the data models used is a metaclass or symbolic data model
    the output is a symbolic data model.

    Table:

    | `x1`   | `x2`   | Logical And (`&`) |
    | ------ | ------ | ----------------- |
    | `x1`   | `x2`   | `x1 + x2`         |
    | `x1`   | `None` | `None`            |
    | `None` | `x2`   | `None`            |
    | `None` | `None` | `None`            |

    Args:
        x1 (JsonDataModel | SymbolicDataModel): The first input data model.
        x2 (JsonDataModel | SymbolicDataModel): The second input data model.
        name (str): Optional. The name of the operation.
        description (str): Optional. The description of the operation.

    Returns:
        (JsonDataModel | SymbolicDataModel | None): The resulting data model or
            None if the condition is not met.
    """
    if any_symbolic_data_models(x1, x2):
        return await And(
            name=name,
            description=description,
        ).symbolic_call(x1, x2)
    return await And(
        name=name,
        description=description,
    )(x1, x2)


class Or(Operation):
    """Perform a logical `Or` operation between data models."""

    def __init__(
        self,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
        )

    async def call(self, x1, x2):
        if x1 and x2:
            json = concatenate_json(x1.get_json(), x2.get_json())
            schema = concatenate_schema(x1.get_schema(), x2.get_schema())
            return JsonDataModel(json=json, schema=schema, name=self.name)
        elif x1 and not x2:
            return JsonDataModel(
                json=x1.get_json(), schema=x1.get_schema(), name=self.name
            )
        elif not x1 and x2:
            return JsonDataModel(
                json=x2.get_json(), schema=x2.get_schema(), name=self.name
            )
        else:
            return None

    async def compute_output_spec(self, x1, x2):
        return SymbolicDataModel(schema=x1.get_schema(), name=self.name)


@synalinks_export(["synalinks.ops.logical_or", "synalinks.ops.json.logical_or"])
async def logical_or(x1, x2, name=None, description=None):
    """Perform a logical `Or` between two data models.

    If one of the input is `None`, then output the other one.
    If both inputs are provided, the output is a concatenation
    of the two given data models.

    If any of the data models used is a metaclass or symbolic data model
    the output is a symbolic data model.

    This operation is implemented in the Python `|` operator.

    Table:

    | `x1`   | `x2`   | Logical Or (`|`) |
    | ------ | ------ | ---------------- |
    | `x1`   | `x2`   | `x1 + x2`        |
    | `x1`   | `None` | `x1`             |
    | `None` | `x2`   | `x2`             |
    | `None` | `None` | `None`           |

    Args:
        x1 (JsonDataModel | SymbolicDataModel): The first input data model.
        x2 (JsonDataModel | SymbolicDataModel): The second input data model.
        name (str): Optional. The name of the operation.
        description (str): Optional. The description of the operation.

    Returns:
        (JsonDataModel | SymbolicDataModel | None): The resulting data model or
            None if the condition is not met.
    """
    if any_symbolic_data_models(x1, x2):
        return await Or(
            name=name,
            description=description,
        ).symbolic_call(x1, x2)
    return await Or(
        name=name,
        description=description,
    )(x1, x2)


class Xor(Operation):
    """Perform a logical `Xor` operation between data models."""

    def __init__(
        self,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
        )

    async def call(self, x1, x2):
        if x1 and x2:
            return None
        elif x1 and not x2:
            return JsonDataModel(
                json=x1.get_json(), schema=x1.get_schema(), name=self.name
            )
        elif not x1 and x2:
            return JsonDataModel(
                json=x2.get_json(), schema=x2.get_schema(), name=self.name
            )
        else:
            return None

    async def compute_output_spec(self, x1, x2):
        return SymbolicDataModel(schema=x1.get_schema(), name=self.name)


@synalinks_export(["synalinks.ops.logical_xor", "synalinks.ops.json.logical_xor"])
async def logical_xor(x1, x2, name=None, description=None):
    """Perform a logical `Xor` between two data models.

    If one of the input is `None`, then output the other one.
    If both inputs are provided, the output is `None`.

    If any of the data models used is a metaclass or symbolic data model
    the output is a symbolic data model.

    This operation is implemented in the Python `^` operator.

    Table:

    | `x1`   | `x2`   | Logical Xor (`^`)|
    | ------ | ------ | ---------------- |
    | `x1`   | `x2`   | `None`           |
    | `x1`   | `None` | `x1`             |
    | `None` | `x2`   | `x2`             |
    | `None` | `None` | `None`           |

    Args:
        x1 (JsonDataModel | SymbolicDataModel): The first input data model.
        x2 (JsonDataModel | SymbolicDataModel): The second input data model.
        name (str): Optional. The name of the operation.
        description (str): Optional. The description of the operation.

    Returns:
        (JsonDataModel | SymbolicDataModel | None): The resulting data model or
            None if the condition is not met.
    """
    if any_symbolic_data_models(x1, x2):
        return await Xor(
            name=name,
            description=description,
        ).symbolic_call(x1, x2)
    return await Xor(
        name=name,
        description=description,
    )(x1, x2)


class Factorize(Operation):
    """Factorize a data model by grouping similar properties together."""

    def __init__(
        self,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
        )

    async def call(self, x):
        if not x:
            return None
        json = factorize_json(x.get_json())
        schema = factorize_schema(x.get_schema())
        return JsonDataModel(json=json, schema=schema, name=self.name)

    async def compute_output_spec(self, x):
        schema = factorize_schema(x.get_schema())
        return SymbolicDataModel(schema=schema, name=self.name)


@synalinks_export(["synalinks.ops.factorize", "synalinks.ops.json.factorize"])
async def factorize(x, name=None, description=None):
    """Factorize a data model by grouping similar properties together.

    Factorization consist in grouping the same properties into lists.
    The property key of the resulting grouped property is changed to its plural form.
    For example `action` become `actions`, or `query` become `queries`.

    If the data models used is a metaclass or symbolic data model
    the output is a symbolic data model.

    This operation is implemented in `.factorize()`

    Args:
        x (JsonDataModel | SymbolicDataModel): the input data model.
        name (str): Optional. The name of the operation.
        description (str): Optional. The description of the operation.

    Returns:
        (JsonDataModel | SymbolicDataModel): The resulting data model.
    """
    if any_symbolic_data_models(x):
        return await Factorize(
            name=name,
            description=description,
        ).symbolic_call(x)
    return await Factorize(
        name=name,
        description=description,
    )(x)


class OutMask(Operation):
    """Mask specific fields of a data model."""

    def __init__(
        self,
        mask=None,
        recursive=True,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
        )
        self.mask = mask
        self.recursive = recursive

    async def call(self, x):
        if not x:
            return None
        json = out_mask_json(x.get_json(), mask=self.mask, recursive=self.recursive)
        schema = out_mask_schema(x.get_schema(), mask=self.mask, recursive=self.recursive)
        return JsonDataModel(json=json, schema=schema, name=self.name)

    async def compute_output_spec(self, x):
        schema = out_mask_schema(x.get_schema(), mask=self.mask, recursive=self.recursive)
        return SymbolicDataModel(schema=schema, name=self.name)


@synalinks_export(["synalinks.ops.out_mask", "synalinks.ops.json.out_mask"])
async def out_mask(x, mask=None, recursive=True, name=None, description=None):
    """Mask specific fields of a data model.

    Out masking consist in removing the properties that match with the keys given
    in the mask. The masking process ignore the numerical suffixes that could be added
    by other operations.

    If the data models used is a metaclass or symbolic data model
    the output is a symbolic data model.

    Args:
        x (JsonDataModel | SymbolicDataModel): the input data model.
        mask (list): the input mask (list of keys).
        recursive (bool): Whether or not to remove
            recursively for nested objects (default True).
        name (str): Optional. The name of the operation.
        description (str): Optional. The description of the operation.

    Returns:
        (JsonDataModel | SymbolicDataModel): The resulting data model.
    """
    if mask is None:
        raise ValueError("You should specify the `mask` argument")
    if any_symbolic_data_models(x):
        return await OutMask(
            mask=mask,
            recursive=recursive,
            name=name,
            description=description,
        ).symbolic_call(x)
    return await OutMask(
        mask=mask,
        recursive=recursive,
        name=name,
        description=description,
    )(x)


class InMask(Operation):
    """Keep specific fields of a data model."""

    def __init__(
        self,
        mask=None,
        recursive=True,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
        )
        self.mask = mask
        self.recursive = recursive

    async def call(self, x):
        if not x:
            return None
        json = in_mask_json(x.get_json(), mask=self.mask, recursive=self.recursive)
        schema = in_mask_schema(x.get_schema(), mask=self.mask, recursive=self.recursive)
        return JsonDataModel(json=json, schema=schema, name=self.name)

    async def compute_output_spec(self, x):
        schema = in_mask_schema(x.get_schema(), mask=self.mask, recursive=self.recursive)
        return SymbolicDataModel(schema=schema, name=self.name)


@synalinks_export(["synalinks.ops.in_mask", "synalinks.ops.json.in_mask"])
async def in_mask(x, mask=None, recursive=True, name=None, description=None):
    """Keep specific fields of a data model.

    In masking consists in keeping the properties that match with the keys given
    in the mask. The masking process ignores the numerical suffixes that could be added
    by other operations.

    If the data models used is a metaclass or symbolic data model
    the output is a symbolic data model.

    Args:
        x (JsonDataModel | SymbolicDataModel): the input data model
        mask (list): the input mask (list of keys)
        recursive (bool): Whether or not to keep
            recursively for nested objects (default True).
        name (str): Optional. The name of the operation.
        description (str): Optional. The description of the operation.

    Returns:
        (JsonDataModel | SymbolicDataModel): The resulting data model.
    """
    if mask is None:
        raise ValueError("You should specify the `mask` argument")
    if any_symbolic_data_models(x):
        return await InMask(
            mask=mask,
            recursive=recursive,
            name=name,
            description=description,
        ).symbolic_call(x)
    return await InMask(
        mask=mask,
        recursive=recursive,
        name=name,
        description=description,
    )(x)


class Prefix(Operation):
    """Add a prefix to **all** the data model fields (non-recursive)."""

    def __init__(
        self,
        prefix=None,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
        )
        self.prefix = prefix

    async def call(self, x):
        if not x:
            return None
        json = prefix_json(x.get_json(), self.prefix)
        schema = prefix_schema(x.get_schema(), self.prefix)
        return JsonDataModel(json=json, schema=schema, name=self.name)

    async def compute_output_spec(self, x):
        schema = prefix_schema(x.get_schema(), self.prefix)
        return SymbolicDataModel(schema=schema, name=self.name)


@synalinks_export(["synalinks.ops.prefix", "synalinks.ops.json.prefix"])
async def prefix(x, prefix=None, name=None, description=None):
    """Add a prefix to **all** the data model fields (non-recursive).

    If the data models used is a metaclass or symbolic data model
    the output is a symbolic data model.

    Args:
        x (JsonDataModel | SymbolicDataModel): the input data model
        prefix (str): the prefix to add.
        name (str): Optional. The name of the operation.
        description (str): Optional. The description of the operation.

    Returns:
        (JsonDataModel | SymbolicDataModel): The resulting data model.
    """
    if prefix is None:
        raise ValueError("You should specify the `prefix` argument")
    if any_symbolic_data_models(x):
        return await Prefix(
            prefix=prefix,
            name=name,
            description=description,
        ).symbolic_call(x)
    return await Prefix(
        prefix=prefix,
        name=name,
        description=description,
    )(x)


class Suffix(Operation):
    """Add a suffix to **all** the data model fields (non-recursive)."""

    def __init__(
        self,
        suffix=None,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
        )
        self.suffix = suffix

    async def call(self, x):
        if not x:
            return None
        json = suffix_json(x.get_json(), self.suffix)
        schema = suffix_schema(x.get_schema(), self.suffix)
        return JsonDataModel(json=json, schema=schema, name=self.name)

    async def compute_output_spec(self, x):
        schema = suffix_schema(x.get_schema(), self.suffix)
        return SymbolicDataModel(schema=schema, name=self.name)


@synalinks_export(["synalinks.ops.suffix", "synalinks.ops.json.suffix"])
async def suffix(x, suffix=None, name=None, description=None):
    """Add a suffix to **all** the data model fields (non-recursive).

    If the data models used is a metaclass or symbolic data model
    the output is a symbolic data model.

    Args:
        x (JsonDataModel | SymbolicDataModel): the input data model
        suffix (str): the suffix to add.
        name (str): Optional. The name of the operation.
        description (str): Optional. The description of the operation.

    Returns:
        (JsonDataModel | SymbolicDataModel): The resulting data model
    """
    if suffix is None:
        raise ValueError("You should specify the `suffix` argument")
    if any_symbolic_data_models(x):
        return await Suffix(
            suffix=suffix,
            name=name,
            description=description,
        ).symbolic_call(x)
    return await Suffix(
        suffix=suffix,
        name=name,
        description=description,
    )(x)


class Not(Operation):
    """Negation/invert operator to be used in logical flows.
    
    When used the output is always `None`.
    """
    def __init__(
        self,
        name=None,
        description=None,
    ):
        super().__init__(
            name=name,
            description=description,
        )

    async def call(self, x):
        return None

    async def compute_output_spec(self, x):
        return SymbolicDataModel(schema=x.get_schema(), name=self.name)


@synalinks_export(["synalinks.ops.logical_not", "synalinks.ops.json.logical_not"])
async def logical_not(x, name=None, description=None):
    """Negation/invert operator to be used in logical flows.
    
    When used the output is always `None`.
    
    If the data models used is a metaclass or symbolic data model
    the output is a symbolic data model with the same schema than the input.
    
    Args:
        x (JsonDataModel | SymbolicDataModel): the input data model
        name (str): Optional. The name of the operation.
        description (str): Optional. The description of the operation.

    Returns:
        (`None` | SymbolicDataModel): The resulting data model
    """
    if any_symbolic_data_models(x):
        return await Not(
            name=name,
            description=description,
        ).symbolic_call(x)
    return await Not(
        name=name,
        description=description,
    )(x)