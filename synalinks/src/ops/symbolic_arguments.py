# Modified from: keras/src/ops/symbolic_arguments.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src import tree
from synalinks.src.backend import SymbolicDataModel


class SymbolicArguments:
    def __init__(self, *args, **kwargs):
        self.args = tree.map_structure(lambda x: x, args)
        self.kwargs = tree.map_structure(lambda x: x, kwargs)
        self._flat_arguments = tree.flatten((self.args, self.kwargs))

        # Used to avoid expensive `tree` operations in the most common case.
        if (
            not self.kwargs
            and len(self.args) == 1
            and isinstance(self.args[0], SymbolicDataModel)
        ):
            self._single_positional_data_model = self.args[0]
        else:
            self._single_positional_data_model = None

        self.symbolic_data_models = []
        for arg in self._flat_arguments:
            if isinstance(arg, SymbolicDataModel):
                self.symbolic_data_models.append(arg)

    def convert(self, conversion_fn):
        args = tree.map_structure(conversion_fn, self.args)
        kwargs = tree.map_structure(conversion_fn, self.kwargs)
        return args, kwargs

    def fill_in(self, data_model_dict):
        """Maps SymbolicDataModels to computed values using `data_model_dict`.

        `data_model_dict` maps `SymbolicDataModel` instances to their current values.
        """
        if self._single_positional_data_model is not None:
            # Performance optimization for most common case.
            # Approx. 70x faster.
            return (data_model_dict[id(self._single_positional_data_model)],), {}

        def switch_fn(x):
            if isinstance(x, SymbolicDataModel):
                return data_model_dict.get(id(x), None)
            return x

        return self.convert(switch_fn)
