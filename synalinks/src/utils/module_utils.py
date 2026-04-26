# Modified from: keras/src/utils/module_utils.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import importlib


class LazyModule:
    def __init__(self, name, pip_name=None):
        self.name = name
        pip_name = pip_name or name
        self.pip_name = pip_name
        self.module = None
        self._available = None

    @property
    def available(self):
        if self._available is None:
            try:
                self.initialize()
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def initialize(self):
        try:
            self.module = importlib.import_module(self.name)
        except ImportError:
            raise ImportError(
                f"This requires the {self.name} module. "
                f"You can install it via `pip install {self.pip_name}`"
            )

    def __getattr__(self, name):
        if name == "_api_export_path":
            raise AttributeError
        if self.module is None:
            self.initialize()
        return getattr(self.module, name)

    def __repr__(self):
        return f"LazyModule({self.name})"


scipy = LazyModule("scipy")
matplotlib = LazyModule("matplotlib")
optree = LazyModule("optree")
