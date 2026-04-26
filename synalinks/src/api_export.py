# Modified from: keras/src/api_export.py
# Original authors: François Chollet et al. (Keras Team)
# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

import namex

# These dicts reference "canonical names" only
# (i.e. the first name an object was registered with).
REGISTERED_NAMES_TO_OBJS = {}
REGISTERED_OBJS_TO_NAMES = {}


def register_internal_serializable(path, symbol):
    global REGISTERED_NAMES_TO_OBJS
    if isinstance(path, (list, tuple)):
        name = path[0]
    else:
        name = path
    REGISTERED_NAMES_TO_OBJS[name] = symbol
    REGISTERED_OBJS_TO_NAMES[symbol] = name


def get_symbol_from_name(name):
    return REGISTERED_NAMES_TO_OBJS.get(name, None)


def get_name_from_symbol(symbol):
    return REGISTERED_OBJS_TO_NAMES.get(symbol, None)


class synalinks_export(namex.export):
    def __init__(self, path):
        super().__init__(package="synalinks", path=path)

    def __call__(self, symbol):
        register_internal_serializable(self.path, symbol)
        return super().__call__(symbol)
