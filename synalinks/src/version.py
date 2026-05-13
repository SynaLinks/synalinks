# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)

from synalinks.src.api_export import synalinks_export

# Unique source of truth for the version number.
__version__ = "0.8.030"


@synalinks_export("synalinks.version")
def version():
    return __version__
