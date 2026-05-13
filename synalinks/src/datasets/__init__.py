# License Apache 2.0: (c) 2025-2026 Yoan Sallami (Synalinks Team)
"""Re-export built-in datasets at the top level.

Each module under ``built_in/`` is exposed on the public API as
``synalinks.datasets.<name>``, so callers can use any of these forms:

```python
import synalinks

(x_train, y_train), (x_test, y_test) = synalinks.datasets.gsm8k.load_data()

# Or import the module directly:
from synalinks.datasets import gsm8k
from synalinks.datasets.gsm8k import load_data
```
"""

import sys

from synalinks.src.datasets.built_in import arc_challenge as arc_challenge
from synalinks.src.datasets.built_in import arcagi as arcagi
from synalinks.src.datasets.built_in import arcagi1_tasks as arcagi1_tasks
from synalinks.src.datasets.built_in import bbh as bbh
from synalinks.src.datasets.built_in import bbq as bbq
from synalinks.src.datasets.built_in import boolq as boolq
from synalinks.src.datasets.built_in import drop as drop
from synalinks.src.datasets.built_in import gsm8k as gsm8k
from synalinks.src.datasets.built_in import hellaswag as hellaswag
from synalinks.src.datasets.built_in import hotpotqa as hotpotqa
from synalinks.src.datasets.built_in import humaneval as humaneval
from synalinks.src.datasets.built_in import ifeval as ifeval
from synalinks.src.datasets.built_in import lambada as lambada
from synalinks.src.datasets.built_in import logiqa as logiqa
from synalinks.src.datasets.built_in import mmlu as mmlu
from synalinks.src.datasets.built_in import squad as squad
from synalinks.src.datasets.built_in import truthfulqa as truthfulqa
from synalinks.src.datasets.built_in import winogrande as winogrande

# Alias each built-in module under the top-level ``synalinks.src.datasets``
# package so ``import synalinks.src.datasets.<name>`` and
# ``from synalinks.src.datasets.<name> import ...`` resolve without the
# ``built_in`` segment.
_BUILT_IN = (
    "arc_challenge",
    "arcagi",
    "arcagi1_tasks",
    "bbh",
    "bbq",
    "boolq",
    "drop",
    "gsm8k",
    "hellaswag",
    "hotpotqa",
    "humaneval",
    "ifeval",
    "lambada",
    "logiqa",
    "mmlu",
    "squad",
    "truthfulqa",
    "winogrande",
)
for _name in _BUILT_IN:
    sys.modules[f"synalinks.src.datasets.{_name}"] = sys.modules[
        f"synalinks.src.datasets.built_in.{_name}"
    ]
del _BUILT_IN, _name
