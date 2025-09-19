# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import json
import logging

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import any_symbolic_data_models
from synalinks.src.hooks.hook import Hook

_SYMBOLIC_LOG_TEMPLATE = """
\033[94m---
# {name}
Call ID: {call_id}
Module Name: {module_name}
Module Description: {module_description}
Data Model JSON Schema:
{data_model_schema}
---\033[0m
"""

_DATA_LOG_TEMPLATE = """
\033[92m---
# {name}
Call ID: {call_id}
Module Name: {module_name}
Module Description: {module_description}
Data Model JSON:
{data_model_json}
---\033[0m
"""

_EXCEPTION_TEMPLATE = """
\033[91m---
# Exception
Call ID: {call_id}
Module Name: {module_name}
Module Description: {module_description}
Exception: {exception}
---\033[0m
"""


@synalinks_export("synalinks.hooks.Logger")
class Logger(Hook):
    """Logger hook for logging module calls.

    This hook is set by default when you enables logging.

    Example:

    ```python
    import synalinks

    synalinks.enable_logging()
    ```
    """

    def _maybe_setup_logger(self):
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger(self.module.name)

    def on_call_begin(
        self,
        call_id,
        inputs=None,
    ):
        self._maybe_setup_logger()
        if not inputs:
            return
        module_name = self.module.name
        module_description = self.module.description
        flatten_inputs = tree.flatten(inputs)
        if any_symbolic_data_models(inputs):
            data_models_schemas = [
                dm.get_schema() for dm in flatten_inputs if dm is not None
            ]
            if data_models_schemas:
                self.logger.info(
                    _SYMBOLIC_LOG_TEMPLATE.format(
                        name="Symbolic Call Start",
                        call_id=call_id,
                        module_name=module_name,
                        module_description=module_description,
                        data_model_schema=json.dumps(
                            data_models_schemas,
                            indent=2,
                        ),
                    ),
                )
        else:
            data_models_jsons = [dm.get_json() for dm in flatten_inputs if dm is not None]
            if data_models_jsons:
                self.logger.info(
                    _DATA_LOG_TEMPLATE.format(
                        name="Call Start",
                        call_id=call_id,
                        module_name=module_name,
                        module_description=module_description,
                        data_model_json=json.dumps(
                            data_models_jsons,
                            indent=2,
                        ),
                    )
                )

    def on_call_end(
        self,
        call_id,
        outputs=None,
        exception=None,
    ):
        self._maybe_setup_logger()
        module_name = self.module.name
        module_description = self.module.description
        if exception:
            self.logger.error(
                _EXCEPTION_TEMPLATE.format(
                    call_id=call_id,
                    exception=exception,
                    module_name=module_name,
                    module_description=module_description,
                )
            )
        if not outputs:
            return
        flatten_outputs = tree.flatten(outputs)
        if any_symbolic_data_models(outputs):
            data_models_schemas = [
                dm.get_schema() for dm in flatten_outputs if dm is not None
            ]
            if data_models_schemas:
                self.logger.info(
                    _SYMBOLIC_LOG_TEMPLATE.format(
                        name="Symbolic Call End",
                        call_id=call_id,
                        module_name=module_name,
                        module_description=module_description,
                        data_model_schema=json.dumps(
                            data_models_schemas,
                            indent=2,
                        ),
                    ),
                )
        else:
            data_models_jsons = [
                dm.get_json() for dm in flatten_outputs if dm is not None
            ]
            if data_models_jsons:
                self.logger.info(
                    _DATA_LOG_TEMPLATE.format(
                        name="Call End",
                        call_id=call_id,
                        module_name=module_name,
                        module_description=module_description,
                        data_model_json=json.dumps(
                            data_models_jsons,
                            indent=2,
                        ),
                    )
                )
