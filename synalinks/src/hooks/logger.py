# License Apache 2.0: (c) 2025 Yoan Sallami (Synalinks Team)

import logging

import orjson

from synalinks.src import tree
from synalinks.src.api_export import synalinks_export
from synalinks.src.backend import any_symbolic_data_models
from synalinks.src.hooks.hook import Hook

_SYMBOLIC_LOG_TEMPLATE = """
Call ID: {call_id}
Parent call ID: {parent_call_id}
Module: {module}
Module Name: {module_name}
Module Description: {module_description}
Data Model JSON Schema:
{data_model_schema}
"""

_DATA_LOG_TEMPLATE = """
Call ID: {call_id}
Parent call ID: {parent_call_id}
Module: {module}
Module Name: {module_name}
Module Description: {module_description}
Data Model JSON:
{data_model_json}
"""

_EXCEPTION_TEMPLATE = """
# Exception
Call ID: {call_id}
Parent call ID: {parent_call_id}
Module: {module}
Module Name: {module_name}
Module Description: {module_description}
Exception: {exception}
"""

_KWARGS_LOG_TEMPLATE = """
Call ID: {call_id}
Parent call ID: {parent_call_id}
Module: {module}
Module Name: {module_name}
Module Description: {module_description}
Keyword Arguments:
{kwargs_json}
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
            self.logger = logging.getLogger(f"synalinks.{self.module.name}")

    def _serialize_kwargs(self, kwargs):
        """Serialize kwargs to JSON-compatible format."""
        if not kwargs:
            return {}
        serialized = {}
        for key, value in kwargs.items():
            if key == "training":
                serialized[key] = value
            elif hasattr(value, "get_json"):
                serialized[key] = value.get_json()
            elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                serialized[key] = value
        return serialized

    def on_call_begin(
        self,
        call_id,
        parent_call_id=None,
        inputs=None,
        kwargs=None,
    ):
        self._maybe_setup_logger()
        module_name = self.module.name
        module_description = self.module.description
        flatten_inputs = tree.flatten(inputs)
        if any_symbolic_data_models(inputs):
            data_models_schemas = [
                dm.get_schema() for dm in flatten_inputs if dm is not None
            ]
            if data_models_schemas:
                self.logger.debug(
                    _SYMBOLIC_LOG_TEMPLATE.format(
                        name="Symbolic Call Start",
                        call_id=call_id,
                        parent_call_id=parent_call_id,
                        module=str(self.module.__class__.__name__),
                        module_name=module_name,
                        module_description=module_description,
                        data_model_schema=orjson.dumps(
                            data_models_schemas,
                            option=orjson.OPT_INDENT_2,
                        ).decode(),
                    ),
                )
        else:
            data_models_jsons = [dm.get_json() for dm in flatten_inputs if dm is not None]
            if data_models_jsons:
                self.logger.info(
                    _DATA_LOG_TEMPLATE.format(
                        name="Call Start",
                        call_id=call_id,
                        parent_call_id=parent_call_id,
                        module=str(self.module.__class__.__name__),
                        module_name=module_name,
                        module_description=module_description,
                        data_model_json=orjson.dumps(
                            data_models_jsons,
                            option=orjson.OPT_INDENT_2,
                        ).decode(),
                    )
                )
            # Log kwargs if no data models but kwargs present (e.g., Tool modules)
            elif kwargs:
                serialized_kwargs = self._serialize_kwargs(kwargs)
                # Filter out 'training' from display
                display_kwargs = {
                    k: v for k, v in serialized_kwargs.items() if k != "training"
                }
                if display_kwargs:
                    self.logger.info(
                        _KWARGS_LOG_TEMPLATE.format(
                            call_id=call_id,
                            parent_call_id=parent_call_id,
                            module=str(self.module.__class__.__name__),
                            module_name=module_name,
                            module_description=module_description,
                            kwargs_json=orjson.dumps(
                                display_kwargs,
                                option=orjson.OPT_INDENT_2,
                            ).decode(),
                        )
                    )

    def on_call_end(
        self,
        call_id,
        parent_call_id=None,
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
                    parent_call_id=parent_call_id,
                    exception=exception,
                    module=str(self.module.__class__.__name__),
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
                self.logger.debug(
                    _SYMBOLIC_LOG_TEMPLATE.format(
                        name="Symbolic Call End",
                        call_id=call_id,
                        parent_call_id=parent_call_id,
                        module=str(self.module.__class__.__name__),
                        module_name=module_name,
                        module_description=module_description,
                        data_model_schema=orjson.dumps(
                            data_models_schemas,
                            option=orjson.OPT_INDENT_2,
                        ).decode(),
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
                        parent_call_id=parent_call_id,
                        module=str(self.module.__class__.__name__),
                        module_name=module_name,
                        module_description=module_description,
                        data_model_json=orjson.dumps(
                            data_models_jsons,
                            option=orjson.OPT_INDENT_2,
                        ).decode(),
                    )
                )
